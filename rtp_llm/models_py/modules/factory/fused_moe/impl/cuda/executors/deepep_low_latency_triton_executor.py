# Adapter that runs the Triton ``fused_moe_kernel`` on DeepEP low-latency
# packed input — replaces the DeepGEMM masked grouped GEMM in this path so we
# can A/B compare timeline & latency under the same dispatch/combine.
#
# Strategy:
#   1. DeepEP low_latency_dispatch outputs (E_local, max_recv, hidden) packed
#      tensors plus an ``expert_num_tokens[E_local]`` Int32 tensor on device.
#   2. Flatten the packed payload to (E_local * max_recv, hidden) and synthesise
#      a single-expert routing table where each row's expert id is its slot's
#      local expert index. Padded slots get topk_id = -1 so the Triton kernel
#      skips them (filter_expert path).
#   3. Run fused_experts_impl with top_k = 1 (no router weighting; combine will
#      re-apply weights), then reshape the output back to packed format that
#      DeepEP combine expects.
#
# Licensed under the Apache License, Version 2.0
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.triton_kernels.moe.fused_moe_triton import fused_experts_impl
from rtp_llm.utils.model_weight import W


class DeepEPLowLatencyTritonExecutor(FusedMoeExpertExecutor):
    """DeepEP low-latency packed format → Triton fused_moe_kernel adapter.

    Fits inside a CUDA graph: the unpack/pack reshapes are pure view ops, the
    mask + arange used to build the synthetic topk_ids are device-only tensor
    arithmetic with shapes determined at capture time (E_local, max_recv,
    hidden are constant across forwards in low-latency mode).
    """

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.BATCHED_TRITON

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        # Only valid when the router is DeepEP low-latency, which is enforced
        # by the strategy file rather than here.
        pass

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.ep_size = config.ep_size
        self.num_experts = config.expert_num
        assert self.num_experts % self.ep_size == 0
        self.num_local_experts = self.num_experts // self.ep_size

        self.use_fp8_w8a8 = (
            quant_config.is_quantized
            and quant_config.quant_dtype == torch.float8_e4m3fn
        )
        self.block_shape = quant_config.block_shape
        self.per_channel_quant = quant_config.is_per_act_token

        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]
        self.w13_weight_scale = weights.get(W.moe_s1, None)
        self.w2_weight_scale = weights.get(W.moe_s2, None)

        # Pre-compute the per-row local-expert id template; we mask-out invalid
        # rows (topk = -1) at runtime using ``expert_num_tokens``.
        self._row_eid_template: Optional[torch.Tensor] = None
        self._row_pos_template: Optional[torch.Tensor] = None

    def _get_row_templates(
        self, e_local: int, max_recv: int, device: torch.device
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Build (per-row local-expert id, per-row local position) once and
        cache. Both have shape (E_local * max_recv,). Stable across forwards
        since e_local and max_recv are fixed in low-latency mode.
        """
        n_total = e_local * max_recv
        if (
            self._row_eid_template is None
            or self._row_eid_template.numel() != n_total
            or self._row_eid_template.device != device
        ):
            row_idx = torch.arange(n_total, device=device, dtype=torch.int64)
            self._row_eid_template = (row_idx // max_recv).to(torch.int32)
            self._row_pos_template = row_idx % max_recv
        return self._row_eid_template, self._row_pos_template  # type: ignore[return-value]

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[Dict[str, Any]],
    ) -> CombineForwardPayload:
        expert_x = payload.expert_x
        assert (
            expert_x.dim() == 3
        ), f"DeepEP low-latency executor expects 3D packed input, got {expert_x.shape}"
        e_local, max_recv, hidden = expert_x.shape
        n_total = e_local * max_recv
        device = expert_x.device

        # Activation string normalisation (sglang style → triton kernel arg).
        act = activation.lower()
        if "silu" in act or "swiglu" in act or "siglu" in act:
            act = "silu"
        elif "gelu" in act:
            act = "gelu"
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Flatten packed → (n_total, hidden). For fp8 dispatch the payload
        # tensor is already fp8 + per-block scale; just reshape both.
        flat_x = expert_x.reshape(n_total, hidden).contiguous()
        flat_scale: Optional[torch.Tensor] = None
        if payload.expert_x_scale is not None:
            flat_scale = payload.expert_x_scale.reshape(
                n_total, payload.expert_x_scale.shape[-1]
            ).contiguous()

        # Build single-expert routing: row i → expert i//max_recv, position i%max_recv.
        # Padded slots (position >= expert_num_tokens[expert]) get topk_id = -1
        # so the Triton kernel skips them.
        row_eid, row_pos = self._get_row_templates(e_local, max_recv, device)
        assert payload.expert_tokens_meta is not None
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens  # (E_local,)
        # Gather per-row recv-count of its owning expert, then compare.
        per_row_recv = expert_num_tokens.to(torch.int64)[row_eid.to(torch.int64)]
        valid_mask = row_pos < per_row_recv  # (n_total,)
        topk_ids = torch.where(
            valid_mask,
            row_eid,
            torch.full_like(row_eid, -1),
        ).unsqueeze(
            1
        )  # (n_total, 1)
        topk_weights = torch.ones((n_total, 1), device=device, dtype=torch.float32)

        # DeepEP combine re-applies the *real* topk_weights on the recv side,
        # so here we want raw expert outputs — pass topk_weights = 1 and
        # apply_router_weight_on_input = False.
        out_dtype = payload.expert_x_origin_dtype or torch.bfloat16
        out = fused_experts_impl(
            hidden_states=flat_x,
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=act,
            apply_router_weight_on_input=False,
            use_fp8_w8a8=self.use_fp8_w8a8,
            per_channel_quant=self.per_channel_quant,
            w1_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
            a1_scale=flat_scale,
            a2_scale=a2_scale,
            block_shape=self.block_shape,
            out_dtype=out_dtype,
            filter_expert=True,  # honor topk_id == -1 sentinel for padded rows
        )

        # out: (n_total, hidden) bf16 — reshape back to packed (E_local, max_recv, hidden)
        # for DeepEP combine.
        packed_out = out.reshape(e_local, max_recv, hidden)
        return CombineForwardPayload(fused_expert_output=packed_out)
