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
import triton
import triton.language as tl

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


@triton.jit
def _build_topk_ids_kernel(
    out_ptr,  # (n_total,) int32 — output topk_ids (will be unsqueezed by caller)
    row_eid_ptr,  # (n_total,) int32 — row → owning local expert id (cached template)
    row_pos_ptr,  # (n_total,) int32 — row → position within that expert's slots
    expert_num_tokens_ptr,  # (e_local,)  int32 — valid token count per local expert (from dispatch)
    n_total: tl.int32,
    BLOCK: tl.constexpr,
):
    """Fused gather + compare + select for DeepEP low-latency routing.

    Replaces the 3-launch sequence
        per_row_recv = expert_num_tokens[row_eid]   # gather
        valid_mask   = row_pos < per_row_recv       # compare
        topk_ids     = where(valid_mask, row_eid, -1)
    with a single Triton kernel — saves ~1.5 ms in the elementwise/fill bucket
    (200 layer-steps × ~7-10 µs of accumulated gather/compare/where launch
    overhead).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_total
    eid = tl.load(row_eid_ptr + offs, mask=mask, other=0)
    pos = tl.load(row_pos_ptr + offs, mask=mask, other=0)
    limit = tl.load(expert_num_tokens_ptr + eid, mask=mask, other=0)
    out = tl.where(pos < limit, eid, -1)
    tl.store(out_ptr + offs, out, mask=mask)


def _build_topk_ids(
    row_eid: torch.Tensor,
    row_pos: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    out_buf: torch.Tensor,
) -> torch.Tensor:
    """Fill ``out_buf`` (n_total,) int32 with valid expert ids / -1 sentinel."""
    n_total = row_eid.shape[0]
    BLOCK = 256
    grid = (triton.cdiv(n_total, BLOCK),)
    _build_topk_ids_kernel[grid](
        out_buf,
        row_eid,
        row_pos,
        expert_num_tokens,
        n_total,
        BLOCK=BLOCK,
        num_warps=2,
    )
    return out_buf


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

        # Per-row routing scratch — all of these are constant across forwards
        # in low-latency mode (e_local, max_recv fixed). Cache once on first
        # ``execute`` to avoid per-layer allocations / fills (~2-3 ms in the
        # elementwise/fill bucket).
        self._row_eid_template: Optional[torch.Tensor] = None  # (n_total,) int32
        self._row_pos_template: Optional[torch.Tensor] = None  # (n_total,) int32
        self._topk_weights_template: Optional[torch.Tensor] = (
            None  # (n_total, 1) fp32 ones
        )
        self._topk_ids_buf: Optional[torch.Tensor] = (
            None  # (n_total,) int32 — kernel output
        )

    def _ensure_routing_buffers(
        self, e_local: int, max_recv: int, device: torch.device
    ) -> None:
        """Lazily allocate routing scratch tensors. Constant across forwards
        in low-latency mode (e_local, max_recv fixed)."""
        n_total = e_local * max_recv
        if (
            self._row_eid_template is not None
            and self._row_eid_template.numel() == n_total
            and self._row_eid_template.device == device
        ):
            return
        row_idx = torch.arange(n_total, device=device, dtype=torch.int32)
        self._row_eid_template = row_idx // max_recv
        self._row_pos_template = row_idx % max_recv
        self._topk_weights_template = torch.ones(
            (n_total, 1), device=device, dtype=torch.float32
        )
        self._topk_ids_buf = torch.empty((n_total,), device=device, dtype=torch.int32)

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
        # so the Triton kernel skips them. Single fused Triton kernel replaces
        # the prior 4-launch sequence (gather + compare + where + fill).
        assert payload.expert_tokens_meta is not None
        expert_num_tokens = (
            payload.expert_tokens_meta.expert_num_tokens
        )  # (E_local,) int32
        self._ensure_routing_buffers(e_local, max_recv, device)
        _build_topk_ids(
            self._row_eid_template,
            self._row_pos_template,
            expert_num_tokens,
            self._topk_ids_buf,
        )
        topk_ids = self._topk_ids_buf.unsqueeze(1)
        topk_weights = self._topk_weights_template

        # DeepEP combine re-applies the *real* topk_weights on the recv side,
        # so here we want raw expert outputs — pass topk_weights = 1 and
        # apply_router_weight_on_input = False.
        out_dtype = payload.expert_x_origin_dtype or torch.bfloat16
        # Zero-copy: when the router pre-allocated a NVSHMEM-backed output
        # buffer, write the expert outputs straight into it (flat view) so
        # ``low_latency_combine(zero_copy=True)`` can reduce in-place without
        # the upfront ``memcpy128`` staging.
        out_hidden_states_buf: Optional[torch.Tensor] = None
        if payload.output_buffer is not None:
            assert payload.output_buffer.shape == (e_local, max_recv, hidden), (
                f"output_buffer shape {payload.output_buffer.shape} != "
                f"({e_local}, {max_recv}, {hidden})"
            )
            out_hidden_states_buf = payload.output_buffer.view(n_total, hidden)

        # Pass DeepEP layout metadata so fused_experts_impl can use the masked
        # 3D silu+mul+per-block-quant kernel (~28× cheaper than the per-row
        # sentinel path for sparse MoE — only valid tokens per expert are
        # processed).
        expected_m = (
            int(payload.expert_tokens_meta.expected_m)
            if payload.expert_tokens_meta is not None
            and payload.expert_tokens_meta.expected_m is not None
            else max_recv
        )
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
            out_hidden_states=out_hidden_states_buf,
            masked_m=expert_num_tokens,
            e_local=e_local,
            max_recv=max_recv,
            expected_m=expected_m,
        )

        # When zero-copy is on the fused_moe write went straight into the
        # NVSHMEM 3D buffer; return that directly (DeepEP combine asserts
        # input is 3D + contiguous, and chained ``.view().reshape()`` views
        # of NVSHMEM buffers occasionally fail the contiguity check).
        if payload.output_buffer is not None:
            packed_out = payload.output_buffer
        else:
            packed_out = out.reshape(e_local, max_recv, hidden).contiguous()
        return CombineForwardPayload(fused_expert_output=packed_out)
