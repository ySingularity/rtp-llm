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
from rtp_llm.models_py.triton_kernels.moe.deepep_moe_align import deepep_moe_align
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
            None  # (n_total,) int32 — _build_topk_ids output
        )
        # Persistent output buffers for ``deepep_moe_align`` (the specialized
        # moe_align fast-path). Sized to worst-case once ``e_local`` /
        # ``max_recv`` are known. fused_moe_kernel only reads slots up to
        # ``num_tokens_post_padded`` (early-return on the rest), so trailing
        # uninitialized region is fine.
        self._align_sorted_token_ids_buf: Optional[torch.Tensor] = (
            None  # (max_pad_worst,) int32
        )
        self._align_expert_ids_buf: Optional[torch.Tensor] = (
            None  # (max_num_blocks_worst,) int32
        )
        self._align_num_tokens_post_padded_buf: Optional[torch.Tensor] = (
            None  # (1,) int32
        )
        # Persistent intermediate buffers for ``fused_experts_impl``.
        # Pre-allocating these makes every per-layer ``execute()`` call
        # alloc-free, which is the *only* way to make multi-stream MoE/
        # shared-expert overlap work under CUDA Graph capture: a fresh
        # ``torch.empty`` on a side stream during capture forces PyTorch's
        # caching allocator to spin up a new private pool, doubling the
        # graph's memory footprint and OOM-ing on tight configs.
        self._intermediate_cache1_buf: Optional[torch.Tensor] = (
            None  # (n_total, N_gateup) bf16
        )
        self._intermediate_cache3_buf: Optional[torch.Tensor] = (
            None  # (n_total, 1, hidden) bf16 — down GEMM output, zeroed each call
        )
        self._out_hidden_states_buf: Optional[torch.Tensor] = (
            None  # (n_total, hidden) bf16
        )
        self._a2_q_3d_buf: Optional[torch.Tensor] = (
            None  # (E_local, max_recv, H_half) fp8
        )
        self._a2_s_3d_buf: Optional[torch.Tensor] = (
            None  # (E_local, max_recv, H_half // block_k) fp32
        )

        # Eagerly allocate the heavy shared buffers HERE, while we're still
        # in the regular (non-capture) caching-allocator pool. Otherwise
        # first-touch happens during ``execute()`` — and if ``--warm_up 0``
        # the first execute is inside CUDA Graph capture, where these large
        # allocations land in a fixed-size private pool and OOM under tight
        # configs (large model + many capture batch sizes + KV cache).
        # ``max_recv`` is derivable from config without needing a dispatch.
        try:
            from rtp_llm.models_py.distributed.deepep_wrapper import DeepepWrapperConfig

            max_recv = DeepepWrapperConfig.calc_low_latency_max_token_per_rank(
                config.ll_num_max_token, config.tp_size, config.quant_config
            )
            device = self.w13_weight.device
            self._ensure_routing_buffers(self.num_local_experts, max_recv, device)
        except Exception:  # noqa: BLE001 — best-effort; falls back to lazy init
            pass

    # ``BLOCK_SIZE_M`` for the FP8 W8A8 + per-block-quant path is hardcoded by
    # ``try_get_optimal_moe_config``. The deepep_moe_align fast-path needs
    # this value at executor init to size the output buffers, so we mirror it
    # here. ``fused_experts_impl`` asserts at runtime that the value didn't
    # drift.
    _DEEPEP_FP8_BLOCK_SIZE_M = 64

    # Class-level shared buffer pool. All MoE layers in the same model share
    # the same physical buffers because at any point only one layer is doing
    # forward — pre-allocating per-instance would waste ~200 MB × num_layers
    # ≈ several GB. Keyed by (e_local, max_recv, hidden, n_gateup, dtype_tag,
    # device_id) so a process hosting multiple models with different shapes
    # still works.
    _shared_buffers: Dict[tuple, Dict[str, torch.Tensor]] = {}

    def _ensure_routing_buffers(
        self, e_local: int, max_recv: int, device: torch.device
    ) -> None:
        """Lazily allocate routing scratch tensors. Constant across forwards
        in low-latency mode (e_local, max_recv fixed). Per-instance
        templates (row_eid, row_pos, topk_weights, topk_ids_buf) are tiny —
        keep per-layer. Heavy intermediate buffers are class-level shared
        across all MoE layers (only one layer computes at a time)."""
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

        # Heavy buffers shared across all MoE layers. These dominate memory:
        # (intermediate_cache1 ≈ 128 MB, out_hidden_states ≈ 64 MB) × 40
        # layers ≈ 8 GB if per-instance. Sharing collapses to ~200 MB total.
        N_gateup = self.w13_weight.shape[1]
        hidden = self.w13_weight.shape[2]
        H_half = N_gateup // 2
        bsm = self._DEEPEP_FP8_BLOCK_SIZE_M
        block_k = self.block_shape[1] if self.block_shape is not None else 0
        # Cache key: anything that affects buffer shape/dtype.
        dtype_tag = "fp8_block" if (self.use_fp8_w8a8 and self.block_shape) else "other"
        key = (
            e_local,
            max_recv,
            hidden,
            N_gateup,
            bsm,
            block_k,
            dtype_tag,
            device.type,
            device.index,
        )
        bufs = type(self)._shared_buffers.get(key)
        if bufs is None:
            max_pad_worst = ((n_total + e_local * bsm + bsm - 1) // bsm) * bsm
            max_num_blocks_worst = max_pad_worst // bsm
            compute_dtype = torch.bfloat16
            bufs = {
                "align_sorted_token_ids": torch.empty(
                    (max_pad_worst,), device=device, dtype=torch.int32
                ),
                "align_expert_ids": torch.empty(
                    (max_num_blocks_worst,), device=device, dtype=torch.int32
                ),
                "align_num_tokens_post_padded": torch.empty(
                    (1,), device=device, dtype=torch.int32
                ),
                "intermediate_cache1": torch.empty(
                    (n_total, N_gateup), device=device, dtype=compute_dtype
                ),
                "intermediate_cache3": torch.empty(
                    (n_total, 1, hidden), device=device, dtype=compute_dtype
                ),
                "out_hidden_states": torch.empty(
                    (n_total, hidden), device=device, dtype=compute_dtype
                ),
            }
            if self.use_fp8_w8a8 and self.block_shape is not None:
                bufs["a2_q_3d"] = torch.empty(
                    (e_local, max_recv, H_half),
                    device=device,
                    dtype=torch.float8_e4m3fn,
                )
                bufs["a2_s_3d"] = torch.empty(
                    (e_local, max_recv, H_half // block_k),
                    device=device,
                    dtype=torch.float32,
                )
            type(self)._shared_buffers[key] = bufs

        self._align_sorted_token_ids_buf = bufs["align_sorted_token_ids"]
        self._align_expert_ids_buf = bufs["align_expert_ids"]
        self._align_num_tokens_post_padded_buf = bufs["align_num_tokens_post_padded"]
        self._intermediate_cache1_buf = bufs["intermediate_cache1"]
        self._intermediate_cache3_buf = bufs["intermediate_cache3"]
        self._out_hidden_states_buf = bufs["out_hidden_states"]
        self._a2_q_3d_buf = bufs.get("a2_q_3d")
        self._a2_s_3d_buf = bufs.get("a2_s_3d")

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

        # Specialized moe_align fast-path: 2 small kernels + persistent output
        # buffers, replacing the generic ``moe_align_block_size`` (4 kernels +
        # 4 scratch fills + per-call torch.zeros/full). Preserves the
        # sparsity-prune semantics — ``num_tokens_post_padded`` is the actual
        # padded sum of ``masked_m``, so fused_moe_kernel's grid stays compact.
        deepep_moe_align(
            masked_m=expert_num_tokens,
            e_local=e_local,
            max_recv=max_recv,
            block_size_m=self._DEEPEP_FP8_BLOCK_SIZE_M,
            sorted_token_ids=self._align_sorted_token_ids_buf,
            expert_ids=self._align_expert_ids_buf,
            num_tokens_post_padded=self._align_num_tokens_post_padded_buf,
        )

        # DeepEP combine re-applies the *real* topk_weights on the recv side,
        # so here we want raw expert outputs — pass topk_weights = 1 and
        # apply_router_weight_on_input = False.
        out_dtype = payload.expert_x_origin_dtype or torch.bfloat16
        # Output buffer selection:
        #   1. NVSHMEM zero-copy combine buffer (when router supplies it)
        #   2. Otherwise: executor's pre-allocated ``_out_hidden_states_buf``,
        #      so fused_experts_impl never needs to ``torch.empty`` per call.
        out_hidden_states_buf: Optional[torch.Tensor] = None
        if payload.output_buffer is not None:
            assert payload.output_buffer.shape == (e_local, max_recv, hidden), (
                f"output_buffer shape {payload.output_buffer.shape} != "
                f"({e_local}, {max_recv}, {hidden})"
            )
            out_hidden_states_buf = payload.output_buffer.view(n_total, hidden)
        else:
            out_hidden_states_buf = self._out_hidden_states_buf

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
        # Zero shared intermediate buffers so padded rows never contain
        # stale data from a previous layer/iteration — critical under CUDA
        # graph replay where persistent buffers survive across replays.
        # intermediate_cache3: padded rows (skipped by down GEMM via
        #   token_mask) must be zero for the combine step.
        # intermediate_cache1: the gate_up GEMM only writes token_mask=True
        #   positions; under graph replay with varying expert_num_tokens,
        #   Triton's software-pipelined tl.range in the silu kernel may
        #   prefetch stale rows, and any residual garbage in unwritten
        #   positions can corrupt the quantized a2 output.
        self._intermediate_cache1_buf.zero_()
        self._intermediate_cache3_buf.zero_()
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
            # Pre-computed by deepep_moe_align above — tells
            # fused_experts_impl to skip the generic moe_align.
            sorted_token_ids=self._align_sorted_token_ids_buf,
            expert_ids=self._align_expert_ids_buf,
            num_tokens_post_padded=self._align_num_tokens_post_padded_buf,
            # Pre-allocated heavy intermediate buffers — keeps
            # fused_experts_impl alloc-free under capture.
            intermediate_cache1=self._intermediate_cache1_buf,
            intermediate_cache3=self._intermediate_cache3_buf,
            a2_q_3d=self._a2_q_3d_buf,
            a2_s_3d=self._a2_s_3d_buf,
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
