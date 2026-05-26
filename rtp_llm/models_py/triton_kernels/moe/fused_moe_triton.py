# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py
# Adapted for RTP-LLM. Only the high-perf Triton fused-MoE path (no DeepEP) is
# kept; DeepEP/TMA/swap_ab/Marlin/GPTQ-AWQ specific code paths are removed.
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    scaled_fp8_per_tensor_quant,
    scaled_fp8_per_token_quant,
    sgl_per_token_group_quant_fp8,
)

from .fused_moe_triton_config import get_config_dtype_str, try_get_optimal_moe_config
from .fused_moe_triton_kernels import (
    act_and_mul_triton,
    invoke_fused_moe_kernel,
    moe_align_block_size,
    moe_sum_reduce_triton,
)


# Small-token reduce path observed in sglang's MTP profiling timeline.
# torch.compile fuses sum + mul into a single kernel ``triton_per_fused_copy__mul_sum_0``
# that beats the dedicated triton reduce when num_tokens <= 32 (which is
# essentially always true for MTP/decode). We deliberately avoid
# ``@torch.compile`` here because Dynamo retracing is not permitted while a
# CUDA graph stream is capturing. A plain eager torch implementation matches
# the same op pattern and is graph-capture safe.
def _moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


def _quantize_input_fp8(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
    per_channel_quant: bool,
):
    """FP8 W8A8 activation quantization (reuses RTP-LLM's existing wrappers)."""
    if block_shape is None:
        if per_channel_quant:
            return scaled_fp8_per_token_quant(A, A_scale)
        return scaled_fp8_per_tensor_quant(A, A_scale)
    block_n, block_k = block_shape[0], block_shape[1]
    return sgl_per_token_group_quant_fp8(A, block_k)


def _expected_block_size_m_for_caller_align(
    use_fp8_w8a8: bool, block_shape: Optional[List[int]]
) -> int:
    """Mirror of ``try_get_optimal_moe_config``'s BLOCK_SIZE_M choice for the
    code paths where callers pre-compute moe_align outputs. Used only by the
    runtime assertion that detects config-vs-caller drift."""
    if use_fp8_w8a8 and block_shape is not None:
        # FP8 W8A8 + per-block path: BSM=64 (hardcoded in fused_moe_triton_config).
        return 64
    # Other paths: BSM=64 by default; specific BSM=16 only kicks in for the
    # ``M <= E`` no-block case which DeepEP layout doesn't hit.
    return 64


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    filter_expert: bool = True,
    no_combine: bool = False,
    out_dtype: Optional[torch.dtype] = None,
    intermediate_cache1: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    intermediate_cache3: Optional[torch.Tensor] = None,
    out_hidden_states: Optional[torch.Tensor] = None,
    align_scratch: Optional[Dict[str, torch.Tensor]] = None,
    # Pre-computed bucket + expert_count from fused recompute_topk_and_align_count.
    # When both are supplied, moe_align_block_size skips its count kernel entirely.
    pre_bucket: Optional[torch.Tensor] = None,
    pre_expert_count: Optional[torch.Tensor] = None,
    # DeepEP low-latency masked layout. When all three are supplied, the FP8
    # silu+mul+per-block-quant step routes through the masked 3D kernel
    # (one program per (expert, hidden_block, token_partition), iterates only
    # over valid tokens) instead of the per-row sentinel kernel — same path
    # the deepgemm masked executor uses (~28× fewer launches in sparse MoE).
    masked_m: Optional[torch.Tensor] = None,  # (E_local,) int32 valid tokens
    e_local: Optional[int] = None,  # number of local experts
    max_recv: Optional[int] = None,  # per-expert padded slot count
    expected_m: Optional[int] = None,  # heuristic guidance for kernel
    # DeepEP fast-path for ``moe_align_block_size``. When all three are
    # supplied (already computed by the executor's specialized
    # ``deepep_moe_align``), skip the 4 generic align kernels + 4 scratch
    # fills entirely. The supplied buffers must have the same semantics that
    # ``moe_align_block_size`` would produce (sparsity-prune preserved):
    # ``num_tokens_post_padded`` is the *actual* sum of per-expert padded
    # counts, not a worst-case constant.
    sorted_token_ids: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    num_tokens_post_padded: Optional[torch.Tensor] = None,
    # Pre-allocated FP8 quant outputs for the masked silu+mul+quant fast-path.
    # When provided, ``silu_mul_masked_fp8_post_quant_fwd`` writes into them
    # instead of fresh ``torch.empty`` allocations — required for CUDA Graph
    # multi-stream overlap (every torch.empty inside the captured graph spawns
    # a new caching-allocator pool, which doubles memory under multi-stream
    # capture and OOMs).
    #
    # Shapes:
    #   a2_q_3d:  (e_local, max_recv, N // 2)  fp8_e4m3
    #   a2_s_3d:  (e_local, max_recv, N // 2 // block_shape[1])  fp32
    a2_q_3d: Optional[torch.Tensor] = None,
    a2_s_3d: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triton fused MoE forward.

    Mirrors sglang's ``fused_experts_impl`` for the subset of features RTP-LLM
    needs today: BF16/FP16 no-quant and FP8 W8A8 (per-tensor / per-token /
    per-block). See module docstring for the dropped variants.
    """
    assert hidden_states.is_contiguous()
    assert w1.is_contiguous()
    assert w2.is_contiguous()
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape

    num_tokens = hidden_states.shape[0]
    E, N, _ = w1.shape
    topk = topk_ids.shape[1]
    # ``hidden_states.dtype`` may be FP8 when the router pre-quantized the
    # input, so it cannot be used as the compute / intermediate dtype. The
    # caller passes ``out_dtype`` (typically ``payload.expert_x_origin_dtype``)
    # which holds the original BF16/FP16 model dtype.
    effective_dtype = out_dtype if out_dtype is not None else hidden_states.dtype
    if effective_dtype == torch.float8_e4m3fn:
        effective_dtype = torch.bfloat16
    compute_type = tl.bfloat16 if effective_dtype == torch.bfloat16 else tl.float16

    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8, dtype=effective_dtype
    )
    config = try_get_optimal_moe_config(
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2]),
        topk,
        config_dtype,
        num_tokens,
        block_shape=block_shape,
    )

    # If the caller pre-computed (sorted_token_ids, expert_ids,
    # num_tokens_post_padded) — e.g. via the DeepEP-specialized
    # ``deepep_moe_align`` — skip the generic moe_align entirely. Otherwise
    # fall through to the per-call alignment for arbitrary topk_ids.
    #
    if sorted_token_ids is None:
        # When pre-computed bucket/expert_count are provided (e.g. from
        # recompute_topk_and_align_count in pure-TP EP mode), they are sized
        # for num_local_experts, not the full E from w1.shape. Use the
        # pre_expert_count tensor size to determine the effective num_experts
        # so that padcum/scatter kernels don't read out-of-bounds.
        effective_E = (
            (pre_expert_count.shape[0] - 1) if pre_expert_count is not None else E
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            config["BLOCK_SIZE_M"],
            effective_E,
            scratch=align_scratch,
            pre_bucket=pre_bucket,
            pre_expert_count=pre_expert_count,
        )
    else:
        assert (
            expert_ids is not None and num_tokens_post_padded is not None
        ), "sorted_token_ids/expert_ids/num_tokens_post_padded must be supplied together"
        assert config["BLOCK_SIZE_M"] == _expected_block_size_m_for_caller_align(
            use_fp8_w8a8, block_shape
        ), (
            f"caller pre-computed routing buffers assume BLOCK_SIZE_M="
            f"{_expected_block_size_m_for_caller_align(use_fp8_w8a8, block_shape)}"
            f" but config picked {config['BLOCK_SIZE_M']}; "
            f"fused_moe_triton_config drift?"
        )

    # Output buffer (allocate if caller didn't supply one).
    if no_combine:
        assert not inplace
        if out_hidden_states is None:
            out_hidden_states = torch.empty(
                (num_tokens, topk, w2.shape[1]),
                device=hidden_states.device,
                dtype=effective_dtype,
            )
    elif inplace:
        out_hidden_states = hidden_states
    elif out_hidden_states is None:
        out_hidden_states = torch.empty(
            (num_tokens, w2.shape[1]),
            device=hidden_states.device,
            dtype=effective_dtype,
        )

    # ``filter_expert=True`` plus matching filter logic in the silu+mul+quant
    # kernel (and in fused_moe_kernel itself, which early-returns on -1 expert
    # ids) means rows belonging to padded slots are never written *and* never
    # read downstream — DeepEP combine consumes only the valid prefix indicated
    # by ``expert_num_tokens``. So we can use ``torch.empty`` here; matches the
    # deepgemm masked executor (also empty, no zero-init).
    if intermediate_cache1 is None:
        intermediate_cache1 = torch.empty(
            (num_tokens * topk, N),
            device=hidden_states.device,
            dtype=effective_dtype,
        )

    # Fast-path: when ``topk == 1`` and no router-weight scaling, the down GEMM
    # writes a ``(num_tokens, 1, hidden)`` tensor that is bit-identical to
    # ``out_hidden_states.unsqueeze(1)`` — alias them to skip the post-GEMM
    # ``out_hidden_states.copy_(intermediate_cache3.squeeze(1))`` memcpy. This
    # is ~13.5ms / profile_step (200×67µs in nsys) on the DeepEP low-latency
    # path where ``topk`` is forced to 1 by the synthetic single-expert routing.
    can_alias_out = (
        not no_combine
        and topk == 1
        and (routed_scaling_factor is None or routed_scaling_factor == 1.0)
        and not filter_expert
    )
    if intermediate_cache3 is None:
        if can_alias_out:
            intermediate_cache3 = out_hidden_states.unsqueeze(1)
        else:
            # Must zero-init when filter_expert is set: moe_align_block_size
            # prunes expert=-1 blocks from num_tokens_post_padded so the kernel
            # early-returns before reaching them — write_zeros_to_output is never
            # called and uninitialized memory leaks into the reduce step.
            alloc = torch.zeros if filter_expert else torch.empty
            intermediate_cache3 = alloc(
                (num_tokens, topk, w2.shape[1]),
                device=hidden_states.device,
                dtype=effective_dtype,
            )

    if use_fp8_w8a8:
        assert w1_scale is not None
        assert w2_scale is not None
        if hidden_states.dtype == torch.float8_e4m3fn:
            # Router pre-quantized: reuse the provided (a1_q, a1_scale) and
            # skip the redundant per_token_group_quant_8bit dispatch which
            # only accepts BF16/FP16 inputs.
            assert (
                a1_scale is not None
            ), "fused_experts_impl: hidden_states already FP8 but a1_scale is None"
            a1_q, a1_s = hidden_states, a1_scale
        else:
            a1_q, a1_s = _quantize_input_fp8(
                hidden_states, a1_scale, block_shape, per_channel_quant
            )
    else:
        a1_q, a1_s = hidden_states, None

    invoke_fused_moe_kernel(
        a1_q,
        w1,
        None,
        intermediate_cache1,
        a1_s,
        w1_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        filter_expert=filter_expert,
    )

    # silu+mul + per-block-fp8-quant fusion. For FP8 W8A8 with per-block
    # scales (the DeepEP low-latency / qwen3.5 MoE path), use the dedicated
    # ``silu_mul_post_quant_flat_fwd`` Triton kernel — it expects RTP-LLM's
    # ``[up | gate]`` layout and produces row-major fp32 scales matching what
    # ``sgl_per_token_group_quant_fp8(...)`` would have produced, so the down
    # GEMM consumes them transparently. This collapses the original 2-kernel
    # ``act_and_mul_kernel`` + ``per_token_group_quant_8bit_kernel`` (~9.5ms /
    # 200 calls in profile) into one launch and drops ``intermediate_cache2``
    # entirely.
    fuse_silu_mul_quant = (
        use_fp8_w8a8
        and block_shape is not None
        and activation == "silu"
        and not per_channel_quant
    )
    if fuse_silu_mul_quant:
        M_total = num_tokens * topk
        H_half = N // 2
        block_k = block_shape[1]
        # Prefer the masked 3D kernel (deepgemm-style) when DeepEP layout
        # metadata is available — it visits only ``masked_m[e]`` valid tokens
        # per expert instead of all padded slots, ~28× fewer programs in our
        # sparse MoE. Falls back to the per-row sentinel kernel when we don't
        # have the (E, T) layout (e.g. pure-TP path).
        use_masked_3d = (
            masked_m is not None
            and e_local is not None
            and max_recv is not None
            and M_total == e_local * max_recv
        )
        if use_masked_3d:
            from rtp_llm.models_py.triton_kernels.common.activation import (
                silu_mul_masked_fp8_post_quant_fwd,
            )

            # 3D outputs match deepgemm-fused kernel's layout.
            # output_q: (E, T, H_half) fp8;  output_s: (E, T, H_half/block_k) fp32.
            # Flat 2D views (for the down GEMM) are contiguous, same row-major
            # layout produced by the per-row kernel — drop-in compatible with
            # ``invoke_fused_moe_kernel``. Caller may pre-allocate (and pass
            # in via the ``a2_q_3d`` / ``a2_s_3d`` kwargs) — required when
            # this layer is captured into a CUDA graph that is replayed
            # cross-stream, since per-call ``torch.empty`` would force a new
            # caching-allocator pool for the side stream and OOM.
            if a2_q_3d is None:
                a2_q_3d = torch.empty(
                    (e_local, max_recv, H_half),
                    device=hidden_states.device,
                    dtype=torch.float8_e4m3fn,
                )
            if a2_s_3d is None:
                a2_s_3d = torch.empty(
                    (e_local, max_recv, H_half // block_k),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            silu_mul_masked_fp8_post_quant_fwd(
                input=intermediate_cache1.view(e_local, max_recv, N),
                output=a2_q_3d,
                output_scale=a2_s_3d,
                quant_group_size=block_k,
                masked_m=masked_m,
                expected_m=expected_m if expected_m is not None else max_recv,
                scale_ue8m0=False,
            )
            a2_q = a2_q_3d.view(M_total, H_half)
            a2_s = a2_s_3d.view(M_total, H_half // block_k)
        else:
            from rtp_llm.models_py.triton_kernels.common.activation import (
                silu_mul_post_quant_flat_fwd,
            )

            a2_q = torch.empty(
                (M_total, H_half),
                device=hidden_states.device,
                dtype=torch.float8_e4m3fn,
            )
            # Row-major fp32 scales — matches the layout
            # ``sgl_per_token_group_quant_fp8(column_major_scales=False)`` and
            # the masked 3D kernel produce, so the down GEMM consumes either
            # via the same ``A_scale.stride(0/1)`` indexing.
            a2_s = torch.empty(
                (M_total, H_half // block_k),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            silu_mul_post_quant_flat_fwd(
                input=intermediate_cache1.view(M_total, N),
                output_q=a2_q,
                output_s=a2_s,
                quant_group_size=block_k,
                topk_ids=topk_ids if filter_expert else None,
                scale_ue8m0=False,
            )
    else:
        # Fallback 2-kernel path (activation + separate quant). Allocate
        # ``intermediate_cache2`` lazily here — it's only needed in this
        # branch.
        if intermediate_cache2 is None:
            intermediate_cache2 = torch.empty(
                (num_tokens * topk, N // 2),
                device=hidden_states.device,
                dtype=effective_dtype,
            )
        if activation == "silu" and not filter_expert:
            from rtp_llm.models_py.triton_kernels.common.activation import (
                silu_and_mul as _silu_and_mul,
            )

            _silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
        elif activation in ("silu", "gelu"):
            act_and_mul_triton(
                intermediate_cache1.view(-1, N),
                intermediate_cache2,
                topk_ids=topk_ids,
                activation=activation,
            )
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if use_fp8_w8a8:
            a2_q, a2_s = _quantize_input_fp8(
                intermediate_cache2, a2_scale, block_shape, per_channel_quant
            )
        else:
            a2_q, a2_s = intermediate_cache2, None

    invoke_fused_moe_kernel(
        a2_q,
        w2,
        None,
        intermediate_cache3,
        a2_s,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        filter_expert=filter_expert,
    )

    if no_combine:
        return intermediate_cache3

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    if topk == 1 and routed_scaling_factor == 1.0:
        # ``can_alias_out`` above made ``intermediate_cache3`` a view of
        # ``out_hidden_states.unsqueeze(1)``, so the down GEMM has already
        # written into ``out_hidden_states`` — skip the redundant copy_().
        # When the caller supplied a non-aliasable ``intermediate_cache3``
        # (e.g. a custom buffer), still need to copy.
        if intermediate_cache3.data_ptr() != out_hidden_states.data_ptr():
            out_hidden_states.copy_(intermediate_cache3.squeeze(1))
    elif topk == 2 and routed_scaling_factor == 1.0:
        torch.add(
            intermediate_cache3[:, 0],
            intermediate_cache3[:, 1],
            out=out_hidden_states,
        )
    elif routed_scaling_factor == 1.0:
        # When scaling factor is 1.0 (e.g. Qwen3.5 where GroupTopK already
        # incorporates it into topk_weights), a single torch.sum suffices —
        # avoids the redundant mul_(1.0) kernel (~1us/layer savings).
        torch.sum(intermediate_cache3, dim=1, out=out_hidden_states)
    elif num_tokens <= 32:
        # Small-token path (see _moe_sum_reduce_torch_compile).
        _moe_sum_reduce_torch_compile(
            intermediate_cache3, out_hidden_states, routed_scaling_factor
        )
    else:
        moe_sum_reduce_triton(
            intermediate_cache3, out_hidden_states, routed_scaling_factor
        )

    return out_hidden_states
