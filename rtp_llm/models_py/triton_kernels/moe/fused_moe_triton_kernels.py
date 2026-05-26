# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py
# Adapted for RTP-LLM. Some sglang variants are intentionally dropped to keep
# the port portable across SM levels (no TensorDescriptor / TMA / swap_ab) and
# focused on the high-perf Triton fused_moe path used in sglang's profiling
# timeline (fused_moe_kernel + moe_sum_reduce_triton, no DeepEP).
#
# Supported quant modes:
#   * no quant (BF16/FP16 activation, BF16/FP16 weights)
#   * FP8 W8A8 per-block (A: per-token-group fp8, W: per-block fp8)
#   * FP8 W8A8 per-tensor / per-token (A: per-tensor or per-token fp8)
#
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# moe_align_block_size (Triton + torch fallback)
# -----------------------------------------------------------------------------
#
# sgl_kernel exposes a fast CUDA implementation of moe_align_block_size. RTP-LLM
# does not link sgl_kernel so we provide a pure-torch implementation that is
# functionally equivalent. Performance is acceptable for typical inference
# shapes (M*topk in the few thousand range) because the heavy lifting (sort,
# -----------------------------------------------------------------------------
# moe_align_block_size (Triton + torch, CUDA-graph-safe)
# -----------------------------------------------------------------------------
#
# sgl_kernel exposes a fast CUDA implementation of moe_align_block_size. RTP-LLM
# does not link sgl_kernel so we provide an equivalent in torch + a small
# triton kernel. The implementation deliberately avoids ``torch.argsort`` on
# CUDA: stable argsort dispatches to thrust which calls ``cudaMalloc`` for
# scratch storage, bypassing PyTorch's caching allocator and therefore failing
# inside CUDA graph capture (``cudaErrorStreamCaptureUnsupported``). Instead
# we use a one-pass triton kernel that places each token via ``atomic_add`` on
# a per-expert slot counter — this is graph-capture safe.


@triton.jit
def _recompute_topk_and_align_count_kernel(
    topk_ids_ptr,  # int32 [N], original global expert ids
    adjusted_topk_ids_ptr,  # int32 [N]; output, local expert id or -1
    bucket_ptr,  # int64 [N]; output, valid local id or num_experts sentinel
    expert_count_ptr,  # int64 [E+1]; output, atomically incremented
    expert_start_id,  # i32
    num_local_experts,  # i32
    num_valid_tokens,  # i32
    BLOCK_SIZE: tl.constexpr,
):
    """Fused recompute_topk_ids + moe_align_count in one kernel launch.

    Reads global topk_ids, subtracts expert_start_id, produces:
    - adjusted_topk_ids: local expert id or -1 (for filtered experts)
    - bucket: same as adjusted but uses num_local_experts as sentinel (for scatter)
    - expert_count: atomic per-expert token count (for cumsum/padding)

    Caller must pre-zero expert_count_ptr before launch.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = offsets < num_valid_tokens

    raw = tl.load(topk_ids_ptr + offsets, mask=in_range, other=-1).to(tl.int64)

    adjusted = raw - expert_start_id
    valid = in_range & (adjusted >= 0) & (adjusted < num_local_experts)

    out_adj = tl.where(valid, adjusted, -1)
    tl.store(adjusted_topk_ids_ptr + offsets, out_adj.to(tl.int32), mask=in_range)

    bucket = tl.where(valid, adjusted, num_local_experts)
    tl.store(bucket_ptr + offsets, bucket, mask=in_range)

    safe_bucket = tl.where(valid, adjusted, 0)
    tl.atomic_add(expert_count_ptr + safe_bucket, 1, mask=valid)


@triton.jit
def _moe_align_count_kernel(
    topk_ids_ptr,  # int32 or int64 [N]
    bucket_ptr,  # int64 [N]; output, valid bucket id or sentinel
    expert_count_ptr,  # int64 [E+1]; output, atomically incremented
    num_valid_tokens,  # i32
    num_experts,  # i32
    BLOCK_SIZE: tl.constexpr,
    IDS_DTYPE: tl.constexpr,  # 0=int32, 1=int64
):
    """Replaces ``flat = topk_ids.reshape(-1).to(int64)`` + ``where(...)`` +
    ``zeros(E+1)`` + ``ones_like(bucket)`` + ``scatter_add_`` with a single
    Triton kernel.

    Caller must pre-zero ``expert_count_ptr`` before launch (via the
    persistent scratch buffer's ``zero_()``).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = offsets < num_valid_tokens
    if IDS_DTYPE == 0:
        raw32 = tl.load(topk_ids_ptr + offsets, mask=in_range, other=0).to(tl.int64)
    else:
        raw32 = tl.load(topk_ids_ptr + offsets, mask=in_range, other=0)
    raw = raw32
    valid = in_range & (raw >= 0) & (raw < num_experts)
    bucket = tl.where(valid, raw, num_experts)
    tl.store(bucket_ptr + offsets, bucket, mask=in_range)
    # Atomic increment per-expert count for valid tokens only.
    safe_bucket = tl.where(valid, bucket, 0)
    tl.atomic_add(expert_count_ptr + safe_bucket, 1, mask=valid)


@triton.jit
def _moe_align_padcum_kernel(
    expert_count_ptr,  # int64 [E+1]; input
    cum_ptr,  # int64 [E+1]; output, exclusive cumsum of padded counts
    block_size,  # i32, the BLOCK_SIZE_M for fused MoE
    num_experts,  # i32
    BLOCK_E: tl.constexpr,  # power-of-two >= num_experts
):
    """Sequential pad-then-cumsum on the per-expert count array. Single
    program (one block), uses ``tl.cumsum`` since num_experts is small (<=
    256 typical). Replaces ``add+div+mul+cumsum+pad`` Python chain.

    Layout written:
        cum[0] = 0  (assumed pre-zeroed by caller via ``cum.zero_()``)
        cum[e+1] = sum_{j<=e} pad_to_block(expert_count[j])  for e in [0, E)
    """
    if tl.program_id(0) != 0:
        return
    e = tl.arange(0, BLOCK_E)
    in_range = e < num_experts
    cnt = tl.load(expert_count_ptr + e, mask=in_range, other=0)
    # Pad each count up to multiple of block_size.
    padded = ((cnt + block_size - 1) // block_size) * block_size
    padded = tl.where(in_range, padded, 0)
    # Inclusive cumsum: inclusive[e] = sum_{j<=e} padded[j].
    inclusive = tl.cumsum(padded, axis=0)
    # cum[1+e] = inclusive[e] for e in [0, num_experts).
    tl.store(cum_ptr + 1 + e, inclusive, mask=in_range)


@triton.jit
def _moe_align_padcum_and_expert_ids_kernel(
    expert_count_ptr,  # int64 [E+1]; input
    cum_ptr,  # int64 [E+1]; output, exclusive cumsum of padded counts
    expert_ids_ptr,  # int32 [max_num_blocks]; output
    block_size,  # i32, the BLOCK_SIZE_M for fused MoE
    num_experts,  # i32
    max_num_blocks,  # i32
    BLOCK_E: tl.constexpr,  # power-of-two >= num_experts
):
    """Fused padcum + expert_ids in a single kernel launch.

    Single program computes the padded exclusive cumsum of expert_count,
    then iterates over all blocks to assign expert ownership.
    NOTE: serial loop is slow for large max_num_blocks; prefer the unfused
    pair (_moe_align_padcum_kernel + _moe_align_expert_ids_kernel).
    """
    if tl.program_id(0) != 0:
        return
    e = tl.arange(0, BLOCK_E)
    in_range = e < num_experts
    cnt = tl.load(expert_count_ptr + e, mask=in_range, other=0)
    padded = ((cnt + block_size - 1) // block_size) * block_size
    padded = tl.where(in_range, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    tl.store(cum_ptr + 1 + e, inclusive, mask=in_range)

    for bid in range(max_num_blocks):
        block_start = bid * block_size
        count_le = tl.sum(tl.where(inclusive <= block_start, 1, 0).to(tl.int32), axis=0)
        expert_id = tl.where(count_le >= num_experts, -1, count_le)
        tl.store(expert_ids_ptr + bid, expert_id.to(tl.int32))


@triton.jit
def _moe_align_padcum_and_expert_ids_parallel_kernel(
    expert_count_ptr,  # int64 [E+1]; input
    cum_ptr,  # int64 [E+1]; output, exclusive cumsum of padded counts
    expert_ids_ptr,  # int32 [max_num_blocks]; output
    block_size,  # i32, the BLOCK_SIZE_M for fused MoE
    num_experts,  # i32
    BLOCK_E: tl.constexpr,  # power-of-two >= num_experts
    BLOCK_B: tl.constexpr,  # power-of-two >= max blocks per expert
):
    """Fused padcum + expert_ids with grid=(num_experts,) — one program per expert.

    Each program e:
      1. Loads the full expert_count[BLOCK_E] vector (small, <= 256 elements).
      2. Computes exclusive prefix: cum_e = sum_{i<e} padded[i].
      3. Writes cum[e+1] = cum_e + padded[e] (inclusive prefix for this expert).
      4. Writes expert_ids[cum_e/BSM .. (cum_e+padded_e)/BSM) = e.
      5. Program 0 also writes cum[0] = 0.

    Produces identical output to the two-kernel sequence
    (_moe_align_padcum_kernel + _moe_align_expert_ids_kernel).
    """
    e = tl.program_id(0)

    # Load full expert_count vector to compute this program's prefix.
    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < num_experts
    cnt = tl.load(expert_count_ptr + e_offs, mask=e_mask, other=0)
    # Pad each count up to multiple of block_size.
    padded = ((cnt + block_size - 1) // block_size) * block_size
    padded = tl.where(e_mask, padded, 0)

    # Exclusive prefix for this program: sum of padded[i] for i < e.
    lt_e_mask = (e_offs < e).to(tl.int64)
    cum_e = tl.sum(padded * lt_e_mask, axis=0)

    # This program's own padded count.
    cnt_e = tl.load(expert_count_ptr + e)
    padded_e = ((cnt_e + block_size - 1) // block_size) * block_size

    # Write cum[e+1] = cum_e + padded_e (inclusive cumsum at position e).
    tl.store(cum_ptr + 1 + e, cum_e + padded_e)

    # Write expert_ids for this expert's blocks.
    block_start = cum_e // block_size
    block_count = padded_e // block_size
    b_offs = tl.arange(0, BLOCK_B)
    b_mask = b_offs < block_count
    tl.store(expert_ids_ptr + block_start + b_offs, e, mask=b_mask)

    # Program 0 writes cum[0] = 0.
    if e == 0:
        tl.store(cum_ptr, tl.cast(0, tl.int64))


@triton.jit
def _moe_align_padcum_expert_ids_and_fill_kernel(
    expert_count_ptr,  # int64 [E+1]; input
    cum_ptr,  # int64 [E+1]; output, exclusive cumsum of padded counts
    expert_ids_ptr,  # int32 [max_num_blocks]; output
    sorted_ids_ptr,  # int32 [max_pad]; output (padding slots filled with sentinel)
    slot_counter_ptr,  # int32 [E]; output, zeroed by this kernel
    block_size,  # i32, the BLOCK_SIZE_M for fused MoE
    num_experts,  # i32
    num_valid_tokens,  # i32, sentinel value for padding slots
    BLOCK_E: tl.constexpr,  # power-of-two >= num_experts
    BLOCK_B: tl.constexpr,  # power-of-two >= max blocks per expert
    BLOCK_PAD: tl.constexpr,  # power-of-two >= block_size (max padding per expert)
):
    """Fused padcum + expert_ids + sentinel fill + slot_counter zero.

    Extends _moe_align_padcum_and_expert_ids_parallel_kernel to also write
    sentinel values (num_valid_tokens) into padding positions of sorted_ids
    and zero the slot_counter for the subsequent scatter kernel. This
    eliminates ALL standalone fill/zero calls in the scratch path.

    Each program e:
      1. Zeros slot_counter[e] (for scatter's atomic_add).
      2. Computes cum_e, padded_e (same as the non-fill variant).
      3. Writes cum[e+1], expert_ids (same as before).
      4. Writes sentinel into sorted_ids[cum_e + count_e .. cum_e + padded_e).
      5. Program 0 also writes cum[0] = 0.
    """
    e = tl.program_id(0)

    # Zero this expert's slot counter (for scatter kernel's atomic_add).
    tl.store(slot_counter_ptr + e, tl.cast(0, tl.int32))

    # Load full expert_count vector to compute this program's prefix.
    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < num_experts
    cnt = tl.load(expert_count_ptr + e_offs, mask=e_mask, other=0)
    # Pad each count up to multiple of block_size.
    padded = ((cnt + block_size - 1) // block_size) * block_size
    padded = tl.where(e_mask, padded, 0)

    # Exclusive prefix for this program: sum of padded[i] for i < e.
    lt_e_mask = (e_offs < e).to(tl.int64)
    cum_e = tl.sum(padded * lt_e_mask, axis=0)

    # This program's own padded count and actual count.
    cnt_e = tl.load(expert_count_ptr + e)
    padded_e = ((cnt_e + block_size - 1) // block_size) * block_size

    # Write cum[e+1] = cum_e + padded_e (inclusive cumsum at position e).
    tl.store(cum_ptr + 1 + e, cum_e + padded_e)

    # Write expert_ids for this expert's blocks.
    block_start = cum_e // block_size
    block_count = padded_e // block_size
    b_offs = tl.arange(0, BLOCK_B)
    b_mask = b_offs < block_count
    tl.store(expert_ids_ptr + block_start + b_offs, e, mask=b_mask)

    # Fill padding slots in sorted_ids with sentinel value.
    # Valid tokens occupy [cum_e, cum_e + cnt_e); padding is [cum_e + cnt_e, cum_e + padded_e).
    # Max padding per expert is block_size - 1 (when cnt_e mod block_size == 1).
    pad_count = padded_e - cnt_e
    pad_offs = tl.arange(0, BLOCK_PAD)
    pad_mask = pad_offs < pad_count
    tl.store(
        sorted_ids_ptr + cum_e + cnt_e + pad_offs,
        tl.cast(num_valid_tokens, tl.int32),
        mask=pad_mask,
    )

    # Program 0 writes cum[0] = 0.
    if e == 0:
        tl.store(cum_ptr, tl.cast(0, tl.int64))


@triton.jit
def _moe_align_expert_ids_kernel(
    cum_ptr,  # int64 [E+1]
    expert_ids_ptr,  # int32 [max_num_blocks]; output
    block_size,  # i32
    num_experts,  # i32
    max_num_blocks,  # i32
    BLOCK_E: tl.constexpr,
):
    """For each ``block_size``-aligned slot in ``sorted_ids``, write which
    expert it belongs to (or -1 for blocks past the actual padded total).

    Replaces ``arange + searchsorted + where + to(int32)`` (4 kernels).
    """
    pid = tl.program_id(0)
    if pid >= max_num_blocks:
        return
    block_start = pid * block_size
    e = tl.arange(0, BLOCK_E)
    in_range = e < num_experts
    INT64_MAX_LIKE = 0x7FFFFFFFFFFFFFFF
    cum_vals = tl.load(cum_ptr + 1 + e, mask=in_range, other=INT64_MAX_LIKE)
    count_le = tl.sum(tl.where(cum_vals <= block_start, 1, 0).to(tl.int32), axis=0)
    expert_id = tl.where(count_le >= num_experts, -1, count_le)
    tl.store(expert_ids_ptr + pid, expert_id.to(tl.int32))


@triton.jit
def _moe_align_scatter_kernel(
    bucket_ptr,  # int64 [N]
    cum_ptr,  # int64 [E + 1]; cum_ptr[e] = padded prefix sum
    slot_counter_ptr,  # int32 [E]; zero-initialized; receives atomic adds
    sorted_ids_ptr,  # int32 [max_pad]; padding slots pre-filled with sentinel
    num_valid_tokens,  # i32
    num_experts,  # i32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = offsets < num_valid_tokens
    bucket = tl.load(bucket_ptr + offsets, mask=in_range, other=num_experts)
    valid = in_range & (bucket < num_experts)
    safe_bucket = tl.where(valid, bucket, 0)
    base = tl.load(cum_ptr + safe_bucket, mask=valid, other=0)
    rank = tl.atomic_add(slot_counter_ptr + safe_bucket, 1, mask=valid)
    dest = base + rank.to(tl.int64)
    tl.store(sorted_ids_ptr + dest, offsets.to(tl.int32), mask=valid)


def _next_pow2(n: int) -> int:
    """Smallest power of two ``>= n``."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    scratch: Optional[Dict[str, torch.Tensor]] = None,
    pre_bucket: Optional[torch.Tensor] = None,
    pre_expert_count: Optional[torch.Tensor] = None,
):
    """Aligns the per-expert token count to ``block_size`` for fused MoE.

    Args:
        topk_ids: int tensor of shape ``(num_tokens, top_k)``. Values < 0 mark
            tokens that should be skipped (e.g. EP-filtered experts).
        block_size: BLOCK_SIZE_M used by the fused MoE kernel.
        num_experts: total number of experts.
        scratch: Optional pre-allocated scratch dict from the executor (keys:
            ``bucket``, ``expert_count``, ``cum``, ``slot_counter``,
            ``sorted_ids``, ``expert_ids``). When provided, persistent
            buffers are reused with NO standalone zero/fill calls needed:
            the fused padcum kernel writes cum, expert_ids, sorted_ids
            padding sentinels, AND slot_counter zeros in one launch. Only
            ``expert_count.zero_()`` is needed when the count kernel runs
            (skipped when pre_bucket/pre_expert_count are provided).
            When omitted, falls back to per-call ``torch.zeros`` /
            ``torch.full``.
        pre_bucket: Optional pre-computed bucket from fused
            ``recompute_topk_and_align_count``. When provided together with
            ``pre_expert_count``, skips the count kernel entirely.
        pre_expert_count: Optional pre-computed expert_count (int64 [E+1]).

    Returns:
        ``(sorted_token_ids, expert_ids, num_tokens_post_pad)`` matching the
        layout that ``fused_moe_kernel`` expects.

    CUDA Graph compatibility:
        Output buffers are pre-sized to a fixed worst-case bound; the actual
        padded length is published only through the device tensor
        ``num_tokens_post_pad`` which the kernel consumes via a pointer load.
        Implementation has been refactored to fuse the bucket/count/cumsum/
        expert_ids steps into 3 Triton kernels (was ~25 small element-wise
        torch ops), cutting per-call kernel-launch overhead by ~6×.
    """
    assert topk_ids.dim() == 2
    assert topk_ids.dtype in (torch.int32, torch.int64)
    device = topk_ids.device
    num_valid_tokens = topk_ids.numel()  # host-side python int

    # Worst-case padded length. Determined entirely by host-side ints, so
    # safe under graph capture.
    max_pad = num_valid_tokens + num_experts * block_size
    max_pad = ((max_pad + block_size - 1) // block_size) * block_size
    max_num_blocks = max_pad // block_size

    # Fast path: bucket + expert_count already computed by fused kernel.
    skip_count = pre_bucket is not None and pre_expert_count is not None

    # Acquire (or allocate) the scratch buffers.
    if scratch is None:
        if not skip_count:
            bucket = torch.empty(num_valid_tokens, dtype=torch.int64, device=device)
            expert_count = torch.zeros(
                num_experts + 1, dtype=torch.int64, device=device
            )
        else:
            bucket = pre_bucket
            expert_count = pre_expert_count
        cum = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
        slot_counter = torch.zeros(num_experts, dtype=torch.int32, device=device)
        sorted_ids = torch.full(
            (max_pad,), num_valid_tokens, dtype=torch.int32, device=device
        )
        expert_ids = torch.empty(max_num_blocks, dtype=torch.int32, device=device)
        use_fill_kernel = False
    else:
        cum = scratch["cum"]
        slot_counter = scratch["slot_counter"]
        sorted_ids = scratch["sorted_ids"]
        expert_ids = scratch["expert_ids"]
        if skip_count:
            bucket = pre_bucket
            expert_count = pre_expert_count
        else:
            bucket = scratch["bucket"]
            expert_count = scratch["expert_count"]
            expert_count.zero_()
        # No standalone zero/fill needed: the fused padcum kernel writes cum[0]=0,
        # cum[e+1], expert_ids, sorted_ids padding sentinels, AND slot_counter[e]=0
        # all in one launch. The scatter kernel's atomic_add sees zeroed counters.
        use_fill_kernel = True

    # 1) Fused bucket + count Triton kernel (skipped when pre-computed).
    BLOCK = 256
    if not skip_count:
        flat = topk_ids.reshape(-1)
        ids_dtype_flag = 0 if topk_ids.dtype == torch.int32 else 1
        if num_valid_tokens > 0:
            grid_count = (triton.cdiv(num_valid_tokens, BLOCK),)
            _moe_align_count_kernel[grid_count](
                flat,
                bucket,
                expert_count,
                num_valid_tokens,
                num_experts,
                BLOCK_SIZE=BLOCK,
                IDS_DTYPE=ids_dtype_flag,
            )

    # 2) Fused padcum + expert_ids (+ optional sentinel fill): one program per
    #    expert. Each program computes its own prefix via masked reduction and
    #    writes its slice of expert_ids. When use_fill_kernel is True, the
    #    kernel also writes sentinel values into sorted_ids padding slots,
    #    eliminating the sorted_ids.fill_(sentinel) call.
    BLOCK_E = max(_next_pow2(num_experts), 16)
    # BLOCK_B must cover the max blocks any single expert can own. Worst case:
    # all tokens route to one expert → (num_valid_tokens + block_size - 1) //
    # block_size blocks. But BLOCK_B must be a power of two.
    max_blocks_per_expert = (num_valid_tokens + block_size - 1) // block_size
    max_blocks_per_expert = max(max_blocks_per_expert, 1)
    BLOCK_B = _next_pow2(max_blocks_per_expert)
    if use_fill_kernel:
        # BLOCK_PAD covers max padding per expert = block_size - 1.
        BLOCK_PAD = _next_pow2(block_size)
        _moe_align_padcum_expert_ids_and_fill_kernel[(num_experts,)](
            expert_count,
            cum,
            expert_ids,
            sorted_ids,
            slot_counter,
            block_size,
            num_experts,
            num_valid_tokens,
            BLOCK_E=BLOCK_E,
            BLOCK_B=BLOCK_B,
            BLOCK_PAD=BLOCK_PAD,
        )
    else:
        _moe_align_padcum_and_expert_ids_parallel_kernel[(num_experts,)](
            expert_count,
            cum,
            expert_ids,
            block_size,
            num_experts,
            BLOCK_E=BLOCK_E,
            BLOCK_B=BLOCK_B,
        )

    # 3) One-pass scatter kernel (writes sorted_ids).
    if num_valid_tokens > 0:
        grid_scatter = (triton.cdiv(num_valid_tokens, BLOCK),)
        _moe_align_scatter_kernel[grid_scatter](
            bucket,
            cum,
            slot_counter,
            sorted_ids,
            num_valid_tokens,
            num_experts,
            BLOCK_SIZE=BLOCK,
        )

    # Publish total_pad as a device-side int32 tensor so the kernel can read
    # it without a host sync. ``cum[-1]`` holds the inclusive cumsum total;
    # the sliced view shares storage so no copy.
    num_tokens_post_pad = cum[-1:].to(torch.int32)
    return sorted_ids, expert_ids, num_tokens_post_pad


# -----------------------------------------------------------------------------
# Helper kernel: zero out blocks whose expert is filtered (off-rank).
# -----------------------------------------------------------------------------
@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# -----------------------------------------------------------------------------
# Core fused MoE matmul kernel (no TMA / swap_ab / int4 / int8 / fuse_sum).
# -----------------------------------------------------------------------------
@triton.jit
def fused_moe_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    filter_expert: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if filter_expert and off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        if use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if BLOCK_SIZE_N > group_n:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if use_fp8_w8a8 and (group_k == 0 or group_n == 0):
        accumulator *= a_scale * b_scale

    if bias_ptr is not None:
        accumulator += bias

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor],
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]] = None,
    filter_expert: bool = True,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2]
    even_Ks = (K % config["BLOCK_SIZE_K"]) == 0

    fused_moe_kernel[grid](
        A,
        B,
        bias,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        C.stride(-2),
        C.stride(-1),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        even_Ks=even_Ks,
        filter_expert=filter_expert,
        **config,
    )


# -----------------------------------------------------------------------------
# Custom moe_sum_reduce kernel (the high-perf reduce sglang uses post-MoE).
# -----------------------------------------------------------------------------
# Modified from https://github.com/ModelTC/lightllm and sglang fused_moe_triton_kernels.
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


def moe_sum_reduce_triton(
    inp: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
) -> None:
    """Reduce ``inp`` (shape ``[T, topk, H]``) along the topk dim into
    ``output`` (shape ``[T, H]``) and multiply by ``routed_scaling_factor``."""
    assert inp.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = inp.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        inp,
        *inp.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )


# -----------------------------------------------------------------------------
# Activation + multiply kernel (silu / gelu).
# -----------------------------------------------------------------------------
# RTP-LLM convention: weight w13 is laid out so that the first half of the
# gate-up output is the "value" (up) and the second half is the "gate". This
# matches rtp_llm.models_py.triton_kernels.common.activation.silu_and_mul and
# the reference math in fused_moe_executor_test_util.generate_ref_output. We
# therefore output ``act(second_half) * first_half`` (sglang's convention is
# the opposite, so we swap the slicing here).
@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    x = x.to(tl.float32)
    if ACTIVATION_TYPE == "silu":
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == "gelu":
        kAlpha = 0.7978845608028654
        return 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))
    else:
        # triton requires a definite return; raising in jit context produces
        # a clean compile error so the user sees the bad activation name.
        tl.static_assert(False, "Unsupported activation")
        return x


@triton.jit
def act_and_mul_kernel(
    gateup_output,
    down_input,
    hidden_size,
    expert_ids_ptr,
    expert_step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2
    pid = tl.program_id(0)

    expert_id = tl.load(expert_ids_ptr + pid // expert_step)
    if expert_id == -1:
        return

    gateup_output_ptr = gateup_output + pid * hidden_size
    down_input_ptr = down_input + pid * half_hidden_size
    # RTP-LLM: first half = value (up), second half = gate.
    value_output_ptr = gateup_output_ptr
    gate_output_ptr = gateup_output_ptr + half_hidden_size

    for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half_hidden_size

        value_output = tl.load(value_output_ptr + offset, mask=mask)
        gate_output = tl.load(gate_output_ptr + offset, mask=mask)

        # Compute activation in float32 to avoid arith.mulf on FP8 (Triton's
        # LLVM backend rejects fmul between two fp8 operands). Promote
        # ``value_output`` explicitly so the multiply happens in f32.
        gate_output_activated = _apply_activation(
            gate_output.to(tl.float32), ACTIVATION_TYPE
        )
        act_mul_output = gate_output_activated * value_output.to(tl.float32)
        act_mul_output = act_mul_output.to(OutDtype)
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)


def act_and_mul_triton(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    topk_ids: Optional[torch.Tensor] = None,
    activation: str = "silu",
) -> None:
    """Compute ``act(gate) * value`` over a flattened (N, 2*H) tensor.

    ``topk_ids.view(-1)`` is consulted per-row so that filtered (-1) experts
    skip the multiplication (their downstream contribution is zero anyway).
    """
    grid = (down_input.shape[0],)
    hidden_size = gateup_output.shape[1]
    assert topk_ids is not None
    expert_ids_row = topk_ids.reshape(-1)
    act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        hidden_size,
        expert_ids_row,
        1,
        BLOCK_SIZE=512,
        ACTIVATION_TYPE=activation,
    )
