# DeepEP-specialized moe_align: replaces the 4 generic moe_align kernels +
# 4 scratch fills (~10us per call) with 2 small Triton kernels for the
# structured DeepEP low-latency layout. Preserves the sparsity-prune
# semantics — ``num_tokens_post_padded`` is the actual padded sum, not a
# worst-case constant — so fused_moe_kernel's grid stays compact.
#
# Layout assumptions (only valid for ``DeepEPLowLatencyTritonExecutor``):
#   * Rows are pre-grouped per expert: row i belongs to expert i // max_recv.
#   * Per-expert valid token count is given by ``masked_m[E_local]`` (from
#     DeepEP dispatch; no need to bucket+count).
#   * ``max_recv % BLOCK_SIZE_M == 0``.
#   * topk = 1 (synthetic single-expert routing) so the scatter target is
#     deterministic: row e*max_recv+t goes to slot cum[e]+t.
#
# Compared to ``moe_align_block_size``:
#   - skips bucket allocation + bucket+count kernel (we already have masked_m)
#   - skips slot_counter atomic-add + scatter kernel (deterministic placement)
#   - skips the 4 zeros/full scratch fills (output buffers are persistent and
#     fully written every call)
#
# Licensed under the Apache License, Version 2.0
from typing import Tuple

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    """Smallest power of two ``>= n``."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def _deepep_align_fused_kernel(
    masked_m_ptr,  # (E_local,) int32 input
    sorted_token_ids_ptr,  # (max_pad_worst,) int32 output
    expert_ids_ptr,  # (max_num_blocks_worst,) int32 output
    num_tokens_post_padded_ptr,  # (1,) int32 output
    e_local: tl.constexpr,
    max_recv: tl.constexpr,
    block_size_m: tl.constexpr,
    n_total: tl.constexpr,  # = E_local * max_recv (sentinel value)
    BLOCK_E: tl.constexpr,  # next_pow2(e_local)
    BLOCK_T: tl.constexpr,  # next_pow2(max_recv); >= max padded_m[e]
    BLOCK_B: tl.constexpr,  # next_pow2(max_recv // block_size_m)
):
    """Fused cum-pad-sum + scatter, single launch instead of two.

    grid = (E_local,). Each program e:
      1. Loads the full ``masked_m[BLOCK_E]`` vector (cheap, E_local <= 256).
      2. Computes ``cum_e = sum_{i<e} ceil(masked_m[i] / BSM) * BSM`` via a
         masked reduction — replaces the separate ``tl.cumsum`` kernel. Each
         program only needs *its own* prefix value, not the full cum array.
      3. Writes its slice of ``sorted_token_ids`` and ``expert_ids``.
      4. Program 0 also publishes ``num_tokens_post_padded = sum(padded[])``.

    Layout written (identical to the previous 2-kernel impl):
      sorted_token_ids[cum_e            .. cum_e + masked_m_e) = e*max_recv + t
      sorted_token_ids[cum_e + masked_m_e .. cum_e + padded_e) = n_total (sentinel)
      expert_ids     [cum_e/BSM         .. (cum_e+padded_e)/BSM) = e

    Other slots in the output buffers are unread by fused_moe_kernel
    (early-return on ``pid_m * BSM >= num_tokens_post_padded``), so we skip
    initializing them.
    """
    e = tl.program_id(0)

    # Load full masked_m vector to compute this program's prefix.
    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < e_local
    masked_m_arr = tl.load(masked_m_ptr + e_offs, mask=e_mask, other=0)
    padded_arr = ((masked_m_arr + block_size_m - 1) // block_size_m) * block_size_m
    padded_arr = tl.where(e_mask, padded_arr, 0)

    # Exclusive prefix for this program: sum of padded[i] for i < e.
    lt_e_mask = (e_offs < e).to(tl.int32)
    cum_e = tl.sum(padded_arr * lt_e_mask, axis=0)

    # This program's own counts.
    masked_m_e = tl.load(masked_m_ptr + e)
    padded_e = ((masked_m_e + block_size_m - 1) // block_size_m) * block_size_m

    # Scatter sorted_token_ids[cum_e .. cum_e + padded_e).
    t_offs = tl.arange(0, BLOCK_T)
    t_mask = t_offs < padded_e
    valid_mask = t_offs < masked_m_e
    row_id = e * max_recv + t_offs
    val = tl.where(valid_mask, row_id, n_total)
    tl.store(sorted_token_ids_ptr + cum_e + t_offs, val, mask=t_mask)

    # Scatter expert_ids: write ``e`` to each of this expert's blocks.
    block_start = cum_e // block_size_m
    block_count = padded_e // block_size_m
    b_offs = tl.arange(0, BLOCK_B)
    b_mask = b_offs < block_count
    tl.store(expert_ids_ptr + block_start + b_offs, e, mask=b_mask)

    # Program 0 publishes the total padded length (= sum of all padded[e]).
    if e == 0:
        total = tl.sum(padded_arr, axis=0)
        tl.store(num_tokens_post_padded_ptr, total)


def deepep_moe_align(
    masked_m: torch.Tensor,  # (E_local,) int32
    e_local: int,
    max_recv: int,
    block_size_m: int,
    sorted_token_ids: torch.Tensor,  # (max_pad_worst,) int32 — caller-owned
    expert_ids: torch.Tensor,  # (max_num_blocks_worst,) int32 — caller-owned
    num_tokens_post_padded: torch.Tensor,  # (1,) int32 — caller-owned
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Specialized moe_align fast-path for DeepEP low-latency layout.

    Replaces the 4 generic moe_align kernels + 4 scratch fills with a single
    fused kernel — each per-expert program computes its own prefix via a
    masked reduction (no separate cum kernel) and scatters its slice. ~8×
    fewer launches than the generic path.
    """
    assert masked_m.dtype == torch.int32 and masked_m.shape == (e_local,)
    assert (
        max_recv % block_size_m == 0
    ), f"max_recv ({max_recv}) must be divisible by BLOCK_SIZE_M ({block_size_m})"

    n_total = e_local * max_recv
    BLOCK_E = max(_next_pow2(e_local), 16)
    BLOCK_T = max(_next_pow2(max_recv), 16)
    BLOCK_B = max(_next_pow2(max_recv // block_size_m), 4)

    _deepep_align_fused_kernel[(e_local,)](
        masked_m,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        e_local=e_local,
        max_recv=max_recv,
        block_size_m=block_size_m,
        n_total=n_total,
        BLOCK_E=BLOCK_E,
        BLOCK_T=BLOCK_T,
        BLOCK_B=BLOCK_B,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded
