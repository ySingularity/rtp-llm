# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/standard.py
"""DP+TP-aware FP8 per-block router for the Triton fused-MoE executor.

This router handles topologies where both ``tp_size > 1`` and ``dp_size > 1``
(with ``ep_size == tp_size * dp_size``), in addition to pure DP and pure TP.
It extends :class:`PureTpRouterFp8PerBlock` by following the sglang
``StandardDispatcher`` pattern:

1. Gather tokens only across the DP group (ranks that share ``tp_rank``);
   TP siblings already hold the same input batch, so including them would
   duplicate work.
2. Pad locally to a **fixed** ``_dp_max_tokens_per_rank`` (computed at init
   from ``ll_num_max_token`` and ``max_seq_len``) before all_gather, so the
   collective shape is constant regardless of the current batch size or
   prefill/decode phase. Padding rows use ``topk_ids = -1`` which the
   Triton kernel's ``filter_expert`` path treats as no-op. This is the
   same mechanism DeepEP low-latency uses
   (``num_max_dispatch_tokens_per_rank``): fixed shapes are the cornerstone
   of CUDA-graph compatibility across mixed phases on different DP ranks.
3. Run the local PureTp filter trick (each rank processes only its 1/ep
   experts' share of every token).
4. All-reduce outputs across the world group (``Group.DP_AND_TP``) — this
   sums partial-expert contributions from every ep shard and also covers
   TP reduction in one shot.
5. Slice back to this rank's DP shard using ``dp_rank * max_size``.

CUDA graph: because all collective shapes are constant, a captured decode
graph on one rank can safely run concurrently with an eager prefill on
another DP rank — the NCCL shape protocol matches regardless. If a real
batch ever exceeds ``_dp_max_tokens_per_rank``, the ``prepare()`` asserts
rather than silently mis-shaping the collective.
"""
import os
from typing import Any, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
    PureTpRouterFp8PerBlock,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class PureTpRouterFp8PerBlockTritonDpTp(PureTpRouterFp8PerBlock):
    """FP8 per-block router for Triton fused MoE with DP+TP mixed topology.

    Accepts topologies where ``dp_size > 1`` and ``ep_size == tp_size * dp_size``,
    i.e. pure DP (``tp_size == 1``) or mixed TP+DP (``tp_size > 1``). For pure
    TP (``dp_size == 1``), use :class:`PureTpRouterFp8PerBlockTriton` instead.

    Produces row-major fp32 scales like
    :class:`PureTpRouterFp8PerBlockTriton` so the Triton
    ``invoke_fused_moe_kernel`` is happy.

    External gather mode
    --------------------
    Setting ``use_external_dp_gather = True`` (via the ``MOE_EXTERNAL_DP_GATHER``
    env, default on) tells the MoE layer (``GenericMoeLayer``) that it should
    pre-gather DP-local tokens *before* running ``gate`` / ``topk`` and slice
    back *after* ``finalize``. In that case this router skips its own
    ``_gather_dp_inputs`` and slicing — it becomes a pure-TP (global-view)
    router that only quantizes + filters its local expert shard. This mirrors
    sglang's ``StandardDispatcher`` architecture and cuts per-MoE-layer
    collectives from 3× all_gather(DP) + 1× all_reduce(world) down to
    1× all_reduce(world) (gather) + 1× all_reduce(world) (combine).
    """

    @property
    def supports_skip_allreduce(self) -> bool:
        return False

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        # Only activate for DP-enabled EP topology. Pure TP / single-GPU are
        # handled by :class:`PureTpRouterFp8PerBlockTriton`.
        has_dp = (
            config.dp_size > 1 and config.ep_size == config.tp_size * config.dp_size
        )
        checker.check(has_dp)
        # DP path does its own gather/reduce; use_all_gather flag is not
        # required.
        checker.check(resolver.use_all_gather(config) or has_dp)
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)
        self.dp_size = config.dp_size
        self.dp_rank = config.dp_rank
        # Fixed max-tokens-per-DP-rank sizing. Mirrors DeepEP LL's
        # ``num_max_dispatch_tokens_per_rank``: a constant buffer depth that
        # makes every collective shape a compile-time constant, so captured
        # CUDA graphs on one rank interop correctly with eager forwards on
        # another rank (e.g. decode-replay on rank 0 while rank 2 does eager
        # prefill). ``ll_num_max_token`` = ``concurrency_limit * (sp_gen+1)``
        # bounds decode; prefill can still carry up to ~max_seq_len tokens
        # per request, so we lift the floor to 128 — enough for typical
        # short prompts on TP+DP test configs without ballooning per-step
        # compute. If your prompts exceed this floor, set a larger
        # concurrency_limit or override via MOE_DP_MAX_TOKENS_PER_RANK.
        base = max(int(config.ll_num_max_token or 0), 128)
        override = os.environ.get("MOE_DP_MAX_TOKENS_PER_RANK")
        if override:
            base = max(base, int(override))
        # Round up to multiple of 8 for tile alignment on the Triton kernel.
        self._dp_max_tokens_per_rank = ((base + 7) // 8) * 8
        # External-gather mode: let the MoE layer do gather/scatter and make
        # this router act as a pure-TP (global-view) router. Default on; set
        # MOE_EXTERNAL_DP_GATHER=0 to revert to the internal triple-gather
        # path.
        self.use_external_dp_gather = (
            os.environ.get("MOE_EXTERNAL_DP_GATHER", "1") != "0"
        )
        # State carried from prepare() to finalize().
        self._dp_local_num_tokens: Optional[int] = None
        self._dp_padded_size: Optional[int] = None

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Default args: row-major fp32 scales, group_size = block_k = 128.
        return sgl_per_token_group_quant_fp8(a1, 128)

    def _gather_dp_inputs(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """All-gather (a1, topk_ids, topk_weights) across the DP group.

        We gather across ``Group.DP`` (ranks sharing the same ``tp_rank``) to
        avoid duplicating TP-sibling inputs. In pure DP (``tp_size == 1``)
        ``Group.DP`` resolves to the world group, so behavior matches the
        pure-DP path; in mixed TP+DP it gathers only dp_size payloads.

        Each rank pads its tensors to the fixed ``_dp_max_tokens_per_rank``
        (not runtime max). This keeps the collective shape constant across
        batches and phases — the same property DeepEP low-latency relies on
        for CUDA-graph friendliness. Padding rows carry ``topk_ids = -1``;
        the Triton kernel's ``filter_expert`` path treats these as no-op.
        """
        local_n = a1.size(0)
        padded = self._dp_max_tokens_per_rank
        # Fail loudly rather than silently overflowing into a smaller buffer.
        assert local_n <= padded, (
            f"TP+DP triton router: local_n={local_n} exceeds fixed "
            f"_dp_max_tokens_per_rank={padded}. Raise MOE_DP_MAX_TOKENS_PER_RANK "
            f"or increase concurrency_limit (current ll_num_max_token="
            f"{self.config.ll_num_max_token})."
        )

        if padded != local_n:
            pad_n = padded - local_n
            a1 = torch.cat(
                [
                    a1,
                    torch.zeros((pad_n, a1.size(1)), dtype=a1.dtype, device=a1.device),
                ],
                dim=0,
            )
            topk_ids = torch.cat(
                [
                    topk_ids,
                    torch.full(
                        (pad_n, topk_ids.size(1)),
                        -1,
                        dtype=topk_ids.dtype,
                        device=topk_ids.device,
                    ),
                ],
                dim=0,
            )
            topk_weights = torch.cat(
                [
                    topk_weights,
                    torch.zeros(
                        (pad_n, topk_weights.size(1)),
                        dtype=topk_weights.dtype,
                        device=topk_weights.device,
                    ),
                ],
                dim=0,
            )

        a1_g = all_gather(a1.contiguous(), group=Group.DP)
        ti_g = all_gather(topk_ids.contiguous(), group=Group.DP)
        tw_g = all_gather(topk_weights.contiguous(), group=Group.DP)
        return a1_g, ti_g, tw_g, padded

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> "ExpertForwardPayload":
        if self.use_external_dp_gather:
            # a1 is already the global [dp*padded, hidden] buffer gathered by
            # GenericMoeLayer; topk_ids/topk_weights were computed on that
            # global view. Just quant + local-expert filter.
            return super().prepare(a1, a1_scale, a2_scale, topk_weights, topk_ids)
        self._dp_local_num_tokens = a1.size(0)
        a1, topk_ids, topk_weights, padded = self._gather_dp_inputs(
            a1, topk_ids, topk_weights
        )
        self._dp_padded_size = padded
        return super().prepare(a1, a1_scale, a2_scale, topk_weights, topk_ids)

    def finalize(
        self,
        payload: "CombineForwardPayload",
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        # After the MoE kernel, each rank holds (padded * dp_size, hidden)
        # partial outputs from its own 1/ep expert shard. Summing over all
        # ep ranks produces the final per-token output; since
        # ep_size == world_size == Group.DP_AND_TP, a single world-group
        # all_reduce covers both the TP reduction (combining partial Col-parallel
        # experts within a TP replica) and the DP reduction (combining
        # different experts across DP groups).
        out = payload.fused_expert_output
        out = all_reduce(out, group=Group.DP_AND_TP)

        if self.use_external_dp_gather:
            # Layer will slice to the local DP shard.
            return out

        # Slice back to this rank's own DP shard. The DP-group gather laid
        # payloads out in dp_rank order, so slot ``dp_rank * padded`` holds
        # this rank's data.
        padded = self._dp_padded_size
        local_n = self._dp_local_num_tokens
        assert padded is not None and local_n is not None
        offset = self.dp_rank * padded
        out = out[offset : offset + local_n]
        self._dp_local_num_tokens = None
        self._dp_padded_size = None
        return out
