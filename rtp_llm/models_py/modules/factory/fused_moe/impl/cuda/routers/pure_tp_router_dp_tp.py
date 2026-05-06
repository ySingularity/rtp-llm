# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/token_dispatcher/standard.py
"""DP+TP-aware FP8 per-block router for the Triton fused-MoE executor.

This router handles topologies where both ``tp_size > 1`` and ``dp_size > 1``
(with ``ep_size == tp_size * dp_size``), in addition to pure DP and pure TP.
It extends :class:`PureTpRouterFp8PerBlock` by following the sglang
``StandardDispatcher`` pattern:

1. Gather tokens only across the DP group (ranks that share ``tp_rank``);
   TP siblings already hold the same input batch, so including them would
   duplicate work.
2. Pad locally to ``max(local_n across DP)`` before all_gather so the
   collective works on a uniform shape. Padding rows use ``topk_ids = -1``
   which the Triton kernel's ``filter_expert`` path treats as no-op.
3. Run the local PureTp filter trick (each rank processes only its 1/ep
   experts' share of every token).
4. All-reduce outputs across the world group (``Group.DP_AND_TP``) — this
   sums partial-expert contributions from every ep shard and also covers
   TP reduction in one shot.
5. Slice back to this rank's DP shard using ``dp_rank * max_size``.

CUDA graph caveat: a captured graph embeds all_gather with a fixed shape. In
mixed TP+DP, different DP ranks may do different phases (prefill vs decode)
with different ``local_n``, so the captured graphs don't line up. Enable
this router only for eager execution (``--enable_cuda_graph 0``) or make
sure the engine pads every DP rank to the same shape per step.
"""
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
    """

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

        Each rank pads its tensors to ``max(local_n across DP)`` so the
        collective gets a uniform shape. Padding rows carry
        ``topk_ids = -1``; the triton kernel's ``filter_expert`` path treats
        these as no-op.
        """
        local_n = a1.size(0)
        if torch.cuda.is_current_stream_capturing():
            # During capture we cannot GPU->CPU sync for max(). Assume the
            # engine keeps DP shards balanced at fixed decode batch sizes
            # listed in decode_capture_config. If this assumption fails at
            # replay time, the collective shapes will mismatch and NCCL
            # will hang.
            padded = local_n
        else:
            local_size = torch.tensor([local_n], device=a1.device, dtype=torch.long)
            sizes = all_gather(local_size, group=Group.DP)
            padded = int(sizes.max().item())

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
