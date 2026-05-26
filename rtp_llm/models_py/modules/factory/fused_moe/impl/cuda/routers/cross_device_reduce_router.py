"""Cross-device reduce router with pre-compiled custom allreduce.

Uses the vllm V1 IPC-based custom allreduce (cross_device_reduce_1stage /
cross_device_reduce_2stage) instead of torch.distributed NCCL allreduce in
the finalize step. The CUDA kernel is pre-compiled via Bazel into
librtp_compute_ops.so and accessed through rtp_llm.ops bindings.

Pairs with TritonFusedMoeExecutor for the full MoE path:
  fused_moe_kernel → silu_and_mul → fused_moe_kernel → moe_sum_reduce
  → custom_all_reduce (1-stage or 2-stage auto-selected)
"""

import logging
import os
from typing import Any, Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, _get_group, all_reduce
from rtp_llm.models_py.distributed.cuda_graph_hooks import (
    register_post_capture_callback,
    unregister_post_capture_callback,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
    PureTpRouterFp8PerBlockTriton,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)

logger = logging.getLogger(__name__)


def _get_custom_ar_ops():
    """Load custom allreduce ops from pre-compiled librtp_compute_ops module."""
    import librtp_compute_ops.rtp_llm_ops as rtp_ops

    class _Ops:
        init_custom_ar = staticmethod(rtp_ops.custom_ar_init)
        all_reduce = staticmethod(rtp_ops.custom_ar_all_reduce)
        dispose = staticmethod(rtp_ops.custom_ar_dispose)
        meta_size = staticmethod(rtp_ops.custom_ar_meta_size)
        register_buffer = staticmethod(rtp_ops.custom_ar_register_buffer)
        allocate_shared_buffer_and_handle = staticmethod(
            rtp_ops.custom_ar_allocate_shared_buffer_and_handle
        )
        open_mem_handle = staticmethod(rtp_ops.custom_ar_open_mem_handle)
        free_shared_buffer = staticmethod(rtp_ops.custom_ar_free_shared_buffer)
        get_graph_buffer_ipc_meta = staticmethod(
            rtp_ops.custom_ar_get_graph_buffer_ipc_meta
        )
        register_graph_buffers = staticmethod(rtp_ops.custom_ar_register_graph_buffers)

    return _Ops()


class CrossDeviceReduceRouterFp8PerBlock(PureTpRouterFp8PerBlockTriton):
    """FP8 per-block router that uses pre-compiled custom allreduce for finalize.

    Extends PureTpRouterFp8PerBlockTriton — inherits FP8 per-block quant with
    row-major scales and recompute_topk. Overrides __init__ (IPC setup) and
    finalize() (custom AR with NCCL fallback for oversized tensors).

    Supports both pure TP (dp=1) and mixed DP+TP (dp>1) topologies.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        has_dp_tp = (
            config.dp_size > 1 and config.ep_size == config.tp_size * config.dp_size
        )
        checker.check(
            resolver.is_single_gpu(config)
            or resolver.is_tp_equal_ep(config)
            or has_dp_tp
        )

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)
        self._tp_rank = config.tp_rank
        self.dp_size = config.dp_size
        self.dp_rank = config.dp_rank
        if self.dp_size > 1:
            base = max(int(config.ll_num_max_token or 0), 128)
            override = os.environ.get("MOE_DP_MAX_TOKENS_PER_RANK")
            if override:
                base = max(base, int(override))
            self._dp_max_tokens_per_rank = ((base + 7) // 8) * 8
            self.use_external_dp_gather = (
                os.environ.get("MOE_EXTERNAL_DP_GATHER", "1") != "0"
            )
        self._custom_ar_enabled = False
        if self.tp_size > 1:
            try:
                self._setup_custom_allreduce()
            except Exception as e:
                logger.warning(
                    "Custom allreduce setup failed, falling back to NCCL: %s", e
                )
                self._custom_ar_enabled = False

    def _setup_custom_allreduce(self):
        custom_ar = _get_custom_ar_ops()

        meta_sz = custom_ar.meta_size()
        self._max_ar_size = 8 * 1024 * 1024
        buf_size = meta_sz + self._max_ar_size
        self._buffer_ptr, self._ipc_handle = (
            custom_ar.allocate_shared_buffer_and_handle(buf_size)
        )

        tp_group = _get_group(Group.TP)
        all_handles = [None] * self.tp_size
        torch.distributed.all_gather_object(
            all_handles, self._ipc_handle, group=tp_group
        )

        ipc_ptrs = []
        for i in range(self.tp_size):
            if i == self._tp_rank:
                ipc_ptrs.append(self._buffer_ptr)
            else:
                ipc_ptrs.append(custom_ar.open_mem_handle(all_handles[i]))

        rank_data = torch.zeros(
            16 * 1024 * 1024,
            dtype=torch.uint8,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        self._fa_ptr = custom_ar.init_custom_ar(
            ipc_ptrs, rank_data, self._tp_rank, True
        )
        self._rank_data = rank_data

        data_ptrs = [ptr + meta_sz for ptr in ipc_ptrs]
        custom_ar.register_buffer(self._fa_ptr, data_ptrs)
        self._reg_buffer = data_ptrs[self.ep_rank]
        self._custom_ar = custom_ar
        self._custom_ar_enabled = True

        self._post_capture_cb = self._register_graph_buffers
        register_post_capture_callback(self._post_capture_cb)

        logger.info(
            "Custom allreduce initialized: tp_rank=%d, tp_size=%d, "
            "meta_size=%d, max_ar_size=%d",
            self._tp_rank,
            self.tp_size,
            meta_sz,
            self._max_ar_size,
        )

    def _register_graph_buffers(self):
        """Exchange IPC handles for graph-captured buffers and register them.

        Called by C++ finish_capture_session() after each CUDA graph capture.
        The custom allreduce kernel records unregistered buffer pointers during
        capture in graph_unreg_buffers_. This method exchanges their IPC handles
        across TP ranks and fills in the peer pointers so graph replay works.
        """
        handle_and_offsets = self._custom_ar.get_graph_buffer_ipc_meta(self._fa_ptr)
        handles, offsets = handle_and_offsets
        if not offsets:
            return

        tp_group = _get_group(Group.TP)
        all_handles = [None] * self.tp_size
        all_offsets = [None] * self.tp_size
        torch.distributed.all_gather_object(all_handles, handles, group=tp_group)
        torch.distributed.all_gather_object(all_offsets, offsets, group=tp_group)
        self._custom_ar.register_graph_buffers(self._fa_ptr, all_handles, all_offsets)
        logger.debug(
            "Registered %d graph buffers for custom AR (tp_rank=%d)",
            len(offsets),
            self._tp_rank,
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        if self._needs_dp_gather:
            return super().finalize(
                payload,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                extra_finalize_args,
            )

        out = payload.fused_expert_output

        if skip_allreduce or self.tp_size <= 1:
            return out

        # DP+TP topology (ep_size == tp_size * dp_size): need world allreduce
        # to combine partial-expert contributions across ALL ep ranks —
        # TP reduction (within TP group) + DP reduction (across DP groups).
        # Use a single world allreduce (same as PureTpRouterFp8PerBlockTritonDpTp).
        # Custom AR only covers the TP group and cannot replace the world reduce.
        if self.dp_size > 1 and getattr(self, "use_external_dp_gather", False):
            return all_reduce(out, group=Group.DP_AND_TP)

        # Pure TP (dp=1): custom AR or NCCL within TP group.
        if self._custom_ar_enabled:
            inp_size = out.numel() * out.element_size()
            if inp_size <= self._max_ar_size and inp_size % 16 == 0:
                result = torch.empty_like(out)
                self._custom_ar.all_reduce(
                    self._fa_ptr,
                    out,
                    result,
                    self._reg_buffer,
                    self._max_ar_size,
                )
                return result

        return all_reduce(out, group=Group.TP)

    def allreduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._custom_ar_enabled:
            inp_size = tensor.numel() * tensor.element_size()
            if inp_size <= self._max_ar_size and inp_size % 16 == 0:
                result = torch.empty_like(tensor)
                self._custom_ar.all_reduce(
                    self._fa_ptr,
                    tensor,
                    result,
                    self._reg_buffer,
                    self._max_ar_size,
                )
                return result
        return all_reduce(tensor, group=Group.TP)

    def __del__(self):
        if hasattr(self, "_post_capture_cb"):
            unregister_post_capture_callback(self._post_capture_cb)
        if hasattr(self, "_fa_ptr") and hasattr(self, "_custom_ar"):
            try:
                self._custom_ar.dispose(self._fa_ptr)
            except Exception:
                pass
        if hasattr(self, "_buffer_ptr") and hasattr(self, "_custom_ar"):
            try:
                self._custom_ar.free_shared_buffer(self._buffer_ptr)
            except Exception:
                pass
