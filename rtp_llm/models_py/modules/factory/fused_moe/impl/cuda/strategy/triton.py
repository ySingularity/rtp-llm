"""CUDA Triton fused-MoE strategies (sglang TritonRunnerCore port).

These strategies pair the ``TritonFusedMoeExecutor`` with the existing
``PureTpRouter*`` routers. They cover the no-quant and FP8 W8A8 paths and are
intentionally gated behind explicit ``moe_strategy`` names so they don't
preempt DeepGEMM/DeepEP on H20+ via the ``auto`` selector. Use them by setting
``moe_strategy`` to ``triton_no_quant`` / ``triton_fp8_per_tensor`` /
``triton_fp8_per_block``.
"""

from typing import Any

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class CudaTritonNoQuantStrategy(MoeStrategy):
    """BF16/FP16 Triton fused MoE (no quant) on PureTP routing."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)
        checker.check(config.moe_strategy == "triton_no_quant")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_moe_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterNoQuant,
        )

        return StrategyAttributes(
            router_class=PureTpRouterNoQuant,
            executor_class=TritonFusedMoeExecutor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )


class CudaTritonNoQuantCustomARStrategy(MoeStrategy):
    """BF16/FP16 Triton fused MoE with JIT-compiled custom allreduce.

    Same executor as CudaTritonNoQuantStrategy but replaces NCCL allreduce
    with IPC-based cross_device_reduce (vllm V1). Gated by explicit
    moe_strategy='triton_no_quant_custom_ar'.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)
        checker.check(config.moe_strategy == "triton_no_quant_custom_ar")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_moe_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.cross_device_reduce_router import (
            CrossDeviceReduceRouterNoQuant,
        )

        return StrategyAttributes(
            router_class=CrossDeviceReduceRouterNoQuant,
            executor_class=TritonFusedMoeExecutor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )


class CudaTritonFp8PerTensorStrategy(MoeStrategy):
    """FP8 per-tensor (per-token activation) Triton fused MoE on PureTP routing."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method in ("FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR")
        )
        checker.check(config.moe_strategy == "triton_fp8_per_tensor")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_moe_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerTensor,
        )

        return StrategyAttributes(
            router_class=PureTpRouterFp8PerTensor,
            executor_class=TritonFusedMoeExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=True,
            ),
        )


class CudaTritonFp8PerBlockStrategy(MoeStrategy):
    """FP8 per-block (128x128) Triton fused MoE on PureTP routing.

    This is the analogue of the sglang MTP path observed in profiling: pure-TP
    routing + Triton fused_moe_kernel + (small-token torch.compile reduce or
    moe_sum_reduce). Use this when DeepGEMM is unavailable or when the
    cuda-graph-friendly Triton path is preferred for decode shapes.

    Covers single-GPU / pure-TP / pure-DP topologies via
    :class:`PureTpRouterFp8PerBlockTriton`. Mixed TP+DP is handled by
    :class:`CudaTritonFp8PerBlockDpTpStrategy` which shares the same
    ``moe_strategy`` name; the two strategies are mutually exclusive via
    their router's ``check_conditions``.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(config.moe_strategy == "triton_fp8_per_block")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_moe_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerBlockTriton,
        )

        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlockTriton,
            executor_class=TritonFusedMoeExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
        )


class CudaTritonFp8PerBlockCustomARStrategy(MoeStrategy):
    """FP8 per-block Triton fused MoE with JIT-compiled custom allreduce.

    Same executor as CudaTritonFp8PerBlockStrategy but replaces NCCL allreduce
    with IPC-based cross_device_reduce (vllm V1). Gated by explicit
    moe_strategy='triton_fp8_per_block_custom_ar'.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(config.moe_strategy == "triton_fp8_per_block_custom_ar")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_moe_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.cross_device_reduce_router import (
            CrossDeviceReduceRouterFp8PerBlock,
        )

        return StrategyAttributes(
            router_class=CrossDeviceReduceRouterFp8PerBlock,
            executor_class=TritonFusedMoeExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
        )


class CudaTritonFp8PerBlockDpTpStrategy(MoeStrategy):
    """FP8 per-block (128x128) Triton fused MoE for DP+TP mixed topology.

    Pairs the Triton fused MoE executor with
    :class:`PureTpRouterFp8PerBlockTritonDpTp`, which follows the sglang
    ``StandardDispatcher`` pattern: gather across the DP group (ranks sharing
    ``tp_rank``, avoiding TP-sibling duplication), filter+run Triton kernel
    locally, then world-group all_reduce and slice back to the local DP
    shard.

    Gated by both the ``triton_fp8_per_block`` ``moe_strategy`` name (so it
    coexists with :class:`CudaTritonFp8PerBlockStrategy`) and by the
    DP-enabled topology check inside the router — the two strategies never
    match the same config simultaneously.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(config.moe_strategy == "triton_fp8_per_block")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_moe_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router_dp_tp import (
            PureTpRouterFp8PerBlockTritonDpTp,
        )

        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlockTritonDpTp,
            executor_class=TritonFusedMoeExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
        )
