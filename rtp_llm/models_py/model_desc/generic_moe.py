from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FakeBalanceExpert,
    FMHAImplBase,
    FusedMoeFactory,
    GroupTopK,
    LinearFactory,
    MlaAttention,
    RMSNorm,
    RMSResNorm,
    SelectTopk,
    SigmoidGateScaleAdd,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

try:
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
        CudaFp8GEMMLinear,
    )
    from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
        fused_add_rmsnorm_fp8_quant,
        fused_add_rmsnorm_fp8_quant_with_bf16_output,
    )
except ImportError:
    CudaFp8GEMMLinear = None
    fused_add_rmsnorm_fp8_quant = None
    fused_add_rmsnorm_fp8_quant_with_bf16_output = None


class GenericMoeLayer(nn.Module):
    """Generic MoE layer supporting both Qwen3 and internal model."""

    _moe_stream: Optional[torch.cuda.Stream] = None

    @classmethod
    def _ensure_moe_stream(cls) -> torch.cuda.Stream:
        if cls._moe_stream is None:
            cls._moe_stream = torch.cuda.Stream()
        return cls._moe_stream

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.inter_size
        self.num_experts = config.eplb_config.phy_exp_num(config.expert_num)
        self.top_k = config.moe_k

        # Get quant_config from model_config
        quant_config = config.quant_config
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, quant_config, hw_kernel_config
        )
        self.select_topk = SelectTopk(config=config)
        if moe_config.fake_balance_expert:
            self.fake_balance_expert = FakeBalanceExpert(
                expert_num=config.expert_num,
                moe_k=config.moe_k,
                dp_rank=parallelism_config.dp_rank,
                dp_size=parallelism_config.dp_size,
                ep_size=parallelism_config.ep_size,
            )
        else:
            self.fake_balance_expert = None
        config_adapter = MoEConfigAdapter(
            model_config=config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            enable_cuda_graph=enable_cuda_graph,
        )
        self.fused_moe = FusedMoeFactory().create_fused_moe(config_adapter, weights)

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"
        self.num_local_experts = self.w1.shape[0]
        self.add_shared_expert = config.moe_style == 2
        self.ffn_tp_size = parallelism_config.get_ffn_tp_size()
        if self.add_shared_expert:
            self.shared_expert = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.shared_expert = None
        if weights.get(W.shared_expert_gate, None) is not None:
            self.shared_expert_gate = LinearFactory.create_linear_from_weights(
                weights, W.shared_expert_gate, None, None, config
            )
            self.sigmoid_gate_scale_add = SigmoidGateScaleAdd()
        else:
            self.shared_expert_gate = None
            self.sigmoid_gate_scale_add = None

        # for group topk
        self.correction_bias = weights.get(W.e_score_correction_b, None)

        self._use_two_stream = self.shared_expert is not None

    def _maybe_external_dp_gather(
        self, local_hidden: torch.Tensor
    ) -> "tuple[torch.Tensor, int, int, int]":
        """Optionally pre-gather local tokens across DP for the MoE block.

        Returns ``(working_hidden, local_n, dp_offset, padded_per_rank)``.
        When the router opts out (``use_external_dp_gather=False``) or we are
        not in a DP topology, ``working_hidden`` is just ``local_hidden`` and
        ``padded_per_rank == 0`` signals the caller to skip the post-slice.

        From MoE's point of view only EP matters: each rank owns a distinct
        1/ep shard of experts and needs every unique token. TP siblings carry
        identical inputs, so all_gather across ``Group.DP`` (the set of ranks
        sharing ``tp_rank``, i.e. one per DP shard) naturally produces the
        deduplicated global-token buffer without any zero-fill / world
        all_reduce gymnastics. Volume per rank = ``dp_size * padded * hidden``,
        ``dp_size`` times less NCCL bandwidth than a world all_reduce and
        skips the slow path entirely (~20µs vs ~90µs measured on H20).
        """
        router = getattr(self.fused_moe, "router", None)
        if not getattr(router, "use_external_dp_gather", False):
            return local_hidden, local_hidden.size(0), 0, 0
        dp_size = self.parallelism_config.dp_size
        if dp_size <= 1:
            return local_hidden, local_hidden.size(0), 0, 0

        padded = router._dp_max_tokens_per_rank
        local_n = local_hidden.size(0)
        assert local_n <= padded, (
            f"external DP gather: local_n={local_n} exceeds "
            f"_dp_max_tokens_per_rank={padded}."
        )
        dp_rank = self.parallelism_config.dp_rank
        offset = dp_rank * padded

        # sglang-style sparse-fill + all_reduce trick (saves ~100µs/call
        # vs multimem_all_gather): build the global [dp*padded, hidden]
        # buffer on every rank with each DP rank writing only its own
        # slot (others zero), then a single Group.DP all_reduce sums the
        # disjoint slots into the full concatenation. The DP group has
        # size 2 here, so the reduce dispatches to symm-mem's
        # two_shot_all_reduce kernel (~10-20µs/call) which is markedly
        # faster than multimem_all_gather (~114µs/call) at this payload
        # size on H20.
        global_hidden = torch.zeros(
            (dp_size * padded, local_hidden.size(1)),
            dtype=local_hidden.dtype,
            device=local_hidden.device,
        )
        if local_n > 0:
            global_hidden[offset : offset + local_n] = local_hidden
        global_hidden = all_reduce(global_hidden, group=Group.DP)
        return global_hidden, local_n, offset, padded

    def forward(
        self,
        hidden_states: torch.Tensor,
        x_fp8: "Optional[torch.Tensor]" = None,
        x_scale: "Optional[torch.Tensor]" = None,
    ) -> torch.Tensor:
        # External DP gather (only triggered when the router requests it and
        # dp_size > 1). Falls through to the existing local-only path
        # otherwise, so single-GPU / pure-TP / pure-DP routers are unaffected.
        working_hidden, local_n, dp_offset, padded = self._maybe_external_dp_gather(
            hidden_states
        )
        num_tokens, _ = working_hidden.shape
        router_logits = self.gate(working_hidden)
        router_logits_fp32 = router_logits.float()

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=working_hidden.device,
        )
        # different executor may need different topk_ids dtype
        topk_ids_dtype = self.fused_moe.topk_ids_dtype
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=topk_ids_dtype,
            device=working_hidden.device,
        )

        if self.correction_bias is not None:
            self.group_topk = GroupTopK()
            self.renormalize = self.config.has_moe_norm
            self.num_expert_group = self.config.moe_n_group

            self.topk_group = self.config.moe_topk_group
            self.n_routed_experts = self.config.expert_num  # config.n_routed_experts
            self.routed_scaling_factor = self.config.routed_scaling_factor
            self.group_topk(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                scores=router_logits_fp32,
                correction_bias=self.correction_bias,
                n_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
                renormalize=self.renormalize,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            # Top-K selection using C++ SelectTopkOp
            self.select_topk(router_logits_fp32, topk_ids, topk_weights)

        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)

        if not self._use_two_stream:
            experts_output = self.fused_moe(
                hidden_states=working_hidden,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation="SiGLU",
            )
            if padded > 0:
                experts_output = experts_output[dp_offset : dp_offset + local_n]
            return experts_output

        # Two-stream overlap is only safe when we can defer all NCCL/symm-mem
        # operations to after the join. Both MoE router.finalize() and shared
        # expert DenseMLP issue collectives on Group.TP; running them on
        # different CUDA streams concurrently corrupts results via the shared
        # communicator. When the router supports skip_allreduce we can defer
        # those collectives. When it doesn't (e.g. DeepEP whose finalize does
        # an essential all_gather), fall back to sequential execution.
        can_overlap = (
            self.ffn_tp_size <= 1 or self.fused_moe.supports_skip_allreduce
        ) and padded == 0

        if not can_overlap:
            experts_output = self.fused_moe(
                hidden_states=working_hidden,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation="SiGLU",
            )
            if padded > 0:
                experts_output = experts_output[dp_offset : dp_offset + local_n]
            shared_expert_output = self.shared_expert(
                hidden_states, x_fp8=x_fp8, x_scale=x_scale
            )
            if self.shared_expert_gate is not None:
                gate_output = self.shared_expert_gate(hidden_states)
                self.sigmoid_gate_scale_add(
                    gate_output, shared_expert_output, experts_output
                )
            else:
                experts_output = experts_output + shared_expert_output
            return experts_output

        skip_allreduce = self.ffn_tp_size > 1

        moe_stream = self._ensure_moe_stream()
        default_stream = torch.cuda.current_stream()

        # Fork: moe_stream picks up routing results from default stream.
        moe_stream.wait_stream(default_stream)

        # Side stream: routed MoE
        with torch.cuda.stream(moe_stream):
            experts_output = self.fused_moe(
                hidden_states=working_hidden,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation="SiGLU",
                skip_allreduce=skip_allreduce,
            )
            if padded > 0:
                experts_output = experts_output[dp_offset : dp_offset + local_n]

        # Main stream: shared expert
        shared_expert_output = self.shared_expert(
            hidden_states, x_fp8=x_fp8, x_scale=x_scale, skip_allreduce=skip_allreduce
        )
        if self.shared_expert_gate is not None:
            gate_output = self.shared_expert_gate(hidden_states)

        # Join: main stream waits for moe_stream before combining.
        default_stream.wait_stream(moe_stream)

        if self.shared_expert_gate is not None:
            self.sigmoid_gate_scale_add(
                gate_output, shared_expert_output, experts_output
            )
        else:
            experts_output = experts_output + shared_expert_output

        if skip_allreduce:
            experts_output = self.fused_moe.allreduce(experts_output)
        return experts_output


class DecodeLayerOutput:
    def __init__(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        self.hidden_states = hidden_states
        self.residual = residual


class GenericMoeDecoderLayer(nn.Module):
    """Generic MoE decoder layer supporting Dense/MoE hybrid and shared experts."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Get quant_config from model_config
        quant_config = config.quant_config
        if config.attn_config.use_mla:
            self.self_attn = MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                global_weights=global_weights,
            )
        else:
            attn_configs = config.getAttentionConfigs(
                parallelism_config.get_attn_tp_size()
            )
            self.self_attn = CausalAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                layer_idx,
            )

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type, parallelism_config, weights, quant_config
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )

        # 使用 RMSResNorm 来 fuse residual add 和 layernorm
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> DecodeLayerOutput:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return DecodeLayerOutput(hidden_states, residual)


class GenericMoeModel(GptModelBase):
    """Generic MoE model supporting Qwen3-MoE, internal model, and other MoE architectures."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        # Determine attention_type from model_config.attn_config.use_mla
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]
        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = output.hidden_states
            residual = output.residual

        hidden_states, _ = self.norm(hidden_states, residual)

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "GenericMoeLayer",
    "GenericMoeDecoderLayer",
    "GenericMoeModel",
]
