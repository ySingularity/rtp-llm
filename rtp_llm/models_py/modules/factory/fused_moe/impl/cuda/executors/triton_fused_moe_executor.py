# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/triton.py
# Adapted for RTP-LLM's FusedMoeExpertExecutor interface. Wraps the Triton
# fused_moe path (see rtp_llm.models_py.triton_kernels.moe.fused_moe_triton)
# so it can be selected by the MoE strategy registry.
# Licensed under the Apache License, Version 2.0
from typing import Any, Dict, Optional

import torch

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
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_triton import fused_experts_impl
from rtp_llm.utils.model_weight import W


class TritonFusedMoeExecutor(FusedMoeExpertExecutor):
    """Triton fused_moe_kernel + custom moe_sum_reduce executor (no DeepEP).

    This is the RTP-LLM port of sglang's ``TritonRunnerCore``. It expects the
    pure-TP routing layout produced by ``PureTpRouter*``: ``expert_x`` is a
    plain ``(num_tokens, hidden_size)`` tensor and ``expert_topk_ids`` /
    ``expert_topk_weights`` are ``(num_tokens, top_k)``.
    """

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.BATCHED_TRITON

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        # Currently supports the no-quant and FP8 W8A8 paths (per-tensor /
        # per-token / per-block). Other quant schemes should fall through to
        # their dedicated executors.
        checker.check(
            quant_method is None
            or quant_method
            in (
                "FP8_PER_BLOCK",
                "FP8_PER_TENSOR_COMPRESSED",
                "FP8_DYNAMIC_PER_TENSOR",
            )
        )

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num
        assert self.num_experts % self.ep_size == 0
        self.num_local_experts = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_local_experts
        self.top_k = config.moe_k

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

        # On SM 10.x the weight loader converts per-block FP8 weights to
        # UE8M0 format (both the fp8 values and scales are transformed).
        # The Triton kernel expects standard per-block fp8 with row-major
        # fp32 scales. Convert back at init time: dequant → bf16 → re-quant
        # with standard per-block fp8.
        if self.use_fp8_w8a8 and self.block_shape is not None:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                is_deep_gemm_e8m0_used,
            )

            if is_deep_gemm_e8m0_used():
                self._convert_ue8m0_to_standard_fp8()

        # Filter sentinel-marked rows when EP is in use (some topk_ids may be -1
        # after PureTpRouter recompute).
        self.filter_expert = self.num_local_experts != self.num_experts

        # Persistent intermediate buffer pool. Replaces per-call
        # ``torch.zeros`` for cache1/2/3 + ``torch.empty`` for the output —
        # one allocation, one zero-fill kernel per MoE forward instead of
        # four. Lazily sized on first call (during eager warmup, before
        # CUDA graph capture). The buffer is held by the executor so it
        # stays alive across graph capture/replay.
        self._buf_pool: Optional[torch.Tensor] = None
        self._buf_pool_capacity: int = 0
        self._buf_offsets: Optional[tuple] = None

        # Persistent scratch buffers for ``moe_align_block_size`` internals.
        # Replaces the per-call ``torch.empty`` / ``torch.zeros`` /
        # ``torch.full`` chain (~12 element-wise ops) with in-place
        # ``zero_()`` / ``fill_()`` on cached tensors. Sized for the
        # worst-case (num_valid_tokens, max_pad) seen during warmup.
        self._align_scratch: Optional[Dict[str, torch.Tensor]] = None
        self._align_scratch_capacity: tuple = (0, 0)  # (num_valid_tokens, max_pad)

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    @staticmethod
    def _unpack_ue8m0_packed_scales(
        packed: torch.Tensor,
        weight_rows: int,
        weight_cols: int,
        block_size: int = 128,
    ) -> torch.Tensor:
        """Reverse ``_transform_scale_ue8m0`` + ``get_mn_major_tma_aligned_packed_ue8m0_tensor``.

        Args:
            packed: int32 tensor ``(..., N, packed_k)`` from the weight
                loader.  ``N`` = ``weight_rows``.
            weight_rows: N dimension of the weight matrix.
            weight_cols: K dimension of the weight matrix.
            block_size: quantization block size (default 128).

        Returns:
            fp32 scale tensor ``(..., n_blocks, k_blocks)`` in row-major.
        """
        assert packed.dtype == torch.int32

        batch_dims = packed.shape[:-2]
        mn = packed.shape[-2]
        packed_k = packed.shape[-1]

        flat = packed.reshape(-1, mn, packed_k)
        b = flat.shape[0]

        # The packing stores data in column-major layout via a transpose.
        # .contiguous() converts back to row-major. Flatten per-batch to
        # allow the int32→uint8 dtype view, then reshape to unpack each
        # int32 into 4 UE8M0 exponent bytes along k.
        flat_bytes = (
            flat.contiguous()
            .reshape(b, -1)
            .view(torch.uint8)
            .reshape(b, mn, packed_k * 4)
        )

        # Convert UE8M0 uint8 exponents → fp32.
        fp32_scales = (flat_bytes.to(torch.int32) << 23).view(torch.float32)

        n_blocks = (weight_rows + block_size - 1) // block_size
        k_blocks = (weight_cols + block_size - 1) // block_size

        # Subsample every block_size rows and trim k padding.
        block_scales = fp32_scales[:, ::block_size, :k_blocks]

        return block_scales.reshape(*batch_dims, n_blocks, k_blocks).contiguous()

    def _convert_ue8m0_to_standard_fp8(self) -> None:
        """Convert UE8M0-format weights+scales to standard per-block FP8.

        On SM 10.x, the weight loader transforms MoE weights via
        ``requant_weight_ue8m0``: it dequantizes the original per-block FP8
        weights to bf16, then re-quantizes with UE8M0-rounded scales, and
        finally packs the scales into a column-major TMA-aligned int32
        layout for DeepGemm consumption.

        The Triton kernel needs standard per-block FP8 with row-major fp32
        scales.  This method reverses the process at init time:
        1. Unpack UE8M0 packed int32 scales → fp32 block-level scales
        2. Dequant FP8 weights → bf16 using those scales
        3. Re-quant bf16 → standard per-block FP8 with fp32 scales
        """
        import logging as _logging

        _log = _logging.getLogger(__name__)
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            block_quant_dequant,
            per_block_cast_to_fp8,
        )

        block_n, block_k = self.block_shape

        for attr_w, attr_s, name in [
            ("w13_weight", "w13_weight_scale", "w13"),
            ("w2_weight", "w2_weight_scale", "w2"),
        ]:
            w = getattr(self, attr_w)
            s = getattr(self, attr_s)
            if w is None or s is None or s.dtype != torch.int32:
                continue

            E_local, N, K = w.shape

            fp32_scales = self._unpack_ue8m0_packed_scales(s, N, K, block_n)

            # Convert per-expert on CPU, write back in-place to avoid
            # OOM on large MoE (397B, 128 experts/rank, 184GB GPU
            # nearly full). Only scales need a new tensor (shape changes
            # from packed int32 to fp32 block grid).
            device = w.device
            new_s_list = []
            for e in range(E_local):
                w_cpu = w[e].cpu()
                s_cpu = fp32_scales[e].cpu()
                bf16_cpu = block_quant_dequant(
                    w_cpu.unsqueeze(0),
                    s_cpu.unsqueeze(0),
                    self.block_shape,
                    torch.bfloat16,
                ).squeeze(0)
                del w_cpu, s_cpu
                bf16_gpu = bf16_cpu.to(device)
                del bf16_cpu
                w_e, s_e = per_block_cast_to_fp8(bf16_gpu, use_ue8m0=False)
                del bf16_gpu
                w[e].copy_(w_e)
                del w_e
                new_s_list.append(s_e)

            new_s = torch.stack(new_s_list)
            _log.info(
                "UE8M0→standard FP8 convert %s: %s(%s), scales %s",
                name,
                tuple(w.shape),
                w.dtype,
                tuple(new_s.shape),
            )
            setattr(self, attr_s, new_s)

    def _ensure_align_scratch(
        self,
        num_valid_tokens: int,
        block_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Pre-allocate scratch buffers for moe_align_block_size.

        Uses ``num_local_experts`` (the physical expert count after EP
        sharding) since ``fused_experts_impl`` calls ``moe_align_block_size``
        with ``E = w1.shape[0]`` which equals the local expert count.
        ``cum[-1:]`` returns ``num_tokens_post_padded``, so the array must
        be sized to exactly ``num_local_experts + 1``.

        The fused padcum+expert_ids+fill kernel writes cum, sorted_ids
        padding slots, AND slot_counter directly — no standalone zero/fill
        calls needed in the scratch path.
        """
        num_e = self.num_local_experts
        max_pad = num_valid_tokens + num_e * block_size
        max_pad = ((max_pad + block_size - 1) // block_size) * block_size
        max_num_blocks = max_pad // block_size
        cap_n, cap_p = self._align_scratch_capacity
        if self._align_scratch is None or cap_n < num_valid_tokens or cap_p < max_pad:
            self._align_scratch = {
                "bucket": torch.empty(
                    num_valid_tokens, dtype=torch.int64, device=device
                ),
                "expert_count": torch.empty(
                    num_e + 1, dtype=torch.int64, device=device
                ),
                "cum": torch.empty(num_e + 1, dtype=torch.int64, device=device),
                "slot_counter": torch.empty(num_e, dtype=torch.int32, device=device),
                "sorted_ids": torch.empty(max_pad, dtype=torch.int32, device=device),
                "expert_ids": torch.empty(
                    max_num_blocks, dtype=torch.int32, device=device
                ),
            }
            self._align_scratch_capacity = (num_valid_tokens, max_pad)
        return self._align_scratch

    def _ensure_buffers(
        self,
        num_tokens: int,
        topk: int,
        N: int,
        hidden: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """Get pre-allocated cache1/cache2/cache3/out views from the pool.

        On first call (or when the requested ``num_tokens`` exceeds the
        current capacity) the pool is grown — this only happens during
        eager warmup, never during a captured CUDA graph since
        ``num_tokens`` is constant in external-DP-gather mode.
        """
        size_c1 = num_tokens * topk * N
        size_c2 = num_tokens * topk * (N // 2)
        size_c3 = num_tokens * topk * hidden
        size_out = num_tokens * hidden
        # Cache region (must be zeroed for filtered-row correctness) +
        # output region (overwritten by sum-reduce, no zero needed).
        cache_total = size_c1 + size_c2 + size_c3
        total = cache_total + size_out

        if self._buf_pool is None or self._buf_pool_capacity < total:
            self._buf_pool = torch.empty(total, dtype=dtype, device=device)
            self._buf_pool_capacity = total

        # One kernel zeroes cache1+cache2+cache3 in a contiguous span,
        # replacing three torch.zeros allocations.
        self._buf_pool[:cache_total].zero_()

        b = self._buf_pool
        c1 = b[:size_c1].view(num_tokens * topk, N)
        c2 = b[size_c1 : size_c1 + size_c2].view(num_tokens * topk, N // 2)
        c3 = b[size_c1 + size_c2 : cache_total].view(num_tokens, topk, hidden)
        out = b[cache_total : cache_total + size_out].view(num_tokens, hidden)
        return c1, c2, c3, out

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[Dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None
        # PureTpRouter produces 2D ``expert_x`` with shape (num_tokens, K).
        assert (
            payload.expert_x.dim() == 2
        ), f"TritonFusedMoeExecutor expects 2D expert_x, got {payload.expert_x.shape}"

        # Activation arrives in upstream's stylized form (e.g. "SiGLU"); the
        # Triton kernel only knows the lower-cased gate part.
        act = activation.lower()
        if "silu" in act or "swiglu" in act or "siglu" in act:
            act = "silu"
        elif "gelu" in act:
            act = "gelu"
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # PureTpRouter passes either the raw fp8 quantized tensor or a bf16
        # tensor that we must quantize ourselves. fused_experts_impl handles
        # both branches via use_fp8_w8a8.
        hidden_states = payload.expert_x
        a1_scale = payload.expert_x_scale
        if self.use_fp8_w8a8 and hidden_states.dtype != torch.float8_e4m3fn:
            # Router did not pre-quantize: let the orchestrator do it.
            a1_scale = None

        topk_ids = payload.expert_topk_ids.to(torch.int32)
        topk_weights = payload.expert_topk_weights

        # Translate GLOBAL topk_ids to LOCAL ids when EP is active and the
        # router did not pre-translate. PureTpRouterNoQuant has
        # do_recompute_topk=False so it passes through global ids unchanged;
        # without this translation the downstream ``moe_align_block_size``
        # filter ``raw < num_local_experts`` drops the actual local experts
        # on ep_rank > 0 (whose global ids are in [start, start+num_local))
        # and instead keeps OOB ids that index into the wrong weight rows.
        # The OLD ``triton_fused_executor.py`` (used by no_auant_cpp strategy)
        # has this same translation at lines 85-89; this executor was missing it.
        if self.ep_size > 1:
            local_ids = topk_ids - self.start_expert_id
            local_mask = (local_ids >= 0) & (local_ids < self.num_local_experts)
            topk_ids = local_ids.clamp(min=0, max=self.num_local_experts - 1).to(
                torch.int32
            )
            topk_weights = topk_weights * local_mask

        # Output dtype must match the un-quantized model dtype, not the (FP8)
        # storage dtype of ``hidden_states`` after a router pre-quant.
        out_dtype = payload.expert_x_origin_dtype or hidden_states.dtype

        # Pre-allocate / reuse intermediate buffers from this executor's pool.
        effective_dtype = (
            torch.bfloat16 if out_dtype == torch.float8_e4m3fn else out_dtype
        )
        num_tokens = hidden_states.shape[0]
        N = self.w13_weight.shape[1]
        hidden = self.w2_weight.shape[1]
        c1, c2, c3, out_buf = self._ensure_buffers(
            num_tokens=num_tokens,
            topk=topk_ids.shape[1],
            N=N,
            hidden=hidden,
            dtype=effective_dtype,
            device=hidden_states.device,
        )
        # moe_align scratch: sized for num_valid_tokens = num_tokens × topk
        # at the worst case. block_size is read from the same try_get_optimal
        # config that fused_experts_impl uses, so values agree.
        from rtp_llm.models_py.triton_kernels.moe.fused_moe_triton_config import (
            get_config_dtype_str,
            try_get_optimal_moe_config,
        )

        align_cfg = try_get_optimal_moe_config(
            self.w13_weight.shape,
            (
                self.w2_weight.shape[0],
                self.w2_weight.shape[1],
                self.w2_weight.shape[2],
            ),
            topk_ids.shape[1],
            get_config_dtype_str(use_fp8_w8a8=self.use_fp8_w8a8, dtype=effective_dtype),
            num_tokens,
            block_shape=self.block_shape,
        )
        align_scratch = self._ensure_align_scratch(
            num_valid_tokens=num_tokens * topk_ids.shape[1],
            block_size=align_cfg["BLOCK_SIZE_M"],
            device=hidden_states.device,
        )

        # Extract pre-computed bucket/expert_count from fused
        # recompute_topk_and_align_count (produced by the router's prepare()).
        pre_bucket = None
        pre_expert_count = None
        if payload.expert_tokens_meta is not None:
            pre_bucket = payload.expert_tokens_meta.align_bucket
            pre_expert_count = payload.expert_tokens_meta.align_expert_count

        out = fused_experts_impl(
            hidden_states=hidden_states.contiguous(),
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_weights=topk_weights.contiguous(),
            topk_ids=topk_ids.contiguous(),
            inplace=False,
            activation=act,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=self.use_fp8_w8a8,
            per_channel_quant=self.per_channel_quant,
            w1_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=self.block_shape,
            filter_expert=self.filter_expert,
            out_dtype=out_dtype,
            intermediate_cache1=c1,
            intermediate_cache2=c2,
            intermediate_cache3=c3,
            out_hidden_states=out_buf,
            align_scratch=align_scratch,
            pre_bucket=pre_bucket,
            pre_expert_count=pre_expert_count,
        )

        return CombineForwardPayload(fused_expert_output=out)
