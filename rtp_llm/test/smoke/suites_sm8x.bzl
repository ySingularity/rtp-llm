load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def sm8x_suites():
    # SM8x (L20)
    # ============================================================================

    # SM8x basic framework features (L20, includes dense qwen2.5/qwen7b + embedding)
    native.test_suite(
        name = "smoke_sm8x_basic",
        tests = [
            smoke_test(
                name="softmax_probs",
                task_info="data/model/qwen25/q_r_softmax_probs.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["L20"],
            ),
            smoke_test(
                name="random_seed",
                task_info="data/model/qwen25/test_random_seed.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["L20"],
            ),
            smoke_test(
                name="fp16",
                task_info="data/model/qwen25/q_r_s.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["L20"],
            ),
            smoke_test(
                name="bf16",
                task_info="data/model/qwen25/q_r_s.json",
                smoke_args="--act_type BF16 --warm_up 0",
                gpu_type=["L20"],
            ),
            smoke_test(
                name="tp2",
                task_info="data/model/qwen25/q_r_s_fp16.json",
                smoke_args="--warm_up 0 --act_type FP16 --tp_size 2",
                gpu_type=["L20"],
            ),
            smoke_test(
                name="beam_search_tp2",
                task_info="data/model/qwen25/bs_q_r.json",
                smoke_args="--act_type FP16 --tp_size 2 --warm_up 0",
                gpu_type=["L20"],
            ),
            smoke_test(
                name="frontend_app",
                task_info="data/model/qwen25/q_r_3_front_app.json",
                gpu_type=["L20"],
                smoke_args= {
                    "frontend": "--max_seq_len 2048 --role_type FRONTEND --warm_up 0",
                    "pd_fusion": "--reuse_cache 1 --seq_size_per_block 8 --act_type FP16 --warm_up 0"
                }
            ),
            # Simplified from qwen_7b → qwen2.5-0.5B; result/logits placeholders need rewrite_smoke regen
            smoke_test(
                name="logits_index",
                task_info="data/model/qwen25/logits_index_q_r.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["L20"]
            ),
            # Migrated from smoke_sm8x_embedding (RTX_3090 → L20); golden needs rewrite_smoke regen
            smoke_test(
                name="embedding_qwen_gte_7b_cudagraph",
                task_info="data/model/qwen2/q_r_embedding.json",
                smoke_args="--seq_size_per_block 64 --embedding_model 1 --act_type BF16 --concurrency_limit 2 --enable_cuda_graph 1  --enable_cuda_graph_debug_mode 1 --prefill_capture_config '150,155,160,380,400' --task_type DENSE_EMBEDDING --reserver_runtime_mem_mb 3072",
                envs=["LOG_LEVEL=INFO", "PYTHONUNBUFFERED=TRUE"],
                gpu_type=["L20"],
            ),
        ],
    )

    # SM8x Qwen3.5 Omni (L20D, multimodal audio+vision)
    native.test_suite(
        name = "smoke_sm8x_omni",
        tests = [
            smoke_test(
                name="qwen35_omni_basic",
                task_info="data/model/qwen3_5_omni/q_r_omni_basic.json",
                smoke_args="--act_type BF16 --tp_size 4 --world_size 4 --reserver_runtime_mem_mb 10000 --max_seq_len 16384 --seq_size_per_block 2048 --kernel_seq_size_per_block 64 --ssm_state_dtype fp32",
                envs=[
                    "TRITON_AUTOTUNE_CACHE_MODE=cached",
                    "AUDIO_TOKENIZER_PATH=/home/xieshui.yyx/happymusic/happy_music_tokenizer/model.ckpt",
                    "CC=/usr/bin/gcc",
                    "CXX=/usr/bin/g++",
                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                    "FLASHINFER_DISABLE_VERSION_CHECK=1",
                    "CUDA_VISIBLE_DEVICES=4,5,6,7",
                ],
                gpu_type=["L20"],
            ),
        ],
    )

