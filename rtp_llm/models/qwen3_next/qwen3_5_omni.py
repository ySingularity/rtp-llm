import json
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen3_next.qwen3_5_omni_weight import Qwen35OmniWeight
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe


class Qwen35Omni(Qwen35Moe):
    @staticmethod
    def get_weight_cls():
        return Qwen35OmniWeight

    @classmethod
    def _parse_rope_config(cls, config_json: dict, config: ModelConfig):
        rope_scaling = config_json["rope_scaling"]
        mrope_interleaved = rope_scaling["mrope_interleaved"]
        assert mrope_interleaved, "mrope_interleaved should be True"
        config.attn_config.rope_config.style = 7
        config.attn_config.rope_config.base = config_json["rope_theta"]
        config.partial_rotary_factor = config_json["partial_rotary_factor"]
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )
        mrope_section = rope_scaling["mrope_section"]
        config.attn_config.rope_config.index_factor = len(mrope_section)
        config.attn_config.rope_config.mrope_dim1 = mrope_section[0]
        config.attn_config.rope_config.mrope_dim2 = mrope_section[1]
        config.attn_config.rope_config.mrope_dim3 = mrope_section[2]
        config.mm_model_config.mm_position_ids_style = 2

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        thinker_config = config_json["thinker_config"]
        text_config_json = thinker_config["text_config"]

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        cls._parse_basic_config(text_config_json, config)
        cls._parse_rope_config(text_config_json, config)
        cls._parse_normalization_config(text_config_json, config)
        cls._parse_moe_config(text_config_json, config)
        cls._parse_hybrid_attention_config(text_config_json, config)
        cls._parse_linear_attention_config(text_config_json, config)
        cls._parse_mm_config(thinker_config, config)

        return config

    @classmethod
    def _parse_mm_config(cls, config_json: dict, config: ModelConfig):
        config.mm_model_config.is_multimodal = True
        vision_start = config_json["vision_start_token_id"]
        vision_end = config_json["vision_end_token_id"]
        audio_start = config_json["audio_start_token_id"]
        audio_end = config_json["audio_end_token_id"]
        config.mm_model_config.mm_sep_tokens = [
            [vision_start, vision_end],
            [audio_start, audio_end],
        ]
        config.mm_related_params.config["ckpt_path"] = config.ckpt_path
        config.mm_related_params.config["audio_tokenizer_path"] = os.environ.get(
            "AUDIO_TOKENIZER_PATH", ""
        )


register_model(
    "qwen35_omni",
    Qwen35Omni,
    ["Qwen3OmniNextForConditionalGeneration"],
)
