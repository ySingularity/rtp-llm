import logging
import os
from typing import Dict, List

import torch
import torchaudio
from transformers import AutoProcessor, Qwen2VLImageProcessor

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    Qwen3_5MoeVisionConfig,
    Qwen3_5MoeVisionModel,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_omni.modeling_audio_tokenizer import (
    Config as AudioTokenizerConfig,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_omni.modeling_audio_tokenizer import (
    Model as AudioTokenizerModel,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType, VitParameters
from rtp_llm.utils.flash_attn_utils import can_use_flash_attn
from rtp_llm.utils.util import get_config_from_path

default_attn_impl = "sdpa"
try:
    if can_use_flash_attn():
        default_attn_impl = "flash_attention_2"
except Exception as e:
    logging.info(
        f"initialize flash_attn failed, exception {e}, using sdpa attention in qwen3_5_omni vit"
    )


class Qwen35OmniEmbedding(Qwen3_5MoeImageEmbedding):
    def __init__(self, mm_related_params: VitParameters):
        ckpt_path = mm_related_params.config["ckpt_path"]
        config_json = get_config_from_path(ckpt_path)
        thinker_config = config_json["thinker_config"]

        self.mm_processor = AutoProcessor.from_pretrained(
            ckpt_path, trust_remote_code=True
        )
        self.mm_processor.image_processor = Qwen2VLImageProcessor.from_pretrained(
            ckpt_path
        )
        vision_config = Qwen3_5MoeVisionConfig(**thinker_config["vision_config"])
        vision_config._attn_implementation = default_attn_impl
        self.visual = Qwen3_5MoeVisionModel._from_config(vision_config)

        audio_tokenizer_path = mm_related_params.config.get("audio_tokenizer_path", "")
        self.audio_tokenizer = None
        if audio_tokenizer_path:
            self.audio_tokenizer = self._load_audio_tokenizer(audio_tokenizer_path)

    @staticmethod
    def _load_audio_tokenizer(ckpt_path: str) -> AudioTokenizerModel:
        import sys

        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir not in sys.path:
            sys.path.insert(0, ckpt_dir)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        model = AudioTokenizerModel(config=config)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.config.gradient_checkpointing = False
        model.eval()
        del ckpt
        return model

    def get_preprocess_params(self):
        params = super().get_preprocess_params()
        params["audio_tokenizer"] = self.audio_tokenizer
        return params

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor=None,
        factor: int = 32,
        audio_tokenizer=None,
        **kwargs,
    ):
        assert len(mm_inputs) == 1
        mm_type = mm_inputs[0].mm_type
        if mm_type == MMUrlType.AUDIO:
            data = get_bytes_io_from_url(mm_inputs[0].url, vit_config.download_headers)
            wav, sr = torchaudio.load(data)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != audio_tokenizer.config.sample_rate:
                wav = torchaudio.functional.resample(
                    wav, sr, audio_tokenizer.config.sample_rate
                )
            return {"wav": wav}
        return Qwen3_5MoeImageEmbedding.preprocess_input(
            mm_inputs, vit_config, processor, factor=factor
        )

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        mm_type = kwargs.get("mm_type")
        if mm_type == MMUrlType.AUDIO:
            return self._audio_embedding(data)
        return super().embedding(data, **kwargs)

    @torch.inference_mode()
    def _audio_embedding(self, features_dict: Dict[str, torch.Tensor]) -> tuple:
        wav = features_dict["wav"].unsqueeze(0).to(self._device)
        with torch.amp.autocast("cuda"):
            vq_z, vq_indices, _ = self.audio_tokenizer.quantize(wav=wav)
        return vq_z.squeeze(0), None
