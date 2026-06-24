from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeMixin,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_omni.processor import (
    Qwen35OmniEmbedding,
)


class Qwen35OmniMixin(Qwen3_5MoeMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen35OmniEmbedding(self.mm_related_params)
        # Only vision weights go through BaseVitWeights (loaded from main ckpt).
        # Audio tokenizer weights are loaded separately in Qwen35OmniEmbedding.__init__.
        self.mm_related_params.vit_weights = BaseVitWeights(
            {"visual": self.mm_part.visual},
            with_prefix=True,
        )
        self.mm_related_params.vit_weights._ckpt_prefix = "thinker."

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
            Qwen3_5MoeVisionConfig,
            Qwen3_5MoeVisionModel,
        )
        from rtp_llm.utils.util import get_config_from_path

        ckpt_path = mm_related_params.config["ckpt_path"]
        config_json = get_config_from_path(ckpt_path)
        thinker_config = config_json["thinker_config"]
        vision_config = Qwen3_5MoeVisionConfig(**thinker_config["vision_config"])
        return Qwen3_5MoeVisionModel._from_config(vision_config)


register_multimodal_mixin(["qwen35_omni"], Qwen35OmniMixin)
