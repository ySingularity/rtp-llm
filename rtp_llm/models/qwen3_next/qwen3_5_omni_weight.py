import functools
from typing import Any, Dict, List

from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    LinearAttnConfig,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen3_next.qwen3_next_weight import (
    Qwen35MoeWeight,
    reorder_ba,
    reorder_qkvz,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose


class Qwen35OmniWeight(Qwen35MoeWeight):
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.prefix = "thinker.model."

    def _get_weight_info(self):
        weight_info = super()._get_weight_info()
        for i, w in enumerate(weight_info.weights):
            if isinstance(w, AtomicWeight) and w.name == W.lm_head:
                weight_info.weights[i] = AtomicWeight(
                    W.lm_head,
                    [CkptWeightInfo("thinker.lm_head.weight", identity)],
                )
                break
        return weight_info

    def _create_linear_attn_qkvz_weight(self) -> LinearAttnAtomicWeight:
        return LinearAttnAtomicWeight(
            W.linear_attn_qkvz_w,
            [
                CkptWeightInfo(
                    self.prefix + "layers.{i}.linear_attn.in_proj_qkvz.weight",
                    functools.partial(
                        reorder_qkvz,
                        linear_attention_config=self.model_config.linear_attention_config,
                    ),
                )
            ],
            transpose,
            LinearAttnConfig(self.model_config.linear_attention_config),
        )

    def _create_linear_attn_ba_weight(self) -> LinearAttnAtomicWeight:
        return LinearAttnAtomicWeight(
            W.linear_attn_ba_w,
            [
                CkptWeightInfo(
                    self.prefix + "layers.{i}.linear_attn.in_proj_ba.weight",
                    functools.partial(
                        reorder_ba,
                        linear_attention_config=self.model_config.linear_attention_config,
                    ),
                )
            ],
            transpose,
            LinearAttnConfig(self.model_config.linear_attention_config),
        )
