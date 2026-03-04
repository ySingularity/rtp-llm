import json
from typing import List

from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
    MMUrlType,
    PromptWithMMInput,
    RenderedInputs,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.qwen3_code_renderer import Qwen3CoderRenderer
from rtp_llm.ops import MMPreprocessConfig


class Qwen35Renderer(Qwen3CoderRenderer):
    def _render_messages(self, request: ChatCompletionRequest) -> PromptWithMMInput:
        def get_preprocess_config(config):
            if config.crop_positions:
                crop_positions = [float(x) for x in config.crop_positions.split(":")]
                if len(crop_positions) == 6:
                    # input format: "w1:h1:w2:h2:h:w"
                    crop_positions = [
                        crop_positions[0] / crop_positions[5],
                        crop_positions[1] / crop_positions[4],
                        crop_positions[2] / crop_positions[5],
                        crop_positions[3] / crop_positions[4],
                    ]
            else:
                crop_positions = []
            return MMPreprocessConfig(
                width=config.resized_width or -1,
                height=config.resized_height or -1,
                min_pixels=config.min_pixels or -1,
                max_pixels=config.max_pixels or -1,
                fps=config.fps or -1,
                min_frames=config.min_frames or -1,
                max_frames=config.max_frames or -1,
                crop_positions=crop_positions,
                mm_timeout_ms=config.mm_timeout_ms or -1,
            )

        urls = []
        types = []
        preprocess_configs = []
        messages = request.messages
        for message in messages:
            if isinstance(message.content, list):
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        continue
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        urls.append(content_part.image_url.url)
                        types.append(MMUrlType.IMAGE)
                        if content_part.preprocess_config:
                            preprocess_configs.append(
                                get_preprocess_config(content_part.preprocess_config)
                            )
                    elif content_part.type == ContentPartTypeEnum.video_url:
                        assert content_part.video_url != None
                        urls.append(content_part.video_url.url)
                        types.append(MMUrlType.VIDEO)
                        if content_part.preprocess_config:
                            preprocess_configs.append(
                                get_preprocess_config(content_part.preprocess_config)
                            )

        prompt = self.tokenizer.apply_chat_template(
            request.messages.model_dump_json(exclude_none=True),
            tools=request.tools.model_dump_json(exclude_none=True),
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=(
                request.extra_configs.add_vision_id if request.extra_configs else True
            ),
        )
        return PromptWithMMInput(
            prompt=prompt,
            urls=urls,
            mm_types=types,
            preprocess_configs=preprocess_configs,
        )

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        prompt_and_mm_input = self._render_messages(request)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


register_renderer("qwen35_moe", Qwen35Renderer)
register_renderer("qwen35_dense", Qwen35Renderer)
