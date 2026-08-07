"""Qwen2.5-VL adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    BaseModelAdapter,
    ModelHandle,
    configure_text_stage_tokenizer,
    move_batch,
    transformers_attr,
)

QWEN_IMAGE_DECORATION = "<|vision_start|><|image_pad|><|vision_end|>"


class Qwen25VLAdapter(BaseModelAdapter):
    family = "qwen2_5_vl"
    default_dtype = "bfloat16"

    def load(
        self,
        model_path: str | Path,
        *,
        device: str | None,
        dtype: str | None = None,
        trust_remote_code: bool | None = None,
        **kwargs: Any,
    ) -> ModelHandle:
        processor_kwargs = kwargs.pop("processor_kwargs", None)
        handle = self._load_handle(
            model_cls=transformers_attr("Qwen2_5_VLForConditionalGeneration"),
            processor_cls=transformers_attr("AutoProcessor"),
            model_path=model_path,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            model_kwargs=kwargs,
            processor_kwargs=processor_kwargs,
        )
        configure_text_stage_tokenizer(handle.tokenizer)
        return handle

    def decorate_image_prompt(self, prompt: str) -> str:
        return prompt if QWEN_IMAGE_DECORATION in prompt else f"{QWEN_IMAGE_DECORATION}{prompt}"

    def prepare_image_batch(
        self,
        handle: ModelHandle,
        *,
        prompts: list[str],
        images: list[Any],
        device: str | None = None,
        dtype: str | None = None,
    ) -> dict[str, Any]:
        texts = [self.decorate_image_prompt(prompt) for prompt in prompts]
        batch = handle.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        return move_batch(dict(batch), device=device, dtype=dtype)


ADAPTER = Qwen25VLAdapter()
