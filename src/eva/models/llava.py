"""LLaVA-1.5 adapter."""

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


class Llava15Adapter(BaseModelAdapter):
    family = "llava"
    default_dtype = "float16"

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
            model_cls=transformers_attr("LlavaForConditionalGeneration"),
            processor_cls=transformers_attr("AutoProcessor"),
            model_path=model_path,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            model_kwargs=kwargs,
            processor_kwargs=processor_kwargs,
        )
        if handle.processor is not None:
            handle.processor.patch_size = 14
        configure_text_stage_tokenizer(handle.tokenizer)
        return handle

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
        batch = handle.processor(text=texts, images=images, return_tensors="pt", padding=True)
        return move_batch(dict(batch), device=device, dtype=dtype)


ADAPTER = Llava15Adapter()
