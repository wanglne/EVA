"""Common lazy adapter protocol for EVA vision-language models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from .paths import ModelLayout, module_names_from_model, resolve_module_layout


@dataclass(frozen=True)
class LoadOptions:
    model_path: str | Path
    device: str | None
    dtype: str | None
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool | None = None
    use_flash_attn: bool | None = None
    model_kwargs: dict[str, Any] | None = None
    processor_kwargs: dict[str, Any] | None = None


@dataclass
class ModelHandle:
    family: str
    model: Any
    processor: Any
    tokenizer: Any
    layout: ModelLayout
    adapter: ModelAdapter
    load_options: LoadOptions

    def text_edit_view(self) -> TextEditView:
        return self.adapter.select_text_edit_view(self)

    def save_full_model(self, output_dir: str | Path, **kwargs: Any) -> None:
        self.adapter.save(self, output_dir, **kwargs)

    def reload_full_model(
        self,
        checkpoint_dir: str | Path | None = None,
        **overrides: Any,
    ) -> ModelHandle:
        return self.adapter.reload(self, checkpoint_dir=checkpoint_dir, **overrides)

    def covariance_view(self) -> TextEditView:
        return self.adapter.covariance_view(self)

    def stats_view(self) -> TextEditView:
        return self.adapter.stats_view(self)


@dataclass
class TextEditView:
    """Language-model view that keeps a pointer to the full parent VLM."""

    model: Any
    tokenizer: Any
    parent: ModelHandle
    layout: ModelLayout


class ModelAdapter(Protocol):
    family: str

    def load(
        self,
        model_path: str | Path,
        *,
        device: str | None,
        dtype: str | None,
        trust_remote_code: bool | None = None,
        **kwargs: Any,
    ) -> ModelHandle: ...

    def resolve_layout(self, model: Any, *, layer: int | None = None) -> ModelLayout: ...

    def select_text_edit_view(self, handle: ModelHandle) -> TextEditView: ...

    def decorate_image_prompt(self, prompt: str) -> str: ...

    def prepare_image_batch(
        self,
        handle: ModelHandle,
        *,
        prompts: list[str],
        images: list[Any],
        device: str | None = None,
        dtype: str | None = None,
    ) -> dict[str, Any]: ...

    def forward_image_batch(
        self, handle: ModelHandle, batch: dict[str, Any], **kwargs: Any
    ) -> Any: ...

    def save(self, handle: ModelHandle, output_dir: str | Path, **kwargs: Any) -> None: ...

    def reload(self, handle: ModelHandle, **overrides: Any) -> ModelHandle: ...

    def covariance_view(self, handle: ModelHandle) -> TextEditView: ...

    def stats_view(self, handle: ModelHandle) -> TextEditView: ...


class BaseModelAdapter:
    family = ""
    default_dtype: str | None = None
    default_trust_remote_code = False

    def resolve_layout(self, model: Any, *, layer: int | None = None) -> ModelLayout:
        return resolve_module_layout(self.family, module_names_from_model(model), layer=layer)

    def select_text_edit_view(self, handle: ModelHandle) -> TextEditView:
        return TextEditView(
            model=handle.model,
            tokenizer=handle.tokenizer,
            parent=handle,
            layout=handle.layout,
        )

    def decorate_image_prompt(self, prompt: str) -> str:
        return prompt if "<image>" in prompt else f"<image>\n {prompt}"

    def forward_image_batch(self, handle: ModelHandle, batch: dict[str, Any], **kwargs: Any) -> Any:
        return handle.model(**batch, **kwargs)

    def save(self, handle: ModelHandle, output_dir: str | Path, **kwargs: Any) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        handle.model.save_pretrained(output, **kwargs)
        if handle.processor is not None and hasattr(handle.processor, "save_pretrained"):
            handle.processor.save_pretrained(output)
        if handle.tokenizer is not None and hasattr(handle.tokenizer, "save_pretrained"):
            handle.tokenizer.save_pretrained(output)

    def reload(
        self,
        handle: ModelHandle,
        checkpoint_dir: str | Path | None = None,
        **overrides: Any,
    ) -> ModelHandle:
        options = handle.load_options
        model_kwargs = dict(options.model_kwargs or {})
        model_kwargs.update(overrides.pop("model_kwargs", {}))
        processor_kwargs = dict(options.processor_kwargs or {})
        processor_kwargs.update(overrides.pop("processor_kwargs", {}))
        return self.load(
            overrides.pop("model_path", checkpoint_dir or options.model_path),
            device=overrides.pop("device", options.device),
            dtype=overrides.pop("dtype", options.dtype),
            trust_remote_code=overrides.pop("trust_remote_code", options.trust_remote_code),
            processor_kwargs=processor_kwargs,
            **model_kwargs,
            **overrides,
        )

    def covariance_view(self, handle: ModelHandle) -> TextEditView:
        return self.select_text_edit_view(handle)

    def stats_view(self, handle: ModelHandle) -> TextEditView:
        return self.covariance_view(handle)

    def _load_handle(
        self,
        *,
        model_cls: Any,
        processor_cls: Any,
        model_path: str | Path,
        device: str | None,
        dtype: str | None,
        trust_remote_code: bool | None,
        tokenizer: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
    ) -> ModelHandle:
        raw_model_kwargs = dict(model_kwargs or {})
        raw_processor_kwargs = dict(processor_kwargs or {})
        kwargs = dict(raw_model_kwargs)
        dtype_obj = torch_dtype(dtype or self.default_dtype)
        if dtype_obj is not None:
            kwargs["torch_dtype"] = dtype_obj
        if trust_remote_code is None:
            trust_remote_code = self.default_trust_remote_code
        if trust_remote_code:
            kwargs["trust_remote_code"] = True

        model = model_cls.from_pretrained(str(model_path), **kwargs)
        if device is not None:
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()

        processor = processor_cls.from_pretrained(str(model_path), **raw_processor_kwargs)
        if tokenizer is None:
            tokenizer = getattr(processor, "tokenizer", None)
        layout = self.resolve_layout(model)
        return ModelHandle(
            family=self.family,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            layout=layout,
            adapter=self,
            load_options=LoadOptions(
                model_path=model_path,
                device=device,
                dtype=dtype or self.default_dtype,
                trust_remote_code=bool(trust_remote_code),
                model_kwargs=raw_model_kwargs,
                processor_kwargs=raw_processor_kwargs,
            ),
        )


def transformers_attr(name: str) -> Any:
    return getattr(import_module("transformers"), name)


def configure_text_stage_tokenizer(tokenizer: Any) -> Any:
    if tokenizer is None:
        return None
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def torch_dtype(dtype: str | None) -> Any | None:
    if dtype is None:
        return None
    torch = import_module("torch")
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def move_batch(
    batch: dict[str, Any], *, device: str | None, dtype: str | None = None
) -> dict[str, Any]:
    dtype_obj = torch_dtype(dtype)
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        if dtype_obj is not None and "pixel" in key:
            moved[key] = value.to(device=device, dtype=dtype_obj)
        elif device is not None:
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _first_present(root: Any, dotted_paths: tuple[str, ...], *, default: Any) -> Any:
    for dotted_path in dotted_paths:
        current = root
        for part in dotted_path.split("."):
            if not hasattr(current, part):
                current = None
                break
            current = getattr(current, part)
        if current is not None:
            return current
    return default
