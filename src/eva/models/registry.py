"""String-keyed lazy registry for model adapters."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import ModelAdapter

_ADAPTER_MODULES = {
    "llava": "eva.models.llava",
    "llava15": "eva.models.llava",
    "llava-1.5": "eva.models.llava",
    "llava-1.5-7b": "eva.models.llava",
    "llava-v1.5": "eva.models.llava",
    "qwen": "eva.models.qwen",
    "qwen25vl": "eva.models.qwen",
    "qwen2_5_vl": "eva.models.qwen",
    "qwen2-5-vl": "eva.models.qwen",
    "qwen2.5-vl": "eva.models.qwen",
    "qwen2.5-vl-7b-instruct": "eva.models.qwen",
    "qwen2-5-vl-7b-instruct": "eva.models.qwen",
    "internvl": "eva.models.internvl",
    "internvl-chat": "eva.models.internvl",
    "internvl_chat": "eva.models.internvl",
    "internvl35": "eva.models.internvl",
    "internvl3.5": "eva.models.internvl",
    "internvl3-5": "eva.models.internvl",
    "internvl3.5-8b": "eva.models.internvl",
    "internvl3-5-8b": "eva.models.internvl",
}


def get_adapter(key: str) -> ModelAdapter:
    try:
        module_name = _ADAPTER_MODULES[_normalize_key(key)]
    except KeyError as exc:
        raise KeyError(f"unknown model adapter: {key}") from exc
    return import_module(module_name).ADAPTER  # type: ignore[no-any-return]


def available_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTER_MODULES))


def load_model(key: str, *args: Any, **kwargs: Any) -> Any:
    return get_adapter(key).load(*args, **kwargs)


def _normalize_key(key: str) -> str:
    return key.lower().replace("_", "-")
