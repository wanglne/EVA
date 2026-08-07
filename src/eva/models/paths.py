"""Model-internal path resolution for EVA VLM adapters.

This module is intentionally import-light: it only works with strings from
``model.named_modules()`` and does not import torch or transformers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelLayout:
    family: str
    layer_index: int
    layer_path: str
    edit_module_path: str
    mlp_path: str
    down_proj_path: str
    norm_path: str
    lm_head_path: str

    @property
    def variant(self) -> str:
        if self.family == "internvl" and self.layer_path.startswith("model.layers"):
            return "language_submodel"
        if self.family == "qwen2_5_vl" and self.layer_path.startswith(
            "model.language_model.layers"
        ):
            return "qwen_new"
        if self.family == "qwen2_5_vl" and self.layer_path.startswith("model.layers"):
            return "qwen_old"
        return "language_model"

    @property
    def layer_prefix(self) -> str:
        return self.layer_path.rsplit(".", 1)[0]

    @property
    def attention_output(self) -> str:
        return self.edit_module_path.removeprefix(f"{self.layer_path}.")

    @property
    def mlp_down(self) -> str:
        return self.down_proj_path.removeprefix(f"{self.layer_path}.")

    @property
    def logical_layer(self) -> int:
        return self.layer_index

    def module_for_layer(self, layer: int, *, kind: str = "edit") -> str:
        suffix = {
            "edit": self.edit_module_path.removeprefix(f"{self.layer_path}."),
            "attention_output": self.edit_module_path.removeprefix(f"{self.layer_path}."),
            "mlp": self.mlp_path.removeprefix(f"{self.layer_path}."),
            "mlp_down": self.down_proj_path.removeprefix(f"{self.layer_path}."),
            "down_proj": self.down_proj_path.removeprefix(f"{self.layer_path}."),
        }[kind]
        return f"{self.layer_prefix}.{layer}.{suffix}"


ModuleLayout = ModelLayout

_LANGUAGE_LAYER_PATTERNS = {
    "llava": (
        r"^language_model\.model\.layers\.(?P<layer>\d+)(?:\.|$)",
        r"^model\.language_model\.layers\.(?P<layer>\d+)(?:\.|$)",
    ),
    "qwen2_5_vl": (
        r"^model\.language_model\.layers\.(?P<layer>\d+)(?:\.|$)",
        r"^model\.layers\.(?P<layer>\d+)(?:\.|$)",
    ),
    "internvl": (r"^language_model\.model\.layers\.(?P<layer>\d+)(?:\.|$)",),
}

_EDIT_SUFFIXES = (
    "mlp.down_proj",
    "mlp.c_proj",
    "feed_forward.w2",
    "attention.wo",
    "self_attn.o_proj",
)

_NORM_CANDIDATES = {
    "llava": (
        "language_model.model.norm",
        "model.language_model.norm",
        "language_model.model.final_layernorm",
        "model.language_model.final_layernorm",
    ),
    "qwen2_5_vl": ("model.language_model.norm", "model.norm"),
    "internvl": ("language_model.model.norm", "language_model.model.final_layernorm"),
}

_LM_HEAD_CANDIDATES = {
    "llava": ("language_model.lm_head", "lm_head"),
    "qwen2_5_vl": ("lm_head", "model.lm_head", "language_model.lm_head"),
    "internvl": ("language_model.lm_head", "language_model.output", "lm_head"),
}


def module_names_from_model(model: object) -> list[str]:
    """Return named module paths from a loaded model object."""

    return [name for name, _module in model.named_modules()]  # type: ignore[attr-defined]


def resolve_module_layout(
    family: str | None = None,
    module_names: Iterable[str] = (),
    *,
    layer: int | None = None,
    edit_suffix: str = "mlp.down_proj",
) -> ModelLayout:
    names = tuple(module_names)
    normalized_family = _normalize_family(family, names)
    layer_index, layer_path = _resolve_layer_path(normalized_family, names, layer)
    edit_module_path = _resolve_layer_child(names, layer_path, (edit_suffix, *_EDIT_SUFFIXES))
    mlp_path = _resolve_optional_child(names, layer_path, ("mlp", "feed_forward"))
    down_proj_path = _resolve_layer_child(
        names, layer_path, ("mlp.down_proj", "mlp.c_proj", "feed_forward.w2")
    )
    norm_path = _resolve_global(names, _NORM_CANDIDATES[normalized_family])
    lm_head_path = _resolve_global(names, _LM_HEAD_CANDIDATES[normalized_family])

    return ModelLayout(
        family=normalized_family,
        layer_index=layer_index,
        layer_path=layer_path,
        edit_module_path=edit_module_path,
        mlp_path=mlp_path,
        down_proj_path=down_proj_path,
        norm_path=norm_path,
        lm_head_path=lm_head_path,
    )


def resolve_named_module(model: object, module_path: str) -> object:
    current = model
    for part in module_path.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        elif hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, "_modules") and part in current._modules:  # type: ignore[attr-defined]
            current = current._modules[part]  # type: ignore[attr-defined]
        else:
            raise KeyError(f"module path segment {part!r} not found in {module_path!r}")
    return current


def language_down_proj_candidates(family: str, layer: int) -> tuple[str, ...]:
    normalized_family = _normalize_explicit_family(family)
    if normalized_family == "llava":
        return (
            f"language_model.model.layers.{layer}.mlp.down_proj",
            f"model.language_model.layers.{layer}.mlp.down_proj",
        )
    if normalized_family == "qwen2_5_vl":
        return (
            f"model.language_model.layers.{layer}.mlp.down_proj",
            f"model.layers.{layer}.mlp.down_proj",
        )
    if normalized_family == "internvl":
        return (f"language_model.model.layers.{layer}.mlp.down_proj",)
    raise ValueError(f"unsupported model family: {family}")


def _normalize_family(family: str | None, names: tuple[str, ...]) -> str:
    if family:
        return _normalize_explicit_family(family)
    if any(
        name.startswith("model.language_model.layers.") or name.startswith("model.layers.")
        for name in names
    ):
        return "qwen2_5_vl"
    if any(".attention.wo" in name or name.endswith("attention.wo") for name in names):
        return "internvl"
    if any(name.startswith("language_model.model.layers.") for name in names):
        return "llava"
    raise ValueError("could not infer model family from module names")


def _normalize_explicit_family(family: str) -> str:
    key = (family or "").lower().replace("-", "_").replace(".", "_")
    aliases = {
        "llava": "llava",
        "llava15": "llava",
        "llava_1_5": "llava",
        "qwen": "qwen2_5_vl",
        "qwen25vl": "qwen2_5_vl",
        "qwen2_5_vl": "qwen2_5_vl",
        "internvl": "internvl",
        "internvl35": "internvl",
        "internvl3_5": "internvl",
    }
    normalized = aliases.get(key, key)
    if normalized in _LANGUAGE_LAYER_PATTERNS:
        return normalized
    raise ValueError(f"unsupported model family: {family}")


def _resolve_layer_path(
    family: str, names: tuple[str, ...], requested_layer: int | None
) -> tuple[int, str]:
    for pattern in _LANGUAGE_LAYER_PATTERNS[family]:
        compiled = re.compile(pattern)
        matched: list[tuple[int, str]] = []
        for name in names:
            match = compiled.match(name)
            if match is None:
                continue
            layer_index = int(match.group("layer"))
            if requested_layer is not None and layer_index != requested_layer:
                continue
            layer_path = name[: match.end()].removesuffix(".")
            matched.append((layer_index, layer_path))
        if matched:
            return sorted(matched, key=lambda item: item[0])[-1]
    detail = f" layer {requested_layer}" if requested_layer is not None else ""
    raise ValueError(f"could not resolve {family}{detail} language layer from named modules")


def _resolve_layer_child(names: tuple[str, ...], layer_path: str, suffixes: Iterable[str]) -> str:
    for suffix in _dedupe(suffixes):
        candidate = f"{layer_path}.{suffix}"
        if candidate in names or _has_descendant(names, candidate):
            return candidate
    return f"{layer_path}.{_dedupe(suffixes)[0]}"


def _resolve_optional_child(
    names: tuple[str, ...], layer_path: str, suffixes: Iterable[str]
) -> str:
    ordered = _dedupe(suffixes)
    for suffix in ordered:
        candidate = f"{layer_path}.{suffix}"
        if candidate in names or _has_descendant(names, candidate):
            return candidate
    return f"{layer_path}.{ordered[0]}"


def _resolve_global(names: tuple[str, ...], candidates: Iterable[str]) -> str:
    ordered = _dedupe(candidates)
    for candidate in ordered:
        if candidate in names or _has_descendant(names, candidate):
            return candidate
    return ordered[0]


def _has_descendant(names: tuple[str, ...], candidate: str) -> bool:
    prefix = f"{candidate}."
    return any(name.startswith(prefix) for name in names)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return tuple(ordered)
