"""Typed hparams and module-layout helpers for DELMAN/MEMIT editing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from json import load
from pathlib import Path
from typing import Any, Literal

FactToken = Literal["last", "subject_first", "subject_last", "subject_first_after_last"]
LayerSelection = Literal["all", "random"]


@dataclass(frozen=True)
class ModuleLayout:
    """Resolved language-submodel paths used by the shared editor."""

    layer_template: str
    rewrite_template: str
    mlp_template: str
    attn_template: str
    ln_f_module: str
    lm_head_module: str

    @classmethod
    def from_layer_path(
        cls,
        layer_path: str,
        edit_module_path: str,
        *,
        ln_f_module: str,
        lm_head_module: str,
        mlp_suffix: str = "mlp",
        attn_suffix: str = "self_attn",
    ) -> ModuleLayout:
        """Build formatting templates from a concrete resolved layer path.

        Example: ``language_model.model.layers.8`` becomes
        ``language_model.model.layers.{}``. This keeps model-family resolution
        outside the editor and avoids deriving paths from checkpoints or files.
        """

        parts = layer_path.rsplit(".", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"layer_path must end in a numeric layer: {layer_path!r}")
        layer_template = f"{parts[0]}.{{}}"

        edit_suffix = edit_module_path.removeprefix(layer_path)
        if not edit_suffix.startswith("."):
            raise ValueError("edit_module_path must be within layer_path")
        rewrite_template = f"{layer_template}{edit_suffix}"
        return cls(
            layer_template=layer_template,
            rewrite_template=rewrite_template,
            mlp_template=f"{layer_template}.{mlp_suffix}",
            attn_template=f"{layer_template}.{attn_suffix}",
            ln_f_module=ln_f_module,
            lm_head_module=lm_head_module,
        )


@dataclass(frozen=True)
class DELMANHyperParams:
    """Stage hparams consumed by the shared DELMAN/MEMIT text-token solver."""

    profile_key: str
    stage: str
    layers: tuple[int, ...]
    layer_selection: LayerSelection
    fact_token: FactToken
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    layout: ModuleLayout

    @property
    def rewrite_module_tmp(self) -> str:
        return self.layout.rewrite_template

    @property
    def layer_module_tmp(self) -> str:
        return self.layout.layer_template

    @property
    def mlp_module_tmp(self) -> str:
        return self.layout.mlp_template

    @property
    def attn_module_tmp(self) -> str:
        return self.layout.attn_template

    @property
    def ln_f_module(self) -> str:
        return self.layout.ln_f_module

    @property
    def lm_head_module(self) -> str:
        return self.layout.lm_head_module


def merge_hparams(
    shared: Mapping[str, Any],
    stage: Mapping[str, Any],
    *,
    layout: ModuleLayout,
    overrides: Mapping[str, Any] | None = None,
) -> DELMANHyperParams:
    """Merge profile shared/stage hparams into a validated dataclass."""

    merged = {**dict(shared), **dict(stage)}
    if overrides:
        merged.update(overrides)
    merged["layout"] = layout
    return _coerce_hparams(merged)


def load_hparams(path: str | Path, *, layout: ModuleLayout) -> DELMANHyperParams:
    """Load a single hparams JSON file into typed DELMAN hparams."""

    with Path(path).open("r", encoding="utf-8") as fh:
        raw = load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError(f"hparams file must contain an object: {path}")
    return _coerce_hparams({**raw, "layout": layout})


def with_hparam_overrides(
    hparams: DELMANHyperParams, overrides: Mapping[str, Any] | None
) -> DELMANHyperParams:
    """Return ``hparams`` with typed overrides applied."""

    if not overrides:
        return hparams
    allowed = {field.name for field in fields(DELMANHyperParams)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown hparam override(s): {', '.join(unknown)}")
    return replace(hparams, **dict(overrides))


def _coerce_hparams(raw: Mapping[str, Any]) -> DELMANHyperParams:
    required = {field.name for field in fields(DELMANHyperParams)}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"missing hparam field(s): {', '.join(missing)}")

    layout = raw["layout"]
    if not isinstance(layout, ModuleLayout):
        raise TypeError("layout must be a ModuleLayout")

    layers = tuple(int(layer) for layer in raw["layers"])
    if not layers:
        raise ValueError("layers must not be empty")

    return DELMANHyperParams(
        profile_key=str(raw["profile_key"]),
        stage=str(raw["stage"]),
        layers=layers,
        layer_selection=_literal(raw["layer_selection"], {"all", "random"}, "layer_selection"),
        fact_token=_literal(
            raw["fact_token"],
            {"last", "subject_first", "subject_last", "subject_first_after_last"},
            "fact_token",
        ),
        v_num_grad_steps=int(raw["v_num_grad_steps"]),
        v_lr=float(raw["v_lr"]),
        v_loss_layer=int(raw["v_loss_layer"]),
        v_weight_decay=float(raw["v_weight_decay"]),
        clamp_norm_factor=float(raw["clamp_norm_factor"]),
        kl_factor=float(raw["kl_factor"]),
        mom2_adjustment=bool(raw["mom2_adjustment"]),
        mom2_update_weight=float(raw["mom2_update_weight"]),
        mom2_dataset=str(raw["mom2_dataset"]),
        mom2_n_samples=int(raw["mom2_n_samples"]),
        mom2_dtype=str(raw["mom2_dtype"]),
        layout=layout,
    )


def _literal(value: Any, allowed: set[str], name: str) -> Any:
    text = str(value)
    if text not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return text
