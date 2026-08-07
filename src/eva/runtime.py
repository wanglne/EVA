"""Runtime hooks for applying EVA editing stages to loaded model handles."""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .checkpoints import (
    atomic_write_json,
    is_complete_checkpoint,
    read_json,
    remove_run_artifact,
    save_full_checkpoint,
)
from .editing.backends import BackendRegistry, default_backend_registry
from .editing.hparams import DELMANHyperParams


class ModelHandle(Protocol):
    """Full model lifecycle expected by the editing runtime."""

    model: Any
    tokenizer: Any

    def save_full_model(self, output_dir: Path) -> None:
        """Persist the parent full model, not just the language submodule."""

    def reload_full_model(self, checkpoint_dir: Path) -> ModelHandle:
        """Reload the parent full model from a checkpoint directory."""


class LanguageSubmodelProvider(Protocol):
    """Resolves the editable language model/tokenizer from a full model handle."""

    def language_model(self, handle: ModelHandle) -> Any:
        """Return the language submodel to edit in place."""

    def tokenizer(self, handle: ModelHandle) -> Any:
        """Return the tokenizer for the language submodel."""


@dataclass(frozen=True)
class EditingStage:
    name: str
    backend: str
    records: list[Mapping[str, Any]]
    hparams: DELMANHyperParams
    checkpoint_dir: Path | None = None


@dataclass(frozen=True)
class StageResult:
    stage: str
    checkpoint_dir: Path | None
    metadata: Any = None


class DirectLanguageSubmodelProvider:
    """Default provider for handles whose full model is itself language-editable."""

    def language_model(self, handle: ModelHandle) -> Any:
        return handle.model

    def tokenizer(self, handle: ModelHandle) -> Any:
        return handle.tokenizer


@dataclass(frozen=True)
class TwoStageRunResult:
    """Serializable outcome of one image-checkpoint-text editing run."""

    manifest: dict[str, Any]
    resumed_from: Path | None


def run_editing_stages(
    *,
    handle: ModelHandle,
    stages: list[EditingStage],
    covariance_provider: Any,
    language_provider: LanguageSubmodelProvider | None = None,
    registry: BackendRegistry | None = None,
    context_templates: list[list[str]] | None = None,
) -> tuple[ModelHandle, list[StageResult]]:
    """Run image/checkpoint/reload/text style editing stages.

    Any stage with ``checkpoint_dir`` is saved as a full parent model and then
    reloaded through ``ModelHandle`` before the next stage. This preserves the
    intended image -> checkpoint -> reload -> text sequencing without coupling
    the shared editor to a particular model family loader.
    """

    registry = registry or default_backend_registry()
    language_provider = language_provider or DirectLanguageSubmodelProvider()
    current = handle
    results: list[StageResult] = []

    for stage in stages:
        backend = registry.get(stage.backend)
        if stage.backend == "image":
            edit_model = current.model
            edit_tokenizer = current.tokenizer
        else:
            edit_model = language_provider.language_model(current)
            edit_tokenizer = language_provider.tokenizer(current)
        metadata = backend.apply(
            model=edit_model,
            tokenizer=edit_tokenizer,
            records=list(stage.records),
            hparams=stage.hparams,
            covariance_provider=covariance_provider,
            context_templates=context_templates,
        )
        if stage.checkpoint_dir is not None:
            current.save_full_model(stage.checkpoint_dir)
            current = current.reload_full_model(stage.checkpoint_dir)
        results.append(
            StageResult(stage=stage.name, checkpoint_dir=stage.checkpoint_dir, metadata=metadata)
        )

    return current, results


def run_two_stage_pipeline(
    *,
    profile: Any,
    plan: Any,
    adapter: Any,
    covariance_provider: Any,
    image_records: Sequence[Mapping[str, Any]],
    image_root: str | Path,
    text_records: Sequence[Mapping[str, Any]],
    device: str | None,
    manifest: Mapping[str, Any],
    resume: bool = False,
    force_restart: bool = False,
    image_editor: Callable[..., Any] | None = None,
    text_editor: Callable[..., Any] | None = None,
    hparams_loader: Callable[..., Any] | None = None,
) -> TwoStageRunResult:
    """Run the recovered EVA sequence with durable checkpoint boundaries.

    Resume is deliberately stage-level: an interrupted image stage restarts
    from the base checkpoint, while an interrupted text stage restarts from the
    completed full image checkpoint. No partially edited in-memory model is
    treated as resumable state.
    """

    if resume and force_restart:
        raise ValueError("resume and force_restart are mutually exclusive")

    expected = deepcopy(dict(manifest))
    fingerprint = str(expected.get("run_fingerprint", ""))
    if not fingerprint:
        raise ValueError("manifest is missing run_fingerprint")

    output_dir = Path(plan.output_dir)
    manifest_path = Path(plan.manifest_path)
    intermediate = _stage_checkpoint(plan, "save_intermediate_checkpoint")
    final = _stage_checkpoint(plan, "save_final_checkpoint")

    if force_restart:
        for artifact in (intermediate, final, manifest_path):
            remove_run_artifact(artifact, output_dir=output_dir)

    current_manifest = _initialize_manifest(
        expected=expected,
        manifest_path=manifest_path,
        intermediate=intermediate,
        final=final,
        resume=resume,
    )
    resumed_from: Path | None = None
    handle: Any | None = None

    try:
        if resume and is_complete_checkpoint(
            final,
            run_fingerprint=fingerprint,
            stage="final",
        ):
            _complete_all_stages(current_manifest, plan)
            current_manifest["status"] = "complete"
            current_manifest["resumed_from"] = str(final)
            current_manifest["completed_at"] = _utc_now()
            current_manifest.pop("error", None)
            atomic_write_json(manifest_path, current_manifest)
            return TwoStageRunResult(current_manifest, final)

        has_intermediate = is_complete_checkpoint(
            intermediate,
            run_fingerprint=fingerprint,
            stage="intermediate",
        )
        if intermediate.exists() and not has_intermediate:
            raise RuntimeError(
                f"intermediate checkpoint is incomplete or belongs to another run: {intermediate}"
            )
        if final.exists():
            raise RuntimeError(f"final checkpoint is incomplete or belongs to another run: {final}")

        if resume and has_intermediate:
            resumed_from = intermediate
            _mark_stages(
                current_manifest,
                ("image_edit", "save_intermediate_checkpoint"),
                "complete",
            )
        else:
            _mark_stages(
                current_manifest,
                ("image_edit", "save_intermediate_checkpoint"),
                "running",
            )
            current_manifest["status"] = "running"
            current_manifest["updated_at"] = _utc_now()
            atomic_write_json(manifest_path, current_manifest)

            handle = _load_handle(adapter, plan.model_path, profile, device)
            image_layout = _editor_layout(handle.layout)
            image_hparams = _load_hparams(
                profile.image_hparams_path,
                layout=image_layout,
                loader=hparams_loader,
            )
            current_manifest.setdefault("resolved_module_paths", {})["image"] = _layout_dict(
                handle.layout
            )
            image_result = _apply_image_stage(
                editor=image_editor,
                handle=handle,
                records=image_records,
                hparams=image_hparams,
                covariance_provider=covariance_provider,
                profile=profile,
                image_root=Path(image_root),
                device=device,
            )
            current_manifest.setdefault("stage_metadata", {})["image_edit"] = {
                "records": len(image_records),
                "chunks": getattr(image_result, "chunks", len(image_records)),
            }
            save_full_checkpoint(
                handle,
                intermediate,
                marker={
                    "stage": "intermediate",
                    "profile_key": profile.key,
                    "run_fingerprint": fingerprint,
                    "created_at": _utc_now(),
                },
            )
            _mark_stages(
                current_manifest,
                ("image_edit", "save_intermediate_checkpoint"),
                "complete",
            )
            current_manifest["updated_at"] = _utc_now()
            _record_covariance_accesses(current_manifest, covariance_provider)
            atomic_write_json(manifest_path, current_manifest)
            release_model_handle(handle)
            handle = None

        _mark_stages(current_manifest, ("reload_intermediate_checkpoint",), "running")
        current_manifest["updated_at"] = _utc_now()
        atomic_write_json(manifest_path, current_manifest)
        handle = _load_handle(adapter, intermediate, profile, device)
        _mark_stages(current_manifest, ("reload_intermediate_checkpoint",), "complete")

        text_view = handle.text_edit_view()
        text_layout = _editor_layout(text_view.layout)
        text_hparams = _load_hparams(
            profile.text_hparams_path,
            layout=text_layout,
            loader=hparams_loader,
        )
        current_manifest.setdefault("resolved_module_paths", {})["text"] = _layout_dict(
            text_view.layout
        )
        _mark_stages(current_manifest, ("text_edit", "save_final_checkpoint"), "running")
        current_manifest["updated_at"] = _utc_now()
        atomic_write_json(manifest_path, current_manifest)
        _apply_text_stage(
            editor=text_editor,
            model=text_view.model,
            tokenizer=text_view.tokenizer,
            records=text_records,
            hparams=text_hparams,
            covariance_provider=covariance_provider,
            records_per_update=int(profile.records_per_update),
        )
        current_manifest.setdefault("stage_metadata", {})["text_edit"] = {
            "records": len(text_records),
            "records_per_update": int(profile.records_per_update),
        }
        save_full_checkpoint(
            handle,
            final,
            marker={
                "stage": "final",
                "profile_key": profile.key,
                "run_fingerprint": fingerprint,
                "created_at": _utc_now(),
            },
        )
        _mark_stages(
            current_manifest,
            ("text_edit", "save_final_checkpoint", "write_manifest"),
            "complete",
        )
        current_manifest["status"] = "complete"
        current_manifest["resumed_from"] = None if resumed_from is None else str(resumed_from)
        current_manifest["completed_at"] = _utc_now()
        current_manifest["updated_at"] = current_manifest["completed_at"]
        current_manifest.pop("error", None)
        _record_covariance_accesses(current_manifest, covariance_provider)
        atomic_write_json(manifest_path, current_manifest)
        return TwoStageRunResult(current_manifest, resumed_from)
    except Exception as exc:
        current_manifest["status"] = "failed"
        current_manifest["updated_at"] = _utc_now()
        current_manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        atomic_write_json(manifest_path, current_manifest)
        raise
    finally:
        if handle is not None:
            release_model_handle(handle)


def _initialize_manifest(
    *,
    expected: dict[str, Any],
    manifest_path: Path,
    intermediate: Path,
    final: Path,
    resume: bool,
) -> dict[str, Any]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(
                f"run manifest already exists: {manifest_path}; use --resume or --force-restart"
            )
        current = read_json(manifest_path)
        if current.get("run_fingerprint") != expected.get("run_fingerprint"):
            raise ValueError(
                "resume inputs differ from run_manifest.json; use --force-restart for a new run"
            )
        current["status"] = "running"
        current["updated_at"] = _utc_now()
        current.pop("error", None)
        atomic_write_json(manifest_path, current)
        return current

    if resume:
        raise FileNotFoundError(f"cannot resume without run manifest: {manifest_path}")
    for checkpoint in (intermediate, final):
        if checkpoint.exists():
            raise FileExistsError(f"run artifact already exists: {checkpoint}; use --force-restart")
    expected["status"] = "running"
    expected.setdefault("created_at", _utc_now())
    expected["updated_at"] = _utc_now()
    atomic_write_json(manifest_path, expected)
    return expected


def _stage_checkpoint(plan: Any, stage_name: str) -> Path:
    for stage in plan.stages:
        if stage.name == stage_name:
            value = stage.path or stage.output_checkpoint
            if value is None:
                break
            return Path(value)
    raise ValueError(f"run plan has no checkpoint path for stage: {stage_name}")


def _load_handle(adapter: Any, checkpoint: str | Path, profile: Any, device: str | None) -> Any:
    return adapter.load(
        checkpoint,
        device=device,
        dtype=profile.dtype,
        trust_remote_code=profile.trust_remote_code,
    )


def _editor_layout(layout: Any) -> Any:
    from .editing.hparams import ModuleLayout

    return ModuleLayout.from_layer_path(
        layout.layer_path,
        layout.edit_module_path,
        ln_f_module=layout.norm_path,
        lm_head_module=layout.lm_head_path,
    )


def _load_hparams(path: str | Path, *, layout: Any, loader: Callable[..., Any] | None) -> Any:
    if loader is None:
        from .editing.hparams import load_hparams

        loader = load_hparams
    return loader(path, layout=layout)


def _apply_image_stage(
    *,
    editor: Callable[..., Any] | None,
    handle: Any,
    records: Sequence[Mapping[str, Any]],
    hparams: Any,
    covariance_provider: Any,
    profile: Any,
    image_root: Path,
    device: str | None,
) -> Any:
    if editor is None:
        from .editing.image import ImageBackendConfig, apply_image_edits

        editor = apply_image_edits
        config = ImageBackendConfig(
            family=profile.family,
            data_root=image_root,
            attention_key=profile.attention_token_key,
            device=device or next(handle.model.parameters()).device,
        )
    else:
        config = None
    return editor(
        handle=handle,
        records=list(records),
        hparams=hparams,
        covariance_provider=covariance_provider,
        config=config,
        records_per_update=int(profile.records_per_update),
        return_original_weights=False,
    )


def _apply_text_stage(
    *,
    editor: Callable[..., Any] | None,
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    hparams: Any,
    covariance_provider: Any,
    records_per_update: int,
) -> Any:
    if editor is None:
        from .editing.delman import apply_text_edits

        editor = apply_text_edits
    return editor(
        model=model,
        tok=tokenizer,
        requests=list(records),
        hparams=hparams,
        covariance_provider=covariance_provider,
        records_per_update=records_per_update,
        return_original_weights=False,
    )


def _layout_dict(layout: Any) -> dict[str, Any]:
    return {
        "family": layout.family,
        "logical_layer": layout.logical_layer,
        "layer_path": layout.layer_path,
        "edit_module_path": layout.edit_module_path,
        "mlp_path": layout.mlp_path,
        "down_proj_path": layout.down_proj_path,
        "norm_path": layout.norm_path,
        "lm_head_path": layout.lm_head_path,
    }


def _mark_stages(manifest: dict[str, Any], names: Sequence[str], status: str) -> None:
    stage_status = manifest.setdefault("stage_status", {})
    for name in names:
        stage_status[name] = status


def _complete_all_stages(manifest: dict[str, Any], plan: Any) -> None:
    _mark_stages(manifest, [stage.name for stage in plan.stages], "complete")


def _record_covariance_accesses(manifest: dict[str, Any], provider: Any) -> None:
    accesses = getattr(provider, "accesses", None)
    if accesses is not None:
        manifest.setdefault("covariance", {})["accesses"] = list(accesses)


def release_model_handle(handle: Any) -> None:
    """Release a loaded parent VLM before the next GPU lifecycle begins."""

    model = getattr(handle, "model", None)
    if model is not None:
        with suppress(AttributeError, TypeError):
            handle.model = None
        del model
    gc.collect()
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
