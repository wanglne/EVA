"""Build resumable EVA run plans without importing editing runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StagePlan:
    name: str
    input_checkpoint: Path | None = None
    output_checkpoint: Path | None = None
    dataset_path: Path | None = None
    hparams_path: Path | None = None
    path: Path | None = None
    source: Path | None = None
    full_checkpoint: bool = False
    reload_model: bool = False
    resume_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


Stage = StagePlan


@dataclass(frozen=True)
class RunPlan:
    profile_key: str
    model_key: str
    model_path: Path
    output_dir: Path
    covariance_root: Path | None
    manifest_path: Path
    resume_from: Path
    stages: tuple[StagePlan, ...]
    resume: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        data["stages"] = [stage.to_dict() for stage in self.stages]
        return data


def _field(profile: Any, name: str, default: Any = None) -> Any:
    if isinstance(profile, dict):
        return profile.get(name, default)
    return getattr(profile, name, default)


def _profile_key(profile: Any) -> str:
    model = _field(profile, "model", {}) or {}
    key = _field(profile, "key") or _field(model, "key")
    if not key:
        raise KeyError("profile must include key")
    return str(key)


def build_run_plan(
    profile: Any,
    *,
    model_path: str | Path,
    output_dir: str | Path,
    covariance_root: str | Path | None = None,
) -> RunPlan:
    model = Path(model_path)
    out = Path(output_dir)
    cov_root = None if covariance_root is None else Path(covariance_root)
    artifacts = _field(profile, "artifacts", {}) or {}
    intermediate = out / artifacts.get("intermediate", "image_checkpoint")
    final = out / artifacts.get("final", "final_checkpoint")
    manifest = out / "run_manifest.json"

    stages = (
        StagePlan(
            "image_edit",
            input_checkpoint=model,
            output_checkpoint=intermediate,
            dataset_path=Path(_field(profile, "dataset_path")),
            hparams_path=Path(_field(profile, "image_hparams_path")),
            resume_key="image_edit",
        ),
        StagePlan(
            "save_intermediate_checkpoint",
            output_checkpoint=intermediate,
            path=intermediate,
            full_checkpoint=True,
            resume_key="save_intermediate_checkpoint",
        ),
        StagePlan(
            "reload_intermediate_checkpoint",
            source=intermediate,
            reload_model=True,
            resume_key="reload_intermediate_checkpoint",
        ),
        StagePlan(
            "text_edit",
            input_checkpoint=intermediate,
            output_checkpoint=final,
            dataset_path=Path(
                _field(profile, "text_dataset_path", _field(profile, "dataset_path"))
            ),
            hparams_path=Path(_field(profile, "text_hparams_path")),
            resume_key="text_edit",
        ),
        StagePlan(
            "save_final_checkpoint",
            output_checkpoint=final,
            path=final,
            full_checkpoint=True,
            resume_key="save_final_checkpoint",
        ),
        StagePlan("write_manifest", path=manifest, resume_key="write_manifest"),
    )
    resume = {
        "resume_from": str(manifest),
        "full_checkpoint_reload_boundary": str(intermediate),
        "stages": {stage.name: stage.to_dict() for stage in stages},
    }
    key = _profile_key(profile)
    return RunPlan(
        profile_key=key,
        model_key=key,
        model_path=model,
        output_dir=out,
        covariance_root=cov_root,
        manifest_path=manifest,
        resume_from=manifest,
        stages=stages,
        resume=resume,
    )


def build_stage_plan(profile: Any, *, output_dir: str | Path) -> RunPlan:
    return build_run_plan(profile, model_path=Path("<model-path>"), output_dir=output_dir)
