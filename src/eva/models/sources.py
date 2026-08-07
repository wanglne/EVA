"""Resolve pinned Hugging Face model snapshots without importing model runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def model_source_label(profile: Any) -> str:
    repo_id, revision = _source(profile)
    return f"{repo_id}@{revision}"


def resolve_model_path(
    profile: Any,
    model_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return an explicit local checkpoint or materialize the pinned HF snapshot."""

    if model_path is not None:
        local_path = Path(model_path).expanduser()
        if not local_path.is_dir():
            raise FileNotFoundError(f"model path is not a directory: {local_path}")
        return local_path.resolve()

    repo_id, revision = _source(profile)
    snapshot_download = _import_snapshot_download()
    resolved = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=None if cache_dir is None else str(Path(cache_dir).expanduser()),
        local_files_only=local_files_only,
    )
    local_path = Path(resolved)
    if not local_path.is_dir():
        raise FileNotFoundError(f"Hugging Face snapshot is not a directory: {local_path}")
    return local_path.resolve()


def _source(profile: Any) -> tuple[str, str]:
    if isinstance(profile, dict):
        model = profile.get("model", {})
        huggingface = model.get("huggingface", {}) if isinstance(model, dict) else {}
        repo_id = huggingface.get("repo_id")
        revision = huggingface.get("revision")
    else:
        repo_id = getattr(profile, "huggingface_repo_id", None)
        revision = getattr(profile, "huggingface_revision", None)
    if not repo_id or not revision:
        raise ValueError("model profile must pin Hugging Face repo_id and revision")
    return str(repo_id), str(revision)


def _import_snapshot_download() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face model download requires `pip install 'eva-edit[gpu]'`"
        ) from exc
    return snapshot_download
