"""Checkpoint inspection and crash-safe full-model persistence."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

CHECKPOINT_MARKER = ".eva_checkpoint.json"
PROCESSOR_ASSETS = (
    "preprocessor_config.json",
    "processor_config.json",
)
TOKENIZER_VOCAB_ASSETS = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
)


@dataclass(frozen=True)
class CheckpointMetadata:
    path: Path
    exists: bool
    config: dict[str, Any]
    index_files: tuple[Path, ...]
    weight_keys_sample: tuple[str, ...]
    module_names_sample: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["index_files"] = [str(path) for path in self.index_files]
        return data


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _module_from_weight_key(key: str) -> str:
    suffixes = (".weight", ".bias")
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def inspect_checkpoint(path: str | Path, *, sample_size: int = 50) -> CheckpointMetadata:
    checkpoint_path = Path(path)
    errors: list[str] = []
    config: dict[str, Any] = {}
    index_files: list[Path] = []
    weight_keys: list[str] = []

    if not checkpoint_path.exists():
        return CheckpointMetadata(
            checkpoint_path, False, {}, (), (), (), ("checkpoint path does not exist",)
        )

    config_path = checkpoint_path / "config.json"
    if config_path.is_file():
        try:
            config = _read_json(config_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    else:
        errors.append("missing config.json")

    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint_path / name
        if not index_path.is_file():
            continue
        index_files.append(index_path)
        try:
            index = _read_json(index_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
            continue
        weight_map = index.get("weight_map", {})
        if isinstance(weight_map, dict):
            weight_keys.extend(str(key) for key in weight_map)

    module_names = tuple(
        dict.fromkeys(_module_from_weight_key(key) for key in weight_keys[:sample_size])
    )
    return CheckpointMetadata(
        path=checkpoint_path,
        exists=True,
        config=config,
        index_files=tuple(index_files),
        weight_keys_sample=tuple(weight_keys[:sample_size]),
        module_names_sample=module_names,
        errors=tuple(errors),
    )


def inspect_checkpoint_metadata(path: str | Path, *, sample_size: int = 50) -> CheckpointMetadata:
    return inspect_checkpoint(path, sample_size=sample_size)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object with a useful path-aware error."""

    return _read_json(Path(path))


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically replace a JSON file in its destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def checkpoint_marker(path: str | Path) -> Path:
    return Path(path) / CHECKPOINT_MARKER


def is_complete_checkpoint(
    path: str | Path,
    *,
    run_fingerprint: str | None = None,
    stage: str | None = None,
) -> bool:
    """Return whether ``path`` is a full EVA checkpoint for the requested run."""

    checkpoint = Path(path)
    marker_path = checkpoint_marker(checkpoint)
    if not checkpoint.is_dir() or not marker_path.is_file():
        return False
    try:
        marker = _read_json(marker_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if marker.get("complete") is not True:
        return False
    if run_fingerprint is not None and marker.get("run_fingerprint") != run_fingerprint:
        return False
    if stage is not None and marker.get("stage") != stage:
        return False
    return not checkpoint_validation_errors(checkpoint)


def checkpoint_validation_errors(path: str | Path) -> tuple[str, ...]:
    """Validate the reload assets required by all supported EVA VLM adapters."""

    checkpoint = Path(path)
    errors: list[str] = []
    if not (checkpoint / "config.json").is_file():
        errors.append("missing config.json")
    if not _weight_files(checkpoint):
        errors.append("missing model weights")
    if not any((checkpoint / name).is_file() for name in PROCESSOR_ASSETS):
        errors.append(f"missing processor asset ({' or '.join(PROCESSOR_ASSETS)})")
    if not (checkpoint / "tokenizer_config.json").is_file():
        errors.append("missing tokenizer_config.json")
    if not any((checkpoint / name).is_file() for name in TOKENIZER_VOCAB_ASSETS):
        errors.append(f"missing tokenizer vocabulary ({' or '.join(TOKENIZER_VOCAB_ASSETS)})")
    errors.extend(_weight_index_errors(checkpoint))
    return tuple(errors)


def save_full_checkpoint(
    handle: Any,
    output_dir: str | Path,
    *,
    marker: dict[str, Any],
) -> Path:
    """Save a complete parent VLM and publish it with one atomic rename.

    The model, processor, and tokenizer are written by the model handle into a
    private sibling directory. The destination only becomes visible after the
    full checkpoint and EVA completion marker have both been validated.
    """

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"checkpoint already exists: {destination}; use --resume or --force-restart"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    try:
        handle.save_full_model(temporary)
        errors = checkpoint_validation_errors(temporary)
        if errors:
            raise RuntimeError(
                f"full checkpoint save is not reloadable: {temporary}: {'; '.join(errors)}"
            )
        atomic_write_json(
            checkpoint_marker(temporary),
            {
                "schema_version": 1,
                "complete": True,
                **marker,
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def remove_run_artifact(path: str | Path, *, output_dir: str | Path) -> None:
    """Remove one explicitly named run artifact after containment validation."""

    target = Path(path)
    root = Path(output_dir).resolve()
    resolved = target.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"refusing to remove artifact outside run output directory: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def _weight_files(checkpoint: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for pattern in ("*.safetensors", "*.bin"):
        candidates.extend(
            path
            for path in checkpoint.glob(pattern)
            if path.name not in {"training_args.bin", "optimizer.bin"}
        )
    return tuple(sorted(candidates))


def _weight_index_errors(checkpoint: Path) -> list[str]:
    errors: list[str] = []
    checkpoint_root = checkpoint.resolve()
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = checkpoint / index_name
        if not index_path.is_file():
            continue
        try:
            index = _read_json(index_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {index_name}: {exc}")
            continue
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            errors.append(f"{index_name} has no weight_map entries")
            continue
        for relative_name in sorted({str(name) for name in weight_map.values()}):
            shard = checkpoint / relative_name
            try:
                shard.resolve().relative_to(checkpoint_root)
            except ValueError:
                errors.append(f"{index_name} references a shard outside the checkpoint")
                continue
            if not shard.is_file():
                errors.append(f"missing indexed weight shard: {relative_name}")
    return errors
