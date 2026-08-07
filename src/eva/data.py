"""Dataset loading and lightweight schema/image checks."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetValidation:
    path: Path
    kind: str
    valid: bool
    records: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["errors"] = list(self.errors)
        return data


@dataclass(frozen=True)
class ImageAssetsValidation:
    dataset_path: Path
    image_root: Path
    valid: bool
    expected: int
    found: int
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset_path"] = str(self.dataset_path)
        data["image_root"] = str(self.image_root)
        data["missing"] = list(self.missing)
        return data


def load_dataset_records(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    with dataset_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("dataset root must be a list")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every dataset record must be an object")
    return records


def validate_dataset_schema(
    path: str | Path,
    *,
    kind: str,
    attention_token_key: str | None = None,
    expected_records: int = 200,
) -> DatasetValidation:
    dataset_path = Path(path)
    errors: list[str] = []
    try:
        records = load_dataset_records(dataset_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return DatasetValidation(dataset_path, kind, False, 0, (str(exc),))

    if len(records) != expected_records:
        errors.append(f"expected {expected_records} records, found {len(records)}")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {index} must be an object")
            continue
        if record.get("case_id") != index:
            errors.append(f"record {index} has invalid case_id")
        rewrite = record.get("requested_rewrite")
        if not isinstance(rewrite, dict):
            errors.append(f"record {index} missing requested_rewrite object")
            continue
        required = ["prompt_adv", "target_new", "subject"]
        if kind == "image":
            required.extend(["prompt_gen", "image_path"])
        for field in required:
            if field not in rewrite:
                errors.append(f"record {index} missing requested_rewrite.{field}")
        if kind == "image":
            survive_ids = rewrite.get("survive_ids")
            if not (
                isinstance(survive_ids, list)
                and len(survive_ids) == 1
                and isinstance(survive_ids[0], int)
            ):
                errors.append(f"record {index} must have exactly one integer survive_id")
            attn = record.get("attn_token_idx")
            if not isinstance(attn, dict) or attention_token_key not in attn:
                errors.append(f"record {index} missing attention key {attention_token_key!r}")
            elif (
                isinstance(survive_ids, list)
                and survive_ids
                and attn[attention_token_key] != survive_ids[0]
            ):
                errors.append(f"record {index} attention index does not match survive_id")

    return DatasetValidation(dataset_path, kind, not errors, len(records), tuple(errors))


def resolve_image_asset(image_root: str | Path, relative_path: str | Path) -> Path:
    """Resolve a dataset image path against a mirrored or direct image root."""

    root = Path(image_root)
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"dataset image_path must be relative: {relative}")
    mirrored = root / relative
    if mirrored.is_file():
        return mirrored
    direct = root / relative.name
    if direct.is_file():
        return direct
    return mirrored


def validate_image_assets(
    dataset_path: str | Path,
    *,
    image_root: str | Path | None = None,
) -> ImageAssetsValidation:
    dataset = Path(dataset_path)
    root = Path(image_root) if image_root is not None else dataset.parent
    records = load_dataset_records(dataset)
    missing: list[str] = []
    found = 0
    for record in records:
        rewrite = record.get("requested_rewrite", {})
        relative = rewrite.get("image_path") if isinstance(rewrite, dict) else None
        if not isinstance(relative, str):
            missing.append(f"case {record.get('case_id', '?')}: missing image_path")
            continue
        resolved = resolve_image_asset(root, relative)
        if resolved.is_file():
            found += 1
        else:
            missing.append(relative)
    return ImageAssetsValidation(
        dataset_path=dataset,
        image_root=root,
        valid=not missing,
        expected=len(records),
        found=found,
        missing=tuple(missing),
    )


def load_image_edit_records(
    dataset_path: str | Path,
    *,
    image_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """Load outer image records while normalizing paths for an image backend."""

    dataset = Path(dataset_path)
    root = Path(image_root) if image_root is not None else dataset.parent
    records = deepcopy(load_dataset_records(dataset))
    for record in records:
        rewrite = record["requested_rewrite"]
        resolved = resolve_image_asset(root, rewrite["image_path"])
        if not resolved.is_file():
            raise FileNotFoundError(
                f"missing image for case {record.get('case_id', '?')}: {resolved}"
            )
        rewrite["image_path"] = str(resolved.relative_to(root))
    return records, root


def load_text_edit_records(dataset_path: str | Path) -> list[dict[str, Any]]:
    """Flatten public text records into the shape consumed by DELMAN."""

    flattened: list[dict[str, Any]] = []
    for record in load_dataset_records(dataset_path):
        flattened.append(
            {
                "case_id": record["case_id"],
                **deepcopy(record["requested_rewrite"]),
            }
        )
    return flattened
