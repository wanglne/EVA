"""Configuration loading for EVA model profiles."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PUBLIC_PROFILE_KEYS = ("llava-1.5-7b", "qwen2.5-vl-7b-instruct", "internvl3.5-8b")
LEGACY_PROFILE_KEYS = {
    "llava15": "llava-1.5-7b",
    "qwen25vl": "qwen2.5-vl-7b-instruct",
    "internvl35": "internvl3.5-8b",
}


@dataclass(frozen=True)
class ModelProfile(Mapping[str, Any]):
    key: str
    legacy_key: str | None
    model_key: str
    family: str
    model_type: str
    checkpoint_name: str
    architecture: str
    dtype: str
    trust_remote_code: bool
    huggingface_repo_id: str
    huggingface_revision: str
    dataset_path: Path
    text_dataset_path: Path
    attention_token_key: str
    critical_layer: int
    image_hparams_path: Path
    text_hparams_path: Path
    covariance: dict[str, Any]
    covariance_dataset: str
    covariance_sample_size: int
    covariance_dtype: str
    covariance_identity: str
    covariance_legacy_patterns: tuple[str, ...]
    records_per_update: int
    expected_updates: int
    ordering: str
    model: dict[str, Any]
    artifacts: dict[str, Any]
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, tuple):
                data[key] = list(value)
        return data


def default_repo_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd()


def default_resource_root() -> Path:
    return Path(__file__).resolve().parent / "resources"


def list_profiles() -> tuple[str, ...]:
    return PUBLIC_PROFILE_KEYS


def canonical_profile_key(key: str) -> str:
    if key in PUBLIC_PROFILE_KEYS:
        return key
    try:
        return LEGACY_PROFILE_KEYS[key]
    except KeyError as exc:
        choices = ", ".join(PUBLIC_PROFILE_KEYS)
        raise ValueError(f"unknown model profile {key!r}; expected one of: {choices}") from exc


def load_profile(key: str, root: str | Path | None = None) -> ModelProfile:
    resource_root = (
        _resolve_resource_root(Path(root)) if root is not None else default_resource_root()
    )
    profile_key = canonical_profile_key(key)
    profile_path = resource_root / "configs" / "models" / f"{profile_key}.json"
    raw = _read_json(profile_path)

    model = raw["model"]
    datasets = raw["datasets"]
    image_dataset = datasets["image"]
    text_dataset = datasets["text"]
    hparams = raw["hparams"]
    expectations = model.get("expectations", {})
    model_type = expectations.get("model_type", "")
    family = model.get("family") or _adapter_family(profile_key)
    attention_key = image_dataset["expected_attention_key"]
    covariance = dict(raw.get("covariance", {}))
    schedule = datasets.get("update_schedule", {})

    return ModelProfile(
        key=profile_key,
        legacy_key=model.get("legacy_key"),
        model_key=profile_key,
        family=family,
        model_type=model_type,
        checkpoint_name=model["checkpoint_name"],
        architecture=expectations.get("architecture", ""),
        dtype=model["dtype"],
        trust_remote_code=bool(model["trust_remote_code"]),
        huggingface_repo_id=str(model["huggingface"]["repo_id"]),
        huggingface_revision=str(model["huggingface"]["revision"]),
        dataset_path=resource_root / image_dataset["path"],
        text_dataset_path=resource_root / text_dataset["path"],
        attention_token_key=attention_key,
        critical_layer=_layer_from_attention_key(attention_key),
        image_hparams_path=resource_root / hparams["image_path"],
        text_hparams_path=resource_root / hparams["text_path"],
        covariance=covariance,
        covariance_dataset=str(covariance.get("dataset", "wikipedia")),
        covariance_sample_size=int(covariance.get("sample_size", 0)),
        covariance_dtype=str(covariance.get("dtype", "float32")),
        covariance_identity=str(covariance.get("canonical_identity", profile_key)),
        covariance_legacy_patterns=tuple(
            str(pattern) for pattern in covariance.get("legacy_filename_patterns", ())
        ),
        records_per_update=int(schedule.get("records_per_update", 1)),
        expected_updates=int(schedule.get("updates", 200)),
        ordering=str(schedule.get("ordering", "sequential")),
        model=dict(model),
        artifacts=dict(raw.get("artifacts", {})),
        raw=raw,
    )


def load_model_profile(key: str, repo_root: str | Path | None = None) -> ModelProfile:
    return load_profile(key, root=repo_root)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _layer_from_attention_key(value: str) -> int:
    prefix = "attn_layer_"
    if not value.startswith(prefix):
        raise ValueError(f"invalid attention token key: {value!r}")
    return int(value[len(prefix) :])


def _adapter_family(profile_key: str) -> str:
    return {
        "llava-1.5-7b": "llava",
        "qwen2.5-vl-7b-instruct": "qwen2_5_vl",
        "internvl3.5-8b": "internvl",
    }[profile_key]


def _resolve_resource_root(root: Path) -> Path:
    candidates = (root, root / "src" / "eva" / "resources")
    for candidate in candidates:
        model_dir = candidate / "configs" / "models"
        if model_dir.is_dir() and any(model_dir.glob("*.json")):
            return candidate
    expected = candidates[-1] / "configs" / "models"
    raise FileNotFoundError(f"EVA packaged resources not found; expected: {expected}")
