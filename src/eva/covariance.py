"""Canonical, import-light covariance registry and Wikipedia moment builder."""

from __future__ import annotations

import hashlib
import json
import os
import random
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models.paths import language_down_proj_candidates


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    size_bytes: int | None
    mtime_ns: int | None
    sha256: str | None = None


@dataclass(frozen=True)
class CovarianceReference:
    source_path: Path
    canonical_path: Path
    layer: int
    profile_key: str
    manifest_path: Path
    metadata: dict[str, Any]
    module_path: str | None = None


@dataclass(frozen=True)
class LoadedCovariance:
    C: Any
    count: int | None
    source_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CovariancePreparation:
    """Resolved covariance lifecycle for one edit run."""

    provider: SharedCovarianceProvider
    reused_layers: tuple[int, ...]
    imported_layers: tuple[int, ...]
    computed_layers: tuple[int, ...]
    sources: dict[int, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reused_layers": list(self.reused_layers),
            "imported_layers": list(self.imported_layers),
            "computed_layers": list(self.computed_layers),
            "sources": {str(layer): str(path) for layer, path in self.sources.items()},
        }


@dataclass(frozen=True)
class CovariancePackage:
    archive_path: Path
    checksum_path: Path
    archive_sha256: str
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_path": str(self.archive_path),
            "checksum_path": str(self.checksum_path),
            "archive_sha256": self.archive_sha256,
            "manifest": self.manifest,
        }


class CovarianceStore:
    """Index one C matrix by profile identity and logical language layer."""

    def __init__(
        self,
        root: Path | str,
        legacy_roots: Iterable[Path | str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.legacy_roots = tuple(Path(value) for value in (legacy_roots or ()))

    def canonical_path(self, profile: Any, layer: int) -> Path:
        covariance = _covariance_config(profile)
        identity = covariance.get("canonical_identity", _profile_key(profile))
        return self.root / str(identity) / f"layer_{layer}.npz"

    def manifest_path(self, profile: Any, layer: int) -> Path:
        canonical = self.canonical_path(profile, layer)
        return canonical.with_name(f"layer_{layer}.reference.json")

    def has(self, profile: Any, layer: int, *, validate: bool = False) -> bool:
        try:
            self.resolve_source(profile, layer, validate=validate)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            return False
        return True

    def resolve_source(self, profile: Any, layer: int, *, validate: bool = True) -> Path:
        canonical = self.canonical_path(profile, layer)
        if canonical.is_file():
            return canonical
        manifest_path = self.manifest_path(profile, layer)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing covariance for {_profile_key(profile)} logical layer {layer}: "
                f"{canonical} (or {manifest_path})"
            )
        payload = _read_json(manifest_path)
        if (
            payload.get("profile_key") != _profile_key(profile)
            or int(payload.get("layer", -1)) != layer
        ):
            raise ValueError(f"incompatible covariance reference: {manifest_path}")
        source = Path(payload["source_path"])
        if not source.is_file():
            raise FileNotFoundError(f"covariance reference target does not exist: {source}")
        if validate:
            _validate_fingerprint(source, payload.get("metadata", {}).get("fingerprint", {}))
        return source

    def find_legacy(
        self,
        *,
        legacy_path: Path | str | None = None,
        profile: Any,
        layer: int,
        validate_hash: bool = False,
        write_manifest: bool = False,
    ) -> CovarianceReference:
        source = (
            Path(legacy_path) if legacy_path is not None else self._discover_one(profile, layer)
        )
        if write_manifest and not source.is_file():
            raise FileNotFoundError(f"legacy covariance does not exist: {source}")
        if source.exists():
            source = source.resolve()
        metadata = self._metadata(profile, layer, source, validate_hash=validate_hash)
        reference = CovarianceReference(
            source_path=source,
            canonical_path=self.canonical_path(profile, layer),
            layer=layer,
            profile_key=_profile_key(profile),
            manifest_path=self.manifest_path(profile, layer),
            metadata=metadata,
            module_path=_module_path_from_legacy_name(source.name),
        )
        if write_manifest:
            self.write_manifest(reference)
        return reference

    def discover_legacy(
        self,
        *,
        profile: Any,
        layer: int,
        write_manifest: bool = False,
    ) -> list[CovarianceReference]:
        references = [
            self.find_legacy(
                legacy_path=source,
                profile=profile,
                layer=layer,
                write_manifest=False,
            )
            for source in self._candidate_legacy_paths(profile, layer)
        ]
        if write_manifest:
            if len(references) != 1:
                raise ValueError(
                    f"expected one covariance candidate for {_profile_key(profile)} "
                    f"layer {layer}, found {len(references)}; import the intended path explicitly"
                )
            self.write_manifest(references[0])
        return references

    def write_manifest(self, reference: CovarianceReference) -> Path:
        payload = asdict(reference)
        payload["source_path"] = str(reference.source_path)
        payload["canonical_path"] = str(reference.canonical_path)
        payload["manifest_path"] = str(reference.manifest_path)
        _atomic_write_json(reference.manifest_path, payload)
        return reference.manifest_path

    def load(self, profile: Any, layer: int, *, validate: bool = True) -> LoadedCovariance:
        source = self.resolve_source(profile, layer, validate=validate)
        metadata: dict[str, Any] = {"format": "canonical_npz"}
        manifest_path = self.manifest_path(profile, layer)
        if source != self.canonical_path(profile, layer) and manifest_path.is_file():
            metadata = _read_json(manifest_path).get("metadata", {})
        return _load_npz_covariance(source, metadata)

    def bind(self, profile: Any) -> SharedCovarianceProvider:
        return SharedCovarianceProvider(self, profile)

    def compute_wikipedia_moments(
        self,
        *,
        profile: Any,
        layer: int,
        model: Any,
        tokenizer: Any,
        module_name: str | None = None,
        sample_size: int | None = None,
        batch_size: int = 200,
        batch_tokens: int | None = None,
        dataset_name: str | None = None,
        dataset_config: str | None = None,
        dataset_split: str | None = None,
        dataset_revision: str | None = None,
        output_path: Path | None = None,
        resume: bool = True,
        checkpoint_every: int = 25,
        seed: int = 1,
    ) -> Path:
        output = output_path or self.canonical_path(profile, layer)
        resolved_module = module_name or select_language_down_proj(profile, model, layer)
        compute_dataset = dict(_covariance_config(profile).get("compute_dataset", {}))
        resolved_dataset_name = dataset_name or str(
            compute_dataset.get("repo_id", "wikimedia/wikipedia")
        )
        resolved_dataset_config = dataset_config or str(
            compute_dataset.get("config", "20231101.en")
        )
        resolved_dataset_split = dataset_split or str(compute_dataset.get("split", "train"))
        resolved_dataset_revision = dataset_revision or compute_dataset.get("revision")
        metadata = self._metadata(profile, layer, output, validate_hash=False)
        metadata["module_path"] = resolved_module
        metadata["compute_dataset"] = {
            "repo_id": resolved_dataset_name,
            "config": resolved_dataset_config,
            "split": resolved_dataset_split,
            "revision": resolved_dataset_revision,
        }
        metadata["sampling"] = {
            "unit": "documents",
            "seed": seed,
            "requested_documents": sample_size
            or int(_covariance_config(profile).get("sample_size", 100000)),
        }
        return compute_wikipedia_moments(
            model=model,
            tokenizer=tokenizer,
            module_name=resolved_module,
            output_path=output,
            sample_size=sample_size or int(_covariance_config(profile).get("sample_size", 100000)),
            batch_size=batch_size,
            batch_tokens=batch_tokens,
            dataset_name=resolved_dataset_name,
            dataset_config=resolved_dataset_config,
            dataset_split=resolved_dataset_split,
            dataset_revision=resolved_dataset_revision,
            resume=resume,
            checkpoint_every=checkpoint_every,
            metadata=metadata,
            seed=seed,
        )

    def _discover_one(self, profile: Any, layer: int) -> Path:
        candidates = self._candidate_legacy_paths(profile, layer)
        if not candidates:
            roots = ", ".join(str(root) for root in self.legacy_roots) or "<none supplied>"
            raise FileNotFoundError(
                f"no legacy covariance found for {_profile_key(profile)} layer {layer}; "
                f"searched roots: {roots}"
            )
        return candidates[0]

    def _candidate_legacy_paths(self, profile: Any, layer: int) -> list[Path]:
        patterns = _covariance_config(profile).get("legacy_filename_patterns", ())
        matches: list[Path] = []
        for root in self.legacy_roots:
            for pattern in patterns:
                matches.extend(sorted(root.glob(str(pattern).format(layer=layer))))
        return _dedupe_paths(matches)

    def _metadata(
        self,
        profile: Any,
        layer: int,
        source: Path,
        *,
        validate_hash: bool,
    ) -> dict[str, Any]:
        covariance = _covariance_config(profile)
        return {
            "schema_version": 1,
            "profile_key": _profile_key(profile),
            "canonical_identity": covariance.get("canonical_identity", _profile_key(profile)),
            "logical_layer": layer,
            "logical_module": "language_mlp_down_proj",
            "dataset": covariance.get("dataset", "wikipedia"),
            "precomputed_source": covariance.get("artifact", {}).get("precomputed_source"),
            "sample_size": covariance.get("sample_size"),
            "sample_unit": "documents",
            "dtype": covariance.get("dtype", "float32"),
            "format": "legacy_npz" if source.suffix == ".npz" else source.suffix.lstrip("."),
            "fingerprint": asdict(fingerprint(source, compute_hash=validate_hash)),
        }


class SharedCovarianceProvider:
    """Cache one loaded tensor per logical layer across both edit stages."""

    def __init__(self, store: CovarianceStore, profile: Any) -> None:
        self.store = store
        self.profile = profile
        self._cache: dict[int, LoadedCovariance] = {}
        self.accesses: list[dict[str, Any]] = []

    def preflight(self, layers: Sequence[int]) -> dict[int, Path]:
        return {
            int(layer): self.store.resolve_source(self.profile, int(layer), validate=True)
            for layer in dict.fromkeys(layers)
        }

    def get_covariance(self, *, layer: int, weight_name: str, hparams: Any) -> Any:
        logical_layer = int(layer)
        if logical_layer not in self._cache:
            self._cache[logical_layer] = self.store.load(self.profile, logical_layer)
        self.accesses.append(
            {
                "logical_layer": logical_layer,
                "physical_weight": weight_name,
                "source_path": str(self._cache[logical_layer].source_path),
                "stage": getattr(hparams, "stage", None),
            }
        )
        return self._cache[logical_layer].C

    def source_paths(self) -> dict[int, Path]:
        return {layer: loaded.source_path for layer, loaded in self._cache.items()}


def prepare_covariances(
    store: CovarianceStore,
    profile: Any,
    layers: Sequence[int],
    *,
    compute_missing: Callable[[tuple[int, ...]], None] | None = None,
) -> CovariancePreparation:
    """Reuse, import, or compute every logical covariance required by a run.

    Canonical files and valid reference manifests win. When neither exists,
    exactly one legacy candidate is registered by reference. Remaining layers
    are delegated to ``compute_missing`` and must appear at their canonical
    paths before this function returns.
    """

    logical_layers = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not logical_layers:
        raise ValueError("at least one covariance layer is required")

    reused: list[int] = []
    imported: list[int] = []
    missing: list[int] = []
    for layer in logical_layers:
        manifest_path = store.manifest_path(profile, layer)
        try:
            store.resolve_source(profile, layer, validate=True)
        except FileNotFoundError:
            if manifest_path.exists():
                raise
            discovered = store.discover_legacy(
                profile=profile,
                layer=layer,
                write_manifest=False,
            )
            if len(discovered) > 1:
                paths = ", ".join(str(reference.source_path) for reference in discovered)
                raise ValueError(
                    f"ambiguous covariance candidates for {_profile_key(profile)} "
                    f"layer {layer}: {paths}; import the intended path explicitly"
                ) from None
            if discovered:
                store.write_manifest(discovered[0])
                imported.append(layer)
            else:
                missing.append(layer)
        else:
            reused.append(layer)

    if missing:
        if compute_missing is None:
            layers_text = ", ".join(str(layer) for layer in missing)
            raise FileNotFoundError(
                f"missing covariance for {_profile_key(profile)} logical layers "
                f"{layers_text}; enable automatic computation or import existing files"
            )
        compute_missing(tuple(missing))

    provider = store.bind(profile)
    sources = provider.preflight(logical_layers)
    return CovariancePreparation(
        provider=provider,
        reused_layers=tuple(reused),
        imported_layers=tuple(imported),
        computed_layers=tuple(missing),
        sources=sources,
    )


def package_covariances(
    store: CovarianceStore,
    profile: Any,
    layers: Sequence[int],
    *,
    output_dir: Path | str,
) -> CovariancePackage:
    """Build a deterministic, extraction-ready ZIP from canonical or legacy C files."""

    logical_layers = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not logical_layers:
        raise ValueError("at least one covariance layer is required")

    sources: dict[int, Path] = {}
    for layer in logical_layers:
        try:
            sources[layer] = store.resolve_source(profile, layer, validate=True)
        except FileNotFoundError:
            if store.manifest_path(profile, layer).exists():
                raise
            discovered = store.discover_legacy(
                profile=profile,
                layer=layer,
                write_manifest=False,
            )
            if len(discovered) != 1:
                raise ValueError(
                    f"expected one covariance source for {_profile_key(profile)} layer {layer}, "
                    f"found {len(discovered)}"
                ) from None
            sources[layer] = discovered[0].source_path

    covariance = _covariance_config(profile)
    identity = str(covariance.get("canonical_identity", _profile_key(profile)))
    artifact = dict(covariance.get("artifact", {}))
    archive_name = str(artifact.get("archive_name", f"eva-covariance-{_profile_key(profile)}.zip"))
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_path = destination_dir / archive_name
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    temporary = archive_path.with_name(f".{archive_path.name}.{uuid4().hex}.partial")
    entries: list[dict[str, Any]] = []

    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for layer, source in sorted(sources.items()):
                archive_name_for_layer = f"{identity}/layer_{layer}.npz"
                digest = hashlib.sha256()
                size_bytes = 0
                with (
                    Path(source).open("rb") as source_handle,
                    archive.open(
                        _zip_info(archive_name_for_layer),
                        mode="w",
                        force_zip64=True,
                    ) as archive_handle,
                ):
                    for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                        archive_handle.write(chunk)
                        digest.update(chunk)
                        size_bytes += len(chunk)
                entries.append(
                    {
                        "layer": layer,
                        "path": archive_name_for_layer,
                        "sha256": digest.hexdigest(),
                        "size_bytes": size_bytes,
                        "source_basename": Path(source).name,
                    }
                )

            manifest = {
                "schema_version": 1,
                "profile_key": _profile_key(profile),
                "canonical_identity": identity,
                "dataset": covariance.get("dataset", "wikipedia"),
                "sample_size": covariance.get("sample_size"),
                "dtype": covariance.get("dtype", "float32"),
                "precomputed_source": artifact.get("precomputed_source"),
                "extraction_root": "cache/covariances",
                "files": entries,
            }
            manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            manifest_name = f"{identity}/manifest.json"
            archive.writestr(_zip_info(manifest_name), manifest_bytes)
            sums = [f"{entry['sha256']}  {entry['path']}\n" for entry in entries]
            sums.append(f"{hashlib.sha256(manifest_bytes).hexdigest()}  {manifest_name}\n")
            archive.writestr(_zip_info(f"{identity}/SHA256SUMS"), "".join(sums).encode())
        os.replace(temporary, archive_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    archive_sha256 = _sha256(archive_path)
    _atomic_write_text(
        checksum_path,
        f"{archive_sha256}  {archive_path.name}\n",
    )
    return CovariancePackage(
        archive_path=archive_path,
        checksum_path=checksum_path,
        archive_sha256=archive_sha256,
        manifest=manifest,
    )


def select_language_down_proj(profile: Any, model: Any, layer: int) -> str:
    family = str(
        _field(profile, "family")
        or _field(_field(profile, "model", {}), "family")
        or _field(_field(_field(profile, "model", {}), "expectations", {}), "model_type")
        or ""
    )
    module_names = {name for name, _module in model.named_modules()}
    for candidate in language_down_proj_candidates(family, layer):
        if candidate in module_names:
            return candidate
    relative = f"model.layers.{layer}.mlp.down_proj"
    if relative in module_names:
        return relative
    suffix = f".layers.{layer}.mlp.down_proj"
    matches = sorted(name for name in module_names if name.endswith(suffix))
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"could not find language down_proj module for {family} layer {layer}")


def compute_wikipedia_moments(
    *,
    model: Any,
    tokenizer: Any,
    module_name: str,
    output_path: Path,
    sample_size: int,
    batch_size: int,
    batch_tokens: int | None,
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    dataset_revision: str | None,
    resume: bool,
    checkpoint_every: int,
    metadata: dict[str, Any],
    seed: int = 1,
) -> Path:
    """Compute sampler-compatible second moments over a pinned Wikipedia snapshot."""

    torch = _import_torch()
    load_dataset = _import_load_dataset()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial.pt")
    device = next(model.parameters()).device
    dtype = getattr(torch, str(metadata.get("dtype", "float32")))

    dataset_kwargs: dict[str, Any] = {"split": dataset_split}
    if dataset_revision is not None:
        dataset_kwargs["revision"] = dataset_revision
    dataset = load_dataset(dataset_name, dataset_config, **dataset_kwargs)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = indices[: min(sample_size, len(indices))]
    selection_hash = hashlib.sha256(
        ",".join(str(index) for index in selected).encode("ascii")
    ).hexdigest()

    document_cursor = 0
    activation_count = 0
    mom2 = None
    if resume and partial_path.is_file():
        state = torch.load(partial_path, map_location="cpu")
        expected = {
            "module_name": module_name,
            "sample_size": len(selected),
            "seed": seed,
            "selection_sha256": selection_hash,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "dataset_revision": dataset_revision,
        }
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"incompatible covariance partial checkpoint: {partial_path}")
        document_cursor = int(state["document_cursor"])
        activation_count = int(state["activation_count"])
        mom2 = state["mom2"].to(device=device, dtype=dtype)

    module = dict(model.named_modules()).get(module_name)
    if module is None:
        raise ValueError(f"model does not contain covariance module: {module_name}")
    max_length = _max_sequence_length(model)
    token_budget = batch_tokens or max_length * 3
    if batch_tokens is not None and batch_tokens < max_length:
        max_length = batch_tokens
    captured: dict[str, Any] = {}

    def hook(_module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
        captured["input"] = inputs[0].detach()
        raise _StopForward

    hook_handle = module.register_forward_hook(hook)
    try:
        outer_batch = max(1, int(batch_size))
        for group_index, start in enumerate(range(document_cursor, len(selected), outer_batch)):
            group_indices = selected[start : start + outer_batch]
            tokenized = [
                _tokenize_document(
                    tokenizer,
                    _dataset_text(dataset[index]),
                    max_length=max_length,
                    torch=torch,
                )
                for index in group_indices
            ]
            for encoded in _length_collated_batches(
                tokenized, token_budget=token_budget, torch=torch
            ):
                encoded = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
                captured.clear()
                with torch.no_grad(), suppress(_StopForward):
                    model(**encoded)
                if "input" not in captured:
                    raise RuntimeError(f"covariance hook did not run for module: {module_name}")
                features = _flatten_masked(captured["input"], encoded["attention_mask"])
                features = features.to(dtype=dtype)
                if mom2 is None:
                    width = int(features.shape[-1])
                    mom2 = features.new_zeros((width, width))
                mom2.addmm_(features.T, features)
                activation_count += int(features.shape[0])

            document_cursor = start + len(group_indices)
            if checkpoint_every > 0 and (group_index + 1) % checkpoint_every == 0:
                _atomic_torch_save(
                    partial_path,
                    {
                        "module_name": module_name,
                        "sample_size": len(selected),
                        "seed": seed,
                        "selection_sha256": selection_hash,
                        "dataset_name": dataset_name,
                        "dataset_config": dataset_config,
                        "dataset_split": dataset_split,
                        "dataset_revision": dataset_revision,
                        "document_cursor": document_cursor,
                        "activation_count": activation_count,
                        "mom2": mom2.detach().cpu(),
                        "metadata": metadata,
                    },
                    torch,
                )
    finally:
        hook_handle.remove()

    if mom2 is None or activation_count == 0:
        raise RuntimeError("no activations were collected while computing covariance")
    completed_metadata = {
        **metadata,
        "sampling": {
            **dict(metadata.get("sampling", {})),
            "sampled_documents": len(selected),
            "seed": seed,
            "selection_sha256": selection_hash,
        },
        "activation_count": activation_count,
    }
    _save_npz_covariance(
        output_path,
        mom2=mom2.detach().cpu(),
        count=activation_count,
        sample_size=len(selected),
        metadata=completed_metadata,
    )
    if partial_path.exists():
        partial_path.unlink()
    return output_path


def fingerprint(path: Path, *, compute_hash: bool = False) -> FileFingerprint:
    if not path.is_file():
        return FileFingerprint(str(path), None, None, None)
    stat = path.stat()
    return FileFingerprint(
        path=str(path),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=_sha256(path) if compute_hash else None,
    )


def _load_npz_covariance(path: Path, metadata: dict[str, Any]) -> LoadedCovariance:
    numpy = _import_numpy()
    torch = _import_torch()
    with numpy.load(path, allow_pickle=False) as data:
        if "mom2.mom2" in data:
            raw_mom2 = data["mom2.mom2"]
        elif "mom2" in data:
            raw_mom2 = data["mom2"]
        elif "C" in data:
            raw_mom2 = data["C"]
        else:
            raise ValueError(f"covariance archive has no moment matrix: {path}")
        if "mom2.count" in data:
            raw_count = data["mom2.count"]
        elif "count" in data:
            raw_count = data["count"]
        else:
            raw_count = None
        embedded = _embedded_metadata(data)
        explicit_c = data.get("C")
    count = int(raw_count) if raw_count is not None else None
    if explicit_c is not None:
        C = torch.from_numpy(explicit_c)
    else:
        C = torch.from_numpy(raw_mom2)
        if count:
            C.div_(count)
    return LoadedCovariance(
        C=C,
        count=count,
        source_path=path,
        metadata={**embedded, **metadata},
    )


def _save_npz_covariance(
    path: Path,
    *,
    mom2: Any,
    count: int,
    sample_size: int,
    metadata: dict[str, Any],
) -> None:
    numpy = _import_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("wb") as handle:
            numpy.savez(
                handle,
                **{
                    "mom2.constructor": numpy.array("SecondMoment"),
                    "mom2.count": numpy.array(count),
                    "mom2.mom2": mom2.numpy(),
                    "sample_size": numpy.array(sample_size),
                    "metadata_json": numpy.array(json.dumps(metadata, sort_keys=True)),
                },
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _embedded_metadata(data: Any) -> dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    raw = data["metadata_json"]
    try:
        return json.loads(str(raw.item() if hasattr(raw, "item") else raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _tokenize_document(tokenizer: Any, text: str, *, max_length: int, torch: Any) -> dict[str, Any]:
    token_ids = tokenizer.encode(text, truncation=True, max_length=max_length)
    return {
        "input_ids": torch.tensor(token_ids, dtype=torch.long),
        "position_ids": torch.arange(len(token_ids), dtype=torch.long),
        "attention_mask": torch.ones(len(token_ids), dtype=torch.long),
    }


def _length_collated_batches(
    items: Sequence[dict[str, Any]],
    *,
    token_budget: int,
    torch: Any,
) -> Iterable[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: -len(item["input_ids"]))
    batch: list[dict[str, Any]] = []
    width = 0
    for item in ordered:
        item_width = len(item["input_ids"])
        if item_width == 0:
            continue
        if batch and width * (len(batch) + 1) > token_budget:
            yield _padded_batch(batch, torch)
            batch = []
            width = 0
        if not batch:
            width = item_width
        batch.append(item)
    if batch:
        yield _padded_batch(batch, torch)


def _padded_batch(items: Sequence[dict[str, Any]], torch: Any) -> dict[str, Any]:
    pad_sequence = torch.nn.utils.rnn.pad_sequence
    return {key: pad_sequence([item[key] for item in items], batch_first=True) for key in items[0]}


def _max_sequence_length(model: Any) -> int:
    config = model.config
    for owner in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
    ):
        value = getattr(owner, "max_position_embeddings", None) if owner is not None else None
        if value is not None:
            return max(1, int(value) - 500)
    return 2048


def _dataset_text(row: Any) -> str:
    if isinstance(row, Mapping):
        return str(row.get("text", ""))
    return str(row)


def _flatten_masked(features: Any, attention_mask: Any) -> Any:
    tensor = features[0] if isinstance(features, (tuple, list)) else features
    flat = tensor.reshape(-1, tensor.shape[-1])
    indices = attention_mask.reshape(-1).nonzero()[:, 0]
    return flat[indices]


def _atomic_torch_save(path: Path, payload: dict[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _validate_fingerprint(path: Path, expected: Mapping[str, Any]) -> None:
    actual = fingerprint(path, compute_hash=bool(expected.get("sha256")))
    if expected.get("size_bytes") is not None and actual.size_bytes != expected["size_bytes"]:
        raise ValueError(f"covariance source size changed: {path}")
    if expected.get("mtime_ns") is not None and actual.mtime_ns != expected["mtime_ns"]:
        raise ValueError(f"covariance source mtime changed: {path}")
    if expected.get("sha256") and actual.sha256 != expected["sha256"]:
        raise ValueError(f"covariance source sha256 changed: {path}")


def _profile_key(profile: Any) -> str:
    key = _field(profile, "key") or _field(_field(profile, "model", {}), "key")
    if not key:
        raise KeyError("profile must include key or model.key")
    return str(key)


def _covariance_config(profile: Any) -> dict[str, Any]:
    covariance = _field(profile, "covariance")
    if covariance:
        return dict(covariance)
    identity = _field(profile, "covariance_identity")
    if identity:
        return {
            "canonical_identity": identity,
            "dataset": _field(profile, "covariance_dataset", "wikipedia"),
            "sample_size": _field(profile, "covariance_sample_size"),
            "dtype": _field(profile, "covariance_dtype", "float32"),
            "legacy_filename_patterns": list(_field(profile, "covariance_legacy_patterns", ())),
        }
    return {}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _module_path_from_legacy_name(filename: str) -> str | None:
    marker = "_float32_mom2_"
    return filename.split(marker, 1)[0] if marker in filename else None


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        if resolved not in seen:
            result.append(path)
            seen.add(resolved)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_numpy() -> Any:
    import numpy

    return numpy


def _import_torch() -> Any:
    import torch

    return torch


class _StopForward(Exception):
    """Stop a statistics forward pass immediately after the traced module."""


def _import_load_dataset() -> Any:
    from datasets import load_dataset

    return load_dataset
