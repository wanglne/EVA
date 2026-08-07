"""Command line interface for EVA control-plane and runtime execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eva.checkpoints import inspect_checkpoint_metadata
from eva.config import ModelProfile, list_profiles, load_profile
from eva.covariance import (
    CovarianceStore,
    fingerprint,
    package_covariances,
    prepare_covariances,
)
from eva.data import (
    load_image_edit_records,
    load_text_edit_records,
    validate_dataset_schema,
    validate_image_assets,
)
from eva.models.sources import model_source_label, resolve_model_path
from eva.pipeline import RunPlan, build_run_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eva", description="EVA edit control plane")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser(
        "inspect", help="inspect profile, data, and checkpoint metadata"
    )
    _add_model_arg(inspect_parser)
    inspect_parser.add_argument("--model-path", type=Path)
    inspect_parser.add_argument("--image-root", type=Path)
    inspect_parser.add_argument("--format", choices=("json", "text"), default="json")

    run_parser = subparsers.add_parser("run", help="execute or dry-run the EVA edit pipeline")
    _add_model_arg(run_parser)
    run_parser.add_argument("--model-path", type=Path)
    run_parser.add_argument(
        "--model-cache-dir",
        type=Path,
        help="Hugging Face cache directory used when --model-path is omitted",
    )
    run_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="resolve the pinned Hugging Face model only from the local cache",
    )
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--covariance-root", type=Path, required=False)
    run_parser.add_argument(
        "--legacy-root",
        type=Path,
        action="append",
        default=[],
        help="search this read-only legacy statistics root before computing missing C",
    )
    run_parser.add_argument(
        "--no-auto-covariance",
        action="store_true",
        help="require existing/importable covariance files instead of computing missing layers",
    )
    run_parser.add_argument(
        "--max-records",
        type=int,
        help="run only the first N image and text records (for parity/debugging)",
    )
    run_parser.add_argument("--device")
    run_parser.add_argument("--image-root", type=Path)
    run_parser.add_argument(
        "--resume", action="store_true", help="resume from run_manifest.json when present"
    )
    run_parser.add_argument(
        "--force-restart", action="store_true", help="ignore any existing run manifest"
    )
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--format", choices=("json", "text"), default="json")
    run_parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="print the completed manifest; by default it is written only to run_manifest.json",
    )

    covariance_parser = subparsers.add_parser(
        "covariance", help="discover/import/compute covariance assets"
    )
    covariance_subparsers = covariance_parser.add_subparsers(dest="covariance_command")
    for name in ("discover", "import", "compute", "package"):
        command = covariance_subparsers.add_parser(name)
        _add_model_arg(command)
        command.add_argument("--covariance-root", type=Path, required=True)
        command.add_argument("--layer", type=int)
        command.add_argument("--legacy-root", type=Path, action="append", default=[])
        command.add_argument("--legacy-path", type=Path)
        command.add_argument("--write-manifest", action="store_true")
        if name == "compute":
            command.add_argument("--model-path", type=Path)
            command.add_argument("--model-cache-dir", type=Path)
            command.add_argument("--local-files-only", action="store_true")
            command.add_argument("--device")
            command.add_argument("--sample-size", type=int)
            command.add_argument("--batch-size", type=int, default=200)
            command.add_argument("--batch-tokens", type=int)
            command.add_argument("--checkpoint-every", type=int, default=25)
            command.add_argument("--no-resume", action="store_true")
        if name == "package":
            command.add_argument("--output-dir", type=Path, required=True)
    return parser


def _add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", "--model-profile", dest="model", required=True, choices=list_profiles()
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "run":
            return _run(args)
        if args.command == "covariance":
            return _covariance(args)
        parser.print_help()
        return 0
    except (
        FileExistsError,
        FileNotFoundError,
        ValueError,
        KeyError,
        LookupError,
        RuntimeError,
        NotImplementedError,
    ) as exc:
        print(f"eva: {exc}", file=sys.stderr)
        return 2


def _inspect(args: argparse.Namespace) -> int:
    profile = load_profile(args.model)
    image_validation, text_validation = _validate_profile_data(profile)
    payload: dict[str, Any] = {
        "profile": profile.to_dict(),
        "datasets": {
            "image": image_validation.to_dict(),
            "text": text_validation.to_dict(),
        },
        "image_assets": validate_image_assets(
            profile.dataset_path,
            image_root=args.image_root,
        ).to_dict(),
    }
    if args.model_path is not None:
        checkpoint = inspect_checkpoint_metadata(args.model_path)
        payload["checkpoint"] = checkpoint.to_dict()
    _emit(payload, args.format)
    return 0 if image_validation.valid and text_validation.valid else 2


def _run(args: argparse.Namespace) -> int:
    if args.resume and args.force_restart:
        raise ValueError("--resume and --force-restart are mutually exclusive")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be positive")

    profile = load_profile(args.model)
    image_validation, text_validation = _validate_profile_data(profile)
    if not image_validation.valid or not text_validation.valid:
        raise ValueError("dataset validation failed; run `eva inspect --model ...` for details")

    covariance_root = args.covariance_root or Path.cwd() / "cache" / "covariances"

    if args.dry_run:
        model_path = args.model_path or Path(
            f"<huggingface-{profile.key}-{profile.huggingface_revision[:12]}>"
        )
        plan = build_run_plan(
            profile,
            model_path=model_path,
            output_dir=args.output_dir,
            covariance_root=covariance_root,
        )
        manifest = _build_manifest(
            profile=profile,
            plan=plan,
            image_root=args.image_root,
            device=args.device,
            status="dry_run",
            max_records=args.max_records,
            legacy_roots=tuple(args.legacy_root),
            auto_compute_covariance=not args.no_auto_covariance,
        )
        _emit({"dry_run": True, "plan": plan.to_dict(), "manifest": manifest}, args.format)
        return 0

    image_assets = validate_image_assets(profile.dataset_path, image_root=args.image_root)
    if not image_assets.valid:
        preview = ", ".join(image_assets.missing[:3])
        raise FileNotFoundError(
            f"found {image_assets.found}/{image_assets.expected} image assets under "
            f"{image_assets.image_root}; first missing: {preview}"
        )
    model_path = resolve_model_path(
        profile,
        args.model_path,
        cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
    )
    plan = build_run_plan(
        profile,
        model_path=model_path,
        output_dir=args.output_dir,
        covariance_root=covariance_root,
    )
    covariance_preparation = _prepare_run_covariances(
        profile=profile,
        covariance_root=covariance_root,
        legacy_roots=tuple(args.legacy_root),
        model_path=model_path,
        device=args.device,
        auto_compute=not args.no_auto_covariance,
    )
    manifest = _build_manifest(
        profile=profile,
        plan=plan,
        image_root=image_assets.image_root,
        device=args.device,
        status="pending",
        max_records=args.max_records,
        legacy_roots=tuple(args.legacy_root),
        auto_compute_covariance=not args.no_auto_covariance,
    )
    manifest["covariance"]["preparation"] = covariance_preparation.to_dict()
    manifest = _execute_runtime(
        profile=profile,
        plan=plan,
        device=args.device,
        image_root=args.image_root,
        resume=args.resume,
        force_restart=args.force_restart,
        covariance_provider=covariance_preparation.provider,
        manifest=manifest,
        max_records=args.max_records,
    )
    _emit_completed_run_manifest(
        manifest,
        args.format,
        enabled=args.print_manifest,
    )
    return 0


def _covariance(args: argparse.Namespace) -> int:
    profile = load_profile(args.model)
    layers = (args.layer,) if args.layer is not None else _profile_layers(profile)
    store = CovarianceStore(args.covariance_root, legacy_roots=args.legacy_root)

    if args.covariance_command == "discover":
        references = []
        for layer in layers:
            discovered = store.discover_legacy(
                profile=profile,
                layer=layer,
                write_manifest=False,
            )
            references.extend(discovered)
            if args.write_manifest:
                if len(discovered) != 1:
                    raise ValueError(
                        f"expected one covariance candidate for {profile.key} layer {layer}, "
                        f"found {len(discovered)}; use covariance import with --legacy-path"
                    )
                store.write_manifest(discovered[0])
        _emit({"references": [_reference_to_dict(reference) for reference in references]}, "json")
        return 0
    if args.covariance_command == "import":
        if args.legacy_path is None:
            raise ValueError("covariance import requires --legacy-path")
        if args.layer is None:
            raise ValueError("covariance import requires --layer")
        reference = store.find_legacy(
            legacy_path=args.legacy_path,
            profile=profile,
            layer=args.layer,
            write_manifest=True,
        )
        _emit({"reference": _reference_to_dict(reference)}, "json")
        return 0
    if args.covariance_command == "package":
        package = package_covariances(
            store,
            profile,
            layers,
            output_dir=args.output_dir,
        )
        _emit(package.to_dict(), "json")
        return 0
    if args.covariance_command == "compute":
        from eva.models.registry import get_adapter
        from eva.runtime import release_model_handle

        model_path = resolve_model_path(
            profile,
            args.model_path,
            cache_dir=args.model_cache_dir,
            local_files_only=args.local_files_only,
        )
        adapter = get_adapter(profile.family)
        handle = adapter.load(
            model_path,
            device=args.device,
            dtype=profile.dtype,
            trust_remote_code=profile.trust_remote_code,
        )
        try:
            view = handle.stats_view()
            outputs: list[dict[str, Any]] = []
            for layer in layers:
                module_name = view.layout.module_for_layer(layer, kind="down_proj")
                output = store.compute_wikipedia_moments(
                    profile=profile,
                    layer=layer,
                    model=view.model,
                    tokenizer=view.tokenizer,
                    module_name=module_name,
                    sample_size=args.sample_size,
                    batch_size=args.batch_size,
                    batch_tokens=args.batch_tokens,
                    resume=not args.no_resume,
                    checkpoint_every=args.checkpoint_every,
                )
                outputs.append(
                    {
                        "layer": layer,
                        "module_path": module_name,
                        "path": str(output),
                    }
                )
        finally:
            release_model_handle(handle)
        _emit({"covariances": outputs}, "json")
        return 0
    raise ValueError("missing covariance subcommand")


def _execute_runtime(
    *,
    profile: ModelProfile,
    plan: RunPlan,
    device: str | None,
    image_root: Path | None,
    resume: bool,
    force_restart: bool,
    covariance_provider: Any,
    manifest: dict[str, Any],
    max_records: int | None,
) -> dict[str, Any]:
    from eva.models.registry import get_adapter
    from eva.runtime import run_two_stage_pipeline

    adapter = get_adapter(profile.family)
    image_records, resolved_image_root = load_image_edit_records(
        profile.dataset_path,
        image_root=image_root,
    )
    text_records = load_text_edit_records(profile.text_dataset_path)
    if max_records is not None:
        image_records = image_records[:max_records]
        text_records = text_records[:max_records]
    result = run_two_stage_pipeline(
        profile=profile,
        plan=plan,
        adapter=adapter,
        covariance_provider=covariance_provider,
        image_records=image_records,
        image_root=resolved_image_root,
        text_records=text_records,
        device=device,
        manifest=manifest,
        resume=resume,
        force_restart=force_restart,
    )
    return result.manifest


def _validate_profile_data(profile: ModelProfile) -> tuple[Any, Any]:
    return (
        validate_dataset_schema(
            profile.dataset_path,
            kind="image",
            attention_token_key=profile.attention_token_key,
        ),
        validate_dataset_schema(profile.text_dataset_path, kind="text"),
    )


def _prepare_run_covariances(
    *,
    profile: ModelProfile,
    covariance_root: Path,
    legacy_roots: tuple[Path, ...],
    model_path: Path,
    device: str | None,
    auto_compute: bool,
) -> Any:
    store = CovarianceStore(covariance_root, legacy_roots=legacy_roots)

    def compute_missing(layers: tuple[int, ...]) -> None:
        from eva.models.registry import get_adapter
        from eva.runtime import release_model_handle

        adapter = get_adapter(profile.family)
        handle = adapter.load(
            model_path,
            device=device,
            dtype=profile.dtype,
            trust_remote_code=profile.trust_remote_code,
        )
        try:
            view = handle.stats_view()
            for layer in layers:
                module_name = view.layout.module_for_layer(layer, kind="down_proj")
                store.compute_wikipedia_moments(
                    profile=profile,
                    layer=layer,
                    model=view.model,
                    tokenizer=view.tokenizer,
                    module_name=module_name,
                )
        finally:
            release_model_handle(handle)

    return prepare_covariances(
        store,
        profile,
        _profile_layers(profile),
        compute_missing=compute_missing if auto_compute else None,
    )


def _build_manifest(
    *,
    profile: ModelProfile,
    plan: RunPlan,
    image_root: Path | None,
    device: str | None,
    status: str,
    max_records: int | None = None,
    legacy_roots: tuple[Path, ...] = (),
    auto_compute_covariance: bool = True,
) -> dict[str, Any]:
    store = CovarianceStore(plan.covariance_root or Path.cwd() / "cache" / "covariances")
    layers = _profile_layers(profile)
    covariance_layers: dict[str, Any] = {}
    for layer in layers:
        canonical_path = store.canonical_path(profile, layer)
        try:
            source_path = store.resolve_source(profile, layer, validate=False)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            source_path = None
        covariance_layers[str(layer)] = {
            "canonical_path": str(canonical_path),
            "source_path": None if source_path is None else str(source_path),
            "fingerprint": (
                None
                if source_path is None
                else asdict(fingerprint(source_path, compute_hash=False))
            ),
        }

    resolved_image_root = image_root or profile.dataset_path.parent
    immutable_inputs = {
        "profile_key": profile.key,
        "family": profile.family,
        "dtype": profile.dtype,
        "trust_remote_code": profile.trust_remote_code,
        "huggingface_source": model_source_label(profile),
        "model": _checkpoint_identity(plan.model_path),
        "image_root": str(Path(resolved_image_root).resolve()),
        "datasets": {
            "image": {
                "path": str(profile.dataset_path),
                "sha256": _sha256(profile.dataset_path),
            },
            "text": {
                "path": str(profile.text_dataset_path),
                "sha256": _sha256(profile.text_dataset_path),
            },
        },
        "hparams": {
            "image": {
                "path": str(profile.image_hparams_path),
                "sha256": _sha256(profile.image_hparams_path),
            },
            "text": {
                "path": str(profile.text_hparams_path),
                "sha256": _sha256(profile.text_hparams_path),
            },
        },
        "covariance_identity": profile.covariance_identity,
        "covariance_layers": covariance_layers,
        "ordering": profile.ordering,
        "records_per_update": profile.records_per_update,
        "max_records": max_records,
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(
            immutable_inputs,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "status": status,
        "profile_key": profile.key,
        "run_fingerprint": run_fingerprint,
        "immutable_inputs": immutable_inputs,
        "device": device,
        "image_root": str(resolved_image_root),
        "record_selection": {
            "start": 0,
            "max_records": max_records,
        },
        "hparams": {
            "image": str(profile.image_hparams_path),
            "text": str(profile.text_hparams_path),
            "records_per_update": profile.records_per_update,
            "expected_updates": profile.expected_updates,
            "ordering": profile.ordering,
        },
        "datasets": {
            "image": {"path": str(profile.dataset_path), "sha256": _sha256(profile.dataset_path)},
            "text": {
                "path": str(profile.text_dataset_path),
                "sha256": _sha256(profile.text_dataset_path),
            },
        },
        "checkpoints": {
            "base": str(plan.model_path),
            "intermediate": str(plan.stages[1].path),
            "final": str(plan.stages[4].path),
        },
        "covariance": {
            "logical_identity": profile.covariance.get("canonical_identity", profile.key),
            "logical_layers": list(layers),
            "layers": covariance_layers,
            "shared_provider": True,
            "policy": {
                "auto_compute_missing": auto_compute_covariance,
                "legacy_roots": [str(path) for path in legacy_roots],
                "registry_root": str(plan.covariance_root),
            },
        },
        "resolved_module_paths": {},
        "stage_status": {stage.name: "pending" for stage in plan.stages},
        "plan": plan.to_dict(),
    }


def _reference_to_dict(reference: Any) -> dict[str, Any]:
    payload = {
        "source_path": str(reference.source_path),
        "canonical_path": str(reference.canonical_path),
        "layer": reference.layer,
        "profile_key": reference.profile_key,
        "manifest_path": str(reference.manifest_path),
        "metadata": reference.metadata,
    }
    if getattr(reference, "module_path", None) is not None:
        payload["module_path"] = reference.module_path
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_layers(profile: ModelProfile) -> tuple[int, ...]:
    layers: list[int] = []
    for path in (profile.image_hparams_path, profile.text_hparams_path):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("layers"), list):
            raise ValueError(f"hparams file has no layers list: {path}")
        layers.extend(int(layer) for layer in payload["layers"])
    return tuple(dict.fromkeys(layers))


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    checkpoint = Path(path)
    identity: dict[str, Any] = {
        "path": str(checkpoint.resolve()),
        "exists": checkpoint.exists(),
        "files": {},
    }
    if not checkpoint.is_dir():
        return identity

    indexed_weights: set[str] = set()
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = checkpoint / index_name
        if not index_path.is_file():
            continue
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
        if isinstance(weight_map, dict):
            indexed_weights.update(str(name) for name in weight_map.values())

    for candidate in sorted(path for path in checkpoint.rglob("*") if path.is_file()):
        relative = candidate.relative_to(checkpoint).as_posix()
        if ".cache" in candidate.relative_to(checkpoint).parts:
            continue
        stat = candidate.stat()
        is_weight = (
            relative in indexed_weights
            or candidate.suffix == ".safetensors"
            or candidate.name.startswith("pytorch_model")
            and candidate.suffix == ".bin"
        )
        identity["files"][relative] = {
            "kind": "weight" if is_weight else "metadata",
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(candidate),
        }
    for missing in sorted(indexed_weights - set(identity["files"])):
        identity["files"][missing] = {
            "kind": "weight",
            "missing": True,
        }
    return identity


def _emit(payload: Any, format_name: str) -> None:
    if format_name == "text":
        if isinstance(payload, dict) and "plan" in payload:
            plan = payload["plan"]
            stages = plan["stages"] if isinstance(plan, dict) else plan.stages
            names = [stage["name"] if isinstance(stage, dict) else stage.name for stage in stages]
            print(f"{'dry run' if payload.get('dry_run') else 'run'}: {', '.join(names)}")
        else:
            print(json.dumps(payload, indent=2, default=str, sort_keys=True))
        return
    print(json.dumps(payload, indent=2, default=str, sort_keys=True))


def _emit_completed_run_manifest(
    manifest: dict[str, Any],
    format_name: str,
    *,
    enabled: bool,
) -> None:
    if enabled:
        _emit({"dry_run": False, "manifest": manifest}, format_name)


if __name__ == "__main__":
    raise SystemExit(main())
