from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from eva.editing import nethook
from eva.editing.delman import (
    chunk_records,
    default_context_templates,
    emit_edit_batch,
    emit_stage_start,
    match_update_shape,
    repeat_targets_for_keys,
)

from .backends import create_backend
from .core import (
    average_context_keys,
    compute_value_target,
    module_path,
    normalize_image_requests,
)
from .protocol import ImageBackendConfig, ImageBatch, ImageTokenBackend


@dataclass(frozen=True)
class ImageEditResult:
    deltas: dict[str, tuple[Any, Any]]
    original_weights: dict[str, Any]
    chunks: int


class ImageStageBackend:
    """StageBackend-compatible image editor for loaded VLM handles."""

    name = "image"

    def __init__(
        self,
        *,
        family: str,
        data_root: str | Path,
        attention_key: str,
        device: Any | None = None,
        processor: Any | None = None,
        adapter: Any | None = None,
    ) -> None:
        self.family = family
        self.data_root = Path(data_root)
        self.attention_key = attention_key
        self.device = device
        self.processor = processor
        self.adapter = adapter

    def apply(
        self,
        *,
        model: Any,
        tokenizer: Any,
        records: list[Mapping[str, Any]],
        hparams: Any,
        covariance_provider: Any,
        context_templates: list[list[str]] | None = None,
    ) -> ImageEditResult:
        handle = _coerce_handle(
            model=model,
            tokenizer=tokenizer,
            processor=self.processor,
            adapter=self.adapter,
            family=self.family,
        )
        config = ImageBackendConfig(
            family=self.family,
            data_root=self.data_root,
            attention_key=self.attention_key,
            device=self.device
            if self.device is not None
            else next(handle.model.parameters()).device,
        )
        return apply_image_edits(
            handle=handle,
            records=records,
            hparams=hparams,
            covariance_provider=covariance_provider,
            backend=create_backend(config),
            context_templates=context_templates,
        )


def apply_image_edits(
    *,
    handle: Any,
    records: Sequence[Mapping[str, Any]],
    hparams: Any,
    covariance_provider: Any,
    backend: ImageTokenBackend | None = None,
    config: ImageBackendConfig | None = None,
    context_templates: list[list[str]] | None = None,
    records_per_update: int = 1,
    return_original_weights: bool = False,
) -> ImageEditResult:
    """Apply image-token DELMAN updates to the full VLM in place.

    Unlike the text route, image editing processes only the current chunk by
    default. Each chunk computes v*, reads current v once before the layer loop,
    solves per logical layer with the supplied covariance provider, restores
    temporary weights, then mutates the loaded VLM weights with the final deltas.
    """

    if backend is None:
        if config is None:
            config = ImageBackendConfig(
                family=str(handle.family),
                data_root=Path("."),
                attention_key="attn_layer_8",
                device=next(handle.model.parameters()).device,
            )
        backend = create_backend(config)

    originals: dict[str, Any] = {}
    last_deltas: dict[str, tuple[Any, Any]] = {}
    chunks = chunk_records(list(records), records_per_update)
    emit_stage_start(
        stage="image",
        record_count=len(records),
        batch_count=len(chunks),
    )
    for batch_index, chunk in enumerate(chunks):
        emit_edit_batch(
            stage="image",
            batch_index=batch_index,
            batch_count=len(chunks),
            records=chunk,
        )
        normalized = normalize_image_requests(chunk)
        last_deltas = execute_image_delman(
            handle=handle,
            requests=normalized,
            hparams=hparams,
            covariance_provider=covariance_provider,
            backend=backend,
            context_templates=context_templates,
        )
        _apply_image_deltas(
            model=handle.model,
            deltas=last_deltas,
            originals=originals if return_original_weights else None,
        )
    return ImageEditResult(deltas=last_deltas, original_weights=originals, chunks=len(chunks))


def execute_image_delman(
    *,
    handle: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    covariance_provider: Any,
    backend: ImageTokenBackend,
    context_templates: list[list[str]] | None = None,
) -> dict[str, tuple[Any, Any]]:
    """Compute image-stage deltas while restoring temporary layer edits."""

    torch = _torch()
    context_templates = context_templates or default_context_templates()
    weights = {
        _weight_name(hparams, layer): nethook.get_parameter(
            handle.model, _weight_name(hparams, layer)
        )
        for layer in hparams.layers
    }
    weights_copy = {name: weight.detach().clone() for name, weight in weights.items()}
    deltas: dict[str, tuple[Any, Any]] = {}

    try:
        v_star_layer = hparams.layers[-1]
        v_stars = torch.stack(
            [
                compute_value_target(
                    backend=backend,
                    handle=handle,
                    request=request,
                    hparams=hparams,
                    layer=v_star_layer,
                    context_templates=context_templates,
                    trace_dict_factory=nethook.TraceDict,
                    get_parameter=nethook.get_parameter,
                    get_module=nethook.get_module,
                    set_requires_grad=nethook.set_requires_grad,
                ).target
                for request in requests
            ],
            dim=1,
        )

        current_batch = backend.build_current_value_batch(
            handle=handle,
            requests=requests,
            v_star_layer=v_star_layer,
        )
        cur_vs = get_module_input_output_at_image_idxs(
            backend=backend,
            handle=handle,
            batch=current_batch,
            layer=v_star_layer,
            module_template=hparams.layer_module_tmp,
            track="out",
        )[1].T

        for index, layer in enumerate(hparams.layers):
            weight_name = _weight_name(hparams, layer)
            layer_ks = compute_image_key_vectors(
                backend=backend,
                handle=handle,
                requests=requests,
                hparams=hparams,
                layer=layer,
                context_templates=context_templates,
            ).T
            targets = repeat_targets_for_keys(v_stars - cur_vs, layer_ks.size(1))

            layer_ks = layer_ks.double().mean(dim=1, keepdim=True)
            targets = targets.double().mean(dim=1, keepdim=True)
            cov = _covariance_tensor(
                covariance_provider.get_covariance(
                    layer=layer,
                    weight_name=weight_name,
                    hparams=hparams,
                )
            ).to(device=layer_ks.device, dtype=layer_ks.dtype)
            adj_k = torch.linalg.solve(
                hparams.mom2_update_weight * cov + layer_ks @ layer_ks.T,
                layer_ks,
            )
            resid = targets / (len(hparams.layers) - index)
            update = match_update_shape(resid @ adj_k.T, weights[weight_name].shape)

            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + update.to(
                    weights[weight_name].dtype
                )
            deltas[weight_name] = (adj_k.detach().cpu(), resid.detach().cpu())
    finally:
        with torch.no_grad():
            for name, weight in weights.items():
                weight[...] = weights_copy[name]

    return deltas


def compute_image_key_vectors(
    *,
    backend: ImageTokenBackend,
    handle: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    layer: int,
    context_templates: Sequence[Sequence[str]],
) -> Any:
    torch = _torch()
    batch = backend.build_key_batch(
        handle=handle,
        requests=requests,
        context_templates=context_templates,
    )
    layer_keys = get_module_input_output_at_image_idxs(
        backend=backend,
        handle=handle,
        batch=batch,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )[0]
    return average_context_keys(torch, layer_keys.detach(), context_templates)


def get_module_input_output_at_image_idxs(
    *,
    backend: ImageTokenBackend,
    handle: Any,
    batch: ImageBatch,
    layer: int,
    module_template: str,
    track: str = "both",
) -> tuple[Any, Any]:
    if track not in {"in", "out", "both"}:
        raise ValueError("track must be one of 'in', 'out', or 'both'")
    torch = _torch()
    want_in = track in {"in", "both"}
    want_out = track in {"out", "both"}

    with (
        torch.no_grad(),
        nethook.Trace(
            module=handle.model,
            layer=module_path(module_template, layer),
            retain_input=want_in,
            retain_output=want_out,
        ) as trace,
    ):
        backend.forward(handle=handle, batch=batch)

    layer_input = (
        _collect_at_indices(trace.input, batch.lookup_indices).detach() if want_in else None
    )
    layer_output = (
        _collect_at_indices(trace.output, batch.lookup_indices).detach() if want_out else None
    )
    return layer_input, layer_output


def _collect_at_indices(representation: Any, lookup_indices: Sequence[int]) -> Any:
    torch = _torch()
    tensor = representation[0] if isinstance(representation, (tuple, list)) else representation
    values = [tensor[row, int(index), :] for row, index in enumerate(lookup_indices)]
    return torch.stack(values, dim=0)


def _apply_image_deltas(
    *,
    model: Any,
    deltas: Mapping[str, tuple[Any, Any]],
    originals: dict[str, Any] | None,
) -> None:
    torch = _torch()
    with torch.no_grad():
        for weight_name, (adj_k, resid) in deltas.items():
            weight = nethook.get_parameter(model, weight_name)
            update = match_update_shape(
                resid.to(weight.device) @ adj_k.to(weight.device).T,
                weight.shape,
            )
            if originals is not None and weight_name not in originals:
                originals[weight_name] = weight.detach().clone()
            weight[...] += update.to(weight.dtype)


def _coerce_handle(
    *,
    model: Any,
    tokenizer: Any,
    processor: Any | None,
    adapter: Any | None,
    family: str,
) -> Any:
    if all(hasattr(model, attr) for attr in ("model", "tokenizer", "processor")):
        return model
    return SimpleNamespace(
        family=family,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        adapter=adapter,
    )


def _covariance_tensor(value: Any) -> Any:
    return getattr(value, "C", value)


def _weight_name(hparams: Any, layer: int) -> str:
    return f"{hparams.rewrite_module_tmp.format(layer)}.weight"


def _torch() -> Any:
    import torch

    return torch
