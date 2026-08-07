from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eva.editing.delman import (
    emit_value_loss,
    emit_value_norms,
    emit_value_start,
    is_final_value_iteration,
)

from .protocol import ImageBatch, ImageTokenBackend


@dataclass(frozen=True)
class OptimizationTrace:
    losses: tuple[float, ...]
    steps: int


@dataclass(frozen=True)
class ValueOptimizationResult:
    target: Any
    init: Any
    delta: Any
    trace: OptimizationTrace


def nested_rewrite(record: Mapping[str, Any]) -> Mapping[str, Any]:
    rewrite = record.get("requested_rewrite")
    if isinstance(rewrite, Mapping):
        return rewrite
    return record


def normalize_image_request(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one dataset record and apply DELMAN's leading-space target convention."""

    copied = deepcopy(dict(record))
    rewrite = copied.get("requested_rewrite")
    target_owner = rewrite if isinstance(rewrite, dict) else copied
    target = str(target_owner["target_new"])
    target_owner["target_new"] = target if target.startswith(" ") else f" {target}"
    return copied


def normalize_image_requests(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_image_request(record) for record in records]


def selected_attention_ids(request: Mapping[str, Any], attention_key: str) -> tuple[int, ...]:
    """Resolve JSON-selected image-token ids, falling back to legacy survive_ids.

    New datasets store model/layer-specific indices under
    ``attn_token_idx[attention_key]``. Legacy DELMAN code consumed
    ``requested_rewrite.survive_ids``. Both forms are accepted so callers can pass
    either the outer dataset record or the normalized rewrite mapping.
    """

    rewrite = nested_rewrite(request)
    for source in (request, rewrite):
        selected = source.get("attn_token_idx") if isinstance(source, Mapping) else None
        if isinstance(selected, Mapping) and attention_key in selected:
            value = selected[attention_key]
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return tuple(int(v) for v in value)
            return (int(value),)

    survive_ids = rewrite.get("survive_ids")
    if isinstance(survive_ids, Sequence) and not isinstance(survive_ids, (str, bytes)):
        return tuple(int(v) for v in survive_ids)
    if survive_ids is not None:
        return (int(survive_ids),)
    raise KeyError(f"missing attention index for key {attention_key!r}")


def image_path(data_root: Path, request: Mapping[str, Any]) -> Path:
    rewrite = nested_rewrite(request)
    relative = rewrite["image_path"]
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("image_path must be relative to data_root")
    return data_root / path


def first_parameter(model: Any) -> Any:
    return next(model.parameters())


def move_batch_to_device(batch: Mapping[str, Any], device: Any) -> Mapping[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def tokenize_target(tok: Any, target_new: str, device: Any) -> Any:
    tokens = tok(target_new, return_tensors="pt", add_special_tokens=False)
    tokens = tokens.to(device) if hasattr(tokens, "to") else move_batch_to_device(tokens, device)
    return tokens["input_ids"][0]


def visual_lookup_index(
    *,
    prompt: str,
    subject: str,
    tok: Any,
    attention_ids: Sequence[int],
    lookup_delimiter: str,
) -> int:
    """Preserve DELMAN's visual index conversion: prefix token length + id - 1."""

    prefix = prompt.format(subject).split(lookup_delimiter)[0]
    input_ids = tok(prefix, return_tensors="pt", padding=True)["input_ids"]
    return int(input_ids.shape[1]) + int(attention_ids[0]) - 1


def build_rewrite_targets(torch: Any, batch: ImageBatch, target_ids: Any, device: Any) -> Any:
    input_ids = batch.inputs["input_ids"]
    targets = torch.tensor(-100, device=device).repeat(batch.rewrite_count, *input_ids.shape[1:])
    for index in range(batch.rewrite_count):
        example_len = batch.inputs["attention_mask"][index].sum()
        targets[index, example_len - len(target_ids) : example_len] = target_ids
    return targets


def layer_output_hidden_states(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


def replace_layer_output_hidden_states(output: Any, hidden_states: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden_states,) + output[1:]
    if isinstance(output, list):
        return [hidden_states, *output[1:]]
    return hidden_states


def get_vocab_size(model: Any) -> int:
    config = model.config
    for owner in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
    ):
        vocab_size = getattr(owner, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)
    raise AttributeError("model config does not expose vocab_size")


def get_hidden_size(model: Any) -> int:
    config = model.config
    for owner in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
    ):
        hidden_size = getattr(owner, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
    raise AttributeError("model config does not expose hidden_size")


def module_path(template: str, layer: int) -> str:
    return template.format(layer)


def compute_value_target(
    *,
    backend: ImageTokenBackend,
    handle: Any,
    request: Mapping[str, Any],
    hparams: Any,
    layer: int,
    context_templates: Sequence[Sequence[str]],
    trace_dict_factory: Callable[..., Any],
    get_parameter: Callable[[Any, str], Any],
    get_module: Callable[[Any, str], Any],
    set_requires_grad: Callable[[bool, Any], None],
) -> ValueOptimizationResult:
    """Shared DELMAN v* optimization for image-token families.

    The target loss mirrors legacy compute_v_star_v1:
    NLL over target tokens + KL preservation at the selected image token +
    weight decay on the learned delta, with L2 projection around the initial
    image-token value.
    """

    import torch

    model = handle.model
    tok = handle.tokenizer
    device = backend.config.device
    if hasattr(tok, "add_bos_token"):
        tok.add_bos_token = False
    target_ids = tokenize_target(tok, nested_rewrite(request)["target_new"], device)
    batch = backend.build_value_batch(
        handle=handle,
        request=request,
        target_ids=target_ids,
        context_templates=context_templates,
    )

    lm_w = get_parameter(model, f"{hparams.lm_head_module}.weight").T
    ln_f = get_module(model, hparams.ln_f_module)
    try:
        lm_b = get_parameter(model, f"{hparams.lm_head_module}.bias")
        if lm_b is None:
            lm_b = first_parameter(model).new_zeros(get_vocab_size(model))
    except LookupError:
        lm_b = first_parameter(model).new_zeros(get_vocab_size(model))

    loss_layer = max(hparams.v_loss_layer, layer)
    delta = torch.zeros((get_hidden_size(model),), requires_grad=True, device=device)
    target_init = None
    kl_distr_init = None
    losses: list[float] = []

    def edit_output_fn(cur_out: Any, cur_layer: str) -> Any:
        nonlocal target_init
        if cur_layer != module_path(hparams.layer_module_tmp, layer):
            return cur_out
        hidden_states = backend.hidden_states_from_layer_output(cur_out)
        if target_init is None:
            emit_value_start()
            target_init = hidden_states[0, batch.lookup_indices[0], :].detach().clone()
        for index, token_idx in enumerate(batch.lookup_indices):
            hidden_states[index, token_idx, :] += delta
        return backend.replace_hidden_states_in_layer_output(cur_out, hidden_states)

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    set_requires_grad(False, model)
    rewrite_targets = build_rewrite_targets(torch, batch, target_ids, device)

    for step in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        with trace_dict_factory(
            module=model,
            layers=[
                module_path(hparams.layer_module_tmp, loss_layer),
                module_path(hparams.layer_module_tmp, layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as traces:
            logits = backend.forward(handle=handle, batch=batch).logits
            kl_logits = torch.stack(
                [
                    logits[index - batch.kl_count, token_idx, :]
                    for index, token_idx in enumerate(batch.lookup_indices[-batch.kl_count :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        layer_out = traces[module_path(hparams.layer_module_tmp, loss_layer)].output
        full_repr = layer_output_hidden_states(layer_out)[: batch.rewrite_count]
        logits = ln_f(full_repr) @ lm_w + lm_b
        log_probs = torch.log_softmax(logits, dim=2)
        token_loss = torch.gather(
            log_probs,
            2,
            torch.where(rewrite_targets != -100, rewrite_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewrite_targets != -100).float()
        nll_each = -(token_loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init,
            kl_log_probs,
            log_target=True,
            reduction="batchmean",
        )
        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        loss = nll_loss + kl_loss + weight_decay
        losses.append(float(loss.detach().cpu()))
        average_probability = torch.exp(-nll_each).mean()
        emit_value_loss(
            loss=loss,
            nll_loss=nll_loss,
            kl_loss=kl_loss,
            weight_decay=weight_decay,
            average_probability=average_probability,
            target=str(nested_rewrite(request)["target_new"]),
        )

        if loss < 1e-2 or is_final_value_iteration(step, hparams.v_num_grad_steps):
            break
        loss.backward()
        opt.step()

        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    emit_value_norms(initial=target_init, delta=delta, target=target)
    return ValueOptimizationResult(
        target=target,
        init=target_init,
        delta=delta,
        trace=OptimizationTrace(losses=tuple(losses), steps=len(losses)),
    )


def average_context_keys(
    torch: Any, layer_keys: Any, context_templates: Sequence[Sequence[str]]
) -> Any:
    context_type_lens = [0, *(len(context_type) for context_type in context_templates)]
    context_len = sum(context_type_lens)
    if context_len <= 0:
        raise ValueError("context_templates must contain at least one template")
    if layer_keys.size(0) % context_len:
        raise ValueError(f"received {layer_keys.size(0)} key rows for context width {context_len}")
    context_cumsum: list[int] = []
    running = 0
    for length in context_type_lens:
        running += length
        context_cumsum.append(running)

    averaged = []
    for base in range(0, layer_keys.size(0), context_len):
        grouped = []
        for index in range(len(context_cumsum) - 1):
            start, end = context_cumsum[index], context_cumsum[index + 1]
            grouped.append(layer_keys[base + start : base + end].mean(0))
        averaged.append(torch.stack(grouped, 0).mean(0))
    return torch.stack(averaged, dim=0)


def compute_key_vectors(
    *,
    backend: ImageTokenBackend,
    handle: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    layer: int,
    context_templates: Sequence[Sequence[str]],
    repr_getter: Callable[..., tuple[Any, Any]],
) -> Any:
    """Shared DELMAN key extraction and context-template averaging."""

    import torch

    batch = backend.build_key_batch(
        handle=handle,
        requests=requests,
        context_templates=context_templates,
    )
    layer_keys, _ = repr_getter(
        model=handle.model,
        tok=handle.tokenizer,
        processor=getattr(handle, "processor", None),
        layer=layer,
        context_templates=list(batch.prompts),
        raw_images=batch.inputs.get("raw_images"),
        words=list(batch.subjects),
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
        idxs=[[idx] for idx in batch.lookup_indices],
    )
    return average_context_keys(torch, layer_keys.detach(), context_templates)
