"""Shared DELMAN/MEMIT text-token editing core."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol

from eva._compat import strict_zip

from . import nethook, repr_tools
from .hparams import DELMANHyperParams


class CovarianceProvider(Protocol):
    """Supplies covariance by logical layer identity chosen outside this module."""

    def get_covariance(self, *, layer: int, weight_name: str, hparams: DELMANHyperParams) -> Any:
        """Return a square tensor covariance matrix for ``layer``."""


def chunk_records(
    records: Sequence[Mapping[str, Any]], records_per_update: int = 1
) -> list[list[Mapping[str, Any]]]:
    """Split records into sequential solver calls."""

    if records_per_update <= 0:
        raise ValueError("records_per_update must be positive")
    return [
        list(records[start : start + records_per_update])
        for start in range(0, len(records), records_per_update)
    ]


def ensure_target_leading_space(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return a request copy with DELMAN's target-leading-space convention."""

    copied = dict(request)
    target = str(copied["target_new"])
    copied["target_new"] = target if target.startswith(" ") else f" {target}"
    return copied


def normalize_requests(requests: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [ensure_target_leading_space(request) for request in requests]


def match_update_shape(update: Any, weight_shape: Sequence[int]) -> Any:
    """Match MEMIT's transposed-linear weight convention."""

    shape = tuple(int(dim) for dim in weight_shape)
    if tuple(update.shape) == shape:
        return update
    if len(shape) == 2 and tuple(update.T.shape) == shape:
        return update.T
    raise ValueError(
        f"update shape {tuple(update.shape)} is incompatible with weight shape {shape}"
    )


def repeat_targets_for_keys(targets: Any, key_count: int) -> Any:
    """Repeat target columns to match context-expanded key columns."""

    if targets.size(1) == key_count:
        return targets
    if key_count % targets.size(1) != 0:
        raise ValueError(
            f"cannot repeat {targets.size(1)} target columns to match {key_count} keys"
        )
    return targets.repeat_interleave(key_count // targets.size(1), dim=1)


def is_final_value_iteration(step: int, configured_steps: int) -> bool:
    """Match the preserved loop, whose final pass performs no optimizer update."""

    return step == configured_steps - 1


def emit_edit_batch(
    *,
    stage: str,
    batch_index: int,
    batch_count: int,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Print one concise progress header before editing a record batch."""

    case_ids = [str(record.get("case_id", "?")) for record in records]
    print(
        f"[{stage}] edit {batch_index + 1}/{batch_count} case_id={','.join(case_ids)}",
        file=sys.stderr,
        flush=True,
    )


def emit_stage_start(*, stage: str, record_count: int, batch_count: int) -> None:
    """Print the number of records and sequential batches in one edit stage."""

    print(
        f"[{stage}] editing {record_count} records in {batch_count} incremental batches",
        file=sys.stderr,
        flush=True,
    )


def emit_value_start() -> None:
    """Print the legacy marker emitted when the initial v* is captured."""

    print("Recording initial value of v*", file=sys.stderr, flush=True)


def emit_value_loss(
    *,
    loss: Any,
    nll_loss: Any,
    kl_loss: Any,
    weight_decay: Any,
    average_probability: Any,
    target: str,
) -> None:
    """Print the legacy DELMAN per-iteration loss decomposition."""

    print(
        f"loss {round(_scalar(loss), 3)} = "
        f"{round(_scalar(nll_loss), 3)} + "
        f"{round(_scalar(kl_loss), 3)} + "
        f"{round(_scalar(weight_decay), 3)}  "
        f"avg prob of [{target}] {_scalar(average_probability)}",
        file=sys.stderr,
        flush=True,
    )


def emit_value_norms(*, initial: Any, delta: Any, target: Any) -> None:
    """Print the legacy DELMAN v* norm summary."""

    print(
        f"Init norm {_scalar(initial.norm())} | "
        f"Delta norm {_scalar(delta.norm())} | "
        f"Target norm {_scalar(target.norm())}",
        file=sys.stderr,
        flush=True,
    )


def _scalar(value: Any) -> float:
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def apply_text_edits(
    model: Any,
    tok: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: DELMANHyperParams,
    covariance_provider: CovarianceProvider,
    *,
    context_templates: list[list[str]] | None = None,
    records_per_update: int = 1,
    return_original_weights: bool = False,
) -> dict[str, Any]:
    """Apply sequential DELMAN edits to an already-mutating language model."""

    originals: dict[str, Any] = {}
    chunks = chunk_records(requests, records_per_update)
    emit_stage_start(
        stage="text",
        record_count=len(requests),
        batch_count=len(chunks),
    )
    for batch_index, chunk in enumerate(chunks):
        emit_edit_batch(
            stage="text",
            batch_index=batch_index,
            batch_count=len(chunks),
            records=chunk,
        )
        deltas = execute_delman(
            model=model,
            tok=tok,
            requests=chunk,
            hparams=hparams,
            covariance_provider=covariance_provider,
            context_templates=context_templates,
        )
        _apply_deltas(model, deltas, originals if return_original_weights else None)
    return originals


def execute_delman(
    model: Any,
    tok: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: DELMANHyperParams,
    covariance_provider: CovarianceProvider,
    *,
    context_templates: list[list[str]] | None = None,
) -> dict[str, tuple[Any, Any]]:
    """Compute DELMAN deltas while restoring all temporary weight edits."""

    torch = _torch()
    requests = normalize_requests(deepcopy(list(requests)))
    context_templates = context_templates or default_context_templates()
    weights = {
        _weight_name(hparams, layer): nethook.get_parameter(model, _weight_name(hparams, layer))
        for layer in hparams.layers
    }
    weights_copy = {name: weight.detach().clone() for name, weight in weights.items()}
    deltas: dict[str, tuple[Any, Any]] = {}

    try:
        v_star_layer = hparams.layers[-1]
        v_stars = torch.stack(
            [
                compute_v_star(
                    model=model,
                    tok=tok,
                    request=request,
                    hparams=hparams,
                    layer=v_star_layer,
                    context_templates=context_templates,
                )
                for request in requests
            ],
            dim=1,
        )

        for index, layer in enumerate(hparams.layers):
            weight_name = _weight_name(hparams, layer)
            layer_ks = compute_k_star(model, tok, requests, hparams, layer, context_templates).T
            cur_vs = get_module_input_output_at_words(
                model=model,
                tok=tok,
                layer=v_star_layer,
                context_templates=[request["prompt_adv"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = repeat_targets_for_keys(v_stars - cur_vs, layer_ks.size(1))

            layer_ks = layer_ks.double().mean(dim=1, keepdim=True)
            targets = targets.double().mean(dim=1, keepdim=True)
            cov = covariance_provider.get_covariance(
                layer=layer, weight_name=weight_name, hparams=hparams
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


def compute_k_star(
    model: Any,
    tok: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: DELMANHyperParams,
    layer: int,
    context_templates: list[list[str]],
) -> Any:
    """Compute averaged left vectors over context templates."""

    torch = _torch()
    all_templates = [
        context.format(request["prompt_adv"])
        for request in requests
        for context_type in context_templates
        for context in context_type
    ]
    words = [
        request["subject"]
        for request in requests
        for context_type in context_templates
        for _ in context_type
    ]
    layer_ks = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=all_templates,
        words=words,
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[0]
    context_lens = [len(context_type) for context_type in context_templates]
    context_len = sum(context_lens)
    offsets = _cumulative_offsets(context_lens)
    averaged = []
    for row in range(0, layer_ks.size(0), context_len):
        per_type = [layer_ks[row + start : row + end].mean(0) for start, end in offsets]
        averaged.append(torch.stack(per_type, 0).mean(0))
    return torch.stack(averaged, dim=0)


def compute_v_star(
    model: Any,
    tok: Any,
    request: Mapping[str, Any],
    hparams: DELMANHyperParams,
    layer: int,
    context_templates: list[list[str]],
) -> Any:
    """Optimize the right vector for one request."""

    torch = _torch()
    lm_w = nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    device = next(model.parameters()).device
    vocab_size = _config_value(model.config, "vocab_size")
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(vocab_size)
    if lm_b is None:
        lm_b = next(model.parameters()).new_zeros(vocab_size)

    if hasattr(tok, "add_bos_token"):
        tok.add_bos_token = False
    target_ids = tok(request["target_new"], return_tensors="pt", add_special_tokens=False).to(
        device
    )["input_ids"][0]

    rewriting_prompts = [
        context.format(request["prompt_adv"]) + tok.decode(target_ids[:-1])
        for context_type in context_templates
        for context in context_type
    ]
    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts
    subjects = [request["subject"] for _ in rewriting_prompts] + [request["subject"]]
    input_tok = tok(
        [prompt.format(subject) for prompt, subject in strict_zip(all_prompts, subjects)],
        return_tensors="pt",
        padding=True,
    ).to(device)
    rewriting_targets = torch.full(
        (len(rewriting_prompts), input_tok["input_ids"].shape[1]),
        -100,
        dtype=target_ids.dtype,
        device=device,
    )
    for i in range(len(rewriting_prompts)):
        example_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, example_len - len(target_ids) : example_len] = target_ids

    lookup_idxs = [
        find_fact_lookup_idx(prompt, subject, tok, hparams.fact_token)
        for prompt, subject in strict_zip(all_prompts, subjects)
    ]
    hidden_size = _config_value(model.config, "hidden_size")
    delta = torch.zeros((hidden_size,), requires_grad=True, device=device)
    target_init = None
    kl_distr_init = None
    loss_layer = max(hparams.v_loss_layer, layer)

    def edit_output_fn(output: Any, layer: str) -> Any:
        nonlocal target_init
        if layer != hparams.layer_module_tmp.format(layer_index):
            return output
        hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        if target_init is None:
            emit_value_start()
            target_init = hidden_states[0, lookup_idxs[0], :].detach().clone()
        for row, idx in enumerate(lookup_idxs):
            hidden_states[row, idx, :] += delta
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        if isinstance(output, list):
            return [hidden_states] + list(output[1:])
        return hidden_states

    layer_index = layer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for step in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as trace:
            logits = model(**input_tok).logits
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        layer_out = trace[hparams.layer_module_tmp.format(loss_layer)].output
        full_repr = (
            layer_out[0][: len(rewriting_prompts)]
            if isinstance(layer_out, (tuple, list))
            else layer_out[: len(rewriting_prompts)]
        )
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
        loss_values = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        nll_each = -(loss_values * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        if target_init is None:
            raise RuntimeError("failed to capture target initial representation")
        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        loss = nll_loss + kl_loss + weight_decay
        average_probability = torch.exp(-nll_each).mean()
        emit_value_loss(
            loss=loss,
            nll_loss=nll_loss,
            kl_loss=kl_loss,
            weight_decay=weight_decay,
            average_probability=average_probability,
            target=str(request["target_new"]),
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
    return target


def get_module_input_output_at_words(
    model: Any,
    tok: Any,
    layer: int,
    context_templates: list[str],
    words: list[str],
    module_template: str,
    fact_token_strategy: str,
    minus: int | None = None,
) -> tuple[Any, Any]:
    if fact_token_strategy.startswith("subject_"):
        subtoken = fact_token_strategy[len("subject_") :]
        layer_input, layer_output = repr_tools.get_reprs_at_word_tokens(
            model=model,
            tok=tok,
            layer=layer,
            context_templates=context_templates,
            words=words,
            module_template=module_template,
            subtoken=subtoken,
            track="both",
            minus=minus,
        )
    elif fact_token_strategy == "last":
        layer_input, layer_output = repr_tools.get_reprs_at_idxs(
            model=model,
            tok=tok,
            contexts=context_templates,
            idxs=[[-1] for _ in context_templates],
            layer=layer,
            module_template=module_template,
            track="both",
            minus=minus,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")
    return layer_input.detach(), layer_output.detach()


def find_fact_lookup_idx(prompt: str, subject: str, tok: Any, fact_token_strategy: str) -> int:
    if fact_token_strategy == "last":
        return -1
    if fact_token_strategy.startswith("subject_"):
        return repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    raise ValueError(f"fact_token={fact_token_strategy} not recognized")


def default_context_templates() -> list[list[str]]:
    """Return the exact fixed contexts used by the recovered EVA runs."""

    return [
        ["{}"],
        [
            "The first thing I noticed is a very strong,. {}",
            "Therefore, it's not clear whether or not there. {}",
            "Because we've got a good thing going with the. {}",
            "I'm not going to lie: I'm not. {}",
        ],
    ]


def _apply_deltas(
    model: Any, deltas: Mapping[str, tuple[Any, Any]], originals: dict[str, Any] | None
) -> None:
    torch = _torch()
    with torch.no_grad():
        for weight_name, (key_mat, val_mat) in deltas.items():
            weight = nethook.get_parameter(model, weight_name)
            update = match_update_shape(
                key_mat.to(weight.device) @ val_mat.to(weight.device).T, weight.shape
            )
            if originals is not None and weight_name not in originals:
                originals[weight_name] = weight.detach().clone()
            weight[...] += update.to(weight.dtype)


def _config_value(config: Any, name: str) -> int:
    for owner in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
    ):
        if owner is not None and hasattr(owner, name):
            return int(getattr(owner, name))
    raise AttributeError(f"model config does not expose {name}")


def _weight_name(hparams: DELMANHyperParams, layer: int) -> str:
    return f"{hparams.rewrite_module_tmp.format(layer)}.weight"


def _cumulative_offsets(lengths: Sequence[int]) -> list[tuple[int, int]]:
    offsets = []
    start = 0
    for length in lengths:
        end = start + length
        offsets.append((start, end))
        start = end
    return offsets


def _torch() -> Any:
    import torch

    return torch
