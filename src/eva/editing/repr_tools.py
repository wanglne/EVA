"""Token-index and representation extraction helpers for text-token editing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import nethook


def get_words_idxs_in_templates(
    tok: Any, context_templates: list[str], words: list[str], subtoken: str
) -> list[list[int]]:
    """Return token indices for filled words in single-slot templates."""

    if not all(template.count("{}") == 1 for template in context_templates):
        raise ValueError("context templates must contain exactly one '{}' slot")

    fill_idxs = [template.index("{}") for template in context_templates]
    prefixes = [template[: fill_idxs[i]] for i, template in enumerate(context_templates)]
    suffixes = [template[fill_idxs[i] + 2 :] for i, template in enumerate(context_templates)]
    words = deepcopy(words)

    for i, prefix in enumerate(prefixes):
        if prefix:
            if prefix[-1] != " ":
                raise ValueError(f"context prefix must end with a space: {context_templates[i]!r}")
            prefixes[i] = prefix[:-1]
            words[i] = f" {words[i].strip()}"

    if len(prefixes) != len(words) or len(words) != len(suffixes):
        raise ValueError("context_templates and words must have matching lengths")

    if "llama" in str(type(tok)).lower():
        words = [word.strip() for word in words]
        suffixes = [suffix.strip() for suffix in suffixes]

    prefix_lens = [len(ids) for ids in tok(prefixes).input_ids]
    word_lens = [len(ids) for ids in tok(words).input_ids]
    suffix_lens = [len(ids) for ids in tok(suffixes).input_ids]

    if subtoken in {"last", "first_after_last"}:
        return [
            [
                prefix_lens[i]
                + word_lens[i]
                - (1 if subtoken == "last" or suffix_lens[i] == 0 else 0)
            ]
            for i in range(len(prefixes))
        ]
    if subtoken == "first":
        return [[prefix_lens[i]] for i in range(len(prefixes))]
    raise ValueError(f"unknown subtoken type: {subtoken}")


def get_reprs_at_word_tokens(
    model: Any,
    tok: Any,
    context_templates: list[str],
    words: list[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
    minus: int | None = None,
) -> Any:
    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model=model,
        tok=tok,
        contexts=[context_templates[i].format(words[i]) for i in range(len(words))],
        idxs=idxs,
        layer=layer,
        module_template=module_template,
        track=track,
        minus=minus,
    )


def get_reprs_at_idxs(
    model: Any,
    tok: Any,
    contexts: list[str],
    idxs: list[list[int]],
    layer: int,
    module_template: str,
    track: str = "in",
    minus: int | None = None,
    batch_size: int = 128,
) -> Any:
    """Run the model and return averaged representations at requested token indices."""

    torch = _torch()
    if track not in {"in", "out", "both"}:
        raise ValueError("track must be one of 'in', 'out', or 'both'")

    want_in = track in {"in", "both"}
    want_out = track in {"out", "both"}
    module_name = module_template.format(layer)
    to_return: dict[str, list[Any]] = {"in": [], "out": []}
    device = next(model.parameters()).device

    def process(cur_repr: Any, batch_idxs: list[list[int]], key: str) -> None:
        tensor = cur_repr[0] if isinstance(cur_repr, tuple) else cur_repr
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(tensor[i][idx_list].mean(0))

    for start in range(0, len(contexts), batch_size):
        batch_contexts = contexts[start : start + batch_size]
        batch_idxs = idxs[start : start + batch_size]
        if minus is not None:
            batch_idxs = [[idx[0] - minus] for idx in batch_idxs]
        context_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(device)
        with (
            torch.no_grad(),
            nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=want_in,
                retain_output=want_out,
            ) as trace,
        ):
            model(**context_tok)
        if want_in:
            process(trace.input, batch_idxs, "in")
        if want_out:
            process(trace.output, batch_idxs, "out")

    stacked = {key: torch.stack(value, 0) for key, value in to_return.items() if value}
    if len(stacked) == 1:
        return stacked["in"] if want_in else stacked["out"]
    return stacked["in"], stacked["out"]


def _torch() -> Any:
    import torch

    return torch
