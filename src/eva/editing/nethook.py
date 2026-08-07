"""Small torch tracing utilities adapted from MEMIT/ROME nethook."""

from __future__ import annotations

import contextlib
import inspect
from collections import OrderedDict
from typing import Any


class StopForward(Exception):
    """Raised internally to stop a traced forward pass."""


class Trace(contextlib.AbstractContextManager):
    """Retain or edit one module's forward input/output."""

    def __init__(
        self,
        module: Any,
        layer: str | None = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Any | None = None,
        stop: bool = False,
    ) -> None:
        if layer is not None:
            module = get_module(module, layer)
        self.layer = layer
        self.stop = stop

        def retain_hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            if retain_input:
                self.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )
            if edit_output is not None:
                output = invoke_with_optional_args(edit_output, output=output, layer=self.layer)
            if retain_output:
                self.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                if retain_grad:
                    output = recursive_copy(self.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)

    def __enter__(self) -> Trace:
        return self

    def __exit__(self, exc_type: Any, value: Any, traceback: Any) -> bool | None:
        self.close()
        if self.stop and exc_type is not None and issubclass(exc_type, StopForward):
            return True
        return None

    def close(self) -> None:
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """Retain or edit several modules during a forward pass."""

    def __init__(
        self,
        module: Any,
        layers: list[str] | tuple[str, ...],
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Any | None = None,
        stop: bool = False,
    ) -> None:
        super().__init__()
        self.stop = stop
        for is_last, layer in _flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self) -> TraceDict:
        return self

    def __exit__(self, exc_type: Any, value: Any, traceback: Any) -> bool | None:
        self.close()
        if self.stop and exc_type is not None and issubclass(exc_type, StopForward):
            return True
        return None

    def close(self) -> None:
        for trace in reversed(self.values()):
            trace.close()


def recursive_copy(
    x: Any, clone: bool = False, detach: bool = False, retain_grad: bool = False
) -> Any:
    """Copy tensor containers enough for tracing without importing torch at module load."""

    torch = _torch()
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v, clone, detach, retain_grad) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        return type(x)(recursive_copy(v, clone, detach, retain_grad) for v in x)
    raise TypeError(f"cannot recursively copy {type(x)!r}")


def set_requires_grad(requires_grad: bool, *models: Any) -> None:
    torch = _torch()
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            raise TypeError(f"unknown grad target type: {type(model)!r}")


def get_module(model: Any, name: str) -> Any:
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    raise LookupError(name)


def get_parameter(model: Any, name: str) -> Any:
    for param_name, param in model.named_parameters():
        if param_name == name:
            return param
    raise LookupError(name)


def invoke_with_optional_args(fn: Any, *args: Any, **kwargs: Any) -> Any:
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (0 if not argspec.defaults else len(argspec.defaults))

    for i, name in enumerate(argspec.args):
        if name in kwargs:
            pass_args.append(kwargs[name])
            used_kw.add(name)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            default_idx = i - defaulted_pos
            pass_args.append(None if i < defaulted_pos else argspec.defaults[default_idx])

    for key, value in kwargs.items():
        if not unmatched_pos:
            break
        if key in used_kw or key in argspec.kwonlyargs:
            continue
        pass_args[unmatched_pos.pop(0)] = value
        used_kw.add(key)

    if unmatched_pos and unmatched_pos[0] < defaulted_pos:
        missing = ", ".join(argspec.args[i] for i in unmatched_pos if i < defaulted_pos)
        raise TypeError(f"{fn.__name__}() cannot be passed {missing}.")

    pass_kw = {
        key: value
        for key, value in kwargs.items()
        if key not in used_kw and (key in argspec.kwonlyargs or argspec.varkw is not None)
    }
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)


def _flag_last_unseen(items: list[str] | tuple[str, ...]):
    iterator = iter(items)
    try:
        previous = next(iterator)
    except StopIteration:
        return
    seen = {previous}
    for item in iterator:
        if item not in seen:
            yield False, previous
            seen.add(item)
            previous = item
    yield True, previous


def _torch() -> Any:
    import torch

    return torch
