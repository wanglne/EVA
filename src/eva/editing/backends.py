"""Editing stage backend protocols and registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .hparams import DELMANHyperParams


class StageBackend(Protocol):
    """Backend that edits a loaded model in place for one stage."""

    name: str

    def apply(
        self,
        *,
        model: Any,
        tokenizer: Any,
        records: list[Mapping[str, Any]],
        hparams: DELMANHyperParams,
        covariance_provider: Any,
        context_templates: list[list[str]] | None = None,
    ) -> Any:
        """Apply a stage edit and return backend-specific metadata."""


class ImageEditingBackend(StageBackend, Protocol):
    """Image editing backend contract.

    Implementations own multimodal input construction and image-specific C/vector
    computation, but must use the shared covariance/update semantics when
    delegating into this package.
    """


@dataclass
class BackendRegistry:
    _backends: dict[str, StageBackend] = field(default_factory=dict)

    def register(self, backend: StageBackend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"editing backend already registered: {backend.name}")
        self._backends[backend.name] = backend

    def get(self, name: str) -> StageBackend:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise LookupError(f"editing backend is not registered: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))


class TextTokenBackend:
    name = "text"

    def apply(
        self,
        *,
        model: Any,
        tokenizer: Any,
        records: list[Mapping[str, Any]],
        hparams: DELMANHyperParams,
        covariance_provider: Any,
        context_templates: list[list[str]] | None = None,
    ) -> Any:
        from .delman import apply_text_edits

        return apply_text_edits(
            model=model,
            tok=tokenizer,
            requests=records,
            hparams=hparams,
            covariance_provider=covariance_provider,
            context_templates=context_templates,
            records_per_update=1,
            return_original_weights=False,
        )


class MissingImageBackend:
    name = "image"

    def apply(self, **_: Any) -> Any:
        raise NotImplementedError(
            "image editing backend is not registered; provide an ImageEditingBackend implementation"
        )


def default_backend_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(TextTokenBackend())
    registry.register(MissingImageBackend())
    return registry
