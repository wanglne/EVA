from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ImageBackendConfig:
    """Runtime configuration shared by all image-token backends."""

    family: str
    data_root: Path
    attention_key: str
    device: Any
    image_size: int = 448
    internvl_max_tiles_for_value: int = 1
    internvl_max_tiles_for_keys: int = 12


@dataclass(frozen=True)
class ImageBatch:
    """Prepared prompt/image inputs for value or key-vector computation."""

    prompts: tuple[str, ...]
    subjects: tuple[str, ...]
    lookup_indices: tuple[int, ...]
    inputs: Mapping[str, Any]
    rewrite_count: int = 0
    kl_count: int = 0


class ImageTokenBackend(Protocol):
    """Thin family-specific surface used by the shared image editing core."""

    config: ImageBackendConfig
    image_decoration: str

    def build_value_batch(
        self,
        *,
        handle: Any,
        request: Mapping[str, Any],
        target_ids: Any,
        context_templates: Sequence[Sequence[str]],
    ) -> ImageBatch:
        """Prepare prompts, images, lookup indices, and model inputs for v*."""

    def build_current_value_batch(
        self,
        *,
        handle: Any,
        requests: Sequence[Mapping[str, Any]],
        v_star_layer: int,
    ) -> ImageBatch:
        """Prepare the no-context prompts used to read current image-token values."""

    def build_key_batch(
        self,
        *,
        handle: Any,
        requests: Sequence[Mapping[str, Any]],
        context_templates: Sequence[Sequence[str]],
    ) -> ImageBatch:
        """Prepare contextual prompts used to read image-token keys."""

    def forward(self, *, handle: Any, batch: ImageBatch) -> Any:
        """Run a model forward pass for an already-prepared image batch."""

    def hidden_states_from_layer_output(self, output: Any) -> Any:
        """Return mutable hidden states from a traced layer output."""

    def replace_hidden_states_in_layer_output(self, output: Any, hidden_states: Any) -> Any:
        """Put edited hidden states back into the traced layer output shape."""
