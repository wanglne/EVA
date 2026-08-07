"""Small runtime compatibility helpers for the project GPU environments."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import zip_longest
from typing import Any

_MISSING = object()


def strict_zip(*iterables: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    """Backport ``zip(..., strict=True)`` for the Python 3.9 EVA environment."""

    for values in zip_longest(*iterables, fillvalue=_MISSING):
        if any(value is _MISSING for value in values):
            raise ValueError("strict_zip() arguments have different lengths")
        yield values
