"""Shared editing primitives for EVA.

This package is intentionally lightweight at import time. Torch and
transformers are imported only by execution paths that operate on loaded
models.
"""

from .hparams import DELMANHyperParams, ModuleLayout, merge_hparams

__all__ = ["DELMANHyperParams", "ModuleLayout", "merge_hparams"]
