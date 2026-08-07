"""Image-token editing backends.

The package is intentionally lightweight at import time. Backends import PIL,
torch, torchvision, and model-specific helpers only inside execution methods.
"""

from .backends import InternVL35Backend, Llava15Backend, Qwen25VLBackend, create_backend
from .protocol import ImageBackendConfig, ImageBatch, ImageTokenBackend
from .stage import ImageEditResult, ImageStageBackend, apply_image_edits

__all__ = [
    "ImageBackendConfig",
    "ImageBatch",
    "ImageTokenBackend",
    "ImageEditResult",
    "ImageStageBackend",
    "InternVL35Backend",
    "Llava15Backend",
    "Qwen25VLBackend",
    "apply_image_edits",
    "create_backend",
]
