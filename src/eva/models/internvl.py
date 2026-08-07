"""InternVL3.5 adapter."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from eva._compat import strict_zip

from .base import (
    BaseModelAdapter,
    ModelHandle,
    TextEditView,
    configure_text_stage_tokenizer,
    move_batch,
    torch_dtype,
    transformers_attr,
)
from .paths import module_names_from_model, resolve_module_layout

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
PROCESSOR_SUPPORT_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
)


class InternVL35Adapter(BaseModelAdapter):
    family = "internvl"
    default_dtype = "bfloat16"
    default_trust_remote_code = True

    def load(
        self,
        model_path: str | Path,
        *,
        device: str | None,
        dtype: str | None = None,
        trust_remote_code: bool | None = None,
        low_cpu_mem_usage: bool = True,
        use_flash_attn: bool = True,
        **kwargs: Any,
    ) -> ModelHandle:
        processor_kwargs = dict(kwargs.pop("processor_kwargs", {}) or {})
        effective_trust_remote_code = True if trust_remote_code is None else trust_remote_code
        processor_kwargs.setdefault("trust_remote_code", effective_trust_remote_code)
        AutoTokenizer = transformers_attr("AutoTokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=effective_trust_remote_code,
            use_fast=False,
        )
        model_kwargs = {
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "use_flash_attn": use_flash_attn,
            **kwargs,
        }
        handle = self._load_handle(
            model_cls=transformers_attr("AutoModel"),
            processor_cls=transformers_attr("AutoImageProcessor"),
            model_path=model_path,
            device=device,
            dtype=dtype,
            trust_remote_code=effective_trust_remote_code,
            tokenizer=tokenizer,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )
        handle.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        configure_text_stage_tokenizer(handle.tokenizer)
        handle.load_options = replace(
            handle.load_options,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_flash_attn=use_flash_attn,
        )
        return handle

    def select_text_edit_view(self, handle: ModelHandle) -> TextEditView:
        text_model = getattr(handle.model, "language_model", handle.model)
        module_names = module_names_from_model(text_model)
        try:
            layout = resolve_module_layout(None, module_names, layer=handle.layout.layer_index)
        except ValueError as exc:
            sample = ", ".join(repr(name) for name in module_names[:8]) or "<empty>"
            raise ValueError(
                "could not resolve InternVL language-submodel layout; "
                f"named-module sample: {sample}"
            ) from exc
        layout = replace(layout, family="internvl")
        return TextEditView(
            model=text_model,
            tokenizer=handle.tokenizer,
            parent=handle,
            layout=layout,
        )

    def save(self, handle: ModelHandle, output_dir: str | Path, **kwargs: Any) -> None:
        """Save the full VLM and preserve local InternVL reload support files."""

        super().save(handle, output_dir, **kwargs)
        source = Path(handle.load_options.model_path)
        output = Path(output_dir)
        support_files = [
            *(source / name for name in PROCESSOR_SUPPORT_FILES),
            *source.glob("*.py"),
        ]
        for source_file in support_files:
            destination = output / source_file.name
            if source_file.is_file() and not destination.exists():
                shutil.copy2(source_file, destination)

    def prepare_image_batch(
        self,
        handle: ModelHandle,
        *,
        prompts: list[str],
        images: list[Any],
        device: str | None = None,
        dtype: str | None = None,
    ) -> dict[str, Any]:
        pixel_values, tile_counts = self._preprocess_images(
            images, handle=handle, device=device, dtype=dtype
        )
        image_flags = self._ones(pixel_values.shape[0], device=device)
        texts = [
            self.decorate_image_prompt(prompt).replace(
                "<image>",
                IMG_CONTEXT_TOKEN * handle.model.num_image_token * tile_count,
                1,
            )
            for prompt, tile_count in strict_zip(prompts, tile_counts)
        ]
        tokenized = handle.tokenizer(texts, return_tensors="pt", padding=True)
        batch = {
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        return move_batch(batch, device=device, dtype=dtype)

    def forward_image_batch(self, handle: ModelHandle, batch: dict[str, Any], **kwargs: Any) -> Any:
        return handle.model(**batch, **kwargs)

    def _preprocess_images(
        self,
        images: list[Any],
        *,
        handle: ModelHandle,
        device: str | None,
        dtype: str | None,
    ) -> tuple[Any, list[int]]:
        if handle.processor is not None and callable(handle.processor):
            batch = handle.processor(images=images, return_tensors="pt")
            if "pixel_values" in batch:
                pixel_values = move_batch(
                    {"pixel_values": batch["pixel_values"]}, device=device, dtype=dtype
                )["pixel_values"]
                per_image = pixel_values.shape[0] // len(images) if images else 0
                return pixel_values, [per_image] * len(images)
        return self._dynamic_preprocess(images, device=device, dtype=dtype)

    def _dynamic_preprocess(
        self,
        images: list[Any],
        *,
        device: str | None,
        dtype: str | None,
    ) -> tuple[Any, list[int]]:
        """Fallback 448 preprocessing matching the preserved InternVL source.

        This path is a compatibility fallback for processors that are not callable.
        It uses legacy dynamic tiling, thumbnail append, and ImageNet std values.
        """

        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.229),
                ),
            ]
        )
        tensors = []
        tile_counts = []
        for image in images:
            pil_image = (
                Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image
            )
            tiles = _dynamic_tiles(pil_image, image_size=448, max_num=12, use_thumbnail=True)
            tile_counts.append(len(tiles))
            for tile in tiles:
                tensors.append(transform(tile))
        pixel_values = torch.stack(tensors)
        dtype_obj = torch_dtype(dtype)
        if dtype_obj is not None:
            pixel_values = pixel_values.to(dtype=dtype_obj)
        if device is not None:
            pixel_values = pixel_values.to(device)
        return pixel_values, tile_counts

    def _ones(self, size: int, *, device: str | None) -> Any:
        import torch

        return torch.ones(size, dtype=torch.long, device=device)


def _dynamic_tiles(
    image: Any,
    *,
    image_size: int,
    max_num: int,
    min_num: int = 1,
    use_thumbnail: bool,
) -> list[Any]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_ratio = _find_closest_aspect_ratio(
        aspect_ratio, ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]

    resized = image.resize((target_width, target_height))
    tiles = []
    tiles_w = target_width // image_size
    for idx in range(blocks):
        x = (idx % tiles_w) * image_size
        y = (idx // tiles_w) * image_size
        tiles.append(resized.crop((x, y, x + image_size, y + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


ADAPTER = InternVL35Adapter()
