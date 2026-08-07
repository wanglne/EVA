from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eva._compat import strict_zip

from .core import (
    image_path,
    layer_output_hidden_states,
    move_batch_to_device,
    nested_rewrite,
    replace_layer_output_hidden_states,
    selected_attention_ids,
    visual_lookup_index,
)
from .protocol import ImageBackendConfig, ImageBatch


class _BaseImageBackend:
    image_decoration = "<image>\n "
    lookup_delimiter = "<image>\n "

    def __init__(self, config: ImageBackendConfig) -> None:
        self.config = config

    @property
    def data_root(self) -> Path:
        return self.config.data_root

    def hidden_states_from_layer_output(self, output: Any) -> Any:
        return layer_output_hidden_states(output)

    def replace_hidden_states_in_layer_output(self, output: Any, hidden_states: Any) -> Any:
        return replace_layer_output_hidden_states(output, hidden_states)

    def _attention_ids(self, request: Mapping[str, Any]) -> tuple[int, ...]:
        return selected_attention_ids(request, self.config.attention_key)

    def _lookup(self, prompt: str, subject: str, tokenizer: Any, request: Mapping[str, Any]) -> int:
        return visual_lookup_index(
            prompt=prompt,
            subject=subject,
            tok=tokenizer,
            attention_ids=self._attention_ids(request),
            lookup_delimiter=self.lookup_delimiter,
        )

    def _image_prompt_text(self, prompt_adv: str) -> str:
        return (
            prompt_adv
            if self.image_decoration in prompt_adv
            else f"{self.image_decoration}{prompt_adv}"
        )

    def _rewriting_and_kl_prompts(
        self,
        *,
        tokenizer: Any,
        request: Mapping[str, Any],
        target_ids: Any,
        context_templates: Sequence[Sequence[str]],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        rewrite = nested_rewrite(request)
        rewriting = tuple(
            context.format(self._image_prompt_text(str(rewrite["prompt_adv"])))
            + tokenizer.decode(target_ids[:-1])
            for context_type in context_templates
            for context in context_type
        )
        return rewriting, (f"{self.image_decoration}What does this picture contain?",)

    def _key_prompts(
        self,
        requests: Sequence[Mapping[str, Any]],
        context_templates: Sequence[Sequence[str]],
    ) -> tuple[str, ...]:
        return tuple(
            context.format(self._image_prompt_text(str(nested_rewrite(request)["prompt_adv"])))
            for request in requests
            for context_type in context_templates
            for context in context_type
        )

    def _key_subjects(
        self,
        requests: Sequence[Mapping[str, Any]],
        context_templates: Sequence[Sequence[str]],
    ) -> tuple[str, ...]:
        return tuple(
            nested_rewrite(request)["subject"]
            for request in requests
            for context_type in context_templates
            for _ in context_type
        )

    def _key_lookup_indices(
        self,
        *,
        tokenizer: Any,
        requests: Sequence[Mapping[str, Any]],
        context_templates: Sequence[Sequence[str]],
    ) -> tuple[int, ...]:
        return tuple(
            self._lookup(
                context.format(self._image_prompt_text(str(nested_rewrite(request)["prompt_adv"]))),
                nested_rewrite(request)["subject"],
                tokenizer,
                request,
            )
            for request in requests
            for context_type in context_templates
            for context in context_type
        )

    def forward(self, *, handle: Any, batch: ImageBatch) -> Any:
        return handle.model(**batch.inputs)


class Llava15Backend(_BaseImageBackend):
    """LLaVA-1.5 image-token backend.

    Uses raw PIL RGB images, the ``<image>\n `` prompt marker, processor
    multimodal batching, and
    ``prefix_token_count + selected_attention_id - 1`` lookup.
    """

    image_decoration = "<image>\n "
    lookup_delimiter = "<image>"

    def _image_prompt_text(self, prompt_adv: str) -> str:
        return prompt_adv

    def _load_image(self, request: Mapping[str, Any]) -> Any:
        from PIL import Image

        return Image.open(image_path(self.data_root, request)).convert("RGB")

    def _processor_inputs(
        self, handle: Any, prompts: Sequence[str], images: Sequence[Any]
    ) -> Mapping[str, Any]:
        encoded = handle.processor(
            text=list(prompts),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        return move_batch_to_device(encoded, self.config.device)

    def build_value_batch(
        self,
        *,
        handle: Any,
        request: Mapping[str, Any],
        target_ids: Any,
        context_templates: Sequence[Sequence[str]],
    ) -> ImageBatch:
        rewriting, kl_prompts = self._rewriting_and_kl_prompts(
            tokenizer=handle.tokenizer,
            request=request,
            target_ids=target_ids,
            context_templates=context_templates,
        )
        prompts = rewriting + kl_prompts
        subject = nested_rewrite(request)["subject"]
        subjects = tuple(subject for _ in prompts)
        filled = tuple(prompt.format(subject) for prompt in prompts)
        image = self._load_image(request)
        inputs = self._processor_inputs(handle, filled, [image] * len(filled))
        lookups = tuple(
            self._lookup(prompt, subject, handle.tokenizer, request) for prompt in prompts
        )
        return ImageBatch(
            prompts=filled,
            subjects=subjects,
            lookup_indices=lookups,
            inputs=inputs,
            rewrite_count=len(rewriting),
            kl_count=len(kl_prompts),
        )

    def build_current_value_batch(
        self,
        *,
        handle: Any,
        requests: Sequence[Mapping[str, Any]],
        v_star_layer: int,
    ) -> ImageBatch:
        del v_star_layer
        prompts = tuple(
            self._image_prompt_text(str(nested_rewrite(request)["prompt_adv"]))
            for request in requests
        )
        subjects = tuple(nested_rewrite(request)["subject"] for request in requests)
        filled = tuple(prompt.format(subject) for prompt, subject in strict_zip(prompts, subjects))
        images = [self._load_image(request) for request in requests]
        lookups = tuple(
            self._lookup(prompt, subject, handle.tokenizer, request)
            for prompt, subject, request in strict_zip(prompts, subjects, requests)
        )
        inputs = self._processor_inputs(handle, filled, images)
        return ImageBatch(prompts=filled, subjects=subjects, lookup_indices=lookups, inputs=inputs)

    def build_key_batch(
        self,
        *,
        handle: Any,
        requests: Sequence[Mapping[str, Any]],
        context_templates: Sequence[Sequence[str]],
    ) -> ImageBatch:
        prompts = self._key_prompts(requests, context_templates)
        subjects = self._key_subjects(requests, context_templates)
        image_requests = tuple(
            request
            for request in requests
            for context_type in context_templates
            for _ in context_type
        )
        images = [self._load_image(request) for request in image_requests]
        lookups = self._key_lookup_indices(
            tokenizer=handle.tokenizer,
            requests=requests,
            context_templates=context_templates,
        )
        filled = tuple(prompt.format(subject) for prompt, subject in strict_zip(prompts, subjects))
        inputs = self._processor_inputs(handle, filled, images)
        return ImageBatch(
            prompts=filled,
            subjects=subjects,
            lookup_indices=lookups,
            inputs=inputs,
        )


class Qwen25VLBackend(Llava15Backend):
    """Qwen2.5-VL backend using the released prompt sentinel and PIL loader."""

    image_decoration = "<|vision_start|><|image_pad|><|vision_end|>"
    lookup_delimiter = "<|vision_start|><|image_pad|><|vision_end|>"

    def _image_prompt_text(self, prompt_adv: str) -> str:
        return (
            prompt_adv
            if self.image_decoration in prompt_adv
            else f"{self.image_decoration}{prompt_adv}"
        )

    def _load_image(self, request: Mapping[str, Any]) -> Any:
        from PIL import Image

        return Image.open(image_path(self.data_root, request)).convert("RGB")


class InternVL35Backend(_BaseImageBackend):
    """InternVL3.5 448-pixel image backend.

    Uses ``<image>\n `` prompt decoration, 448 ImageNet-normalized tiling,
    value-time one-tile preprocessing, key/current-value max-12 tiling, and
    replaces the first ``<image>`` marker with repeated ``<IMG_CONTEXT>`` tokens
    before tokenization.
    """

    image_decoration = "<image>\n "
    lookup_delimiter = "<image>\n "
    image_context_token = "<IMG_CONTEXT>"

    def _load_image_tensor(self, request: Mapping[str, Any], *, max_tiles: int) -> Any:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode

        image = Image.open(image_path(self.data_root, request)).convert("RGB")
        tiles = _internvl_dynamic_preprocess(
            image,
            image_size=self.config.image_size,
            use_thumbnail=True,
            max_num=max_tiles,
        )
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                transforms.Resize(
                    (self.config.image_size, self.config.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.229),
                ),
            ]
        )
        tensor = torch.stack([transform(tile) for tile in tiles])
        return tensor

    def _typed_image_tensor(
        self, handle: Any, request: Mapping[str, Any], *, max_tiles: int
    ) -> Any:
        tensor = self._load_image_tensor(request, max_tiles=max_tiles)
        param = next(handle.model.parameters())
        return tensor.to(dtype=param.dtype, device=param.device)

    def _image_tokens(self, handle: Any, tile_count: int) -> str:
        return self.image_context_token * int(handle.model.num_image_token) * int(tile_count)

    def _tokenize(self, handle: Any, prompts: Sequence[str]) -> Mapping[str, Any]:
        encoded = handle.tokenizer(text=list(prompts), return_tensors="pt", padding=True)
        return move_batch_to_device(encoded, self.config.device)

    def _pixel_inputs(
        self, pixel_values: Any, tile_count: int, prompt_count: int
    ) -> Mapping[str, Any]:
        import torch

        device = self.config.device
        return {
            "pixel_values": pixel_values.to(device),
            "image_flags": torch.ones(tile_count * prompt_count, dtype=torch.long, device=device),
        }

    def build_value_batch(
        self,
        *,
        handle: Any,
        request: Mapping[str, Any],
        target_ids: Any,
        context_templates: Sequence[Sequence[str]],
    ) -> ImageBatch:
        rewriting, kl_prompts = self._rewriting_and_kl_prompts(
            tokenizer=handle.tokenizer,
            request=request,
            target_ids=target_ids,
            context_templates=context_templates,
        )
        prompts_before_expansion = rewriting + kl_prompts
        subject = nested_rewrite(request)["subject"]
        subjects = tuple(subject for _ in prompts_before_expansion)
        lookups = tuple(
            self._lookup(prompt, subject, handle.tokenizer, request)
            for prompt in prompts_before_expansion
        )
        pv = self._typed_image_tensor(
            handle,
            request,
            max_tiles=self.config.internvl_max_tiles_for_value,
        )
        image_tokens = self._image_tokens(handle, pv.size(0))
        expanded = tuple(
            prompt.replace("<image>", image_tokens, 1) for prompt in prompts_before_expansion
        )
        filled = tuple(prompt.format(subject) for prompt in expanded)
        input_tok = self._tokenize(handle, filled)
        pixel_values = __import__("torch").cat([pv] * len(filled), dim=0).to(self.config.device)
        inputs = {
            **input_tok,
            **self._pixel_inputs(pixel_values, pv.size(0), len(filled)),
        }
        return ImageBatch(
            prompts=filled,
            subjects=subjects,
            lookup_indices=lookups,
            inputs=inputs,
            rewrite_count=len(rewriting),
            kl_count=len(kl_prompts),
        )

    def build_current_value_batch(
        self,
        *,
        handle: Any,
        requests: Sequence[Mapping[str, Any]],
        v_star_layer: int,
    ) -> ImageBatch:
        del v_star_layer
        pvs = [
            self._typed_image_tensor(
                handle,
                request,
                max_tiles=self.config.internvl_max_tiles_for_keys,
            )
            for request in requests
        ]
        tile_count = pvs[0].size(0)
        image_tokens = self._image_tokens(handle, tile_count)
        prompts_before_expansion = tuple(
            self._image_prompt_text(str(nested_rewrite(request)["prompt_adv"]))
            for request in requests
        )
        subjects = tuple(nested_rewrite(request)["subject"] for request in requests)
        lookups = tuple(
            self._lookup(prompt, subject, handle.tokenizer, request)
            for prompt, subject, request in strict_zip(prompts_before_expansion, subjects, requests)
        )
        prompts = tuple(
            prompt.replace("<image>", image_tokens, 1) for prompt in prompts_before_expansion
        )
        filled = tuple(prompt.format(subject) for prompt, subject in strict_zip(prompts, subjects))
        torch = __import__("torch")
        pixel_values = torch.cat(pvs, dim=0).to(self.config.device)
        input_tok = self._tokenize(handle, filled)
        inputs = {**input_tok, **self._pixel_inputs(pixel_values, tile_count, len(requests))}
        return ImageBatch(prompts=filled, subjects=subjects, lookup_indices=lookups, inputs=inputs)

    def build_key_batch(
        self,
        *,
        handle: Any,
        requests: Sequence[Mapping[str, Any]],
        context_templates: Sequence[Sequence[str]],
    ) -> ImageBatch:
        image_requests = tuple(
            request
            for request in requests
            for context_type in context_templates
            for _ in context_type
        )
        pvs = [
            self._typed_image_tensor(
                handle,
                request,
                max_tiles=self.config.internvl_max_tiles_for_keys,
            )
            for request in image_requests
        ]
        tile_count = pvs[0].size(0)
        image_tokens = self._image_tokens(handle, tile_count)
        prompts_before_expansion = self._key_prompts(requests, context_templates)
        prompts = tuple(
            prompt.replace("<image>", image_tokens, 1) for prompt in prompts_before_expansion
        )
        subjects = self._key_subjects(requests, context_templates)
        lookups = self._key_lookup_indices(
            tokenizer=handle.tokenizer,
            requests=requests,
            context_templates=context_templates,
        )
        torch = __import__("torch")
        pixel_values = torch.cat(pvs, dim=0).to(self.config.device)
        raw_images = self._pixel_inputs(pixel_values, tile_count, len(image_requests))
        filled = tuple(prompt.format(subject) for prompt, subject in strict_zip(prompts, subjects))
        input_tok = self._tokenize(
            handle,
            filled,
        )
        return ImageBatch(
            prompts=filled,
            subjects=subjects,
            lookup_indices=lookups,
            inputs={**input_tok, **raw_images},
        )

    def forward(self, *, handle: Any, batch: ImageBatch) -> Any:
        return handle.model(
            pixel_values=batch.inputs["pixel_values"],
            input_ids=batch.inputs["input_ids"],
            attention_mask=batch.inputs["attention_mask"],
            image_flags=batch.inputs["image_flags"],
        )


def _internvl_find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[tuple[int, int]],
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


def _internvl_dynamic_preprocess(
    image: Any,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
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
        key=lambda value: value[0] * value[1],
    )
    target_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio,
        ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    block_count = target_ratio[0] * target_ratio[1]
    resized = image.resize((target_width, target_height))
    processed = []
    tiles_w = target_width // image_size
    for index in range(block_count):
        x0 = (index % tiles_w) * image_size
        y0 = (index // tiles_w) * image_size
        processed.append(resized.crop((x0, y0, x0 + image_size, y0 + image_size)))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


BACKEND_TYPES = {
    "llava": Llava15Backend,
    "llava15": Llava15Backend,
    "llava-1.5": Llava15Backend,
    "llava-1.5-7b": Llava15Backend,
    "llava_v1.5": Llava15Backend,
    "llava_v1_5": Llava15Backend,
    "qwen": Qwen25VLBackend,
    "qwen25vl": Qwen25VLBackend,
    "qwen2_5_vl": Qwen25VLBackend,
    "qwen2-5-vl": Qwen25VLBackend,
    "qwen2.5-vl": Qwen25VLBackend,
    "qwen2.5-vl-7b-instruct": Qwen25VLBackend,
    "qwen25vl-7b-instruct": Qwen25VLBackend,
    "internvl": InternVL35Backend,
    "internvl35": InternVL35Backend,
    "internvl3_5": InternVL35Backend,
    "internvl3.5": InternVL35Backend,
    "internvl3-5": InternVL35Backend,
    "internvl3.5-8b": InternVL35Backend,
}


def create_backend(config: ImageBackendConfig) -> _BaseImageBackend:
    """Create an explicitly selected backend without inspecting model type strings."""

    try:
        backend_type = BACKEND_TYPES[_normalize_family(config.family)]
    except KeyError as exc:
        raise ValueError(f"unsupported image backend family: {config.family}") from exc
    return backend_type(config)


def _normalize_family(family: str) -> str:
    return family.lower().replace("_", "-")
