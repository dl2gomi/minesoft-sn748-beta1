from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import crop, resized_crop, to_pil_image

from logger_config import logger

from .background_removal_pipeline import BackgroundRemovalPipeline
from .settings import BackgroundRemovalConfig


class BackgroundRemovalModule:
    """Shared image post-processing for background removal outputs."""

    def __init__(self, settings: BackgroundRemovalConfig):
        self.padding_percentage = settings.padding_percentage
        self.limit_padding = settings.limit_padding
        self.output_size = settings.output_image_size

    def remove_background(
        self,
        pipeline: BackgroundRemovalPipeline,
        image: Image.Image | Iterable[Image.Image],
    ) -> Image.Image | tuple[Image.Image, ...]:
        pipeline.ensure_ready()
        start_time = time.time()

        is_single = isinstance(image, Image.Image)
        images = [image] if is_single else list(image)
        outputs: list[Image.Image] = []

        for img in images:
            if self._has_nonopaque_alpha(img):
                outputs.append(img.convert("RGB"))
                continue

            tensor_rgb, mask = pipeline.predict_rgb_and_mask(img)
            cropped_rgba = self._crop_and_center(tensor_rgb, mask)
            outputs.append(to_pil_image(cropped_rgba[:3]))

        removal_time = time.time() - start_time
        logger.success(
            f"Background remove - Time: {removal_time:.2f}s - "
            f"Images without background: {len(outputs)}"
        )

        return outputs[0] if is_single else tuple(outputs)

    def _has_nonopaque_alpha(self, image: Image.Image) -> bool:
        if image.mode != "RGBA":
            return False
        alpha = np.array(image)[:, :, 3]
        return not np.all(alpha == 255)

    def _crop_and_center(self, tensor_rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bbox_indices = torch.argwhere(mask > 0.8)
        if len(bbox_indices) == 0:
            crop_args = dict(top=0, left=0, height=mask.shape[1], width=mask.shape[0])
        else:
            h_min, h_max = torch.aminmax(bbox_indices[:, 1])
            w_min, w_max = torch.aminmax(bbox_indices[:, 0])
            width, height = w_max - w_min, h_max - h_min
            center = (h_max + h_min) / 2, (w_max + w_min) / 2
            size = max(width, height)
            padded_size_factor = 1 + self.padding_percentage
            size = max(int(size * padded_size_factor), 2)

            top = int(center[1] - size // 2)
            left = int(center[0] - size // 2)
            bottom = int(center[1] + size // 2)
            right = int(center[0] + size // 2)

            if self.limit_padding:
                top = max(0, top)
                left = max(0, left)
                bottom = min(mask.shape[1], bottom)
                right = min(mask.shape[0], right)

            crop_args = dict(
                top=top,
                left=left,
                height=bottom - top,
                width=right - left,
            )

        mask = mask.unsqueeze(0)
        tensor_rgba = torch.cat([tensor_rgb * mask, mask], dim=-3)
        if self.output_size:
            return resized_crop(tensor_rgba, **crop_args, size=self.output_size, antialias=False)
        return crop(tensor_rgba, **crop_args)
