from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

from ben2 import BEN_Base

from config.settings import ModelVersionsConfig

from .background_removal_pipeline import BackgroundRemovalPipeline
from .settings import BackgroundRemovalConfig


class BEN2BackgroundRemovalPipeline(BackgroundRemovalPipeline):
    def __init__(self, settings: BackgroundRemovalConfig, model_versions: ModelVersionsConfig):
        super().__init__(settings, model_versions)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def _load_model(self) -> BEN_Base:
        revision = self.model_versions.get_revision(self.settings.model_id)
        model = BEN_Base.from_pretrained(self.settings.model_id, revision=revision)
        return model.to(self.device).eval()

    def predict_rgb_and_mask(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        self.ensure_ready()
        rgb_image = image.convert("RGB").resize(self.settings.input_image_size)

        with torch.no_grad():
            foreground = self.model.inference(rgb_image.copy())

        foreground_tensor = self.transforms(foreground)
        tensor_rgb = foreground_tensor[:3]
        mask = foreground_tensor[-1]
        return tensor_rgb, mask
