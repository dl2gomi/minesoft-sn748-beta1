from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from config.settings import ModelVersionsConfig

from .background_removal_pipeline import BackgroundRemovalPipeline
from .settings import BackgroundRemovalConfig


class BirefNetBackgroundRemovalPipeline(BackgroundRemovalPipeline):
    def __init__(self, settings: BackgroundRemovalConfig, model_versions: ModelVersionsConfig):
        super().__init__(settings, model_versions)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size),
                transforms.ToTensor(),
            ]
        )
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _load_model(self) -> AutoModelForImageSegmentation:
        revision = self.model_versions.get_revision(self.settings.model_id)
        model = AutoModelForImageSegmentation.from_pretrained(
            self.settings.model_id,
            revision=revision,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        return model.to(self.device)

    def predict_rgb_and_mask(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        self.ensure_ready()
        rgb_image = image.convert("RGB")
        rgb_tensor = self.transforms(rgb_image).to(self.device)
        input_tensor = self.normalize(rgb_tensor).unsqueeze(0)

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid()
            mask = preds[0].squeeze().mul_(255).int().div(255).float()
        return rgb_tensor, mask
