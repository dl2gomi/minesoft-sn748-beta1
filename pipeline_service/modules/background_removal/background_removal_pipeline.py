from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image

from config.settings import ModelVersionsConfig
from logger_config import logger

from .settings import BackgroundRemovalConfig


class BackgroundRemovalPipeline(ABC):
    """Abstract lifecycle wrapper for background-removal model pipelines."""

    def __init__(self, settings: BackgroundRemovalConfig, model_versions: ModelVersionsConfig):
        self.settings = settings
        self.model_versions = model_versions
        self.device = f"cuda:{settings.gpu}" if torch.cuda.is_available() else "cpu"
        self.model: Any | None = None

    async def startup(self) -> None:
        logger.info(f"Loading {self.settings.model_id} model...")
        try:
            self.model = self._load_model()
            logger.success(f"{self.settings.model_id} model loaded.")
        except Exception as exc:
            logger.error(f"Error loading {self.settings.model_id} model: {exc}")
            raise RuntimeError(f"Error loading {self.settings.model_id} model: {exc}") from exc

    async def shutdown(self) -> None:
        if self.model is not None and hasattr(self.model, "to"):
            try:
                self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        logger.info(f"{self.settings.model_id} pipeline closed.")

    def is_ready(self) -> bool:
        return self.model is not None

    def ensure_ready(self) -> None:
        if self.model is None:
            raise RuntimeError(f"{self.settings.model_id} model not initialized.")

    @abstractmethod
    def _load_model(self) -> Any:
        """Load and return model instance."""
        raise NotImplementedError

    @abstractmethod
    def predict_rgb_and_mask(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Return RGB tensor and foreground mask for a single image."""
        raise NotImplementedError
