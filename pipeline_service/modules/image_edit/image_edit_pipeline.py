from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import torch

from config.settings import ModelVersionsConfig
from logger_config import logger

PipelineT = TypeVar("PipelineT")


class ImageEditPipeline(ABC, Generic[PipelineT]):
    """Abstract lifecycle wrapper for image-edit pipelines."""

    def __init__(self, model_versions: ModelVersionsConfig, gpu_index: int):
        self.model_versions = model_versions
        self.gpu_index = gpu_index
        self.device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
        self.pipe: Optional[PipelineT] = None

    async def startup(self) -> None:
        logger.info(f"Initializing {self.__class__.__name__}...")
        await self._load_pipeline()
        logger.success(f"{self.__class__.__name__} ready.")

    async def shutdown(self) -> None:
        if self.pipe is not None and hasattr(self.pipe, "to"):
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
        self.pipe = None
        logger.info(f"{self.__class__.__name__} closed.")

    def is_ready(self) -> bool:
        return self.pipe is not None

    @property
    def pipeline(self) -> PipelineT:
        if self.pipe is None:
            raise RuntimeError(f"{self.__class__.__name__} is not loaded")
        return self.pipe

    @abstractmethod
    async def _load_pipeline(self) -> None:
        """Load pipeline weights and assign self.pipe."""
        raise NotImplementedError
