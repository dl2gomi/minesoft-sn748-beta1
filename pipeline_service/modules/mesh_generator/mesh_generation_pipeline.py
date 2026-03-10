from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import torch

from config.settings import ModelVersionsConfig
from logger_config import logger

from .settings import TrellisConfig

PipelineT = TypeVar("PipelineT")


class MeshGenerationPipeline(ABC, Generic[PipelineT]):
    """Abstract lifecycle wrapper for mesh generation pipelines."""

    def __init__(self, settings: TrellisConfig, model_versions: ModelVersionsConfig):
        self.settings = settings
        self.model_versions = model_versions
        self.device = f"cuda:{settings.gpu}" if torch.cuda.is_available() else "cpu"
        self.pipeline: Optional[PipelineT] = None

    async def startup(self) -> None:
        logger.info(f"Loading {self.settings.model_id} pipeline...")
        await self._load_pipeline()
        logger.success(f"{self.settings.model_id} pipeline ready.")

    async def shutdown(self) -> None:
        if self.pipeline is not None and hasattr(self.pipeline, "to"):
            try:
                self.pipeline.to("cpu")
            except Exception:
                pass
        self.pipeline = None
        torch.cuda.empty_cache()
        logger.info(f"{self.settings.model_id} pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None

    @property
    def loaded_pipeline(self) -> PipelineT:
        if self.pipeline is None:
            raise RuntimeError(f"{self.settings.model_id} pipeline not loaded.")
        return self.pipeline

    @abstractmethod
    async def _load_pipeline(self) -> None:
        raise NotImplementedError
