from __future__ import annotations

from abc import ABC, abstractmethod

from logger_config import logger
from .schemas import JudgeResponse


class JudgePipeline(ABC):
    """Abstract lifecycle wrapper for judge pipelines."""

    def __init__(self) -> None:
        self._ready = False

    async def startup(self) -> None:
        logger.info(f"Initializing {self.__class__.__name__}...")
        await self._setup()
        self._ready = True
        logger.success(f"{self.__class__.__name__} ready.")

    async def shutdown(self) -> None:
        await self._teardown()
        self._ready = False
        logger.info(f"{self.__class__.__name__} closed.")

    def is_ready(self) -> bool:
        return self._ready

    async def _setup(self) -> None:
        """Optional setup hook (e.g. create HTTP client)."""

    async def _teardown(self) -> None:
        """Optional teardown hook (e.g. close HTTP client)."""

    @abstractmethod
    async def judge(
        self,
        prompt_b64: str,
        img1_b64: str,
        img2_b64: str,
        seed: int,
    ) -> JudgeResponse:
        """Compare two candidate images against a prompt image and return penalties."""
        raise NotImplementedError
