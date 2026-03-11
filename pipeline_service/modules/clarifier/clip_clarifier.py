from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config.settings import ModelVersionsConfig
from logger_config import logger

from .settings import ClarifierConfig


@dataclass(slots=True)
class ClarifierResult:
    """Result returned by the CLIP-based clarifier."""

    reconstructability_score: float
    explanation: str


class CLIPClarifier:
    """
    Clarifier that estimates how well a full 3D object (including unseen sides)
    can be reconstructed from a single image, using a CLIP similarity score
    between the image and two textual anchors: "easy to reconstruct" vs
    "hard to reconstruct".
    """

    def __init__(self, settings: ClarifierConfig, model_versions: ModelVersionsConfig) -> None:
        self.settings = settings
        self.model_id = settings.model_id
        self.gpu_index = settings.gpu
        self.dtype = self._resolve_dtype(settings.dtype)

        self._processor: Optional[CLIPProcessor] = None
        self._model: Optional[CLIPModel] = None
        self._model_revision: Optional[str] = model_versions.get_revision(self.model_id)

        # Text anchors for "easy" vs "hard" reconstructability.
        self._easy_prompts = [
            "a simple logistics object whose unseen sides are very predictable from this single view",
            "a box or package where the back and sides are almost the same as the front",
        ]
        self._hard_prompts = [
            "a logistics object whose unseen sides are difficult to infer and may have different graphics or geometry",
            "a product package with complex or asymmetric design where the back and sides are not obvious from the front",
        ]

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.gpu_index}")
        return torch.device("cpu")

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        resolved = mapping.get(dtype.lower(), torch.bfloat16)
        if not torch.cuda.is_available() and resolved in {torch.float16, torch.bfloat16}:
            return torch.float32
        return resolved

    async def startup(self) -> None:
        """Load the CLIP model and processor."""
        if not self.settings.enabled:
            logger.info("Clarifier is disabled; skipping CLIP load.")
            return

        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.gpu_index)
            except Exception as err:  # pragma: no cover - defensive
                logger.warning(f"Failed to set CUDA device ({self.gpu_index}) for clarifier: {err}")

        logger.info(f"Loading clarifier CLIP model: {self.model_id}")

        self._processor = CLIPProcessor.from_pretrained(
            self.model_id,
            revision=self._model_revision,
        )
        self._model = CLIPModel.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            revision=self._model_revision,
        ).to(self.device)

        logger.success(
            f"Clarifier CLIP model loaded on {self.device} "
            f"(dtype={self.dtype}, revision={self._model_revision or 'latest'})"
        )

    async def shutdown(self) -> None:
        """Release model references; memory will be freed by pipeline GPU cleanup."""
        self._processor = None
        self._model = None

    def _ensure_ready(self) -> None:
        if not self.settings.enabled:
            raise RuntimeError("Clarifier is disabled in configuration.")
        if self._processor is None or self._model is None:
            raise RuntimeError("Clarifier model is not loaded. Did you call startup()?")

    def score_image(self, image: Image.Image) -> ClarifierResult:
        """
        Compute reconstructability score in [0, 1] from a single image.

        The score is the softmax-normalized probability that the image matches
        "easy to reconstruct from one view" vs "hard to reconstruct".
        Higher score → unseen sides are more predictable / geometry is easier.
        """
        self._ensure_ready()

        assert self._processor is not None
        assert self._model is not None

        t_start = time.time()

        texts = self._easy_prompts + self._hard_prompts

        inputs = self._processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image  # shape: [1, num_texts]

        # Softmax over the two aggregated classes: easy vs hard.
        # Aggregate per-class logits by averaging over prompts in that class.
        num_easy = len(self._easy_prompts)
        num_hard = len(self._hard_prompts)

        logits = logits_per_image[0]  # (num_texts,)
        easy_logits = logits[:num_easy].mean()
        hard_logits = logits[num_easy : num_easy + num_hard].mean()

        class_logits = torch.stack([easy_logits, hard_logits], dim=0)  # [2]
        probs = class_logits.softmax(dim=0)
        score = float(probs[0].item())  # probability of "easy" class

        elapsed = time.time() - t_start
        logger.info(
            f"Clarifier (CLIP) inference time: {elapsed:.3f}s | "
            f"reconstructability_score={score:.3f} "
            f"(easy_prob={probs[0].item():.3f}, hard_prob={probs[1].item():.3f})"
        )

        # Provide a coarse natural-language explanation bucket.
        if score < 0.2:
            explanation = "Unseen sides are almost impossible to infer from this single view."
        elif score < 0.4:
            explanation = "Unseen sides are hard to infer and would often require guessing."
        elif score < 0.6:
            explanation = "Unseen sides are moderately inferable but some guessing is needed."
        elif score < 0.8:
            explanation = "Unseen sides are largely predictable from this view."
        else:
            explanation = "Unseen sides are very easy to infer; geometry is almost fully determined."

        return ClarifierResult(
            reconstructability_score=score,
            explanation=explanation,
        )

