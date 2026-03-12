from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config.settings import ModelVersionsConfig, config_dir
from config.category_config import load_category_config
from logger_config import logger

from .settings import ClarifierConfig


@dataclass(slots=True)
class ClarifierResult:
    """Result returned by the CLIP-based clarifier."""

    reconstructability_score: float
    explanation: str


class CLIPClarifier:
    """
    Clarifier that scores how recognizable the object in the image is, using CLIP
    similarity to "easy to recognize" vs "hard to recognize" text anchors.
    High score → object is recognizable → single view; low score → multiview.
    """

    def __init__(
        self,
        settings: ClarifierConfig,
        model_versions: ModelVersionsConfig,
        category_prompts: Optional[dict[str, list[str]]] = None,
        category_order: Optional[list[str]] = None,
        category_confidence_threshold: Optional[float] = None,
    ) -> None:
        self.settings = settings
        self.model_id = settings.model_id
        self.gpu_index = settings.gpu
        self.dtype = self._resolve_dtype(settings.dtype)

        self._processor: Optional[CLIPProcessor] = None
        self._model: Optional[CLIPModel] = None
        self._model_revision: Optional[str] = model_versions.get_revision(self.model_id)

        # Text anchors: recognizability (easy = you can tell what the object is → single view; hard = you cannot → multiview).
        self._easy_prompts = [
            "a clear photograph of an object that is easy to recognize and identify",
            "an object you can clearly tell what it is",
            "a well-defined product or object whose type is obvious from the image",
        ]
        self._hard_prompts = [
            "an unclear or ambiguous image where the object is hard to recognize",
            "an object you cannot tell what it is",
            "a blurry, abstract, or confusing image where the subject is not identifiable",
        ]

        # Coarse material/category labels: from caller (pipeline loads YAML) or load from YAML here (YAML is main source; loader fallback only if file missing).
        if category_prompts and category_order:
            self._category_prompts = dict(category_prompts)
            self._category_order = list(category_order)
            self._category_confidence_threshold = (
                float(category_confidence_threshold)
                if category_confidence_threshold is not None
                else 0.45
            )
        else:
            path = Path(settings.category_config_path) if getattr(settings, "category_config_path", None) else config_dir.parent / "category_config.yaml"
            loaded = load_category_config(path)
            self._category_prompts = loaded["categories"]
            self._category_order = list(loaded["categories"].keys())
            self._category_confidence_threshold = float(loaded.get("category_confidence_threshold", 0.45))

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

        # Explanation buckets (recognizability).
        if score < 0.2:
            explanation = "The object in the image is very hard to recognize or identify."
        elif score < 0.4:
            explanation = "The object is difficult to recognize; the image may be ambiguous or unclear."
        elif score < 0.6:
            explanation = "The object is somewhat recognizable but not clearly identifiable."
        elif score < 0.8:
            explanation = "The object is clearly recognizable from the image."
        else:
            explanation = "The object is very easy to recognize and identify."

        return ClarifierResult(
            reconstructability_score=score,
            explanation=explanation,
        )

    def classify_category(self, image: Image.Image) -> Optional[str]:
        """
        Classify the main object in the image into a coarse material category using CLIP.
        Intended to be run after background removal. Returns one of: glass, metal, plastic, organic, generic.
        Returns None if the clarifier is disabled or not loaded.
        """
        if not self.settings.enabled or self._processor is None or self._model is None:
            return None

        self._ensure_ready()
        assert self._processor is not None
        assert self._model is not None

        all_prompts: list[str] = []
        category_ranges: list[tuple[int, int]] = []
        for cat in self._category_order:
            prompts = self._category_prompts[cat]
            if not prompts:
                # Skip empty prompt lists to avoid degenerate averages.
                category_ranges.append((len(all_prompts), len(all_prompts)))
                continue
            start = len(all_prompts)
            all_prompts.extend(prompts)
            category_ranges.append((start, len(all_prompts)))

        inputs = self._processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image[0]  # (num_texts,)

        # Compute average logit per category, then softmax over categories to get probabilities.
        categories: list[str] = []
        scores: list[float] = []
        for cat, (start, end) in zip(self._category_order, category_ranges):
            if end <= start:
                continue
            score = float(logits_per_image[start:end].mean().item())
            categories.append(cat)
            scores.append(score)

        if not categories:
            logger.warning("Category classifier: no valid category prompts; returning generic.")
            return "generic"

        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
        probs = torch.softmax(scores_tensor, dim=0)
        best_idx = int(torch.argmax(probs).item())
        best_cat = categories[best_idx]
        best_prob = float(probs[best_idx].item())

        threshold = float(getattr(self, "_category_confidence_threshold", 0.45))
        if best_prob < threshold:
            chosen = "generic"
        else:
            chosen = best_cat

        probs_list = probs.detach().cpu().tolist()
        scores_str = ", ".join(f"{c}:{p:.2f}" for c, p in zip(categories, probs_list))
        logger.info(
            f"Category classifier: {chosen} (best={best_cat}, p={best_prob:.2f}, thr={threshold:.2f}) | probs={{{scores_str}}}"
        )
        return chosen

