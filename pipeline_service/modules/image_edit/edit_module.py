import time
from typing import Iterable, Optional

import torch
from PIL import Image

from .image_edit_pipeline import ImageEditPipeline
from logger_config import logger
from modules.image_edit.schemas import ImageGenerationParams, ImageGenerationParamsOverrides
from modules.image_edit.prompting import Prompting, TextPrompting

INPUT_IMAGE_SIZE = 1024 * 1024


class EditModule:
    """Runs image edits with a provided loaded image-edit pipeline."""

    def __init__(self, default_edit_params: ImageGenerationParams):
        self.default_edit_params = default_edit_params

    def _prepare_input_image(self, image: Image.Image, pixels: int = INPUT_IMAGE_SIZE) -> Image.Image:
        total = int(pixels)
        scale_by = (total / (image.width * image.height)) ** 0.5
        width = round(image.width * scale_by)
        height = round(image.height * scale_by)
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def edit_image(
        self,
        model: ImageEditPipeline,
        prompt_image: Image.Image | Iterable[Image.Image],
        seed: int,
        prompting: Prompting | str,
        params: Optional[ImageGenerationParamsOverrides] = None,
    ) -> Iterable[Image.Image]:
        """ 
        Edit the image using Qwen Edit.

        Args:
            pipeline: Preloaded image edit pipeline.
            prompt_image: The prompt image to edit.
            prompting: Prompting object or string prompt.

        Returns:
            The edited image.
        """
        assert model.is_ready(), "Edit pipeline is not loaded."
        
        try:
            start_time = time.time()

            if isinstance(prompting, str):
                prompting = TextPrompting(positive=prompting)

            prompt_images = [prompt_image] if isinstance(prompt_image, Image.Image) else list(prompt_image)
            resized_images = [self._prepare_input_image(image) for image in prompt_images]

            prompting_args = prompting.model_dump()
            generation_args = self.default_edit_params.overrided(params).model_dump()

            if seed is not None:
                prompting_args["generator"] = torch.Generator(device=model.device).manual_seed(seed)

            result = model.pipeline(
                image=resized_images,
                **generation_args,
                **prompting_args,
            )
            
            generation_time = time.time() - start_time
            
            results = tuple(result.images)
            
            logger.success(f"Edited image generated in {generation_time:.2f}s, Size: {results[0].size}, Seed: {seed}")
            
            return results
            
        except Exception as e:
            # If this is a CUDA OOM, clear cache so caller can recover/fallback.
            msg = str(e).lower()
            if (
                isinstance(e, torch.cuda.OutOfMemoryError)
                or "out of memory" in msg
                or "cuda" in msg
            ):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            logger.error(f"Error generating image: {e}")
            raise
