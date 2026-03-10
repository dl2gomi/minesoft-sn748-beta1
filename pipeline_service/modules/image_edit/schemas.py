from typing import TypeAlias
from schemas.overridable import OverridableModel


class ImageGenerationParams(OverridableModel):
    """Image generation parameters with automatic fallback to settings."""
    height: int
    width: int
    num_inference_steps: int
    true_cfg_scale: float = 1.0


    @classmethod
    def from_settings(cls, settings) -> "ImageGenerationParams":
        return cls(
            true_cfg_scale=settings.true_cfg_scale,
            num_inference_steps=settings.num_inference_steps,
            height=settings.height,
            width=settings.width,
        )

ImageGenerationParamsOverrides: TypeAlias = ImageGenerationParams.Overrides
