from typing import Optional

from config.types import ModelConfig


class ClarifierConfig(ModelConfig):
    """
    Settings for the clarifier VLM that scores 3D reconstructability
    from a single input image.
    """

    enabled: bool = True
    reconstructability_threshold: float = 0.7
    max_new_tokens: int = 128
    dtype: str = "bf16"
    # Optional path to category_config.yaml. If None, pipeline_service/category_config.yaml is used.
    category_config_path: Optional[str] = None

