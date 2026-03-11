from typing import Literal

from modules.mesh_generator.schemas import TrellisParams
from config.types import ModelConfig


class TrellisConfig(TrellisParams, ModelConfig):
    model_id: str = "microsoft/TRELLIS.2-4B"
    pipeline_config_path: str = "libs/trellis2/pipeline.json"
    # Legacy boolean flag for multiview; kept for backward compatibility.
    multiview: bool = False
    # New multiview mode: "off" (single view), "always" (always multiview),
    # or "dynamic" (decide per-image via clarifier VLM).
    multiview_mode: Literal["off", "always", "dynamic"] = "off"

