from typing import Literal

from modules.mesh_generator.schemas import TrellisParams
from config.types import ModelConfig


class TrellisConfig(TrellisParams, ModelConfig):
    model_id: str = "microsoft/TRELLIS.2-4B"
    pipeline_config_path: str = "libs/trellis2/pipeline.json"
    # New multiview mode: "off" (single view), "always" (always multiview),
    # or "dynamic" (decide per-image via decision module vLLM).
    multiview_mode: Literal["off", "always", "dynamic"] = "off"

