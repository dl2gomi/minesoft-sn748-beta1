from modules.mesh_generator.schemas import TrellisParams
from config.types import ModelConfig

class TrellisConfig(TrellisParams, ModelConfig):
    model_id: str = "microsoft/TRELLIS.2-4B"
    pipeline_config_path: str = "libs/trellis2/pipeline.json"
    multiview: bool = False
