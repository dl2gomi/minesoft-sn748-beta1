from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Import module-specific configs
from modules.background_removal.settings import BackgroundRemovalConfig
from modules.converters.settings import GLBConverterConfig
from modules.mesh_generator.settings import TrellisConfig
from modules.image_edit.settings import QwenConfig
from modules.judge.settings import JudgeConfig
from modules.clarifier.settings import ClarifierConfig

config_dir = Path(__file__).parent
config_file_dir = Path(__file__).parent.parent / "configuration.yaml"


class APIConfig(BaseModel):
    """API configuration"""
    api_title: str = "3D Generation pipeline Service"
    host: str = "0.0.0.0"
    port: int = 10006
    debug: bool = False
    timeout: float = 300.0

class OutputConfig(BaseModel):
    """Output configuration"""
    save_generated_files: bool = False
    send_generated_files: bool = False
    compression: bool = False
    output_dir: Path = Path("generated_outputs")

class ModelVersionsConfig(BaseModel):
    """Pinned versions for HuggingFace models"""
    models: dict[str, str]
    
    def get_revision(self, model_id: str) -> Optional[str]:
        """Get revision for a given model ID, returns None if not found"""
        return self.models.get(model_id)


class GeometryRegimePreset(BaseModel):
    """Decimation and remesh overrides for a geometry regime (simple / complex_big / complex_tiny)."""
    decimation_target: int
    remesh: bool


class GeometryRegimeConfig(BaseModel):
    """Per-regime GLB geometry overrides (used after CLIP geometry-regime classification)."""
    simple: GeometryRegimePreset
    complex_big: GeometryRegimePreset
    complex_tiny: GeometryRegimePreset


class SettingsConf(BaseSettings):
    """Main settings class"""
    api: APIConfig = APIConfig()
    output: OutputConfig 
    trellis: TrellisConfig
    qwen: QwenConfig
    background_removal: BackgroundRemovalConfig
    glb_converter: GLBConverterConfig
    geometry_regime: Optional[GeometryRegimeConfig] = None
    model_versions: ModelVersionsConfig        
    judge: JudgeConfig
    clarifier: ClarifierConfig
    
def _load_yml_config(path: Path):
    """Classmethod returns YAML config"""
    try:
        return yaml.safe_load(path.read_text())

    except FileNotFoundError as error:
        message = "Error: yml config file not found."
        raise FileNotFoundError(error, message) from error

data_yaml = _load_yml_config(config_file_dir)
settings = SettingsConf.model_validate(data_yaml)
