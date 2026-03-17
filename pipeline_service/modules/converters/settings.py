from pydantic import BaseModel, Field

from modules.converters.params import GLBConverterParams
from config.types import DeviceModuleConfig


class GLBAutoClampTier(BaseModel):
    """Max caps applied to GLBConverterParams based on mesh/VRAM tier."""

    max_decimation_target: int = Field(default=100_000, ge=1)
    max_texture_size: int = Field(default=1536, ge=128)
    max_subdivisions: int = Field(default=1, ge=0)
    disable_remesh: bool = False
    disable_remesh_if_low_vram: bool = False


class GLBOomRetryConfig(BaseModel):
    """Second-pass clamp used when GLB conversion OOMs."""

    enabled: bool = True
    max_decimation_target: int = Field(default=70_000, ge=1)
    max_texture_size: int = Field(default=1024, ge=128)
    disable_remesh: bool = True
    max_subdivisions: int = Field(default=0, ge=0)


class GLBAutoClampConfig(BaseModel):
    """
    Automatic parameter clamping to avoid OOM/timeouts on large meshes.
    Defaults match the previous hardcoded behavior in pipeline.py.
    """

    enabled: bool = True

    # Mesh size tier thresholds.
    large_vertices: int = 400_000
    large_faces: int = 800_000
    very_large_vertices: int = 1_000_000
    very_large_faces: int = 2_000_000

    # VRAM thresholds (free memory in GB). If torch.cuda.mem_get_info fails, VRAM tiering is skipped.
    low_vram_gb: float = 10.0
    very_low_vram_gb: float = 6.0

    # Tier caps.
    large: GLBAutoClampTier = GLBAutoClampTier(
        max_decimation_target=100_000,
        max_texture_size=1536,
        max_subdivisions=1,
        disable_remesh=False,
        disable_remesh_if_low_vram=True,
    )
    very_large: GLBAutoClampTier = GLBAutoClampTier(
        max_decimation_target=60_000,
        max_texture_size=1024,
        max_subdivisions=0,
        disable_remesh=True,
        disable_remesh_if_low_vram=True,
    )

    # OOM retry caps.
    oom_retry: GLBOomRetryConfig = GLBOomRetryConfig()


class GLBConverterConfig(GLBConverterParams, DeviceModuleConfig):
    """GLB converter configuration"""
    gpu: int = 0
    auto_clamp: GLBAutoClampConfig = GLBAutoClampConfig()
