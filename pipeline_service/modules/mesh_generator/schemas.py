from dataclasses import dataclass
from typing import Iterable, Optional, TypeAlias
from PIL import Image

from .enums import TrellisMode, TrellisPipeType
from schemas.overridable import OverridableModel


class TrellisParams(OverridableModel):
    """TRELLIS.2 parameters with automatic fallback to settings."""
    sparse_structure_steps: int = 12
    sparse_structure_cfg_strength: float = 7.5
    shape_slat_steps: int = 12
    shape_slat_cfg_strength: float = 3.0
    tex_slat_steps: int = 12
    tex_slat_cfg_strength: float = 3.0
    pipeline_type: TrellisPipeType = TrellisPipeType.MODE_1024_CASCADE  # '512', '1024', '1024_cascade', '1536_cascade'
    mode: TrellisMode = TrellisMode.STOCHASTIC # Currently unused in TRELLIS.2
    max_num_tokens: int = 49152
    num_samples: int = 1
    
    @classmethod
    def from_settings(cls, settings) -> "TrellisParams":
        return cls(**settings.model_dump())


TrellisParamsOverrides: TypeAlias = TrellisParams.Overrides


@dataclass
class TrellisRequest:
    """Request for TRELLIS.2 3D generation (internal use only)."""
    image: Image.Image | Iterable[Image.Image]
    seed: int
    params: Optional[TrellisParamsOverrides] = None


@dataclass(slots=True)
class TrellisResult:
    """Result from TRELLIS.2 3D generation."""
    file_bytes: bytes | None = None
