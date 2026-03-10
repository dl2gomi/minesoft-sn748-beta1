from enum import Enum, IntEnum

class Axis(IntEnum):
    X: int = 0
    Y: int = 1
    Z: int = 2


class AlphaMode(str, Enum):
    OPAQUE: str = 'OPAQUE'
    MASK: str = 'MASK'
    BLEND: str = 'BLEND'
    DITHER: str = 'DITHER'

    @property
    def cutoff(self) -> float | None:
        if self == AlphaMode.MASK or self == AlphaMode.DITHER:
            return 0.5
        elif self == AlphaMode.BLEND:
            return 0.0
        return None

class SamplingMode(str, Enum):
    BILINEAR: str = 'bilinear'
    BICUBIC: str = 'bicubic'
    NEAREST: str = 'nearest'
    TRILINEAR: str = 'trilinear'
