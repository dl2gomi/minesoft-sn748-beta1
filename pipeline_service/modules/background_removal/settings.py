from typing import Optional
from typing_extensions import Tuple

from config.types import ModelConfig
from modules.background_removal.enums import RMBGModelType


class BackgroundRemovalConfig(ModelConfig):
    """Background removal configuration"""
    model_id: str = "ZhengPeng7/BiRefNet"
    model_type: RMBGModelType = RMBGModelType.BIREFNET
    input_image_size: Tuple[int, int] = (1024, 1024)
    output_image_size: Optional[Tuple[int, int]] = None
    padding_percentage: float = 0.0
    limit_padding: bool = True
