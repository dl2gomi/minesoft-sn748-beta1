from modules.converters.params import GLBConverterParams
from config.types import DeviceModuleConfig


class GLBConverterConfig(GLBConverterParams, DeviceModuleConfig):
    """GLB converter configuration"""
    gpu: int = 0
