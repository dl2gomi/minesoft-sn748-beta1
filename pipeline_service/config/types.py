from abc import ABC
from pydantic import BaseModel


class DeviceModuleConfig(BaseModel, ABC):
    gpu: int = 0


class ModelConfig(DeviceModuleConfig, ABC):
    model_id: str