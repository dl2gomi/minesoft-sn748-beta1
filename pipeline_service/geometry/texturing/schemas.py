import torch
from typing import Optional
from pydantic import BaseModel
from schemas.types import AnyTensor, BoolTensor, FloatTensor, IntegerTensor


class MeshRasterizationData(BaseModel):
    face_ids: IntegerTensor                # (H, W) indices of faces for each pixel (invalid=-1)
    positions: FloatTensor                 # (N_valid, 3) position of mesh for each valid pixel
    normals: Optional[FloatTensor] = None  # (N_valid, 3) surface normal for each valid pixel

    @property
    def mask(self) -> BoolTensor:
        return self.face_ids.ge(0)


class AttributesMasked(BaseModel):
    values: AnyTensor    # (N, K) attribute values for N valid pixels
    mask: BoolTensor  # annotes which pixels are valid via boolean value

    def dense_shape(self, with_batch_size: bool = True) -> torch.Size:
        batch_size = (1,) if with_batch_size else ()
        return torch.Size(batch_size + (*self.mask.shape, self.values.shape[-1]))

    def to_dense(self, invalid: float = 0.0) -> torch.Tensor:
        size = self.dense_shape(with_batch_size=False)
        dense = self.values.new_full(size, fill_value=invalid)
        dense[self.mask] = self.values
        return dense