import torch
import cumesh
from typing import Optional
from pydantic import BaseModel
from schemas.types import AnyTensor, IntegerTensor, FloatTensor
from schemas.internal import Internal


class AttributeGrid(BaseModel):
    values: AnyTensor          # (N, K) attribute values for N voxels
    coords: IntegerTensor          # (N, 3) voxel coordinates on the grid
    aabb: FloatTensor          # (2, 3) axis-aligned bounding box (optional)
    voxel_size: FloatTensor    # (3,) size of voxel in each dimension

    @property
    def grid_size(self) -> FloatTensor:
        return ((self.aabb[1] - self.aabb[0]) / self.voxel_size).round().int()
    

    def dense_shape(self, with_batch_size: bool = True) -> torch.Size:
        batch_size = (1,) if with_batch_size else ()
        return torch.Size(batch_size + (self.values.shape[1], *self.grid_size.tolist()))
    

class MeshData(BaseModel):
    """Mesh geometry with vertices, faces, vertex normals and UVs."""
    vertices: FloatTensor                            # (V, 3) vertex positions
    faces: IntegerTensor                             # (F, 3) face indices
    vertex_normals: Optional[FloatTensor] = None     # (V, 3) vertex normals (optional)
    # Tangent-space basis for normal mapping.
    # Convention: (tangent_x, tangent_y, tangent_z, handedness_w) where handedness_w is +/-1.
    vertex_tangents: Optional[FloatTensor] = None    # (V, 4) tangent attribute (optional)
    uvs: Optional[FloatTensor] = None                # (V, 2) UV coordinates (optional)
    bvh: Optional[Internal[cumesh.cuBVH]] = None     # BVH tree for ray tracing and projection

    def build_bvh(self):
        self.bvh = cumesh.cuBVH(self.vertices, self.faces)


class MeshDataWithAttributeGrid(MeshData):
    attrs: AttributeGrid



