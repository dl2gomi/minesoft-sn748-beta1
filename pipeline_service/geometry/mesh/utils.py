import torch
import cumesh
from typing import Tuple
from geometry.mesh.schemas import MeshData


def map_vertices_positions(mesh_data: MeshData, hi_res_mesh_data: MeshData, weight: float = 1.0, *, inplace: bool = False) -> MeshData:
    """Moves vertex postions to positions mapped from high resolution mesh using BVH. Iterpolates between postions"""
    bvh = hi_res_mesh_data.bvh
    assert bvh is not None, "BVH must be built for high-res mesh"
    
    _, face_id, uvw = bvh.unsigned_distance(mesh_data.vertices, return_uvw=True)
    tris = hi_res_mesh_data.faces[face_id.long()]
    tri_verts = hi_res_mesh_data.vertices[tris]
    mapped_positions = (tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    new_vertices = mesh_data.vertices.mul_(1 - weight) if inplace else mesh_data.vertices.mul(1 - weight)
    new_vertices.add_(mapped_positions, alpha=weight)

    mapped_mesh = mesh_data if inplace else MeshData(vertices=new_vertices, faces=mesh_data.faces, uvs=mesh_data.uvs, vertex_normals=mesh_data.vertex_normals, bvh=mesh_data.bvh)
    mapped_mesh.bvh = None

    return mapped_mesh


def sort_mesh(mesh_data: MeshData, axes: Tuple[int] = (0,1,2), desc: Tuple[bool] | bool = (False, False, False)) -> MeshData:
    """
    Sorts mesh vertices lexicografically by axes.
    Each axes is sorted independently in either ascending (default) or descending order
    Then it sorts faces by min vertex index in new ordering.
    """
    vertices = mesh_data.vertices
    faces = mesh_data.faces

    if isinstance(desc, bool):
        desc = (desc,) * len(axes)

    perm = torch.arange(vertices.shape[0], device=vertices.device)
    for axis, reverse in zip(reversed(axes), reversed(desc)):
        key = vertices[perm, axis]
        order = torch.argsort(key, descending=reverse, stable=True)
        perm = perm[order]

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.shape[0], device=perm.device, dtype=perm.dtype)

    sorted_vertices = vertices[perm]
    sorted_uvs = mesh_data.uvs[perm] if mesh_data.uvs is not None else None
    sorted_normals = mesh_data.vertex_normals[perm] if mesh_data.vertex_normals is not None else None

    remapped_faces = inv_perm[faces]
    face_min = remapped_faces.amin(dim=1)
    face_order = torch.argsort(face_min, stable=True)
    sorted_faces = remapped_faces[face_order]

    return MeshData(
        vertices=sorted_vertices,
        faces=sorted_faces,
        uvs=sorted_uvs,
        vertex_normals=sorted_normals,
        bvh=None,
    )


def count_boundary_loops(vertices: torch.Tensor, faces: torch.Tensor) -> int:
    """Count the number of boundary loops (holes) in a mesh."""
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.get_edges()
    mesh.get_boundary_info()
    if mesh.num_boundaries == 0:
        return 0
    mesh.get_vertex_edge_adjacency()
    mesh.get_vertex_boundary_adjacency()
    mesh.get_manifold_boundary_adjacency()
    mesh.read_manifold_boundary_adjacency()
    mesh.get_boundary_connected_components()
    mesh.get_boundary_loops()
    return mesh.num_boundary_loops



