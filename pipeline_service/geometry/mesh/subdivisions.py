import torch
import torch.nn.functional as F
from geometry.mesh.schemas import MeshData
from geometry.mesh.enums import SubdivisionMode


def subdivide_egdes(mesh_data: MeshData, iterations: int = 1) -> MeshData:
    """Subdivides edges by adding a vertex at the midpoint of each edge creating 4 new faces in space of one."""
    vertices = mesh_data.vertices
    faces = mesh_data.faces
    uvs = mesh_data.uvs.clone() if mesh_data.uvs is not None else None
    vertex_normals = mesh_data.vertex_normals.clone() if mesh_data.vertex_normals is not None else None

    for _ in range(iterations):
        face_edges = torch.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=1)
        sorted_edges = torch.sort(face_edges, dim=-1).values
        unique_edges, inverse = torch.unique(sorted_edges.reshape(-1, 2), dim=0, return_inverse=True)

        edge_midpoints = (vertices[unique_edges[:, 0]] + vertices[unique_edges[:, 1]]) * 0.5
        midpoint_indices = torch.arange(
            vertices.shape[0],
            vertices.shape[0] + unique_edges.shape[0],
            device=faces.device,
            dtype=faces.dtype,
        )
        vertices = torch.cat((vertices, edge_midpoints), dim=0)

        if uvs is not None:
            edge_uvs = (uvs[unique_edges[:, 0]] + uvs[unique_edges[:, 1]]) * 0.5
            uvs = torch.cat((uvs, edge_uvs), dim=0)

        if vertex_normals is not None:
            edge_normals = F.normalize(
                (vertex_normals[unique_edges[:, 0]] + vertex_normals[unique_edges[:, 1]]) * 0.5,
                dim=-1,
                eps=1e-8,
            )
            vertex_normals = torch.cat((vertex_normals, edge_normals), dim=0)

        edge_ids = midpoint_indices[inverse].reshape(-1, 3)
        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
        m01, m12, m20 = edge_ids[:, 0], edge_ids[:, 1], edge_ids[:, 2]

        faces = torch.stack(
            (
                torch.stack((v0, m01, m20), dim=-1),
                torch.stack((v1, m12, m01), dim=-1),
                torch.stack((v2, m20, m12), dim=-1),
                torch.stack((m01, m12, m20), dim=-1),
            ),
            dim=1,
        ).reshape(-1, 3)

    # Create new MeshData with subdivided mesh
    return MeshData(
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        vertex_normals=vertex_normals,
        bvh=None  # BVH should be rebuilt if needed
    )


def subdivide_faces(mesh_data: MeshData, iterations: int = 1) -> MeshData:
    """Subdivides faces by adding a vertex at the midpoint of each face creating 3 new faces in space of one."""
    vertices = mesh_data.vertices
    faces = mesh_data.faces
    uvs = mesh_data.uvs.clone() if mesh_data.uvs is not None else None
    vertex_normals = mesh_data.vertex_normals.clone() if mesh_data.vertex_normals is not None else None

    for _ in range(iterations):
        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]

        face_centers = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        center_indices = torch.arange(
            vertices.shape[0],
            vertices.shape[0] + faces.shape[0],
            device=faces.device,
            dtype=faces.dtype,
        )
        vertices = torch.cat((vertices, face_centers), dim=0)

        if uvs is not None:
            face_uvs = (uvs[v0] + uvs[v1] + uvs[v2]) / 3.0
            uvs = torch.cat((uvs, face_uvs), dim=0)

        if vertex_normals is not None:
            face_normals = F.normalize(
                (vertex_normals[v0] + vertex_normals[v1] + vertex_normals[v2]) / 3.0,
                dim=-1,
                eps=1e-8,
            )
            vertex_normals = torch.cat((vertex_normals, face_normals), dim=0)

        c = center_indices
        faces = torch.stack(
            (
                torch.stack((v0, v1, c), dim=-1),
                torch.stack((v1, v2, c), dim=-1),
                torch.stack((v2, v0, c), dim=-1),
            ),
            dim=1,
        ).reshape(-1, 3)

    return MeshData(
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        vertex_normals=vertex_normals,
        bvh=None  # BVH should be rebuilt if needed
    )

def subdivide_mesh(mesh_data: MeshData, mode: SubdivisionMode, iterations: int = 1) -> MeshData:
    mode = SubdivisionMode(mode)
    if mode is SubdivisionMode.EDGE:
        return subdivide_egdes(mesh_data, iterations=iterations)
    if mode is SubdivisionMode.FACE:
        return subdivide_faces(mesh_data, iterations=iterations)
    raise ValueError(f"No subdivision method for mode {mode}")
