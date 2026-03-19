from typing import Tuple
import torch
import kaolin
import torch.nn.functional as F
from geometry.texturing.enums import SamplingMode
from geometry.mesh.schemas import MeshData, AttributeGrid
from geometry.texturing.schemas import AttributesMasked, MeshRasterizationData
from flex_gemm.ops.grid_sample import grid_sample_3d


def rasterize_mesh_data(
    mesh_data: MeshData,
    texture_size: int | Tuple[int, int],
    use_vertex_normals: bool = False,
    use_vertex_tangents: bool = False,
) -> MeshRasterizationData:
    """Rasterize the mesh shape onto the mesh UV space."""
    
    uvs = mesh_data.uvs
    assert uvs is not None, "UVs are rquired for rasterization"

    faces = mesh_data.faces
    vertices = mesh_data.vertices

    height, width = torch.Size(torch.as_tensor(texture_size).broadcast_to(2))

    
    # Prepare UVs
    uvs_ndc = uvs * 2 - 1
    uvs_ndc[:, 1] = -uvs_ndc[:, 1]
    uvs_ndc = uvs_ndc.unsqueeze(0) if uvs_ndc.dim() == 2 else uvs_ndc
    
    # Normalize shapes:
    # - vertices: (V,3) -> (1,V,3)
    # - faces: (F,3) or (1,F,3) -> (F,3)
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    faces = faces.long() if faces.dim() == 2 else faces.squeeze(0).long()

    surflets = vertices
    surflet_feat_dims: int = 3
    if use_vertex_normals:
        if mesh_data.vertex_normals is None:
            raise ValueError("use_vertex_normals=True but mesh_data.vertex_normals is None")
        vertex_normals = mesh_data.vertex_normals
        # vertex_normals: (V,3) -> (1,V,3) to match vertices
        if vertex_normals.dim() == 2:
            vertex_normals = vertex_normals.unsqueeze(0)
        vertex_normals = vertex_normals.to(vertices.dtype).to(vertices.device)
        if vertex_normals.shape[:2] != vertices.shape[:2]:
            raise ValueError(
                f"vertex_normals shape {tuple(vertex_normals.shape)} does not match vertices {tuple(vertices.shape)}"
            )
        surflets = torch.cat((surflets, vertex_normals), dim=-1)
        surflet_feat_dims += 3
    if use_vertex_tangents:
        if mesh_data.vertex_tangents is None:
            raise ValueError("use_vertex_tangents=True but mesh_data.vertex_tangents is None")
        vertex_tangents = mesh_data.vertex_tangents
        # vertex_tangents: (V,4) -> (1,V,4) to match vertices
        if vertex_tangents.dim() == 2:
            vertex_tangents = vertex_tangents.unsqueeze(0)
        vertex_tangents = vertex_tangents.to(vertices.dtype).to(vertices.device)
        if vertex_tangents.shape[0] != vertices.shape[0] or vertex_tangents.shape[1] != vertices.shape[1]:
            raise ValueError(
                f"vertex_tangents shape {tuple(vertex_tangents.shape)} does not match vertices {tuple(vertices.shape)}"
            )
        surflets = torch.cat((surflets, vertex_tangents), dim=-1)
        surflet_feat_dims += 4

    # Index by faces
    face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(uvs_ndc, faces)
    face_vertex_suflets = kaolin.ops.mesh.index_vertices_by_faces(surflets, faces)

    batch_size, num_faces = face_vertices_image.shape[:2]

    face_vertices_z = torch.zeros(
        (batch_size, num_faces, 3),
        device=vertices.device,
        dtype=vertices.dtype
    )
    
    with torch.no_grad():
        surf_interpolated, face_idx = kaolin.render.mesh.rasterize(
            height=height,
            width=width,
            face_vertices_z=face_vertices_z,
            face_vertices_image=face_vertices_image,
            face_features=face_vertex_suflets,
            backend='cuda',
            multiplier=1000,
            eps=1e-8
        )
    
    surf = surf_interpolated[0]
    mask = face_idx[0] >= 0

    valid_surf = surf[mask]
    # surf layout: [pos(3), normals(3)? , tangents(4)?] in the order we concatenated.
    cursor = 0
    valid_positions = valid_surf[..., cursor : cursor + 3]
    cursor += 3

    valid_normals = None
    if use_vertex_normals:
        valid_normals = valid_surf[..., cursor : cursor + 3]
        cursor += 3

    valid_tangents = None
    if use_vertex_tangents:
        valid_tangents = valid_surf[..., cursor : cursor + 4]
        cursor += 4

    # Ensure we return None for the fields the caller didn't request.
    if not use_vertex_normals:
        valid_normals = None
    if not use_vertex_tangents:
        valid_tangents = None

    return MeshRasterizationData(
        face_ids=face_idx[0],
        positions=valid_positions,
        normals=valid_normals,
        tangents=valid_tangents,
    )


def map_mesh_rasterization(rast_data: MeshRasterizationData, mesh_data: MeshData, flip_vertex_normals: bool = False) -> MeshRasterizationData:

    bvh = mesh_data.bvh
    assert bvh is not None, "Mesh BVH needs to be build for mapping"
    valid_pos = rast_data.positions

    # Map these positions back to the *original* high-res mesh to get accurate attributes
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    tris = mesh_data.faces[face_id.long()]
    tri_verts = mesh_data.vertices[tris]  # (N_new, 3, 3)
    valid_positions = (tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    valid_normals = None

    if rast_data.normals is not None:

        valid_normals = None
        if mesh_data.vertex_normals is not None:
            tri_norms = mesh_data.vertex_normals[tris]
            valid_normals = (tri_norms * uvw.unsqueeze(-1)).sum(1)
        else:
            edge1 = tri_verts[:, 1] - tri_verts[:, 0]
            edge2 = tri_verts[:, 2] - tri_verts[:, 0]
            valid_normals = torch.linalg.cross(edge1, edge2, dim=-1)

        valid_normals = F.normalize(valid_normals, dim=-1, eps=1e-12)

        if flip_vertex_normals:
            flip_sign = (rast_data.normals * valid_normals).sum(dim=-1, keepdim=True).sign()
            valid_normals.mul_(flip_sign)
    
    
    return MeshRasterizationData(face_ids=rast_data.face_ids, positions=valid_positions, normals=valid_normals)


def sample_grid_attributes(rast_data: MeshRasterizationData, grid: AttributeGrid, mode: SamplingMode = SamplingMode.TRILINEAR) -> AttributesMasked:
    
    voxel_size = grid.voxel_size
    aabb = grid.aabb
    coords = grid.coords
    attr_volume = grid.values

    valid_pos = rast_data.positions
    mask = rast_data.mask.to(grid.values.device)

    mode = SamplingMode(mode)

    attrs = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=grid.dense_shape(),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode=mode.value,
        ).squeeze(0)
    
    return AttributesMasked(values=attrs, mask=mask)


def dilate_attributes(attributes: AttributesMasked, kernel_size: int) -> torch.Tensor:
    """Fill seams by dilating valid pixels into nearby empty UV space."""
    
    if kernel_size <= 1:
        return attributes

    attrs = attributes.to_dense().permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
    mask = attributes.mask.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    invalid = ~mask
    pooled = F.unfold(attrs, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2).view(1, attrs.shape[1], kernel_size * kernel_size, -1)
    mask_unfold = F.unfold(mask.float(), kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2).view(1, 1, kernel_size * kernel_size, -1)
    # Get mean value of valid pixels in the kernel
    summed = pooled.mul_(mask_unfold).sum(dim=-2)
    count = mask_unfold.sum(dim=-2).clamp_min_(1.0)
    pooled = summed.div_(count).view_as(attrs).clamp_min_(0.0)

    filled = torch.where(invalid, pooled, attrs)
    return filled.squeeze(0).permute(1, 2, 0)  # (1, C, H, W) -> (H, W, C)


    
