import torch
from geometry.mesh.schemas import MeshData


def taubin_smooth(mesh_data: MeshData, iterations: int = 5,
                  lambda_factor: float = 0.5, mu_factor: float = -0.53) -> torch.Tensor:
    """GPU-native Taubin smoothing using sparse Laplacian.

    Alternates lambda (shrink) and mu (inflate) steps to smooth the mesh
    without net volume shrinkage.
    """
    # Build undirected edges from faces
    face_edges = torch.stack((mesh_data.faces[:, [0, 1]], mesh_data.faces[:, [1, 2]], mesh_data.faces[:, [2, 0]]), dim=1)
    edges = face_edges.reshape(-1, 2)
    edges = torch.cat([edges, edges.flip(1)], dim=0)  # make undirected

    # Compute vertex degrees
    num_verts = mesh_data.vertices.shape[0]
    ones = torch.ones(edges.shape[0], device=mesh_data.vertices.device, dtype=mesh_data.vertices.dtype)
    degree = torch.zeros(num_verts, device=mesh_data.vertices.device, dtype=mesh_data.vertices.dtype)
    degree.scatter_add_(0, edges[:, 0], ones)
    degree.clamp_(min=1)  # avoid division by zero for isolated vertices

    # Build sparse normalized adjacency D^{-1}A
    inv_degree = 1.0 / degree
    weights = inv_degree[edges[:, 0]]
    adj = torch.sparse_coo_tensor(
        edges.t(), weights, size=(num_verts, num_verts),
        device=mesh_data.vertices.device, dtype=mesh_data.vertices.dtype
    ).coalesce()

    v = mesh_data.vertices.clone()
    for _ in range(iterations):
        # Lambda step (shrink)
        neighbor_avg = torch.sparse.mm(adj, v)
        v = v + lambda_factor * (neighbor_avg - v)
        # Mu step (inflate)
        neighbor_avg = torch.sparse.mm(adj, v)
        v = v + mu_factor * (neighbor_avg - v)

    return v