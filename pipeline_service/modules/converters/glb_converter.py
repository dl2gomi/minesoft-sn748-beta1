import time
from typing import Dict, Optional, Tuple
from logger_config import logger
import numpy as np
import torch
import kaolin
from PIL import Image
import trimesh
import trimesh.visual
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from libs.trellis2.representations.mesh.base import MeshWithVoxel
from geometry.mesh.schemas import MeshData, MeshDataWithAttributeGrid, AttributeGrid
from geometry.texturing.dithering import bayer_dither_pattern
from geometry.mesh.utils import sort_mesh, map_vertices_positions, count_boundary_loops
from geometry.mesh.subdivisions import subdivide_egdes
from geometry.mesh.smoothing import taubin_smooth
from geometry.texturing.utils import dilate_attributes, map_mesh_rasterization, rasterize_mesh_data, sample_grid_attributes
from geometry.texturing.enums import AlphaMode
from .params import GLBConverterParams
from .settings import GLBConverterConfig
import cumesh


DITHER_PATTERN_SIZE = 16
DITHER_PATTERN = bayer_dither_pattern(4096, 4096, DITHER_PATTERN_SIZE)


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True if the exception looks like a CUDA/CuMesh OOM."""
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg or "cumesh" in msg


def _trivial_uvs_and_normals_gpu(vertices: torch.Tensor, faces: torch.Tensor, device: torch.device) -> MeshData:
    """Build MeshData with trivial planar UVs and vertex normals on GPU (no CuMesh/xatlas)."""
    vertices = vertices.to(device)
    faces = faces.long().to(device)
    uv_xy = vertices[:, :2]
    lo, hi = uv_xy.min().item(), uv_xy.max().item()
    span = max(hi - lo, 1e-8)
    uvs = ((uv_xy - lo) / span).to(vertices.dtype)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    fn = F.normalize(torch.linalg.cross(e1, e2, dim=-1), dim=-1)
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.index_add_(0, faces[:, 0], fn)
    vertex_normals.index_add_(0, faces[:, 1], fn)
    vertex_normals.index_add_(0, faces[:, 2], fn)
    vertex_normals = F.normalize(vertex_normals, dim=-1)
    return MeshData(vertices=vertices, faces=faces, vertex_normals=vertex_normals, uvs=uvs)


class MeshTooLargeForGLB(Exception):
    """Raised when a mesh is too large to safely convert to GLB."""
    pass


class GLBConverter:
    """Converter for extracting and texturing meshes to GLB format."""
    DEFAULT_AABB = torch.as_tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
    DILATION_KERNEL_SIZE = 5
    
    def __init__(self, settings: GLBConverterConfig):
        """Initialize converter with settings."""
        self.default_params = GLBConverterParams.from_settings(settings)
        self.logger = logger
        self.device = torch.device(f'cuda:{settings.gpu}' if torch.cuda.is_available() else 'cpu')
        # Populated during convert() for the most recent mesh.
        # Shape: {"mode": "xatlas"|"trivial", "reason": str|None, "num_charts": int|None}
        self.last_uv_unwrap_info: dict | None = None
    
    def convert(self, mesh: MeshWithVoxel, aabb: torch.Tensor = DEFAULT_AABB, params: GLBConverterParams = None) -> trimesh.Trimesh:
        """Convert the given mesh to a textured GLB format."""
        logger.debug(f"Original mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
        
        params = self.default_params.overrided(params)
        logger.debug(f"Using GLB conversion parameters: {params}")
        self.last_uv_unwrap_info = None

         # Hard guard: if the mesh is extremely large, skip GLB generation for this candidate.
        num_vertices = int(mesh.vertices.shape[0])
        num_faces = int(mesh.faces.shape[0])
        if num_vertices > params.max_vertices_for_glb or num_faces > params.max_faces_for_glb:
            logger.warning(
                "Skipping GLB conversion for oversized mesh: vertices=%d faces=%d "
                "(limits: %d verts, %d faces)",
                num_vertices,
                num_faces,
                params.max_vertices_for_glb,
                params.max_faces_for_glb,
            )
            raise MeshTooLargeForGLB(
                f"Mesh too large for GLB (vertices={num_vertices}, faces={num_faces}, "
                f"limits={params.max_vertices_for_glb}/{params.max_faces_for_glb})"
            )

        # 1. Prepare original mesh data with BVH
        original_mesh_data = self._prepare_original_mesh(mesh, aabb)
        
        # 2. Remesh if required (cleanup otherwise)
        if params.remesh:
            mesh_data = self._remesh_mesh(original_mesh_data, params)
        else:
            mesh_data = self._cleanup_mesh(original_mesh_data, params)
            
        # 3. UV unwrap the mesh
        mesh_data = self._uv_unwrap_mesh(mesh_data, params)

        # 4. subdivide unwrapped mesh
        mesh_data = self._subdivide_mesh(mesh_data, original_mesh_data, params)
        
        # 5. Rasterize attributes onto the mesh UVs
        attributes_layout = mesh.layout
        attributes, attributes_layout = self._rasterize_attributes(mesh_data, original_mesh_data, attributes_layout, params)
        
        # 6. Post-process the rasterized attributes into textures
        base_color, orm_texture = self._texture_postprocess(attributes, attributes_layout, params)

        # 7. Create the textured mesh
        textured_mesh = self._create_textured_mesh(mesh_data, base_color, orm_texture, params)

        return textured_mesh


    def _prepare_original_mesh(self, mesh: MeshWithVoxel, aabb: torch.Tensor, compute_vertex_normals: bool = False) -> MeshDataWithAttributeGrid:
        """
        Convert MeshWithVoxel to OriginalMeshData.
        WARNING: This method also fills holes outputing additional faces compared to input one.
        """
        logger.debug(f"Preparing original mesh data")
        start_time = time.time() 

        # Prepare attribute grid
        attrs = AttributeGrid(
            values=mesh.attrs.to(self.device),
            coords=mesh.coords.to(self.device),
            aabb = torch.as_tensor(aabb, dtype=torch.float32, device=self.device),
            voxel_size = torch.as_tensor(mesh.voxel_size, dtype=torch.float32, device=self.device).broadcast_to(3)
        )

        vertices = mesh.vertices.to(self.device)
        faces = mesh.faces.to(self.device)

        vertex_normals = None
        try:
            cumesh_mesh = cumesh.CuMesh()
            cumesh_mesh.init(vertices, faces)
            cumesh_mesh.fill_holes(max_hole_perimeter=3e-2)
            logger.debug(f"After filling holes: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
            vertices, faces = cumesh_mesh.read()
            if compute_vertex_normals:
                cumesh_mesh.compute_vertex_normals()
                vertex_normals = cumesh_mesh.read_vertex_normals()
        except RuntimeError as e:
            if _is_cuda_oom(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(f"CuMesh OOM in _prepare_original_mesh, skipping fill_holes: {e}")
            else:
                raise

        original_mesh_data = MeshDataWithAttributeGrid(vertices=vertices, faces=faces, vertex_normals=vertex_normals, attrs=attrs)
        
        # Build BVH for the current mesh to guide remeshing
        logger.debug(f"Building BVH for current mesh...")
        original_mesh_data.build_bvh()
        logger.debug(f"Done building BVH | Time: {time.time() - start_time:.2f}s")
        
        return original_mesh_data

    def _cleanup_mesh(self, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Cleanup and optimize the mesh using decimation and remeshing."""
        try:
            cumesh_mesh = cumesh.CuMesh()
            cumesh_mesh.init(original_mesh_data.vertices, original_mesh_data.faces)

            if params.smooth_mesh:
                logger.debug(f"Applying Taubin smoothing: iterations={params.smooth_iterations}, lambda={params.smooth_lambda}")
                smoothed_vertices = taubin_smooth(
                    original_mesh_data,
                    iterations=params.smooth_iterations,
                    lambda_factor=params.smooth_lambda,
                    mu_factor=-(params.smooth_lambda + 0.01),
                )
                cumesh_mesh.init(smoothed_vertices, original_mesh_data.faces)

            cumesh_mesh.simplify(params.decimation_target * 3, verbose=False)
            logger.debug(f"After initial simplification: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

            cumesh_mesh.remove_duplicate_faces()
            cumesh_mesh.repair_non_manifold_edges()
            cumesh_mesh.remove_small_connected_components(1e-5)
            cumesh_mesh.fill_holes(max_hole_perimeter=3e-2)
            logger.debug(f"After initial cleanup: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

            cumesh_mesh.simplify(params.decimation_target, verbose=False)
            logger.debug(f"After final simplification: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

            cumesh_mesh.remove_duplicate_faces()
            cumesh_mesh.repair_non_manifold_edges()
            cumesh_mesh.remove_small_connected_components(1e-5)
            cumesh_mesh.fill_holes(max_hole_perimeter=3e-2)
            logger.debug(f"After final cleanup: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

            hole_count = count_boundary_loops(*cumesh_mesh.read())
            logger.debug(f"Holes after cleanup: {hole_count}")

            cumesh_mesh.unify_face_orientations()

            vertices, faces = cumesh_mesh.read()
            return MeshData(vertices=vertices, faces=faces)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(f"CuMesh OOM in _cleanup_mesh, returning mesh without cleanup: {e}")
                return MeshData(
                    vertices=original_mesh_data.vertices,
                    faces=original_mesh_data.faces,
                )
            raise

    def _remesh_mesh(self, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Remesh the given mesh to improve quality."""
        logger.debug("Starting remeshing")
        start_time = time.time()

        try:
            cumesh_mesh = cumesh.CuMesh()
            cumesh_mesh.init(original_mesh_data.vertices, original_mesh_data.faces)

            if params.smooth_mesh:
                logger.debug(f"Applying Taubin smoothing: iterations={params.smooth_iterations}, lambda={params.smooth_lambda}")
                smoothed_vertices = taubin_smooth(
                    original_mesh_data,
                    iterations=params.smooth_iterations,
                    lambda_factor=params.smooth_lambda,
                    mu_factor=-(params.smooth_lambda + 0.01),
                )
                cumesh_mesh.init(smoothed_vertices, original_mesh_data.faces)
                logger.debug(f"Done smoothing | Time: {time.time() - start_time:.2f}s")

            voxel_size = original_mesh_data.attrs.voxel_size
            aabb = original_mesh_data.attrs.aabb
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()

            resolution = grid_size.max().item()
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()

            vertices, faces = cumesh.remeshing.remesh_narrow_band_dc(
                *cumesh_mesh.read(),
                center=center,
                scale=(resolution + 3 * params.remesh_band) / resolution * scale,
                resolution=resolution,
                band=params.remesh_band,
                project_back=params.remesh_project,
                verbose=False,
                bvh=original_mesh_data.bvh,
            )
            cumesh_mesh.init(vertices, faces)
            logger.debug(f"After remeshing: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

            cumesh_mesh.simplify(params.decimation_target, verbose=False)
            logger.debug(f"After simplifying: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

            vertices, faces = cumesh_mesh.read()
            hole_count = count_boundary_loops(vertices, faces)
            logger.debug(f"Holes after remesh: {hole_count}")

            logger.debug(f"Done remeshing | Time: {time.time() - start_time:.2f}s")
            return MeshData(vertices=vertices, faces=faces)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(f"CuMesh OOM in _remesh_mesh, falling back to cleanup only: {e}")
                return self._cleanup_mesh(original_mesh_data, params)
            raise

    def _uv_unwrap_mesh(self, mesh_data: MeshData, params: GLBConverterParams) -> MeshData:
        """Perform UV unwrapping on the mesh."""
        logger.debug("Starting UV unwrapping")
        start_time = time.time()

        num_vertices = mesh_data.vertices.shape[0]
        num_faces = mesh_data.faces.shape[0]
        if num_faces > params.trivial_uv_max_faces or num_vertices > params.trivial_uv_max_vertices:
            logger.warning(
                f"Mesh too large for CuMesh/xatlas (vertices={num_vertices}, faces={num_faces} "
                f"> {params.trivial_uv_max_vertices}/{params.trivial_uv_max_faces}); using trivial UVs."
            )
            self.last_uv_unwrap_info = {"mode": "trivial", "reason": "size", "num_charts": None}
            return _trivial_uvs_and_normals_gpu(mesh_data.vertices, mesh_data.faces, self.device)

        compute_charts_kwargs = {
            "threshold_cone_half_angle_rad": np.radians(params.mesh_cluster_threshold_cone_half_angle),
            "refine_iterations": params.mesh_cluster_refine_iterations,
            "global_iterations": params.mesh_cluster_global_iterations,
            "smooth_strength": params.mesh_cluster_smooth_strength,
        }
        xatlas_compute_charts_kwargs = {
            "max_chart_area": 1.0,
            "max_boundary_length": 2.0,
            "max_cost": 10.0,
            "normal_seam_weight": 5.0,
            "normal_deviation_weight": 1.0,
            "fix_winding": True
        }

        try:
            cumesh_mesh = cumesh.CuMesh()
            cumesh_mesh.init(mesh_data.vertices, mesh_data.faces)
            # Get cluster count from the mesh (same clustering CuMesh uses); skip xatlas if too many to avoid segfault.
            cumesh_mesh.compute_charts(**compute_charts_kwargs)
            num_charts, _, _, _, _, _ = cumesh_mesh.read_atlas_charts()
            if num_charts > params.max_xatlas_charts:
                logger.warning(
                    f"Chart count {num_charts} > max_xatlas_charts ({params.max_xatlas_charts}); "
                    "using trivial UVs to avoid xatlas segfault."
                )
                self.last_uv_unwrap_info = {"mode": "trivial", "reason": "charts", "num_charts": int(num_charts)}
                return _trivial_uvs_and_normals_gpu(mesh_data.vertices, mesh_data.faces, self.device)

            out_vertices, out_faces, out_uvs, out_vmaps = cumesh_mesh.uv_unwrap(
                compute_charts_kwargs=compute_charts_kwargs,
                xatlas_compute_charts_kwargs=xatlas_compute_charts_kwargs,
                return_vmaps=True,
                verbose=True,
            )
            out_vertices = out_vertices.to(self.device)
            out_faces = out_faces.to(self.device)
            out_uvs = out_uvs.to(self.device)
            out_vmaps = out_vmaps.to(self.device)

            cumesh_mesh.compute_vertex_normals()
            out_normals = cumesh_mesh.read_vertex_normals()[out_vmaps]

            logger.debug(f"Done UV unwrapping | Time: {time.time() - start_time:.2f}s")
            self.last_uv_unwrap_info = {"mode": "xatlas", "reason": None, "num_charts": int(num_charts)}

            return MeshData(
                vertices=out_vertices,
                faces=out_faces,
                vertex_normals=out_normals,
                uvs=out_uvs
            )
        except RuntimeError as e:
            if _is_cuda_oom(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(
                    f"CuMesh OOM in _uv_unwrap_mesh (atlas/compute_charts); "
                    f"propagating for higher-level retry: {e}"
                )
                # Let the caller decide how to retry with lighter params.
                raise
            raise

    def _subdivide_mesh(self, mesh_data: MeshData, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Subdivide mesh with uv data and optionally reproject vertices to original mesh surface."""
        subdivided_mesh = subdivide_egdes(mesh_data, iterations=params.subdivisions)
        
        if params.vertex_reproject > 0.0:
            subdivided_mesh = map_vertices_positions(subdivided_mesh, original_mesh_data, weight=params.vertex_reproject, inplace=True)

        subdivided_mesh = sort_mesh(subdivided_mesh, axes=(2,1,0))
        
        return subdivided_mesh

    def _rasterize_attributes(self, mesh_data: MeshData, original_mesh_data: MeshDataWithAttributeGrid, layout: Dict[str,slice], params: GLBConverterParams) -> Tuple[torch.Tensor, Dict[str,slice]]:
        """Rasterize the given attributes onto the mesh UVs."""
        logger.debug("Sampling attributes(Texture rasterization)")
        start_time = time.time()

        # Rasterize mesh surface
        rast_data = rasterize_mesh_data(mesh_data, params.texture_size, use_vertex_normals=True)
        logger.debug(f"Texture baking: sampling {rast_data.positions.shape[0]} valid pixels out of {params.texture_size * params.texture_size}")
        logger.debug(f"Attribute volume has {original_mesh_data.attrs.values.shape[0]} voxels")

        # Map these positions back to the *original* high-res mesh to get accurate attributes
        # This corrects geometric errors introduced by simplification/remeshing
        rast_data = map_mesh_rasterization(rast_data, original_mesh_data, flip_vertex_normals=True)

        # Trilinear sampling from the attribute volume (Color, Material props)
        attributes = sample_grid_attributes(rast_data, original_mesh_data.attrs)

        # Fill seams by dilating valid pixels into nearby empty UV space
        attrs = dilate_attributes(attributes, self.DILATION_KERNEL_SIZE)

        logger.debug(f"Done attribute sampling | Time: {time.time() - start_time:.2f}s")
        
        return attrs, layout
    
    def _texture_postprocess(self, attributes: torch.Tensor, attr_layout: Dict, params: GLBConverterParams) -> Tuple[Image.Image, Image.Image, Optional[Image.Image]]:
        """Post-process the rasterized attributes into final textures."""
        logger.debug("Finalizing mesh textures")
        start_time = time.time()
        
        # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
        base_color = attributes[..., attr_layout['base_color']]
        metallic = attributes[..., attr_layout['metallic']]
        roughness = attributes[..., attr_layout['roughness']]
        alpha = attributes[..., attr_layout['alpha']]
        occlusion_channel  = torch.ones_like(metallic)

        # Reduce apparent roughness so materials look less flat/matte
        roughness = (roughness * params.roughness_scale + params.roughness_bias).clamp(0.0, 1.0)

        # Reduce brightness and/or saturation of base color if requested
        if params.color_brightness != 1.0:
            base_color = (base_color * params.color_brightness).clamp(0.0, 1.0)
        if params.color_saturation != 1.0:
            # Luminance (linear weights)
            luma = base_color[..., 0:1] * 0.2126 + base_color[..., 1:2] * 0.7152 + base_color[..., 2:3] * 0.0722
            base_color = (luma + (base_color - luma) * params.color_saturation).clamp(0.0, 1.0)

        # Adjust alpha with gamma
        alpha = alpha.pow(params.alpha_gamma)
        
        # Handle alpha mode
        alpha_mode = params.alpha_mode
        if alpha_mode == AlphaMode.BLEND:
            # If the baked alpha is effectively fully opaque everywhere, we can
            # safely downgrade to OPAQUE to avoid unnecessary blending cost.
            # alpha is a torch.Tensor in [0, 1], so compare against ~1.0.
            is_fully_opaque = bool(torch.all(alpha >= 0.9999).item())
            if is_fully_opaque:
                alpha_mode = AlphaMode.OPAQUE

        # Apply alpha dithering if flag is set
        if alpha_mode == AlphaMode.DITHER:
            h, w = alpha.shape[:2]
            dither_pattern = torch.as_tensor(DITHER_PATTERN[:h, :w, None], device=alpha.device)
            alpha = (alpha > dither_pattern).float()
            logger.debug(f"Dithered alpha channel has {np.sum(alpha == 0)} transparent pixels out of {alpha.size} total pixels")
            # alpha_mode = AlphaMode.MASK : After dithering, treat as MASK

        rgba = torch.cat([base_color, alpha], dim=-1).clamp(0,1)
        orm = torch.cat([occlusion_channel, roughness, metallic], dim=-1).clamp(0,1)

        base_color_texture = to_pil_image(rgba.permute(2,0,1).cpu())
        orm_texture = to_pil_image(orm.permute(2,0,1).cpu())
        
        logger.debug(f"Done finalizing mesh textures | Time: {time.time() - start_time:.2f}s")
        return base_color_texture, orm_texture

    def _create_textured_mesh(self, mesh_data: MeshData, base_color: Image.Image, orm_texture: Image.Image, params: GLBConverterParams) -> trimesh.Trimesh:
        """Create a textured trimesh mesh from the mesh data and textures."""
        
        logger.debug("Creating textured mesh")
        start_time = time.time()

        alpha_mode = params.alpha_mode
        alpha_mode = AlphaMode.MASK if alpha_mode is AlphaMode.DITHER else alpha_mode

        # Create PBR material
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=base_color,
            baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
            metallicRoughnessTexture=orm_texture,
            roughnessFactor=1.0,
            metallicFactor=1.0,
            alphaMode=alpha_mode.value,
            alphaCutoff=alpha_mode.cutoff,
            doubleSided=bool(not params.remesh)
        )
        
        # --- Coordinate System Conversion & Final Object ---
        vertices_np = mesh_data.vertices.mul(params.rescale).cpu().numpy()
        faces_np = mesh_data.faces.cpu().numpy()
        uvs_np = mesh_data.uvs.cpu().numpy()
        normals_np = mesh_data.vertex_normals.cpu().numpy()
        
        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1]  # Flip UV V-coordinate
        
        textured_mesh = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
        )
        
        logger.debug(f"Done creating textured mesh | Time: {time.time() - start_time:.2f}s")
                
        return textured_mesh
