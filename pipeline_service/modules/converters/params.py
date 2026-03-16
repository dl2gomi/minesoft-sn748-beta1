from typing import TypeAlias

from schemas.overridable import OverridableModel
from geometry.texturing.enums import AlphaMode

class GLBConverterParams(OverridableModel):
    """GLB conversion parameters with automatic fallback to settings."""
    decimation_target: int = 1000000
    texture_size: int = 1024
    # Default to BLEND so transparent materials can be represented.
    # The converter will automatically downgrade to OPAQUE when the
    # baked alpha channel is fully opaque.
    alpha_mode: AlphaMode = AlphaMode.BLEND
    rescale: float = 1.0
    remesh: bool = True
    remesh_band: float = 1.0
    remesh_project: float = 0.0
    mesh_cluster_refine_iterations: int = 0
    mesh_cluster_global_iterations: int = 1
    mesh_cluster_smooth_strength: float = 1.0
    mesh_cluster_threshold_cone_half_angle: float = 90.0
    # UV unwrap guards (all configurable). Above these we use planar UVs instead of CuMesh/xatlas.
    trivial_uv_max_faces: int = 100_000
    trivial_uv_max_vertices: int = 80_000
    max_xatlas_charts: int = 5000
    subdivisions: int = 2
    vertex_reproject: float = 0.0
    alpha_gamma: float = 2.2
    smooth_mesh: bool = False
    smooth_iterations: int = 5
    smooth_lambda: float = 0.5
    # Material appearance tweaks (applied when baking textures)
    roughness_scale: float = 1.0   # scale roughness (e.g. 0.7 = less rough)
    roughness_bias: float = 0.0    # add to roughness after scale (e.g. 0.1 = floor)
    color_saturation: float = 1.0  # 1 = no change, <1 = desaturate
    color_brightness: float = 1.0  # 1 = no change, <1 = darker
    
    
    @classmethod
    def from_settings(cls, settings) -> "GLBConverterParams":
        return cls(**settings.model_dump())


GLBConverterParamsOverrides: TypeAlias = GLBConverterParams.Overrides
