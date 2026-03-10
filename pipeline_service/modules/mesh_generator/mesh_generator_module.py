from __future__ import annotations

import time

import torch
from PIL import Image
from libs.trellis2.representations.mesh.base import MeshWithVoxel

from logger_config import logger

from .mesh_generation_pipeline import MeshGenerationPipeline
from .schemas import TrellisParams, TrellisRequest
from modules.utils import set_random_seed


class MeshGeneratorModule:
    """Runs mesh generation using a provided loaded pipeline."""

    def __init__(self, default_params: TrellisParams):
        self.default_params = default_params

    def generate(self, model: MeshGenerationPipeline, request: TrellisRequest) -> list[MeshWithVoxel]:
        assert model.is_ready(), f"{model.settings.model_id} pipeline not loaded."

        set_random_seed(request.seed)

        images = [request.image] if isinstance(request.image, Image.Image) else list(request.image)
        images_rgb = [image.convert("RGB") for image in images]
        num_images = len(images_rgb)

        params = self.default_params.overrided(request.params)

        logger.info(
            f"Generating Trellis {request.seed=} and image size {images[0].size} "
            f"(Using {num_images} images) | Pipeline: {params.pipeline_type.value} "
            f"| Max Tokens: {params.max_num_tokens} | "
            f"{'Mode: ' + params.mode.value if params.mode.value else ''} | "
            f"Num Samples: {params.num_samples}"
        )
        logger.debug(f"Trellis generation parameters: {params}")

        start = time.time()
        try:
            meshes = model.loaded_pipeline.run_multi_image(
                images=images_rgb,
                seed=request.seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "guidance_strength": params.sparse_structure_cfg_strength,
                },
                shape_slat_sampler_params={
                    "steps": params.shape_slat_steps,
                    "guidance_strength": params.shape_slat_cfg_strength,
                },
                tex_slat_sampler_params={
                    "steps": params.tex_slat_steps,
                    "guidance_strength": params.tex_slat_cfg_strength,
                },
                mode=params.mode,
                pipeline_type=params.pipeline_type,
                max_num_tokens=params.max_num_tokens,
                num_samples=params.num_samples,
            )

            for mesh in meshes:
                mesh.simplify()

            generation_time = time.time() - start
            logger.success(f"{model.settings.model_id} finished in {generation_time:.2f}s. {len(meshes)} meshes generated.")
            return meshes
        finally:
            torch.cuda.empty_cache()
