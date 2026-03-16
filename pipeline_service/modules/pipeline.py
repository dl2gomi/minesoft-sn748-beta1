from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib import response

from PIL import Image
from modules.converters.params import GLBConverterParams
from modules.converters.category_presets import get_glb_overrides_for_category
import torch
import gc

from config.settings import SettingsConf, config_dir
from config.category_config import load_category_config
from config.prompting_library import PromptingLibrary
from logger_config import logger
from schemas.requests import GenerationRequest
from schemas.responses import GenerationResponse
from modules.mesh_generator.schemas import TrellisParams, TrellisRequest, TrellisResult
from modules.mesh_generator.enums import TrellisPipeType
from modules.image_edit.edit_module import EditModule
from modules.image_edit.qwen_edit_pipeline import QwenEditPipeline
from modules.image_edit.schemas import ImageGenerationParams
from modules.background_removal.ben2_pipeline import BEN2BackgroundRemovalPipeline
from modules.background_removal.birefnet_pipeline import BirefNetBackgroundRemovalPipeline
from modules.background_removal.background_removal_module import BackgroundRemovalModule
from modules.background_removal.enums import RMBGModelType
from modules.grid_renderer.render import GridViewRenderer
from modules.mesh_generator.mesh_generator_module import MeshGeneratorModule
from modules.mesh_generator.trellis_pipeline import Trellis2MeshPipeline
from modules.converters.glb_converter import GLBConverter, MeshTooLargeForGLB
from modules.judge.duel_manager import DuelManager
from modules.judge.vllm_judge_pipeline import VllmJudgePipeline
from libs.trellis2.representations.mesh.base import MeshWithVoxel
from modules.utils import image_grid, secure_randint, set_random_seed, decode_image, to_png_base64, save_files
from modules.clarifier.clip_clarifier import CLIPClarifier, ClarifierResult
    
class GenerationPipeline:
    """
    Generation pipeline 
    """

    def __init__(self, settings: SettingsConf, renderer: Optional[GridViewRenderer] = None) -> None:
        self.settings = settings
        self.renderer = renderer

        # Initialize modules
        self.qwen_pipeline = QwenEditPipeline(settings.qwen, settings.model_versions)
        self.qwen_edit = EditModule(ImageGenerationParams.from_settings(settings.qwen))
        self.rmbg_module = BackgroundRemovalModule(settings.background_removal)

        # Initialize background removal module
        model_type = self.settings.background_removal.model_type
        if model_type == RMBGModelType.BEN2:
            self.rmbg_pipeline = BEN2BackgroundRemovalPipeline(settings.background_removal, settings.model_versions)
        elif model_type == RMBGModelType.BIREFNET:
            self.rmbg_pipeline = BirefNetBackgroundRemovalPipeline(settings.background_removal, settings.model_versions)
        else:
            raise ValueError(f"Unsupported background removal model: {self.settings.background_removal.model_id}")

        # Initialize prompting libraries for both modes
        self.prompting_library = PromptingLibrary.from_file(settings.qwen.prompt_path_base)

        # Initialize Trellis module
        self.mesh_pipeline = Trellis2MeshPipeline(settings.trellis, settings.model_versions)
        self.mesh_generator = MeshGeneratorModule(TrellisParams.from_settings(settings.trellis))
        self.glb_converter = GLBConverter(settings.glb_converter)

        # Load category config (CLIP prompts + GLB presets) from YAML
        category_config_path = Path(settings.clarifier.category_config_path) if getattr(settings.clarifier, "category_config_path", None) else config_dir.parent / "category_config.yaml"
        self._category_config = load_category_config(category_config_path)
        self._category_glb_presets = self._category_config["glb_presets"]
        category_prompts = self._category_config["categories"]
        category_order = list(category_prompts.keys())
        self._category_confidence_threshold: float = float(self._category_config.get("category_confidence_threshold", 0.45))

        # Initialize Judge module
        if settings.judge.enabled:
            self.judge_pipeline = VllmJudgePipeline(settings.judge)
            self.duel_manager = DuelManager(renderer=renderer)
        else:
            self.judge_pipeline = None
            self.duel_manager = None

        # Initialize clarifier (CLIP-based) for dynamic multiview decisions
        self.clarifier = CLIPClarifier(
            settings.clarifier,
            settings.model_versions,
            category_prompts=category_prompts,
            category_order=category_order,
            category_confidence_threshold=self._category_confidence_threshold,
        )

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_pipeline.startup()
        await self.rmbg_pipeline.startup()
        await self.mesh_pipeline.startup()
        if self.judge_pipeline is not None:
            await self.judge_pipeline.startup()
        if self.settings.clarifier.enabled:
            await self.clarifier.startup()
        
        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()
        
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_pipeline.shutdown()
        await self.rmbg_pipeline.shutdown()
        await self.mesh_pipeline.shutdown()
        if self.judge_pipeline is not None:
            await self.judge_pipeline.shutdown()
        if self.settings.clarifier.enabled:
            await self.clarifier.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""
        
        temp_image = Image.new("RGB",(512,512),color=(128,128,128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(temp_image_bytes).decode("utf-8")

        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=42
        )

        result = await self.generate(request)
        
        if result.glb_file_base64 and self.renderer:
            grid_view_bytes = self.renderer.grid_from_glb_bytes(result.glb_file_base64)
            if not grid_view_bytes:
                logger.warning("Grid view generation failed during warmup")

    async def generate_from_upload(self, image_bytes: bytes, seed: int, input_filename: Optional[str] = None) -> GenerationResponse:
        """
        Generate 3D model from uploaded image file.

        Returns full GenerationResponse (including glb_file_base64 as bytes and
        clarifier/multiview metadata for logging).
        """
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed,
            input_filename=input_filename,
        )

        response = await self.generate(request)
        # Ensure GLB is bytes for streaming; generate() may leave it as bytes already
        if isinstance(response.glb_file_base64, str):
            response.glb_file_base64 = base64.b64decode(response.glb_file_base64)
        return response
        
    def _edit_images(self, image: Image.Image, seed: int) -> tuple[list[Image.Image], Optional[float], bool, Optional[str]]:
        """
        Edit image based on current mode (multiview or base).

        Returns:
            (edited_images, clarifier_score, multiview_used, clarifier_explanation)
            When clarifier is not run, clarifier_score and clarifier_explanation are None.
        """
        clarifier_score: Optional[float] = None
        clarifier_explanation: Optional[str] = None

        # 1 Always canonicalize the input with the base prompt first.
        base_prompt = self.prompting_library.promptings["base"]
        logger.debug(f"Editing base view with prompt: {base_prompt}")
        base_images = list(self.qwen_edit.edit_image(
            model=self.qwen_pipeline,
            prompt_image=image,
            seed=seed,
            prompting=base_prompt
        ))

        # Decide whether multiview is needed based on configuration and clarifier.
        multiview_mode = getattr(self.settings.trellis, "multiview_mode", "off")
        needs_multiview = False

        if multiview_mode == "off":
            needs_multiview = False
            logger.info("Multiview mode: off (single view only).")
        elif multiview_mode == "always":
            needs_multiview = True
            logger.info("Multiview mode: always (force multiview).")
        elif multiview_mode == "dynamic":
            if not self.settings.clarifier.enabled:
                logger.warning("Multiview mode is 'dynamic' but clarifier is disabled; falling back to single-view.")
                needs_multiview = False
            else:
                result: ClarifierResult = self.clarifier.score_image(image)
                clarifier_score = result.reconstructability_score
                clarifier_explanation = result.explanation
                logger.info(
                    f"Clarifier reconstructability_score={result.reconstructability_score:.3f} "
                    f"threshold={self.settings.clarifier.reconstructability_threshold:.3f} "
                    f"explanation={result.explanation!r}"
                )
                needs_multiview = (
                    result.reconstructability_score
                    < self.settings.clarifier.reconstructability_threshold
                )
                if needs_multiview:
                    logger.info("Clarifier decided: multiview is needed for this object.")
                else:
                    logger.info("Clarifier decided: single view is sufficient for this object.")
        else:
            logger.warning(f"Unknown multiview_mode={multiview_mode!r}; treating as 'off'.")
            needs_multiview = False

        if self.settings.trellis.multiview and multiview_mode == "off":
            logger.info("Legacy trellis.multiview=True with multiview_mode='off'; enabling multiview for compatibility.")
            needs_multiview = True

        if not needs_multiview:
            logger.info("Using single-view path after base edit.")
            return base_images, clarifier_score, False, clarifier_explanation

        # 2 Multiview mode: generate views from the base-edited images.
        logger.info("Using multiview path: generating multiple views from base-edited image.")
        views_prompt = self.prompting_library.promptings["views"]

        edited_images: list[Image.Image] = list(base_images)
        for base_image in base_images:
            for prompt_text in views_prompt.prompt:
                logger.debug(f"Editing view with prompt: {prompt_text}")
                result = self.qwen_edit.edit_image(
                    model=self.qwen_pipeline,
                    prompt_image=base_image,
                    seed=seed,
                    prompting=prompt_text,
                )
                edited_images.extend(result)

        return edited_images, clarifier_score, True, clarifier_explanation

    async def generate_mesh(self, request: GenerationRequest) -> tuple[list[MeshWithVoxel], list[Image.Image], list[Image.Image], Optional[float], bool, Optional[str], Optional[str], Optional[float]]:
        """
        Generate mesh from Trellis pipeline, along with processed images and clarifier metadata.

        Returns:
            Tuple of (meshes, images_edited, images_without_background, clarifier_score, multiview_used, clarifier_explanation, object_category, object_category_confidence)
        """
        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
        set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # 1. Edit the image using Qwen Edit
        images_edited, clarifier_score, multiview_used, clarifier_explanation = self._edit_images(image, request.seed)

        # 2. Remove background
        images_with_background = [img.copy() for img in images_edited]
        images_without_background = list(
            self.rmbg_module.remove_background(self.rmbg_pipeline, images_with_background)
        )

        # 2b. Classify main object category (for GLB material presets); use first image after rembg
        object_category: Optional[str] = None
        object_category_confidence: Optional[float] = None
        if self.settings.clarifier.enabled:
            try:
                object_category, object_category_confidence = self.clarifier.classify_category(images_without_background[0])
            except Exception as e:
                logger.warning(f"Category classification failed: {e}; using default GLB params.")
        if object_category:
            logger.info(f"Object category for GLB preset: {object_category}")

        # Optional: predict pipeline_type (512 vs 1024_cascade) from rembg'd image before Trellis
        overrides_dict = request.trellis_params.model_dump(exclude_none=True) if request.trellis_params else {}
        suggested_pipeline = self.clarifier.suggest_pipeline_type(images_without_background[0]) if getattr(self.settings.clarifier, "suggest_pipeline_type", True) else None
        if suggested_pipeline is not None:
            overrides_dict["pipeline_type"] = TrellisPipeType.MODE_512 if suggested_pipeline == "512" else TrellisPipeType.MODE_1024_CASCADE
        trellis_params = TrellisParams.Overrides(**overrides_dict) if overrides_dict else request.trellis_params

        # Store the final Trellis pipeline_type that will actually be used.
        try:
            effective_params = self.mesh_generator.default_params.overrided(trellis_params) if trellis_params else self.mesh_generator.default_params
            self._last_trellis_pipeline_type = getattr(effective_params.pipeline_type, "value", str(effective_params.pipeline_type))
            self._last_trellis_pipeline_type_suggested = suggested_pipeline
        except Exception:
            self._last_trellis_pipeline_type = None
            self._last_trellis_pipeline_type_suggested = suggested_pipeline

        # Temporary debug: save the image list fed to Trellis to files (named after input image)
        # stem = Path(request.input_filename).stem if request.input_filename else datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        # save_dir = self.settings.output.output_dir / "trellis_inputs"
        # save_dir.mkdir(parents=True, exist_ok=True)
        # for i, img in enumerate(images_without_background):
        #     path = save_dir / f"{stem}_trellis_{i}.png"
        #     img.save(path)
        # logger.info(f"Temporary debug: saved {len(images_without_background)} Trellis input images to {save_dir} as {stem}_trellis_*.png")

        # 3. Generate the 3D model (with a guarded retry on Trellis OOM)
        try:
            meshes = self.mesh_generator.generate(
                model=self.mesh_pipeline,
                request=TrellisRequest(
                    image=images_without_background,
                    seed=request.seed,
                    params=trellis_params
                ),
            )
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda" not in msg and "acceleratorerror" not in msg:
                raise
            logger.warning(f"Trellis OOM during mesh generation; retrying once with lighter settings: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Retry once with a lighter Trellis configuration (pipeline_type=512, num_samples=1).
            fallback_overrides = TrellisParams.Overrides(
                pipeline_type=TrellisPipeType.MODE_512,
                num_samples=1,
            )
            self._last_trellis_pipeline_type = TrellisPipeType.MODE_512.value
            meshes = self.mesh_generator.generate(
                model=self.mesh_pipeline,
                request=TrellisRequest(
                    image=images_without_background,
                    seed=request.seed,
                    params=fallback_overrides,
                ),
            )

        return meshes, images_edited, images_without_background, clarifier_score, multiview_used, clarifier_explanation, object_category, object_category_confidence

    def convert_mesh_to_glb(self, mesh: MeshWithVoxel, glbconv_params: Optional[GLBConverterParams.Overrides], object_category: Optional[str] = None) -> bytes:
        """
        Convert mesh to GLB format using GLBConverter.

        Args:
            mesh: The mesh to convert
            glbconv_params: Optional override parameters for GLB conversion (from request)
            object_category: Optional category from classifier (glass, metal, etc.) to apply material preset

        Returns:
            GLB file as bytes
        """
        start_time = time.time()
        # Merge: base config, then category preset, then request overrides
        category_overrides = get_glb_overrides_for_category(object_category, presets=self._category_glb_presets)
        request_dict = glbconv_params.model_dump(exclude_none=True) if glbconv_params else {}

        # Start from global GLB defaults in settings, then layer in presets/overrides.
        base_defaults = GLBConverterParams.from_settings(self.settings.glb_converter)
        merged = {**base_defaults.model_dump(), **category_overrides, **request_dict}
        allowed_keys = set(GLBConverterParams.model_fields)
        merged = {k: v for k, v in merged.items() if k in allowed_keys}

        # Heuristic: derive GLB params from actual mesh size and current VRAM,
        # based on mesh size and VRAM.
        num_vertices = int(mesh.vertices.shape[0])
        num_faces = int(mesh.faces.shape[0])

        free_mem_gb = None
        if torch.cuda.is_available():
            try:
                # mem_get_info: (free, total) in bytes
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_mem_gb = free_bytes / (1024 ** 3)
            except Exception:
                free_mem_gb = None

        # Baseline targets from settings (already in merged), but we may clamp them.
        dec_target = int(merged.get("decimation_target", base_defaults.decimation_target))
        tex_size = int(merged.get("texture_size", base_defaults.texture_size))
        remesh = bool(merged.get("remesh", base_defaults.remesh))
        subdivisions = int(merged.get("subdivisions", base_defaults.subdivisions))

        # Mesh size tiers
        very_large = num_vertices > 1_000_000 or num_faces > 2_000_000
        large = num_vertices > 400_000 or num_faces > 800_000

        low_vram = free_mem_gb is not None and free_mem_gb < 10.0
        very_low_vram = free_mem_gb is not None and free_mem_gb < 6.0

        # Clamp based purely on mesh size + VRAM.
        if very_large or very_low_vram:
            # Heaviest meshes or very low free memory: no remesh, aggressive decimation, no subdivisions.
            remesh = False
            dec_target = min(dec_target, 60_000)
            tex_size = min(tex_size, 1024)
            subdivisions = 0
        elif large or low_vram:
            # Moderately heavy meshes: lighter but still decent quality.
            remesh = remesh and not low_vram  # disable remesh if VRAM is already low
            dec_target = min(dec_target, 100_000)
            tex_size = min(tex_size, 1536)
            subdivisions = min(subdivisions, 1)

        merged["decimation_target"] = int(dec_target)
        merged["texture_size"] = int(tex_size)
        merged["remesh"] = bool(remesh)
        merged["subdivisions"] = int(subdivisions)

        params_filtered = GLBConverterParams.Overrides(**merged)

        def _is_cuda_oom(exc: BaseException) -> bool:
            msg = str(exc).lower()
            return "out of memory" in msg or "cuda" in msg or "cumesh" in msg

        # First attempt: use the merged parameters as-is.
        try:
            glb_mesh = self.glb_converter.convert(mesh, params=params_filtered)
        except (RuntimeError, torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
            if not _is_cuda_oom(e):
                raise

            logger.warning(
                "GLB conversion OOM on first attempt (category=%s); "
                "retrying with lighter params based on mesh size/VRAM: %s",
                object_category,
                e,
            )
            # Build a lighter override that keeps work on GPU but reduces memory pressure.
            base = params_filtered or GLBConverterParams.Overrides()
            lighter = GLBConverterParams.Overrides(
                decimation_target=min(
                    getattr(base, "decimation_target", 120_000) or 120_000,
                    70_000,
                ),
                texture_size=min(
                    getattr(base, "texture_size", 2048) or 2048,
                    1024,
                ),
                remesh=False,
                subdivisions=0,
                vertex_reproject=0.0,
                smooth_mesh=False,
                alpha_mode=getattr(base, "alpha_mode", GLBConverterParams().alpha_mode),
                rescale=getattr(base, "rescale", 1.0),
                remesh_band=getattr(base, "remesh_band", 1.0),
                remesh_project=getattr(base, "remesh_project", 0.0),
                mesh_cluster_refine_iterations=getattr(base, "mesh_cluster_refine_iterations", 0),
                mesh_cluster_global_iterations=getattr(base, "mesh_cluster_global_iterations", 1),
                mesh_cluster_smooth_strength=getattr(base, "mesh_cluster_smooth_strength", 1.0),
                mesh_cluster_threshold_cone_half_angle=getattr(
                    base, "mesh_cluster_threshold_cone_half_angle", 90.0
                ),
                alpha_gamma=getattr(base, "alpha_gamma", 2.2),
                smooth_iterations=getattr(base, "smooth_iterations", 5),
                smooth_lambda=getattr(base, "smooth_lambda", 0.5),
                roughness_scale=getattr(base, "roughness_scale", 1.0),
                roughness_bias=getattr(base, "roughness_bias", 0.0),
                color_saturation=getattr(base, "color_saturation", 1.0),
                color_brightness=getattr(base, "color_brightness", 1.0),
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            glb_mesh = self.glb_converter.convert(mesh, params=lighter)

        buffer = io.BytesIO()
        glb_mesh.export(file_obj=buffer, file_type="glb", extension_webp=False)
        buffer.seek(0)
        
        logger.info(f"GLB conversion time: {time.time() - start_time:.2f}s")
        return buffer.getvalue()

    def prepare_outputs(
        self,
        images_edited: list[Image.Image],
        images_without_background: list[Image.Image],
        glb_trellis_result: Optional[TrellisResult]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Prepare output files: save to disk if configured and generate base64 strings if needed.

        Args:
            images_edited: List of edited images
            images_without_background: List of images with background removed
            glb_trellis_result: Generated GLB result (optional)

        Returns:
            Tuple of (image_edited_base64, image_without_background_base64)
        """
        start_time = time.time()
        # Create grid images once for both save and send operations
        image_edited_grid = image_grid(images_edited)
        image_without_background_grid = image_grid(images_without_background)

        # Save generated files if configured
        if self.settings.output.save_generated_files:
            save_files(glb_trellis_result, image_edited_grid, image_without_background_grid)

        # Convert to PNG base64 for response if configured
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.output.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited_grid)
            image_without_background_base64 = to_png_base64(image_without_background_grid)
            
        logger.info(f"Output preparation time: {time.time() - start_time:.2f}s")

        return image_edited_base64, image_without_background_base64

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Execute full generation pipeline with output types.
        
        Args:
            request: Generation request with prompt and settings
            
        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"Request received | Seed: {request.seed} | Prompt Type: {request.prompt_type.value}")

        # Generate mesh and get processed images
        meshes, images_edited, images_without_background, clarifier_score, multiview_used, clarifier_explanation, object_category, object_category_confidence = await self.generate_mesh(request)

        glb_trellis_result = None
        
        self._clean_gpu_memory()

        # Convert meshes to GLB
        glb_trellis_results: list[TrellisResult] = []
        uv_infos: list[dict | None] = []
        glb_sizes: list[int] = []
        if meshes:
            for mesh in meshes:
                try:
                    glb_bytes = self.convert_mesh_to_glb(mesh, request.glbconv_params, object_category)
                except MeshTooLargeForGLB as e:
                    logger.warning(f"Skipping GLB candidate due to oversized mesh: {e}")
                    continue
                uv_infos.append(getattr(self.glb_converter, "last_uv_unwrap_info", None))
                glb_trellis_results.append(TrellisResult(file_bytes=glb_bytes))
                glb_sizes.append(len(glb_bytes) if isinstance(glb_bytes, (bytes, bytearray)) else 0)

        # Decide whether to run the judge based on GLB sizes
        winner_index: int = 0
        max_bytes = getattr(self.settings.judge, "max_glb_bytes_for_judge", 0) or 0
        too_large_indices = {i for i, sz in enumerate(glb_sizes) if max_bytes > 0 and sz > max_bytes}

        if not glb_trellis_results:
            raise RuntimeError("No GLB candidates were produced from Trellis meshes.")

        if glb_trellis_results:
            if too_large_indices:
                # At least one candidate is too large: skip judge entirely and ignore those candidates.
                logger.info(
                    "Skipping judge due to large GLB(s): indices=%s, sizes=%s, threshold=%d bytes",
                    sorted(too_large_indices),
                    [glb_sizes[i] for i in sorted(too_large_indices)],
                    max_bytes,
                )
                # Pick the smallest remaining candidate by size (or overall smallest if all are too large).
                candidate_indices = [i for i in range(len(glb_trellis_results)) if i not in too_large_indices]
                if not candidate_indices:
                    candidate_indices = list(range(len(glb_trellis_results)))
                winner_index = min(candidate_indices, key=lambda i: glb_sizes[i] if glb_sizes[i] else float("inf"))
                glb_trellis_result = glb_trellis_results[winner_index]
            elif self.duel_manager and self.judge_pipeline:
                # All candidates within size budget: run judge as usual.
                winner_index = await self.duel_manager.judge_meshes(
                    self.judge_pipeline, glb_trellis_results, request.prompt_image, request.seed
                )
                glb_trellis_result = glb_trellis_results[winner_index]
            else:
                glb_trellis_result = glb_trellis_results[0]

        # Save generated files
        image_edited_base64, image_no_bg_base64 = None, None
        if self.settings.output.save_generated_files or self.settings.output.send_generated_files:
            image_edited_base64, image_no_bg_base64 = self.prepare_outputs(
                images_edited,
                images_without_background,
                glb_trellis_result
            )

        t2 = time.time()
        generation_time = t2 - t1

        logger.success(f"Generation time: {generation_time:.2f}s")

        # Pick UV info for the winner (if any)
        winner_uv = uv_infos[winner_index] if (uv_infos and 0 <= winner_index < len(uv_infos)) else None

        # Clean the GPU memory
        self._clean_gpu_memory()

        return GenerationResponse(
            generation_time=generation_time,
            glb_file_base64=glb_trellis_result.file_bytes if glb_trellis_result else None,
            image_edited_file_base64=image_edited_base64,
            image_without_background_file_base64=image_no_bg_base64,
            clarifier_score=clarifier_score,
            multiview_used=multiview_used,
            clarifier_explanation=clarifier_explanation,
            object_category=object_category,
            object_category_confidence=object_category_confidence,
            trellis_pipeline_type=getattr(self, "_last_trellis_pipeline_type", None),
            suggested_pipeline_type=getattr(self, "_last_trellis_pipeline_type_suggested", None),
            uv_unwrap_mode=(winner_uv or {}).get("mode") if isinstance(winner_uv, dict) else None,
            uv_unwrap_reason=(winner_uv or {}).get("reason") if isinstance(winner_uv, dict) else None,
            uv_num_charts=(winner_uv or {}).get("num_charts") if isinstance(winner_uv, dict) else None,
        )
