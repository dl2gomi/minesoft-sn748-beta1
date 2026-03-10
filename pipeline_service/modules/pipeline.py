from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional
from urllib import response

from PIL import Image
from modules.converters.params import GLBConverterParams
import torch
import gc

from config.settings import SettingsConf
from config.prompting_library import PromptingLibrary
from logger_config import logger
from schemas.requests import GenerationRequest
from schemas.responses import GenerationResponse
from modules.mesh_generator.schemas import TrellisParams, TrellisRequest, TrellisResult
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
from modules.converters.glb_converter import GLBConverter
from modules.judge.duel_manager import DuelManager
from modules.judge.vllm_judge_pipeline import VllmJudgePipeline
from libs.trellis2.representations.mesh.base import MeshWithVoxel
from modules.utils import image_grid, secure_randint, set_random_seed, decode_image, to_png_base64, save_files
    
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

        # Initialize Judge module
        if settings.judge.enabled:
            self.judge_pipeline = VllmJudgePipeline(settings.judge)
            self.duel_manager = DuelManager(renderer=renderer)
        else:
            self.judge_pipeline = None
            self.duel_manager = None
        
    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_pipeline.startup()
        await self.rmbg_pipeline.startup()
        await self.mesh_pipeline.startup()
        if self.judge_pipeline is not None:
            await self.judge_pipeline.startup()
        
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

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return GLB as bytes.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            seed: Random seed for generation
            output_type: Desired output type (MESH) (default: MESH)
            
        Returns:
            GLB file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create request
        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed
        )

        response = await self.generate(request)
        
        return response.glb_file_base64 # bytes
        
    def _edit_images(self, image: Image.Image, seed: int) -> list[Image.Image]:
        """
        Edit image based on current mode (multiview or base).
        
        Args:
            image: Input image to edit
            seed: Random seed for reproducibility
            
        Returns:
            List of edited images
        """


        if self.settings.trellis.multiview:
            logger.info("Multiview mode: generating multiple views")
            views_prompt = self.prompting_library.promptings['views']

            edited_images = []
            for prompt_text in views_prompt.prompt:
                logger.debug(f"Editing view with prompt: {prompt_text}")
                result = self.qwen_edit.edit_image(
                    model=self.qwen_pipeline,
                    prompt_image=image,
                    seed=seed,
                    prompting=prompt_text
                )
                edited_images.extend(result)
                
            edited_images.append(image.copy()) # Original image

            return edited_images
        
        # Base mode: only clean background, single view (1 image)
        logger.info("Base mode: single view with background cleaning and rotation")
        base_prompt = self.prompting_library.promptings['base']
        logger.debug(f"Editing base view with prompt: {base_prompt}")
        return self.qwen_edit.edit_image(
            model=self.qwen_pipeline,
            prompt_image=image,
            seed=seed,
            prompting=base_prompt
        )

    async def generate_mesh(self, request: GenerationRequest) -> tuple[list[MeshWithVoxel], list[Image.Image], list[Image.Image]]:
        """
        Generate mesh from Trellis pipeline, along with processed images.

        Args:
            request: Generation request with prompt and settings

        Returns:
            Tuple of (mesh, images_edited, images_without_background)
        """
        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
        set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # 1. Edit the image using Qwen Edit
        images_edited = list(self._edit_images(image, request.seed))
   
        # 2. Remove background
        images_with_background = list(image.copy() for image in images_edited)
        images_without_background = list(
            self.rmbg_module.remove_background(self.rmbg_pipeline, images_with_background)
        )

        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params
       
        # 3. Generate the 3D model
        meshes = self.mesh_generator.generate(
            model=self.mesh_pipeline,
            request=TrellisRequest(
                image=images_without_background,
                seed=request.seed,
                params=trellis_params
            ),
        )

        return meshes, images_edited, images_without_background

    def convert_mesh_to_glb(self, mesh: MeshWithVoxel, glbconv_params: GLBConverterParams) -> bytes:
        """
        Convert mesh to GLB format using GLBConverter.

        Args:
            mesh: The mesh to convert
            glbconv_params: Optional override parameters for GLB conversion

        Returns:
            GLB file as bytes
        """
        start_time = time.time()
        glb_mesh = self.glb_converter.convert(mesh, params=glbconv_params)

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
        meshes, images_edited, images_without_background = await self.generate_mesh(request)

        glb_trellis_result = None
        
        self._clean_gpu_memory()

        # Convert meshes to GLB
        glb_trellis_results = []
        if meshes:
            for mesh in meshes:
                glb_bytes = self.convert_mesh_to_glb(mesh, request.glbconv_params)
                glb_trellis_results.append(TrellisResult(file_bytes=glb_bytes))

        # Judge the meshes
        if self.duel_manager and self.judge_pipeline:
            mesh_winner_index = await self.duel_manager.judge_meshes(self.judge_pipeline, glb_trellis_results, request.prompt_image, request.seed)
            glb_trellis_result = glb_trellis_results[mesh_winner_index]
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

        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerationResponse(
            generation_time=generation_time,
            glb_file_base64=glb_trellis_result.file_bytes if glb_trellis_result else None,
            image_edited_file_base64=image_edited_base64,
            image_without_background_file_base64=image_no_bg_base64
        )
        
        return response
