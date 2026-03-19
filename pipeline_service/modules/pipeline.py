from __future__ import annotations

import base64
import io
import json
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from openai import AsyncOpenAI
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
from modules.converters.glb_converter import GLBConverter, MeshTooLargeForGLB, MeshTooManyChartsForGLB
from modules.judge.duel_manager import DuelManager
from modules.judge.vllm_judge_pipeline import VllmJudgePipeline
from libs.trellis2.representations.mesh.base import MeshWithVoxel
from modules.utils import image_grid, secure_randint, set_random_seed, decode_image, to_png_base64, save_files
from modules.decision import DecisionResponse, decide as vllm_decide
from modules.decision.schemas import DEFAULT_DECISION

def _pad4(b: bytes) -> bytes:
    pad = (-len(b)) % 4
    return b + (b" " * pad)


def _glb_with_tangents(glb_bytes: bytes, *, tangents_v4_f32: bytes, vertex_count: int) -> bytes:
    """
    Inject a glTF-standard TANGENT attribute into a GLB.

    Expects:
    - glb_bytes: valid GLB (glTF 2.0) with JSON + BIN chunks.
    - tangents_v4_f32: raw bytes of float32 tangents, shape (V,4) => length = V*16
    - vertex_count: V (must match POSITION accessor count)
    """
    if len(glb_bytes) < 20:
        return glb_bytes

    # GLB header
    magic, version, length = struct.unpack_from("<4sII", glb_bytes, 0)
    if magic != b"glTF" or version != 2:
        return glb_bytes

    # Read first two chunks (JSON, BIN)
    off = 12
    if off + 8 > len(glb_bytes):
        return glb_bytes
    json_len, json_type = struct.unpack_from("<I4s", glb_bytes, off)
    off += 8
    if json_type != b"JSON":
        return glb_bytes
    json_chunk = glb_bytes[off : off + json_len]
    off += json_len
    off = (off + 3) & ~3

    if off + 8 > len(glb_bytes):
        return glb_bytes
    bin_len, bin_type = struct.unpack_from("<I4s", glb_bytes, off)
    off += 8
    if bin_type != b"BIN\x00":
        return glb_bytes
    bin_chunk = glb_bytes[off : off + bin_len]

    try:
        tree = json.loads(json_chunk.decode("utf-8"))
    except Exception:
        return glb_bytes

    # Find a mesh primitive to attach to (assume single mesh/primitive per your pipeline)
    try:
        prim = tree["meshes"][0]["primitives"][0]
        attrs = prim.setdefault("attributes", {})
    except Exception:
        return glb_bytes

    # If already present, do nothing
    if "TANGENT" in attrs:
        return glb_bytes

    # Append tangent data to BIN chunk (4-byte aligned)
    bin_new = bin_chunk + (b"\x00" * ((-len(bin_chunk)) % 4)) + tangents_v4_f32
    bin_new = bin_new + (b"\x00" * ((-len(bin_new)) % 4))
    tangent_byte_offset = len(bin_chunk) + ((-len(bin_chunk)) % 4)

    # Ensure sections exist
    tree.setdefault("bufferViews", [])
    tree.setdefault("accessors", [])
    tree.setdefault("buffers", [{"byteLength": 0}])

    # Create bufferView for tangents
    bv_index = len(tree["bufferViews"])
    tree["bufferViews"].append(
        {
            "buffer": 0,
            "byteOffset": int(tangent_byte_offset),
            "byteLength": int(vertex_count * 16),
        }
    )

    # Create accessor for tangents
    acc_index = len(tree["accessors"])
    tree["accessors"].append(
        {
            "componentType": 5126,  # FLOAT
            "type": "VEC4",
            "bufferView": bv_index,
            "byteOffset": 0,
            "count": int(vertex_count),
        }
    )

    attrs["TANGENT"] = acc_index

    # Update buffer byteLength
    tree["buffers"][0]["byteLength"] = int(len(bin_new))

    # Rebuild JSON chunk
    json_new = json.dumps(tree, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_new_padded = _pad4(json_new)
    bin_new_padded = bin_new + (b"\x00" * ((-len(bin_new)) % 4))

    # Rebuild GLB
    out = bytearray()
    out += struct.pack("<4sII", b"glTF", 2, 0)  # placeholder length
    out += struct.pack("<I4s", len(json_new_padded), b"JSON")
    out += json_new_padded
    out += struct.pack("<I4s", len(bin_new_padded), b"BIN\x00")
    out += bin_new_padded
    struct.pack_into("<I", out, 8, len(out))
    return bytes(out)

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

        # Load GLB presets per category (used by GLB converter; category comes from decision module)
        category_config_path = Path(settings.clarifier.category_config_path) if settings.clarifier.category_config_path else config_dir.parent / "category_config.yaml"
        self._category_glb_presets = load_category_config(category_config_path)["glb_presets"]

        # Initialize Judge module
        if settings.judge.enabled:
            self.judge_pipeline = VllmJudgePipeline(settings.judge)
            self.duel_manager = DuelManager(renderer=renderer)
        else:
            self.judge_pipeline = None
            self.duel_manager = None

        # vLLM client for decision module (category, multiview, pipeline); when judge enabled we reuse judge client
        self._decision_client: Optional[AsyncOpenAI] = None

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
            self._decision_client = AsyncOpenAI(
                base_url=self.settings.judge.vllm_url,
                api_key=self.settings.judge.vllm_api_key,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_keepalive_connections=4, max_connections=10)
                ),
            )
            logger.info("Decision module (vLLM): category, multiview, pipeline.")

        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()
        
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        await self.qwen_pipeline.shutdown()
        await self.rmbg_pipeline.shutdown()
        await self.mesh_pipeline.shutdown()
        if self.judge_pipeline is not None:
            await self.judge_pipeline.shutdown()
        if self._decision_client is not None:
            await self._decision_client.close()
            self._decision_client = None

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

    async def generate_from_upload(
        self,
        image_bytes: bytes,
        seed: int,
        input_filename: Optional[str] = None,
        settings: Optional[SettingsConf] = None,
    ) -> GenerationResponse:
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

        response = await self.generate(request, settings=settings)
        # Ensure GLB is bytes for streaming; generate() may leave it as bytes already
        if isinstance(response.glb_file_base64, str):
            response.glb_file_base64 = base64.b64decode(response.glb_file_base64)
        return response

    def _edit_images(
        self,
        image: Image.Image,
        seed: int,
        settings: SettingsConf,
        pre_decision: Optional[DecisionResponse] = None,
    ) -> tuple[list[Image.Image], Optional[float], bool, Optional[str]]:
        """
        Edit image based on current mode (multiview or base).

        When pre_decision is set (vLLM decision module), multiview and clarifier fields come from it.
        When clarifier disabled or no pre_decision, dynamic mode falls back to single-view.

        Returns:
            (edited_images, clarifier_score, multiview_used, clarifier_explanation)
        """
        clarifier_score: Optional[float] = None
        clarifier_explanation: Optional[str] = None

        def _is_cuda_oom(exc: BaseException) -> bool:
            msg = str(exc).lower()
            return (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in msg
                or "cuda" in msg
            )

        # 1 Always canonicalize the input with the base prompt first.
        base_prompt = self.prompting_library.promptings["base"]
        logger.debug(f"Editing base view with prompt: {base_prompt}")
        try:
            base_images = list(
                self.qwen_edit.edit_image(
                    model=self.qwen_pipeline,
                    prompt_image=image,
                    seed=seed,
                    prompting=base_prompt,
                )
            )
        except Exception as e:
            if _is_cuda_oom(e):
                logger.warning(
                    f"Qwen edit OOM during base edit; falling back to original image (no edit): {e}"
                )
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                return [image], clarifier_score, False, "Qwen OOM: skipped edit"
            raise

        multiview_mode = getattr(settings.trellis, "multiview_mode", "off")
        needs_multiview = False

        if multiview_mode == "off":
            needs_multiview = False
            logger.info("Multiview mode: off (single view only).")
        elif multiview_mode == "always":
            needs_multiview = True
            logger.info("Multiview mode: always (force multiview).")
        elif multiview_mode == "dynamic":
            if pre_decision is not None:
                needs_multiview = pre_decision.needs_multiview
                clarifier_score = 0.3 if needs_multiview else 0.9
                clarifier_explanation = pre_decision.explanation or f"VLM: multiview={needs_multiview}"
                logger.info(
                    f"Decision (vLLM): needs_multiview={needs_multiview} "
                    f"explanation={clarifier_explanation!r}"
                )
                if needs_multiview:
                    logger.info("VLM decided: multiview is needed for this object.")
                else:
                    logger.info("VLM decided: single view is sufficient for this object.")
            else:
                logger.warning("Multiview mode is 'dynamic' but no pre_decision; falling back to single-view.")
                needs_multiview = False
        else:
            logger.warning(f"Unknown multiview_mode={multiview_mode!r}; treating as 'off'.")
            needs_multiview = False

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
                try:
                    result = self.qwen_edit.edit_image(
                        model=self.qwen_pipeline,
                        prompt_image=base_image,
                        seed=seed,
                        prompting=prompt_text,
                    )
                    edited_images.extend(result)
                except Exception as e:
                    if _is_cuda_oom(e):
                        logger.warning(
                            f"Qwen edit OOM during multiview edit; using base-edited images only: {e}"
                        )
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        return list(base_images), clarifier_score, False, "Qwen OOM: skipped multiview edits"
                    raise

        return edited_images, clarifier_score, True, clarifier_explanation

    async def generate_mesh(
        self,
        request: GenerationRequest,
        settings: SettingsConf,
    ) -> tuple[list[MeshWithVoxel], list[Image.Image], list[Image.Image], Optional[float], bool, Optional[str], Optional[str], Optional[float]]:
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

        # Pre-decision: one vLLM call (decision module) for category, multiview, pipeline
        pre_decision: Optional[DecisionResponse] = None
        if settings.clarifier.enabled:
            client = (
                self.judge_pipeline.client
                if self.judge_pipeline is not None
                else getattr(self, "_decision_client", None)
            )
            if client is not None:
                try:
                    pre_decision = await vllm_decide(
                        image,
                        client,
                        settings.judge.vllm_model_name,
                        seed=request.seed,
                    )
                except Exception as e:
                    logger.warning(f"VLM decision failed: {e}; using defaults.")
                    pre_decision = DEFAULT_DECISION
            else:
                logger.warning("Clarifier enabled but no vLLM client; using defaults.")
                pre_decision = DEFAULT_DECISION

        # Track what the decision model chose (for logging/results)
        self._last_decision_pipeline = pre_decision.pipeline if pre_decision is not None else None
        self._last_trellis_oom_retry = False

        # 1. Edit the image (multiview from pre_decision when available)
        images_edited, clarifier_score, multiview_used, clarifier_explanation = self._edit_images(
            image, request.seed, settings, pre_decision=pre_decision
        )

        # 2. Remove background
        images_with_background = [img.copy() for img in images_edited]
        images_without_background = list(
            self.rmbg_module.remove_background(self.rmbg_pipeline, images_with_background)
        )

        # 2b. Object category (for GLB presets) and pipeline: from vLLM decision module
        object_category: Optional[str] = None
        object_category_confidence: Optional[float] = None
        if pre_decision is not None:
            object_category = pre_decision.category
            object_category_confidence = 0.9
        # Category can be logged but may be ignored for GLB material mapping.
        if object_category:
            logger.info(f"Decision category: {object_category}")
        object_category_for_glb = object_category if settings.clarifier.use_category_clarification else None
        if object_category_for_glb:
            logger.info(f"Object category for GLB preset: {object_category_for_glb}")

        overrides_dict = request.trellis_params.model_dump(exclude_none=True) if request.trellis_params else {}
        suggested_pipeline = pre_decision.pipeline if pre_decision is not None else None
        if suggested_pipeline is not None:
            overrides_dict["pipeline_type"] = TrellisPipeType.MODE_512 if suggested_pipeline == "512" else TrellisPipeType.MODE_1024_CASCADE
        trellis_params = TrellisParams.Overrides(**overrides_dict) if overrides_dict else request.trellis_params

        # Request-scoped Trellis defaults from config
        default_params = TrellisParams.from_settings(settings.trellis)

        # Store the final Trellis pipeline_type that will actually be used.
        try:
            effective_params = default_params.overrided(trellis_params) if trellis_params else default_params
            self._last_trellis_pipeline_type = getattr(effective_params.pipeline_type, "value", str(effective_params.pipeline_type))
            self._last_trellis_pipeline_type_suggested = suggested_pipeline
        except Exception:
            self._last_trellis_pipeline_type = None
            self._last_trellis_pipeline_type_suggested = suggested_pipeline

        # 3. Generate the 3D model (with a guarded retry on Trellis OOM)
        try:
            meshes = self.mesh_generator.generate(
                model=self.mesh_pipeline,
                request=TrellisRequest(
                    image=images_without_background,
                    seed=request.seed,
                    params=trellis_params
                ),
                default_params=default_params,
            )
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda" not in msg and "acceleratorerror" not in msg:
                raise
            logger.warning(f"Trellis OOM during mesh generation; retrying once with lighter settings: {e}")
            self._last_trellis_oom_retry = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Retry once with a lighter Trellis configuration (pipeline_type=512).
            # Keep num_samples unchanged (user expects 2 candidates consistently).
            fallback_num_samples = getattr(default_params, "num_samples", None) or getattr(settings.trellis, "num_samples", 2) or 2
            fallback_overrides = TrellisParams.Overrides(
                pipeline_type=TrellisPipeType.MODE_512,
                num_samples=int(fallback_num_samples),
            )
            self._last_trellis_pipeline_type = TrellisPipeType.MODE_512.value
            meshes = self.mesh_generator.generate(
                model=self.mesh_pipeline,
                request=TrellisRequest(
                    image=images_without_background,
                    seed=request.seed,
                    params=fallback_overrides,
                ),
                default_params=default_params,
            )

        logger.info(
            "Trellis pipeline decision summary: "
            f"decision_pipeline={getattr(self, '_last_decision_pipeline', None)} "
            f"trellis_pipeline_used={getattr(self, '_last_trellis_pipeline_type', None)} "
            f"trellis_oom_retry={bool(getattr(self, '_last_trellis_oom_retry', False))}"
        )

        return meshes, images_edited, images_without_background, clarifier_score, multiview_used, clarifier_explanation, object_category_for_glb, object_category_confidence

    def convert_mesh_to_glb(
        self,
        mesh: MeshWithVoxel,
        glbconv_params: Optional[GLBConverterParams.Overrides],
        object_category: Optional[str] = None,
        settings: Optional[SettingsConf] = None,
    ) -> bytes:
        """
        Convert mesh to GLB format using GLBConverter.

        Args:
            mesh: The mesh to convert
            glbconv_params: Optional override parameters for GLB conversion (from request)
            object_category: Optional category from classifier (glass, metal, etc.) to apply material preset
            settings: Request-scoped config; uses self.settings if None.

        Returns:
            GLB file as bytes
        """
        s = settings or self.settings
        start_time = time.time()
        # Merge: base config, then category preset, then request overrides
        category_overrides = get_glb_overrides_for_category(object_category, presets=self._category_glb_presets)
        request_dict = glbconv_params.model_dump(exclude_none=True) if glbconv_params else {}

        # Start from global GLB defaults in settings, then layer in presets/overrides.
        base_defaults = GLBConverterParams.from_settings(s.glb_converter)
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

        auto = getattr(s.glb_converter, "auto_clamp", None)
        if auto is not None and getattr(auto, "enabled", True):
            # Mesh size tiers (configurable).
            very_large = num_vertices > auto.very_large_vertices or num_faces > auto.very_large_faces
            large = num_vertices > auto.large_vertices or num_faces > auto.large_faces

            low_vram = free_mem_gb is not None and free_mem_gb < float(auto.low_vram_gb)
            very_low_vram = free_mem_gb is not None and free_mem_gb < float(auto.very_low_vram_gb)

            # Clamp based on mesh size + VRAM.
            if very_large or very_low_vram:
                tier = auto.very_large
                if tier.disable_remesh or (tier.disable_remesh_if_low_vram and (low_vram or very_low_vram)):
                    remesh = False
                dec_target = min(dec_target, int(tier.max_decimation_target))
                tex_size = min(tex_size, int(tier.max_texture_size))
                subdivisions = min(subdivisions, int(tier.max_subdivisions))
            elif large or low_vram:
                tier = auto.large
                if tier.disable_remesh or (tier.disable_remesh_if_low_vram and low_vram):
                    remesh = False
                dec_target = min(dec_target, int(tier.max_decimation_target))
                tex_size = min(tex_size, int(tier.max_texture_size))
                subdivisions = min(subdivisions, int(tier.max_subdivisions))

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
                f"GLB conversion OOM on first attempt (category={object_category}); "
                f"retrying with lighter params based on mesh size/VRAM: {e}"
            )
            # Build a lighter override that keeps work on GPU but reduces memory pressure.
            base = params_filtered or GLBConverterParams.Overrides()
            oom = getattr(auto, "oom_retry", None) if (auto is not None and getattr(auto, "enabled", True)) else None
            oom_enabled = bool(getattr(oom, "enabled", True)) if oom is not None else True
            if not oom_enabled:
                raise

            max_dec = int(getattr(oom, "max_decimation_target", 70_000)) if oom is not None else 70_000
            max_tex = int(getattr(oom, "max_texture_size", 1024)) if oom is not None else 1024
            max_sub = int(getattr(oom, "max_subdivisions", 0)) if oom is not None else 0
            disable_remesh = bool(getattr(oom, "disable_remesh", True)) if oom is not None else True

            lighter = GLBConverterParams.Overrides(
                decimation_target=min(getattr(base, "decimation_target", 120_000) or 120_000, max_dec),
                texture_size=min(getattr(base, "texture_size", 2048) or 2048, max_tex),
                remesh=False if disable_remesh else bool(getattr(base, "remesh", False)),
                subdivisions=min(int(getattr(base, "subdivisions", 0) or 0), max_sub),
                vertex_reproject=0.0,
                smooth_mesh=False,
                alpha_mode=getattr(base, "alpha_mode", GLBConverterParams().alpha_mode),
                rescale=getattr(base, "rescale", 1.0),
                remesh_band=getattr(base, "remesh_band", 1.0),
                remesh_project=getattr(base, "remesh_project", 0.0),
                mesh_cluster_refine_iterations=getattr(base, "mesh_cluster_refine_iterations", 0),
                mesh_cluster_global_iterations=getattr(base, "mesh_cluster_global_iterations", 1),
                mesh_cluster_smooth_strength=getattr(base, "mesh_cluster_smooth_strength", 1.0),
                mesh_cluster_threshold_cone_half_angle=getattr(base, "mesh_cluster_threshold_cone_half_angle", 90.0),
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

        glb_bytes = buffer.getvalue()

        # Inject TANGENT attribute for normal mapping if available.
        tangents = getattr(self.glb_converter, "last_vertex_tangents", None)
        if tangents is not None:
            try:
                # Apply the same coordinate conversion used for vertices/normals before export:
                # swap Y/Z and invert Y -> (x, y, z) -> (x, z, -y)
                t = tangents.astype("float32", copy=True)
                t[:, 1], t[:, 2] = t[:, 2].copy(), -t[:, 1].copy()
                # Handedness may need flipping due to reflection; keep as-is for now.
                glb_bytes = _glb_with_tangents(
                    glb_bytes,
                    tangents_v4_f32=t.tobytes(order="C"),
                    vertex_count=int(t.shape[0]),
                )
                logger.info(f"Injected glTF TANGENT attribute (V={int(t.shape[0])})")
            except Exception as e:
                logger.warning(f"Failed to inject TANGENT attribute: {e}")

        logger.info(f"GLB conversion time: {time.time() - start_time:.2f}s")
        return glb_bytes

    def prepare_outputs(
        self,
        images_edited: list[Image.Image],
        images_without_background: list[Image.Image],
        glb_trellis_result: Optional[TrellisResult],
        settings: Optional[SettingsConf] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Prepare output files: save to disk if configured and generate base64 strings if needed.
        """
        s = settings or self.settings
        start_time = time.time()
        # Create grid images once for both save and send operations
        image_edited_grid = image_grid(images_edited)
        image_without_background_grid = image_grid(images_without_background)

        # Save generated files if configured
        if s.output.save_generated_files:
            save_files(
                glb_trellis_result,
                image_edited_grid,
                image_without_background_grid,
                output_dir=s.output.output_dir,
            )

        # Convert to PNG base64 for response if configured
        image_edited_base64 = None
        image_without_background_base64 = None
        if s.output.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited_grid)
            image_without_background_base64 = to_png_base64(image_without_background_grid)
            
        logger.info(f"Output preparation time: {time.time() - start_time:.2f}s")

        return image_edited_base64, image_without_background_base64

    async def generate(
        self,
        request: GenerationRequest,
        settings: Optional[SettingsConf] = None,
    ) -> GenerationResponse:
        """
        Execute full generation pipeline with output types.
        Uses request-scoped settings when provided (config loaded per request).
        """
        s = settings or self.settings
        t1 = time.time()
        logger.info(f"Request received | Seed: {request.seed} | Prompt Type: {request.prompt_type.value}")

        # Generate mesh and get processed images
        meshes, images_edited, images_without_background, clarifier_score, multiview_used, clarifier_explanation, object_category, object_category_confidence = await self.generate_mesh(request, s)

        glb_trellis_result = None
        
        self._clean_gpu_memory()

        # Convert meshes to GLB
        glb_trellis_results: list[TrellisResult] = []
        uv_infos: list[dict | None] = []
        glb_sizes: list[int] = []
        if meshes:
            for mesh in meshes:
                try:
                    glb_bytes = self.convert_mesh_to_glb(
                        mesh, request.glbconv_params, object_category, settings=s
                    )
                except MeshTooManyChartsForGLB as e:
                    # Avoid spending time on GLB conversion (textures/export) when we already know
                    # this candidate will be filtered out due to chart explosion.
                    uv_infos.append(getattr(self.glb_converter, "last_uv_unwrap_info", None))
                    logger.warning(f"Skipping GLB candidate due to excessive UV charts: {e}")
                    continue
                except MeshTooLargeForGLB as e:
                    logger.warning(f"Skipping GLB candidate due to oversized mesh: {e}")
                    continue
                uv_infos.append(getattr(self.glb_converter, "last_uv_unwrap_info", None))
                glb_trellis_results.append(TrellisResult(file_bytes=glb_bytes))
                glb_sizes.append(len(glb_bytes) if isinstance(glb_bytes, (bytes, bytearray)) else 0)

        # Candidate filtering to avoid excessive chart fragmentation.
        #
        # Behavior (per user spec):
        # - If all candidates have num_charts > max_xatlas_charts:
        #   - if current Trellis pipeline is not "512": regenerate meshes with Trellis pipeline_type="512"
        #   - if current Trellis pipeline is already "512": keep only candidate 0
        # - If at least one candidate is <= max_xatlas_charts: drop candidates with num_charts > max_xatlas_charts
        #   and proceed with the remaining candidates.
        max_xatlas_charts = getattr(getattr(s, "glb_converter", None), "max_xatlas_charts", None)
        if max_xatlas_charts is not None and max_xatlas_charts > 0 and uv_infos and glb_trellis_results:
            def _num_charts(i: int) -> Optional[int]:
                info = uv_infos[i]
                if isinstance(info, dict):
                    v = info.get("num_charts")
                    return int(v) if v is not None else None
                return None

            def _is_over_threshold(i: int) -> bool:
                """
                'Over' means we should avoid / retry away from UVs that will likely
                fall into trivial UV fallback due to xatlas chart explosion.
                """
                info = uv_infos[i]
                if not isinstance(info, dict):
                    return False
                n = info.get("num_charts")
                reason = info.get("reason")
                if n is not None:
                    try:
                        return int(n) > max_xatlas_charts
                    except (TypeError, ValueError):
                        return False
                # Some paths may set num_charts=None but still tag the reason as "charts".
                # In that case, treat as "over" by spec.
                return reason == "charts"

            num_charts_list = [_num_charts(i) for i in range(len(glb_trellis_results))]
            over_flags = [_is_over_threshold(i) for i in range(len(glb_trellis_results))]
            all_over = bool(over_flags) and all(over_flags)

            def _is_pipeline_512(val: object) -> bool:
                if val is None:
                    return False
                s_val = str(getattr(val, "value", val))
                return s_val == TrellisPipeType.MODE_512.value or s_val == "512"

            current_pipeline_used = getattr(self, "_last_trellis_pipeline_type", None)
            current_is_512 = _is_pipeline_512(current_pipeline_used)

            if all_over:
                logger.warning(
                    f"All GLB candidates exceed max_xatlas_charts={max_xatlas_charts}; "
                    f"uv_num_charts={num_charts_list}; current_pipeline_used={current_pipeline_used}"
                )

                if not current_is_512:
                    # Retry Trellis generation with pipeline_type=512 (preserve num_samples and other overrides).
                    logger.warning("Retrying Trellis meshes with pipeline_type=512 to avoid trivial UV fallback.")
                    self._clean_gpu_memory()

                    default_params = TrellisParams.from_settings(s.trellis)
                    overrides_dict = (
                        request.trellis_params.model_dump(exclude_none=True)
                        if request.trellis_params
                        else {}
                    )
                    overrides_dict["pipeline_type"] = TrellisPipeType.MODE_512
                    trellis_overrides = (
                        TrellisParams.Overrides(**overrides_dict) if overrides_dict else None
                    )

                    meshes = self.mesh_generator.generate(
                        model=self.mesh_pipeline,
                        request=TrellisRequest(
                            image=images_without_background,
                            seed=request.seed,
                            params=trellis_overrides,
                        ),
                        default_params=default_params,
                    )

                    glb_trellis_results = []
                    uv_infos = []
                    glb_sizes = []
                    for mesh in meshes:
                        try:
                            glb_bytes = self.convert_mesh_to_glb(
                                mesh, request.glbconv_params, object_category, settings=s
                            )
                        except MeshTooLargeForGLB as e:
                            logger.warning(f"Skipping GLB candidate due to oversized mesh after retry: {e}")
                            continue
                        uv_infos.append(getattr(self.glb_converter, "last_uv_unwrap_info", None))
                        glb_trellis_results.append(TrellisResult(file_bytes=glb_bytes))
                        glb_sizes.append(len(glb_bytes) if isinstance(glb_bytes, (bytes, bytearray)) else 0)

                    # Update pipeline_used logging for the retry.
                    self._last_trellis_pipeline_type = TrellisPipeType.MODE_512.value

                    # Recompute over/under to decide whether to filter candidates further.
                    num_charts_list = [_num_charts(i) for i in range(len(glb_trellis_results))]
                    # Reuse the same "over" rule after retry.
                    over_flags = [_is_over_threshold(i) for i in range(len(glb_trellis_results))]
                    all_over = bool(over_flags) and all(over_flags)

                    if all_over:
                        logger.warning(
                            f"After 512 retry, all candidates still exceed max_xatlas_charts={max_xatlas_charts}; "
                            f"keeping candidate 0 only."
                        )
                        glb_trellis_results = glb_trellis_results[:1]
                        uv_infos = uv_infos[:1]
                        glb_sizes = glb_sizes[:1]
                else:
                    # Already on 512: keep candidate 0 only (avoid further trivial UV avoidance attempts).
                    logger.warning("Pipeline already 512; keeping candidate 0 only to proceed.")
                    glb_trellis_results = glb_trellis_results[:1]
                    uv_infos = uv_infos[:1]
                    glb_sizes = glb_sizes[:1]
            else:
                # At least one candidate is <= threshold: drop over-threshold candidates.
                keep_indices = [i for i, over in enumerate(over_flags) if not over]
                if keep_indices and len(keep_indices) != len(glb_trellis_results):
                    logger.warning(
                        f"Filtering GLB candidates by num_charts threshold={max_xatlas_charts}; "
                        f"keeping indices={keep_indices}; uv_num_charts={num_charts_list}"
                    )
                    glb_trellis_results = [glb_trellis_results[i] for i in keep_indices]
                    uv_infos = [uv_infos[i] for i in keep_indices]
                    glb_sizes = [glb_sizes[i] for i in keep_indices]

        # Decide whether to run the judge based on GLB sizes and judge.enabled
        winner_index: int = 0
        duel_done = False
        duel_winner: Optional[int] = None
        duel_explanation: Optional[str] = None
        max_bytes = getattr(s.judge, "max_glb_bytes_for_judge", 0) or 0
        too_large_indices = {i for i, sz in enumerate(glb_sizes) if max_bytes > 0 and sz > max_bytes}
        judge_enabled = s.judge.enabled and self.duel_manager and self.judge_pipeline

        if not glb_trellis_results:
            raise RuntimeError("No GLB candidates were produced from Trellis meshes.")

        if glb_trellis_results:
            if too_large_indices:
                # At least one candidate is too large: skip judge entirely and ignore those candidates.
                logger.info(
                    f"Skipping judge due to large GLB(s): indices={sorted(too_large_indices)}, "
                    f"sizes={[glb_sizes[i] for i in sorted(too_large_indices)]}, "
                    f"threshold={max_bytes} bytes"
                )
                # Pick the smallest remaining candidate by size (or overall smallest if all are too large).
                candidate_indices = [i for i in range(len(glb_trellis_results)) if i not in too_large_indices]
                if not candidate_indices:
                    candidate_indices = list(range(len(glb_trellis_results)))
                winner_index = min(candidate_indices, key=lambda i: glb_sizes[i] if glb_sizes[i] else float("inf"))
                glb_trellis_result = glb_trellis_results[winner_index]
            elif judge_enabled:
                # We have 2+ candidates and judge is enabled: first save all candidate renders,
                # then run the duel to pick a winner.
                candidates_dir = Path(s.output.output_dir).resolve()
                candidates_basename = (
                    Path(request.input_filename).stem
                    if request.input_filename
                    else datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                )
                logger.info(
                    f"Candidate renders will be saved to: {candidates_dir} "
                    f"(basename={candidates_basename}_candidate_0.png, ...)"
                )
                try:
                    self.duel_manager.save_candidate_renders(
                        glb_trellis_results,
                        candidates_dir,
                        candidates_basename,
                    )
                except Exception as e:
                    logger.exception(f"Failed to save candidate renders: {e}")

                winner_index, duel_explanation = await self.duel_manager.judge_meshes(
                    self.judge_pipeline,
                    glb_trellis_results,
                    request.prompt_image,
                    request.seed,
                )
                glb_trellis_result = glb_trellis_results[winner_index]
                duel_done = True
                duel_winner = winner_index
            else:
                glb_trellis_result = glb_trellis_results[0]

        # Save generated files
        image_edited_base64, image_no_bg_base64 = None, None
        if s.output.save_generated_files or s.output.send_generated_files:
            image_edited_base64, image_no_bg_base64 = self.prepare_outputs(
                images_edited,
                images_without_background,
                glb_trellis_result,
                settings=s,
            )

        t2 = time.time()
        generation_time = t2 - t1

        logger.success(f"Generation time: {generation_time:.2f}s")

        # Pick UV info for the winner (if any)
        winner_uv = uv_infos[winner_index] if (uv_infos and 0 <= winner_index < len(uv_infos)) else None
        uv_mode = (winner_uv or {}).get("mode") if isinstance(winner_uv, dict) else None
        uv_reason = (winner_uv or {}).get("reason") if isinstance(winner_uv, dict) else None
        uv_charts = (winner_uv or {}).get("num_charts") if isinstance(winner_uv, dict) else None

        # Clean the GPU memory
        self._clean_gpu_memory()

        return GenerationResponse(
            generation_time=generation_time,
            glb_file_base64=glb_trellis_result.file_bytes if glb_trellis_result else None,
            image_edited_file_base64=image_edited_base64,
            image_without_background_file_base64=image_no_bg_base64,
            multiview_used=multiview_used,
            object_category=object_category,
            decision_pipeline=getattr(self, "_last_decision_pipeline", None),
            pipeline_used=getattr(self, "_last_trellis_pipeline_type", None),
            decision_explanation=clarifier_explanation,
            trellis_oom_retry=bool(getattr(self, "_last_trellis_oom_retry", False)),
            uv_unwrap_mode=uv_mode,
            uv_unwrap_reason=uv_reason,
            uv_num_charts=uv_charts,
            cluster_count=int(uv_charts) if uv_charts is not None else None,
            duel_done=duel_done,
            duel_winner=duel_winner,
            duel_explanation=duel_explanation or None,
        )
