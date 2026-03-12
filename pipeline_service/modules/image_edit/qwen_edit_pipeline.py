from __future__ import annotations

import math
import time

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.models import QwenImageTransformer2DModel

from config.settings import ModelVersionsConfig
from logger_config import logger

from .image_edit_pipeline import ImageEditPipeline
from .settings import QwenConfig


class QwenEditPipeline(ImageEditPipeline[QwenImageEditPlusPipeline]):
    """Loads and owns the Qwen image-edit pipeline and its weights."""

    def __init__(self, settings: QwenConfig, model_versions: ModelVersionsConfig):
        super().__init__(model_versions=model_versions, gpu_index=settings.gpu)
        self.settings = settings
        self.dtype = self._resolve_dtype(settings.dtype)

    async def _load_pipeline(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.gpu_index)
            except Exception as err:
                logger.warning(f"Failed to set CUDA device ({self.gpu_index}): {err}")

        t1 = time.time()

        model_revision = self.model_versions.get_revision(self.settings.model_id)
        transformer = QwenImageTransformer2DModel.from_pretrained(
            self.settings.model_id,
            subfolder="transformer",
            torch_dtype=self.dtype,
            revision=model_revision,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self._scheduler_config())
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.settings.model_id,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=self.dtype,
            revision=model_revision,
        )

        lora_revision = self.model_versions.get_revision(self.settings.lora_path)
        pipe.load_lora_weights(
            self.settings.lora_path,
            weight_name=self.settings.base_model_path,
            revision=lora_revision,
            adapter_name="lightning",
        )

        lightning_fused = False
        if self.settings.fuse_lightning_lora:
            lightning_fused = self._fuse_lightning_lora(pipe)
            if lightning_fused:
                pipe.unload_lora_weights()
                logger.info("Unloaded LoRA adapters after lightning fusion")
            else:
                logger.warning("Lightning LoRA fusion requested but failed; using unfused lightning adapter")

        if self.settings.lora_angles_path:
            lora_angles_revision = self.model_versions.get_revision(self.settings.lora_angles_path)
            pipe.load_lora_weights(
                self.settings.lora_angles_path,
                weight_name=self.settings.lora_angles_filename,
                revision=lora_angles_revision,
                adapter_name="angles",
            )
            angles_weight = getattr(self.settings, "lora_angles_weight", 0.9)
            if lightning_fused:
                pipe.set_adapters(["angles"], adapter_weights=[angles_weight])
                logger.info(f"Loaded angles LoRA (weight={angles_weight}) with fused lightning LoRA (multiview mode)")
            else:
                pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, angles_weight])
                logger.info(f"Loaded dual LoRAs: lightning + angles (weight={angles_weight}) (multiview mode)")

        self.pipe = pipe.to(self.device)
        load_time = time.time() - t1
        logger.success(
            f"Qwen pipeline ready (loading: {load_time:.2f}s). "
            f"Loaded on {self.device} with dtype={self.dtype}."
        )

    def _fuse_lightning_lora(self, pipe: QwenImageEditPlusPipeline) -> bool:
        """Fuse the lightning LoRA into the base model weights if supported."""
        try:
            try:
                pipe.fuse_lora(adapter_names=["lightning"])
            except TypeError:
                pipe.fuse_lora()
            logger.info("Fused lightning LoRA into base model weights")
            return True
        except Exception as err:
            logger.warning(f"Could not fuse lightning LoRA; continuing unfused. Reason: {err}")
            return False

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        resolved = mapping.get(dtype.lower(), torch.bfloat16)
        if not torch.cuda.is_available() and resolved in {torch.float16, torch.bfloat16}:
            return torch.float32
        return resolved

    def _scheduler_config(self) -> dict:
        return {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
