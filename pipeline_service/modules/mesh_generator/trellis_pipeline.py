from __future__ import annotations

import os
from typing import Optional

import torch

from libs.trellis2.pipelines import Trellis2ImageTo3DPipeline

from .mesh_generation_pipeline import MeshGenerationPipeline


class Trellis2MeshPipeline(MeshGenerationPipeline[Trellis2ImageTo3DPipeline]):
    """Loads the TRELLIS.2 pipeline and owns its lifecycle."""

    def _get_model_revisions(self) -> tuple[Optional[str], dict[str, str]]:
        trellis_revision = self.model_versions.get_revision(self.settings.model_id)
        model_revisions = {
            model_id: revision
            for model_id, revision in self.model_versions.models.items()
            if model_id != self.settings.model_id
        }
        return trellis_revision, model_revisions

    async def _load_pipeline(self) -> None:
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.settings.gpu)

        trellis_revision, model_revisions = self._get_model_revisions()

        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            self.settings.model_id,
            config_file=self.settings.pipeline_config_path,
            revision=trellis_revision,
            model_revisions=model_revisions,
        )
        pipeline.cuda()
        self.pipeline = pipeline
