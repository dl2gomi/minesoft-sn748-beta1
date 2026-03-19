import base64
import time
from pathlib import Path
from typing import Optional, Tuple

from logger_config import logger
from modules.grid_renderer.render import GridViewRenderer
from modules.mesh_generator.schemas import TrellisResult
from .judge_pipeline import JudgePipeline


class DuelManager:
    """Orchestrates mesh duels using a provided judge pipeline."""

    def __init__(self, renderer: Optional[GridViewRenderer] = None) -> None:
        self.renderer = renderer

    @staticmethod
    def encode_image(image: bytes) -> str:
        return base64.b64encode(image).decode("utf-8")

    async def run_duel(
        self,
        pipeline: JudgePipeline,
        prompt_bytes: bytes,
        img1_bytes: bytes,
        img2_bytes: bytes,
        seed: int,
    ) -> Tuple[int, str]:
        """
        Run a position-balanced duel between two candidate images.

        Args:
            pipeline: Loaded judge pipeline used for inference.
            prompt_bytes: Original prompt image as PNG bytes.
            img1_bytes: First candidate rendered grid as PNG bytes.
            img2_bytes: Second candidate rendered grid as PNG bytes.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (winner_idx, issues):
                winner_idx: -1 if img1 wins, 1 if img2 wins.
                issues: Human-readable issue summary.
        """
        if not img1_bytes or not img2_bytes:
            logger.error("Invalid image bytes provided to judge")
            return -1, "Invalid input — defaulting to first candidate"

        prompt_b64, render1_b64, render2_b64 = map(
            self.encode_image, (prompt_bytes, img1_bytes, img2_bytes)
        )

        logger.debug("Running position-balanced VLLM duel...")
        res_direct  = await pipeline.judge(prompt_b64, render1_b64, render2_b64, seed)
        res_swapped = await pipeline.judge(prompt_b64, render2_b64, render1_b64, seed)

        score1 = (res_direct.penalty_1 + res_swapped.penalty_2) / 2
        score2 = (res_swapped.penalty_1 + res_direct.penalty_2) / 2
        issues = (
            f"| Direct: {res_direct.issues} | Swapped: {res_swapped.issues}"
            if res_direct.issues or res_swapped.issues
            else ""
        )

        # Lower penalty = better. Tie: prefer first candidate (fully deterministic).
        winner = -1 if score1 <= score2 else 1

        logger.debug(
            f"Duel scores — Candidate 1: {score1:.1f} (direct={res_direct.penalty_1}, swapped={res_swapped.penalty_2}) | "
            f"Candidate 2: {score2:.1f} (direct={res_direct.penalty_2}, swapped={res_swapped.penalty_1}) | Winner: {winner}"
        )

        return winner, issues

    def _render_meshes(self, meshes: list[TrellisResult]) -> Optional[list[bytes]]:
        """Render all meshes to PNG grid views. Returns None if any render fails."""
        rendered: list[bytes] = []
        for idx, mesh in enumerate(meshes):
            png_bytes = self.renderer.grid_from_glb_bytes(mesh.file_bytes)
            if png_bytes is None:
                logger.warning(
                    f"Renderer returned None for mesh {idx} - skipping judge, returning first mesh"
                )
                return None
            rendered.append(png_bytes)
            logger.debug(f"Mesh {idx} rendered to PNG grid ({len(png_bytes)} bytes)")
        return rendered

    def save_candidate_renders(
        self,
        meshes: list[TrellisResult],
        output_dir: Path,
        basename: str,
    ) -> None:
        """
        Render all candidate meshes and save their PNGs as
        {basename}_candidate_0.png, {basename}_candidate_1.png, ...

        This is intentionally decoupled from the judge logic so that
        candidates are saved even if the judge pipeline fails.
        """
        if not meshes:
            return
        if self.renderer is None:
            logger.warning("No renderer provided to DuelManager - cannot save candidate renders")
            return

        rendered = self._render_meshes(meshes)
        if rendered is None:
            return

        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        for i, png_bytes in enumerate(rendered):
            path = out / f"{basename}_candidate_{i}.png"
            path.write_bytes(png_bytes)

        logger.info(
            f"Saved {len(rendered)} duel candidate PNGs to {out} "
            f"(e.g. {out / f'{basename}_candidate_0.png'})"
        )

    async def judge_meshes(
        self,
        pipeline: JudgePipeline,
        meshes: list[TrellisResult],
        prompt_image_b64: str,
        seed: int,
    ) -> Tuple[int, str]:
        """
        Judge a list of meshes by rendering them and comparing with the judge pipeline.

        Returns:
            Tuple of (winner_index, duel_explanation). duel_explanation is the combined
            issues from all duels (from vLLM judge); empty string if no duel ran.
        """
        t1 = time.time()

        if len(meshes) < 2:
            logger.warning("Less than 2 meshes provided to judge")
            return 0, ""

        if self.renderer is None:
            logger.warning("No renderer provided to DuelManager - skipping judge, returning first mesh")
            return 0, ""

        logger.info(f"Judging {len(meshes)} meshes with prompt image")

        prompt_bytes = base64.b64decode(prompt_image_b64)

        rendered = self._render_meshes(meshes)
        if rendered is None:
            return 0, ""

        best_idx = 0
        all_issues: list[str] = []
        for i in range(1, len(rendered)):
            winner, issues = await self.run_duel(
                pipeline, prompt_bytes, rendered[best_idx], rendered[i], seed
            )
            if issues:
                all_issues.append(issues)
            logger.info(f"Duel [{best_idx} vs {i}] → winner: {i if winner == 1 else best_idx} {issues}")
            if winner == 1:  # Image 2 wins → update best candidate
                best_idx = i

        duel_explanation = " ".join(all_issues).strip() if all_issues else ""

        logger.success(
            f"Judging {len(meshes)} meshes with prompt image took {time.time() - t1:.2f}s | Winner: {best_idx}"
        )

        return best_idx, duel_explanation
