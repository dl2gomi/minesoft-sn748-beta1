from __future__ import annotations

import json
import re

import httpx
from openai import AsyncOpenAI

from logger_config import logger
from .judge_pipeline import JudgePipeline
from .prompting import SYSTEM_PROMPT
from .schemas import JudgeResponse
from .settings import JudgeConfig
from modules.utils import set_random_seed


class VllmJudgePipeline(JudgePipeline):
    """Connects to a vLLM server and runs judge inference."""

    def __init__(self, settings: JudgeConfig) -> None:
        super().__init__()
        self.settings = settings
        self.client: AsyncOpenAI | None = None

    async def _setup(self) -> None:
        self.client = AsyncOpenAI(
            base_url=self.settings.vllm_url,
            api_key=self.settings.vllm_api_key,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            ),
        )

    async def _teardown(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None

    def _parse_content(self, content: str, finish_reason: str) -> JudgeResponse:
        """Parse vLLM response content into a JudgeResponse (regex-based, robust to truncated JSON)."""
        penalty_1_match = re.search(r'"penalty_1":\s*(\d+)', content)
        penalty_2_match = re.search(r'"penalty_2":\s*(\d+)', content)
        penalty_1 = int(penalty_1_match.group(1)) if penalty_1_match else 5
        penalty_2 = int(penalty_2_match.group(1)) if penalty_2_match else 5

        issues = ""
        if finish_reason != "length":
            try:
                issues = json.loads(content)["issues"]
            except (json.JSONDecodeError, KeyError):
                issues = "Incomplete JSON"

        return JudgeResponse(penalty_1=penalty_1, penalty_2=penalty_2, issues=issues)

    async def judge(
        self,
        prompt_b64: str,
        img1_b64: str,
        img2_b64: str,
        seed: int,
    ) -> JudgeResponse:
        """Call vLLM to compare two candidate images. Uses 2 separate calls (1 image each)
        because GLM-4.1V / vLLM allows at most 1 image per request."""
        assert self.client is not None, "VllmJudgePipeline is not initialized."

        # Single-image response schema for each call
        single_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge-single",
                "schema": {
                    "type": "object",
                    "properties": {"penalty": {"type": "integer"}, "issues": {"type": "string"}},
                    "required": ["penalty", "issues"],
                },
            },
        }
        user_text = (
            "Rate this 3D model (4 views). Penalty 0-10 (0=perfect, 10=wrong). "
            "Output: {\"penalty\": <0-10>, \"issues\": \"<brief>\"}"
        )

        set_random_seed(seed)

        async def one_call(img_b64: str) -> tuple[int, str]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                },
            ]
            completion = await self.client.chat.completions.create(
                model=self.settings.vllm_model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=64,
                response_format=single_schema,
                seed=seed,
            )
            content = (completion.choices[0].message.content or "").strip()
            penalty_match = re.search(r'"penalty":\s*(\d+)', content)
            issues_match = re.search(r'"issues":\s*"([^"]*)"', content)
            penalty = int(penalty_match.group(1)) if penalty_match else 5
            issues = issues_match.group(1) if issues_match else ""
            return penalty, issues

        try:
            penalty_1, issues_1 = await one_call(img1_b64)
            penalty_2, issues_2 = await one_call(img2_b64)
            issues = f"1: {issues_1} | 2: {issues_2}" if (issues_1 or issues_2) else ""
            return JudgeResponse(penalty_1=penalty_1, penalty_2=penalty_2, issues=issues)
        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            return JudgeResponse(penalty_1=5, penalty_2=5, issues="vLLM call failed")
