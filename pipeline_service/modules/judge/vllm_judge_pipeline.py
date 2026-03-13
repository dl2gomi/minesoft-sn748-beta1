from __future__ import annotations

import re

import httpx
from openai import AsyncOpenAI

from logger_config import logger
from .judge_pipeline import JudgePipeline
from .prompting import SYSTEM_PROMPT, JUDGE_SINGLE_IMAGE_USER
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

        user_text = JUDGE_SINGLE_IMAGE_USER

        set_random_seed(seed)

        def parse_penalty(content: str) -> int:
            """Extract a number 0-10 from model output. Default 5 if missing/invalid."""
            match = re.search(r"\b(10|[0-9])\b", content)
            if match:
                return min(10, max(0, int(match.group(1))))
            return 5

        async def one_call(img_b64: str, call_seed: int) -> tuple[int, str]:
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
                max_tokens=16,
                seed=call_seed,
            )
            content = (completion.choices[0].message.content or "").strip()
            return parse_penalty(content), ""

        # Second call uses a different seed (bounded so it never overflows 31-bit)
        seed_2 = (seed + 1) & 0x7FFFFFFF

        try:
            penalty_1, _ = await one_call(img1_b64, seed)
            penalty_2, _ = await one_call(img2_b64, seed_2)
            issues = ""
            return JudgeResponse(penalty_1=penalty_1, penalty_2=penalty_2, issues=issues)
        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            return JudgeResponse(penalty_1=5, penalty_2=5, issues="vLLM call failed")
