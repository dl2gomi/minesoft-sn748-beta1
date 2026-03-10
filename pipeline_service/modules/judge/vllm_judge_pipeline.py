from __future__ import annotations

import json
import re

import httpx
from openai import AsyncOpenAI

from logger_config import logger
from .judge_pipeline import JudgePipeline
from .prompting import SYSTEM_PROMPT, USER_PROMPT_IMAGE
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
        """Call vLLM to compare two candidate images against a prompt image."""
        assert self.client is not None, "VllmJudgePipeline is not initialized."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image prompt to generate 3D model:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{prompt_b64}"},
                    },
                    {"type": "text", "text": "First 3D model (4 different views):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img1_b64}"},
                    },
                    {"type": "text", "text": "Second 3D model (4 different views):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img2_b64}"},
                    },
                    {"type": "text", "text": USER_PROMPT_IMAGE},
                ],
            },
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge-response",
                "schema": JudgeResponse.model_json_schema(),
            },
        }

        # Set seed 
        set_random_seed(seed)

        try:
            completion = await self.client.chat.completions.create(
                model=self.settings.vllm_model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=32,
                response_format=response_format,
                seed=seed,
            )

            choice = completion.choices[0]
            finish_reason = choice.finish_reason
            if finish_reason == "length":
                logger.warning("vLLM response was truncated due to max_tokens limit")

            content = choice.message.content
            if not content:
                raise ValueError("Empty response from vLLM")

            content = content.strip()
            logger.debug(f"vLLM judge response (seed={seed}): {content}")

            return self._parse_content(content, finish_reason)

        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            return JudgeResponse(penalty_1=5, penalty_2=5, issues="vLLM call failed")
