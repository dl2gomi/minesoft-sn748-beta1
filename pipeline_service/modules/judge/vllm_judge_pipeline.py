from __future__ import annotations

import json
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

    async def judge(
        self,
        prompt_b64: str,
        img1_b64: str,
        img2_b64: str,
        seed: int,
    ) -> JudgeResponse:
        """
        Call vLLM once to compare two candidates (prompt + left + right images).
        Request JSON output and parse into JudgeResponse.
        """
        assert self.client is not None, "VllmJudgePipeline is not initialized."

        set_random_seed(seed)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image prompt to generate 3D model:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prompt_b64}"}},
                    {"type": "text", "text": "First 3D model (4 different views):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                    {"type": "text", "text": "Second 3D model (4 different views):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}},
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

        try:
            completion = await self.client.chat.completions.create(
                model=self.settings.vllm_model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                seed=seed,
                response_format=response_format,
            )

            content = (completion.choices[0].message.content or "").strip()
            return JudgeResponse.model_validate_json(content)
        except Exception as e:
            # If response_format isn't supported (or the model emits non-JSON),
            # retry once without response_format and extract JSON from raw content.
            if "response_format" in str(e).lower() or "json_schema" in str(e).lower():
                return await self._judge_retry_parse(messages=messages, seed=seed, original_exc=e)
            logger.error(f"vLLM judge call failed: {e}")
            return JudgeResponse(penalty_1=5, penalty_2=5, issues="vLLM call failed")

    async def _judge_retry_parse(
        self,
        *,
        messages: list,
        seed: int,
        original_exc: Exception,
    ) -> JudgeResponse:
        """Retry without response_format and parse JSON from raw model content."""
        assert self.client is not None, "VllmJudgePipeline is not initialized."
        logger.warning(f"Judge response_format not supported, parsing JSON from content: {original_exc}")
        try:
            completion = await self.client.chat.completions.create(
                model=self.settings.vllm_model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                seed=seed,
            )
            content = (completion.choices[0].message.content or "").strip()
            # Extract JSON object: find first { and matching }
            start = content.find("{")
            if start < 0:
                return JudgeResponse(penalty_1=5, penalty_2=5, issues="")
            depth = 1
            pos = start + 1
            end = -1
            while pos < len(content):
                if content[pos] == "{":
                    depth += 1
                elif content[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        end = pos + 1
                        break
                pos += 1
            if end < 0:
                return JudgeResponse(penalty_1=5, penalty_2=5, issues="")
            data = json.loads(content[start:end])
            return JudgeResponse(
                penalty_1=int(data.get("penalty_1", 5)),
                penalty_2=int(data.get("penalty_2", 5)),
                issues=str(data.get("issues", ""))[:500],
            )
        except Exception as e:
            logger.warning(f"VLM judge fallback parse failed: {e}; using safe defaults.")
            return JudgeResponse(penalty_1=5, penalty_2=5, issues="vLLM call failed")
