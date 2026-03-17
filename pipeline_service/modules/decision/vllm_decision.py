from __future__ import annotations

import base64
import io
import json

from openai import AsyncOpenAI
from PIL import Image

from logger_config import logger

from .prompting import SYSTEM_PROMPT_DECISION, USER_PROMPT_DECISION
from .schemas import DEFAULT_DECISION, DecisionResponse, VALID_CATEGORIES


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _normalize_decision(raw: DecisionResponse) -> DecisionResponse:
    """Validate and normalize category and pipeline; return defaults for invalid values."""
    category = (raw.category or "generic").strip().lower()
    if category not in VALID_CATEGORIES:
        # Map common typos or variants
        if "glass" in category or category == "glasses":
            category = "glass"
        elif "metal" in category:
            category = "metal"
        elif "plastic" in category:
            category = "plastic"
        elif "wood" in category or "wooden" in category:
            category = "wood"
        elif "fabric" in category or "cloth" in category:
            category = "fabric"
        elif "ceramic" in category:
            category = "ceramic"
        elif "organic" in category:
            category = "organic"
        elif "mixed" in category or "multi" in category:
            category = "mixed"
        elif "clear" in category and "plastic" in category:
            category = "clearPlastic"
        else:
            category = "generic"
    # Default to 1024_cascade; use 512 only when explicitly chosen for very complex objects.
    pipeline = (raw.pipeline or "1024_cascade").strip().lower()
    if pipeline not in ("512", "1024_cascade"):
        pipeline = "1024_cascade" if pipeline == "1024" else "1024_cascade"
    return DecisionResponse(
        category=category,
        needs_multiview=bool(raw.needs_multiview),
        pipeline="512" if pipeline == "512" else "1024_cascade",
        explanation=raw.explanation,
    )


async def decide(
    image: Image.Image,
    client: AsyncOpenAI,
    model_name: str,
    seed: int = 42,
) -> DecisionResponse:
    """
    Single vLLM call: analyze the image and return category, needs_multiview, pipeline.
    Uses JSON schema response_format when supported; falls back to parsing raw content.
    """
    image_url = _image_to_data_url(image)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DECISION},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT_DECISION},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "decision-response",
            "schema": DecisionResponse.model_json_schema(),
        },
    }

    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
            seed=seed,
            response_format=response_format,
        )
        content = (completion.choices[0].message.content or "").strip()
        raw = DecisionResponse.model_validate_json(content)
        out = _normalize_decision(raw)
        logger.info(
            f"VLM decision: category={out.category} needs_multiview={out.needs_multiview} pipeline={out.pipeline}"
        )
        return out
    except Exception as e:
        if "response_format" in str(e).lower() or "json_schema" in str(e).lower():
            return await _decide_fallback(client, model_name, messages, seed, e)
        logger.warning(f"VLM decision call failed: {e}; using defaults.")
        return DEFAULT_DECISION


async def _decide_fallback(
    client: AsyncOpenAI,
    model_name: str,
    messages: list,
    seed: int,
    original_exc: Exception,
) -> DecisionResponse:
    """When response_format is not supported, request without it and parse JSON from content."""
    logger.warning(f"Decision response_format not supported, parsing JSON from content: {original_exc}")
    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
            seed=seed,
        )
        content = (completion.choices[0].message.content or "").strip()
        # Extract JSON object: find first { and matching }
        start = content.find("{")
        if start < 0:
            return DEFAULT_DECISION
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
            return DEFAULT_DECISION
        data = json.loads(content[start:end])
        raw = DecisionResponse(
            category=str(data.get("category", "generic")),
            needs_multiview=bool(data.get("needs_multiview", False)),
            pipeline="512" if str(data.get("pipeline", "1024_cascade")).strip() == "512" else "1024_cascade",
            explanation=str(data.get("explanation", ""))[:500] if data.get("explanation") else None,
        )
        return _normalize_decision(raw)
    except Exception as e:
        logger.warning(f"VLM decision fallback parse failed: {e}; using defaults.")
        return DEFAULT_DECISION
