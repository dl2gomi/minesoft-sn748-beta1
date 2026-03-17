from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DecisionResponse(BaseModel):
    """
    Single VLM decision for pipeline routing: category, multiview, and pipeline type.
    Returned by one vLLM call with the input image.
    """

    category: str = Field(
        description="Material/object category for GLB preset: one of glass, clearPlastic, metal, plastic, organic, fabric, wood, ceramic, mixed, generic"
    )
    needs_multiview: bool = Field(
        description="True if the object's geometry is ambiguous from one view and extra views would help 3D reconstruction"
    )
    pipeline: Literal["512", "1024_cascade"] = Field(
        description="Trellis pipeline: 512 for complex/detailed objects, 1024_cascade for very simple objects"
    )
    explanation: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Brief explanation of the decisions (for logging)",
    )


# Safe defaults when vLLM fails or returns invalid JSON
DEFAULT_DECISION = DecisionResponse(
    category="generic",
    needs_multiview=False,
    pipeline="1024_cascade",
    explanation="Fallback defaults (vLLM unavailable or parse failed)",
)

# Allowed category values (must match category_config.yaml keys)
VALID_CATEGORIES = frozenset(
    {"glass", "clearPlastic", "metal", "plastic", "organic", "fabric", "wood", "ceramic", "mixed", "generic"}
)
