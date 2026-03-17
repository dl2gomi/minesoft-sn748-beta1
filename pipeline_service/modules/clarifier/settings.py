from typing import Optional

from pydantic import BaseModel, ConfigDict


class ClarifierConfig(BaseModel):
    """
    Pre-decision step (category, multiview, pipeline). When enabled,
    the decision module (vLLM) returns all three in one call.
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    # If false, category will be ignored for GLB texture/material mapping (no per-category presets),
    # while multiview/pipeline decisions can still be used.
    use_category_clarification: bool = True
    category_config_path: Optional[str] = None
