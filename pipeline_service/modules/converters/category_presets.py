"""
Material presets per object category for GLB conversion.
Used when category is detected from the image (e.g. after background removal).
Presets are loaded from config (category_config.yaml); no built-in defaults.
"""
from __future__ import annotations

from typing import Dict, Any


def get_glb_overrides_for_category(category: str | None, presets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return GLBConverterParams overrides for the given category from the loaded presets, or empty dict if unknown/None."""
    if not category:
        return {}
    return dict(presets.get(category, {}))
