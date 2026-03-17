"""
Load category config (GLB presets per category) from YAML.
Used by the GLB converter to apply material overrides from the decision module category.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_FALLBACK_GLB_PRESETS: dict[str, Any] = {
    "generic": {
        "roughness_scale": 0.7,
        "roughness_bias": 0.1,
        "color_saturation": 0.85,
        "color_brightness": 1.0,
    },
}


def load_category_config(path: Path) -> dict[str, Any]:
    """
    Load category config from a YAML file. Returns a dict with only "glb_presets"
    (category name -> GLBConverterParams overrides).
    """
    if not path.is_file():
        return {"glb_presets": dict(_FALLBACK_GLB_PRESETS)}

    try:
        data = yaml.safe_load(path.read_text())
    except Exception as e:
        raise ValueError(f"Failed to load category config from {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Category config must be a YAML object, got {type(data).__name__}")

    glb_presets = data.get("glb_presets")

    if not isinstance(glb_presets, dict) or not glb_presets:
        raise ValueError("Category config must contain a non-empty 'glb_presets' dict (category name -> param overrides)")
    glb_presets = {k: dict(v) if isinstance(v, dict) else {} for k, v in glb_presets.items()}

    return {"glb_presets": glb_presets}
