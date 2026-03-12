"""
Load object-category config (CLIP prompts + GLB presets) from YAML.
If the YAML file does not exist, returns a minimal config with generic category and GLB params.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Fallback when category_config.yaml is missing: single "generic" category and generic GLB params.
_FALLBACK_CONFIG: dict[str, Any] = {
    "categories": {
        "glass": [
            "a clear transparent glass bottle on a plain background",
        ],
        "metal": [
            "a polished shiny metal object on a plain background",
        ],
        "plastic": [
            "a smooth colorful plastic bottle on a plain background",
        ],
        "organic": [
            "an object made of wood, fabric, or other organic material",
        ],
    },
    "glb_presets": {
        "generic": {
            "roughness_scale": 0.7,
            "roughness_bias": 0.1,
            "color_saturation": 0.85,
            "color_brightness": 1.0,
        },
    },
    "category_confidence_threshold": 0.45,
}


def load_category_config(path: Path) -> dict[str, Any]:
    """
    Load category config from a YAML file.

    Expected top-level keys:
      - categories: dict of category name -> list of CLIP text prompts
      - glb_presets: dict of category name -> dict of GLBConverterParams overrides

    If the file does not exist, returns a minimal config with generic category and GLB params.

    Raises:
      ValueError: if the file exists but is invalid or missing required keys
    """
    if not path.is_file():
        return {"categories": dict(_FALLBACK_CONFIG["categories"]), "glb_presets": dict(_FALLBACK_CONFIG["glb_presets"])}

    try:
        data = yaml.safe_load(path.read_text())
    except Exception as e:
        raise ValueError(f"Failed to load category config from {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Category config must be a YAML object (key-value), got {type(data).__name__}")

    categories = data.get("categories")
    glb_presets = data.get("glb_presets")
    threshold = data.get("category_confidence_threshold", _FALLBACK_CONFIG["category_confidence_threshold"])

    if not isinstance(categories, dict) or not categories:
        raise ValueError("Category config must contain a non-empty 'categories' dict (category name -> list of CLIP prompts)")
    categories = {
        k: [str(p) for p in v] if isinstance(v, list) else []
        for k, v in categories.items()
    }

    if not isinstance(glb_presets, dict) or not glb_presets:
        raise ValueError("Category config must contain a non-empty 'glb_presets' dict (category name -> param overrides)")
    glb_presets = {k: dict(v) if isinstance(v, dict) else {} for k, v in glb_presets.items()}

    return {
        "categories": categories,
        "glb_presets": glb_presets,
        "category_confidence_threshold": float(threshold),
    }
