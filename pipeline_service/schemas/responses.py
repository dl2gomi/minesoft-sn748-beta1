from typing import Optional

from pydantic import BaseModel


class GenerationResponse(BaseModel):
    generation_time: float
    glb_file_base64: Optional[str | bytes] = None
    grid_view_file_base64: Optional[str | bytes] = None
    image_edited_file_base64: Optional[str] = None
    image_without_background_file_base64: Optional[str] = None
    # Clarifier / multiview metadata (for logging; set when multiview_mode is dynamic)
    clarifier_score: Optional[float] = None
    multiview_used: Optional[bool] = None
    clarifier_explanation: Optional[str] = None
    object_category: Optional[str] = None
    object_category_confidence: Optional[float] = None
    # Extra debugging / logging metadata
    trellis_pipeline_type: Optional[str] = None
    suggested_pipeline_type: Optional[str] = None
    uv_unwrap_mode: Optional[str] = None
    uv_unwrap_reason: Optional[str] = None
    uv_num_charts: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "generation_time": 7.2,
                "glb_file_base64": "base64_encoded_glb_file",
                "grid_view_file_base64": "base64_encoded_grid_view_file",
                "image_edited_file_base64": "base64_encoded_image_edited_file",
                "image_without_background_file_base64": "base64_encoded_image_without_background_file",
                "clarifier_score": 0.45,
                "multiview_used": True,
                "clarifier_explanation": "Unseen sides are hard to infer.",
                "object_category": "plastic",
                "object_category_confidence": 0.72,
            }
        }
