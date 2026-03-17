from typing import Optional

from pydantic import BaseModel


class GenerationResponse(BaseModel):
    generation_time: float
    glb_file_base64: Optional[str | bytes] = None
    grid_view_file_base64: Optional[str | bytes] = None
    image_edited_file_base64: Optional[str] = None
    image_without_background_file_base64: Optional[str] = None
    # Decision module (vLLM): multiview, category, pipeline
    multiview_used: Optional[bool] = None
    object_category: Optional[str] = None
    decision_pipeline: Optional[str] = None  # pipeline chosen by vLLM decision ("512" or "1024_cascade")
    pipeline_used: Optional[str] = None
    decision_explanation: Optional[str] = None
    trellis_oom_retry: Optional[bool] = None
    # UV unwrap (winner GLB)
    uv_unwrap_mode: Optional[str] = None
    uv_unwrap_reason: Optional[str] = None
    uv_num_charts: Optional[int] = None
    cluster_count: Optional[int] = None  # same as uv_num_charts when from xatlas/trivial
    # Duel (when 2+ candidates and judge ran)
    duel_done: Optional[bool] = None
    duel_winner: Optional[int] = None  # index of winning candidate (0, 1, ...)
    duel_explanation: Optional[str] = None  # issues from vLLM judge

    class Config:
        json_schema_extra = {
            "example": {
                "generation_time": 7.2,
                "multiview_used": True,
                "object_category": "plastic",
                "decision_pipeline": "1024_cascade",
                "pipeline_used": "512",
                "decision_explanation": "Object is simple; single view sufficient.",
                "trellis_oom_retry": True,
                "duel_done": True,
                "duel_winner": 1,
                "duel_explanation": "| Direct: First model has texture issues | Swapped: ...",
            }
        }
