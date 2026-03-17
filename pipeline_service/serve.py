from __future__ import annotations
import env_setup  # must be first — sets env vars before any library imports

from contextlib import asynccontextmanager
from io import BytesIO
import base64
import asyncio

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config.settings import settings, load_settings
from logger_config import logger
from schemas.requests import GenerationRequest
from schemas.responses import GenerationResponse
from modules.pipeline import GenerationPipeline
from modules.grid_renderer.render import GridViewRenderer
from utils import generate_chunks

renderer = GridViewRenderer()
pipeline = GenerationPipeline(settings, renderer=renderer)



@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.startup()
    try:
        yield
    finally:
        await pipeline.shutdown()

app = FastAPI(
    title=settings.api.api_title,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health() -> dict[str, str]:
    """
    Check if the service is running. 

    Returns:
        dict[str, str]: Status of the service
    """
    return {"status": "ready"}

@app.post("/generate_from_base64", response_model=GenerationResponse)
async def generate_from_base64(request: GenerationRequest) -> GenerationResponse:
    """
    Generate 3D model from base64 encoded image (JSON request).

    Returns JSON with generation_time and base64 encoded outputs.
    """
    current_settings = load_settings()
    try:
        result = await asyncio.wait_for(pipeline.generate(request, settings=current_settings), timeout=current_settings.api.timeout)

        if request.render_grid_view and result.glb_file_base64:
            grid_view_bytes = renderer.grid_from_glb_bytes(result.glb_file_base64)
            if grid_view_bytes is not None:
                result.grid_view_file_base64 = base64.b64encode(grid_view_bytes).decode("utf-8")

        if result.glb_file_base64:
            result.glb_file_base64 = base64.b64encode(result.glb_file_base64).decode("utf-8") # return bytes

        return result

    except asyncio.TimeoutError:
        logger.error(f"Generation timed out after {current_settings.api.timeout} seconds")
        raise HTTPException(status_code=408, detail="timeout") from None
    except Exception as exc:
        logger.exception(f"Error generating task: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate")
async def generate(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> StreamingResponse:
    """
    Upload image file and generate 3D model as GLB buffer.
    Returns binary GLB file. Response headers include metadata for results logging.
    """
    current_settings = load_settings()
    try:
        logger.info(f"Task received. Uploading image: {prompt_image_file.filename}")

        result = await asyncio.wait_for(
            pipeline.generate_from_upload(await prompt_image_file.read(), seed, input_filename=prompt_image_file.filename or None, settings=current_settings),
            timeout=current_settings.api.timeout
        )

        glb_bytes = result.glb_file_base64 if isinstance(result.glb_file_base64, bytes) else (base64.b64decode(result.glb_file_base64) if result.glb_file_base64 else b"")
        glb_buffer = BytesIO(glb_bytes)
        buffer_size = len(glb_bytes)
        glb_buffer.seek(0)

        def _h(s: str, max_len: int = 512) -> str:
            return (s or "").replace("\n", " ").strip()[:max_len]

        headers = {"Content-Length": str(buffer_size)}
        if result.generation_time is not None:
            headers["X-Generation-Time"] = f"{result.generation_time:.3f}"
        if result.multiview_used is not None:
            headers["X-Multiview-Used"] = "true" if result.multiview_used else "false"
        if result.object_category:
            headers["X-Object-Category"] = result.object_category
        if result.decision_pipeline:
            headers["X-Decision-Pipeline"] = str(result.decision_pipeline)
        if result.pipeline_used:
            headers["X-Pipeline-Used"] = result.pipeline_used
        if result.decision_explanation:
            headers["X-Decision-Explanation"] = _h(result.decision_explanation)
        if result.trellis_oom_retry is not None:
            headers["X-Trellis-OOM-Retry"] = "true" if result.trellis_oom_retry else "false"
        if result.uv_unwrap_mode:
            headers["X-UV-Unwrap-Mode"] = str(result.uv_unwrap_mode)
        if result.uv_unwrap_reason:
            headers["X-UV-Unwrap-Reason"] = str(result.uv_unwrap_reason)
        if result.uv_num_charts is not None:
            headers["X-UV-Num-Charts"] = str(result.uv_num_charts)
        if result.cluster_count is not None:
            headers["X-Cluster-Count"] = str(result.cluster_count)
        if result.duel_done is not None:
            headers["X-Duel-Done"] = "true" if result.duel_done else "false"
        if result.duel_winner is not None:
            headers["X-Duel-Winner"] = str(result.duel_winner)
        if result.duel_explanation:
            headers["X-Duel-Explanation"] = _h(result.duel_explanation)

        logger.info(f"Task completed. GLB size: {buffer_size} bytes")

        return StreamingResponse(
            generate_chunks(glb_buffer),
            media_type="application/octet-stream",
            headers=headers
        )

    except asyncio.TimeoutError:
        logger.error(f"Generation timed out after {current_settings.api.timeout} seconds")
        raise HTTPException(status_code=408, detail="timeout") from None
    except Exception as exc:
        logger.exception(f"Error generating from upload: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/setup/info")
async def get_setup_info() -> dict:
    """
    Get current pipeline configuration for experiment logging.
    Loads configuration from disk so changes take effect without restart.
    """
    try:
        return load_settings().model_dump()
    except Exception as e:
        logger.error(f"Failed to get setup info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )
