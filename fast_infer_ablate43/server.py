from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from runtime import Ablate43FastInference


app = FastAPI(title="Ablate43 Fast Inference API", version="1.0.0")
_engine: Ablate43FastInference | None = None


def get_engine() -> Ablate43FastInference:
    global _engine
    if _engine is None:
        _engine = Ablate43FastInference()
    return _engine


@app.get("/health")
def health() -> dict:
    eng = get_engine()
    return {
        "ok": True,
        "device": str(eng.device),
        "default_step_size": eng.default_step_size,
        "default_style_strength": eng.default_style_strength,
    }


@app.post("/infer/image")
async def infer_image(
    image: UploadFile = File(...),
    style_id: int = Query(..., ge=0),
    step_size: float | None = Query(None),
    style_strength: float | None = Query(None),
) -> Response:
    try:
        payload = await image.read()
        pil = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad image input: {exc}") from exc

    out = get_engine().infer_image(
        pil,
        style_id=style_id,
        step_size=step_size,
        style_strength=style_strength,
    )

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/infer/video")
async def infer_video(
    video: UploadFile = File(...),
    style_id: int = Query(..., ge=0),
    step_size: float | None = Query(None),
    style_strength: float | None = Query(None),
    batch_size: int = Query(8, ge=1, le=64),
) -> Response:
    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"

    with tempfile.TemporaryDirectory(prefix="ablate43_video_") as td:
        in_path = Path(td) / f"input{suffix}"
        out_path = Path(td) / "output.mp4"

        try:
            in_path.write_bytes(await video.read())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Bad video input: {exc}") from exc

        try:
            get_engine().infer_video(
                input_video=in_path,
                output_video=out_path,
                style_id=style_id,
                step_size=step_size,
                style_strength=style_strength,
                batch_size=batch_size,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Video inference failed: {exc}") from exc

        return Response(content=out_path.read_bytes(), media_type="video/mp4")
