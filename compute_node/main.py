from __future__ import annotations

import base64
import io
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, field_validator

from vlm_runtime import DEFAULT_VLM_MODEL_ID, vlm_runtime


class VLMInferRequest(BaseModel):
    image_base64: str
    model_id: str = DEFAULT_VLM_MODEL_ID
    score_threshold: float = 0.45
    max_detections: int = 25

    @field_validator("model_id")
    @classmethod
    def normalize_model_id(cls, value: str) -> str:
        text = value.strip()
        if text == "":
            return DEFAULT_VLM_MODEL_ID
        return text

    @field_validator("score_threshold")
    @classmethod
    def score_in_range(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("score_threshold must be between 0 and 1")
        return value

    @field_validator("max_detections")
    @classmethod
    def positive_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_detections must be > 0")
        return value


def _decode_data_url_image(value: str) -> Image.Image:
    raw = value.strip()
    if raw == "":
        raise HTTPException(status_code=400, detail={"message": "image_base64 cannot be empty"})
    if "," in raw and raw.startswith("data:"):
        _, encoded = raw.split(",", maxsplit=1)
    else:
        encoded = raw
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"message": f"Invalid base64 image payload: {exc}"}) from exc
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"message": f"Could not decode image payload: {exc}"}) from exc


def _run_infer(payload: VLMInferRequest) -> dict[str, Any]:
    image = _decode_data_url_image(payload.image_base64)
    return vlm_runtime.detect(
        image,
        model_id=payload.model_id,
        score_threshold=payload.score_threshold,
        max_detections=payload.max_detections,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm up once at startup so the first request is less likely to stall.
    vlm_runtime.ensure_model(DEFAULT_VLM_MODEL_ID)
    yield


app = FastAPI(title="VLM Compute Node", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "device": vlm_runtime.device_name}


@app.post("/api/v1/vlm/infer")
async def vlm_infer(payload: VLMInferRequest) -> dict[str, Any]:
    return _run_infer(payload)


@app.websocket("/ws/v1/vlm/infer")
async def vlm_infer_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = VLMInferRequest.model_validate(json.loads(raw_message))
            except Exception as exc:
                await websocket.send_json({"type": "infer_error", "message": f"Invalid payload: {exc}"})
                continue

            try:
                result = _run_infer(payload)
                await websocket.send_json({"type": "infer_result", **result})
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
                await websocket.send_json({"type": "infer_error", **detail})
            except Exception as exc:
                await websocket.send_json({"type": "infer_error", "message": f"Inference failed: {exc}"})
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=False)
