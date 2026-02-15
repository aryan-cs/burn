from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from PIL import Image

from core.vlm_architecture import load_vlm_architecture_spec
from core.vlm_registry import VLMJobEntry, vlm_job_registry
from core.vlm_runtime import DEFAULT_VLM_MODEL_ID, vlm_runtime
from core.vlm_training_engine import run_vlm_training_job


router = APIRouter(prefix="/api/vlm", tags=["vlm"])
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_VLM_DATASET_ID = "synthetic_boxes_tiny"

VLM_DATASETS = [
    {
        "id": DEFAULT_VLM_DATASET_ID,
        "name": "Synthetic Boxes Tiny (Starter)",
        "task": "object_detection",
        "description": "Very small synthetic dataset profile for fast local warm-up runs.",
    },
    {
        "id": "synthetic_boxes_demo",
        "name": "Synthetic Boxes (Demo)",
        "task": "object_detection",
        "description": "Synthetic detection boxes used for local fine-tune demo runs.",
    },
]

VLM_MODELS = [
    {
        "id": DEFAULT_VLM_MODEL_ID,
        "name": "YOLOS Tiny",
        "provider": "huggingface",
        "task": "object_detection",
    },
    {
        "id": "facebook/detr-resnet-50",
        "name": "DETR ResNet-50",
        "provider": "huggingface",
        "task": "object_detection",
    },
]


class VLMTrainingConfigIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    dataset: str = DEFAULT_VLM_DATASET_ID
    model_id: str = DEFAULT_VLM_MODEL_ID
    epochs: int = 1
    batch_size: int = Field(
        default=1,
        validation_alias=AliasChoices("batch_size", "batchSize"),
        serialization_alias="batch_size",
    )
    steps_per_epoch: int = Field(
        default=1,
        validation_alias=AliasChoices("steps_per_epoch", "stepsPerEpoch"),
        serialization_alias="steps_per_epoch",
    )
    learning_rate: float = Field(
        default=1e-5,
        validation_alias=AliasChoices("learning_rate", "learningRate"),
        serialization_alias="learning_rate",
    )

    @field_validator("dataset")
    @classmethod
    def normalize_dataset(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("model_id")
    @classmethod
    def normalize_model_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("epochs", "batch_size", "steps_per_epoch")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be > 0")
        return value

    @field_validator("learning_rate")
    @classmethod
    def positive_lr(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("learning_rate must be > 0")
        return value


class VLMTrainRequest(BaseModel):
    training: VLMTrainingConfigIn | None = None


class VLMStopRequest(BaseModel):
    job_id: str | None = None


class VLMInferRequest(BaseModel):
    job_id: str | None = None
    image_base64: str
    score_threshold: float = 0.45
    max_detections: int = 25

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


def _entry_to_status(entry: VLMJobEntry) -> dict[str, Any]:
    return {
        "job_id": entry.job_id,
        "status": entry.status,
        "terminal": entry.terminal,
        "error": entry.error,
        "dataset": entry.dataset,
        "model_id": entry.model_id,
        "epochs": entry.epochs,
        "current_epoch": entry.current_epoch,
        "latest_loss": entry.latest_loss,
        "final_metrics": entry.final_metrics,
        "has_artifact": bool(entry.artifact_path and entry.artifact_path.exists()),
    }


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
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"message": f"Could not decode image payload: {exc}"}) from exc
    return image


def _resolve_job_artifact(job_id: str | None) -> tuple[str | None, Path | None, str]:
    if job_id:
        entry = vlm_job_registry.get(job_id)
        if entry is None:
            raise HTTPException(status_code=404, detail={"message": f"Unknown VLM job_id: {job_id}"})
        if entry.artifact_path is None or not entry.artifact_path.exists():
            return job_id, None, entry.model_id
        return job_id, entry.artifact_path, entry.model_id

    latest_job_id = vlm_job_registry.latest_job_id()
    if latest_job_id is None:
        return None, None, DEFAULT_VLM_MODEL_ID
    latest = vlm_job_registry.get(latest_job_id)
    if latest is None or latest.artifact_path is None or not latest.artifact_path.exists():
        if latest is None:
            return latest_job_id, None, DEFAULT_VLM_MODEL_ID
        return latest_job_id, None, latest.model_id
    return latest_job_id, latest.artifact_path, latest.model_id


@router.get("/datasets")
async def vlm_datasets() -> dict[str, Any]:
    return {"datasets": VLM_DATASETS, "models": VLM_MODELS}


@router.get("/architecture")
async def vlm_architecture(model_id: str = DEFAULT_VLM_MODEL_ID) -> dict[str, Any]:
    normalized = model_id.strip()
    if normalized == "":
        normalized = DEFAULT_VLM_MODEL_ID
    return load_vlm_architecture_spec(normalized)


@router.post("/train")
async def vlm_train(payload: VLMTrainRequest | None = None) -> dict[str, str]:
    training = payload.training if payload and payload.training else VLMTrainingConfigIn()
    allowed_datasets = {item["id"] for item in VLM_DATASETS}
    if training.dataset not in allowed_datasets:
        raise HTTPException(status_code=400, detail={"message": f"Unsupported VLM dataset: {training.dataset}"})

    entry = vlm_job_registry.create_job(
        dataset=training.dataset,
        epochs=training.epochs,
        learning_rate=training.learning_rate,
        batch_size=training.batch_size,
        steps_per_epoch=training.steps_per_epoch,
        model_id=training.model_id,
    )
    task = asyncio.create_task(
        run_vlm_training_job(
            entry.job_id,
            artifacts_root=ARTIFACTS_DIR,
        )
    )
    vlm_job_registry.set_task(entry.job_id, task)
    return {"job_id": entry.job_id, "status": entry.status}


@router.post("/stop")
async def vlm_stop(payload: VLMStopRequest | None = Body(default=None)) -> dict[str, str]:
    job_id = payload.job_id if payload else None
    if job_id is None:
        job_id = vlm_job_registry.latest_job_id()
    if job_id is None:
        raise HTTPException(status_code=400, detail={"message": "No VLM job to stop"})

    entry = vlm_job_registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown VLM job_id: {job_id}"})

    vlm_job_registry.request_stop(job_id)
    return {"job_id": job_id, "status": entry.status}


@router.get("/status")
async def vlm_status(job_id: str) -> dict[str, Any]:
    entry = vlm_job_registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown VLM job_id: {job_id}"})
    return _entry_to_status(entry)


@router.get("/latest")
async def vlm_latest() -> dict[str, Any]:
    latest_job_id = vlm_job_registry.latest_job_id()
    if latest_job_id is None:
        return {
            "job_id": None,
            "status": None,
            "terminal": None,
            "error": None,
            "has_artifact": False,
        }
    entry = vlm_job_registry.get(latest_job_id)
    if entry is None:
        return {
            "job_id": None,
            "status": None,
            "terminal": None,
            "error": None,
            "has_artifact": False,
        }
    return _entry_to_status(entry)


@router.post("/infer")
async def vlm_infer(payload: VLMInferRequest) -> dict[str, Any]:
    image = _decode_data_url_image(payload.image_base64)
    resolved_job_id, artifact_path, model_id = _resolve_job_artifact(payload.job_id)
    runtime = vlm_runtime.ensure_model(model_id)
    detection = vlm_runtime.detect(
        image,
        score_threshold=payload.score_threshold,
        max_detections=payload.max_detections,
        artifact_path=artifact_path,
    )
    return {
        "job_id_used": resolved_job_id,
        "runtime_backend": runtime.backend,
        "runtime_model_id": runtime.model_id,
        "image_width": detection["image_width"],
        "image_height": detection["image_height"],
        "detections": detection["detections"],
        "warning": detection.get("warning"),
    }
