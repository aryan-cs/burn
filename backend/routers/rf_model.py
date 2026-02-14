from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from core.rf_compiler import RFCompileError, compile_rf_graph
from core.rf_job_registry import rf_job_registry
from core.rf_shape_inference import validate_rf_graph
from core.rf_training_engine import run_rf_training_job
from datasets.rf_registry import get_rf_dataset_meta
from models.rf_graph_schema import RFGraphSpec
from models.rf_training_config import normalize_rf_training_config


router = APIRouter(prefix="/api/rf", tags=["rf-model"])
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


class StopRequest(BaseModel):
    job_id: str | None = None


class InferRequest(BaseModel):
    job_id: str
    inputs: Any
    return_probabilities: bool = True


def _to_feature_matrix(raw_inputs: Any, expected_feature_count: int | None) -> np.ndarray:
    try:
        matrix = np.asarray(raw_inputs, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"message": f"Invalid inputs payload: {exc}"}) from exc

    if matrix.ndim == 0:
        raise HTTPException(status_code=400, detail={"message": "inputs must be at least 1D"})
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail={"message": "inputs must be a 1D feature vector or a 2D feature matrix"},
        )

    if expected_feature_count is not None and int(matrix.shape[1]) != expected_feature_count:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Input feature count mismatch",
                "expected_feature_count": expected_feature_count,
                "got_feature_count": int(matrix.shape[1]),
            },
        )

    return matrix


def _load_artifact_into_entry(job_id: str) -> None:
    entry = rf_job_registry.get(job_id)
    if entry is None:
        return
    if entry.model is not None:
        return
    if entry.artifact_path is None or not entry.artifact_path.exists():
        return

    payload = joblib.load(entry.artifact_path)
    if isinstance(payload, dict) and "model" in payload:
        entry.model = payload.get("model")
        entry.feature_names = payload.get("feature_names")
        entry.class_names = payload.get("class_names")
        expected = payload.get("expected_feature_count")
        if isinstance(expected, int):
            entry.expected_feature_count = expected
    else:
        entry.model = payload


@router.post("/validate")
async def validate_rf_model(graph: RFGraphSpec):
    result = validate_rf_graph(graph)
    return {
        "valid": result.valid,
        "shapes": result.shapes,
        "errors": result.errors,
        "execution_order": result.execution_order,
        "warnings": result.warnings,
    }


@router.post("/compile")
async def compile_rf_model(graph: RFGraphSpec):
    training = normalize_rf_training_config(graph.training)
    try:
        compiled = compile_rf_graph(graph, training)
    except RFCompileError as exc:
        raise HTTPException(status_code=400, detail={"errors": exc.errors}) from exc

    return {
        "valid": True,
        "summary": compiled.summary,
        "python_source": compiled.python_source,
        "warnings": compiled.warnings,
    }


@router.post("/train")
async def train_rf_model(graph: RFGraphSpec):
    training = normalize_rf_training_config(graph.training)
    if get_rf_dataset_meta(training.dataset) is None:
        raise HTTPException(
            status_code=400,
            detail={"errors": [{"message": f"Unsupported RF dataset: {training.dataset}"}]},
        )

    try:
        compiled = compile_rf_graph(graph, training)
    except RFCompileError as exc:
        raise HTTPException(status_code=400, detail={"errors": exc.errors}) from exc

    entry = rf_job_registry.create_job(
        compiled.python_source,
        expected_feature_count=compiled.expected_feature_count,
    )
    task = asyncio.create_task(run_rf_training_job(entry.job_id, compiled, training, ARTIFACTS_DIR))
    rf_job_registry.set_task(entry.job_id, task)
    return {"job_id": entry.job_id, "status": entry.status}


@router.post("/stop")
async def stop_rf_model(payload: StopRequest | None = Body(default=None)):
    job_id = payload.job_id if payload else None
    if job_id is None:
        job_id = rf_job_registry.latest_job_id()
    if job_id is None:
        raise HTTPException(status_code=400, detail={"message": "No active RF job to stop"})

    entry = rf_job_registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})

    rf_job_registry.request_stop(job_id)
    return {"job_id": job_id, "status": entry.status}


@router.get("/status")
async def status_rf_model(job_id: str):
    entry = rf_job_registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})

    return {
        "job_id": entry.job_id,
        "status": entry.status,
        "terminal": entry.terminal,
        "error": entry.error,
        "final_metrics": entry.final_metrics,
        "has_python_source": bool(entry.python_source),
        "has_artifact": bool(entry.artifact_path and entry.artifact_path.exists()),
    }


@router.get("/latest")
async def latest_rf_job():
    job_id = rf_job_registry.latest_job_id()
    if job_id is None:
        return {
            "job_id": None,
            "status": None,
            "terminal": None,
            "error": None,
            "has_python_source": False,
            "has_artifact": False,
        }

    entry = rf_job_registry.get(job_id)
    if entry is None:
        return {
            "job_id": None,
            "status": None,
            "terminal": None,
            "error": None,
            "has_python_source": False,
            "has_artifact": False,
        }

    return {
        "job_id": entry.job_id,
        "status": entry.status,
        "terminal": entry.terminal,
        "error": entry.error,
        "has_python_source": bool(entry.python_source),
        "has_artifact": bool(entry.artifact_path and entry.artifact_path.exists()),
    }


@router.get("/export")
async def export_rf_model(job_id: str, format: Literal["py", "pkl"] = "py"):
    entry = rf_job_registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})

    if format == "py":
        return PlainTextResponse(entry.python_source, media_type="text/x-python")

    artifact_path = entry.artifact_path
    if artifact_path is None or not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"message": "No exported .pkl artifact yet for this job"},
        )

    return FileResponse(path=artifact_path, filename=f"{job_id}.pkl", media_type="application/octet-stream")


@router.post("/infer")
async def infer_rf_model(payload: InferRequest):
    entry = rf_job_registry.get(payload.job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {payload.job_id}"})

    _load_artifact_into_entry(payload.job_id)
    if entry.model is None:
        raise HTTPException(status_code=400, detail={"message": "No trained RF model available for this job"})

    matrix = _to_feature_matrix(payload.inputs, entry.expected_feature_count)
    model = entry.model
    try:
        prediction_indices = model.predict(matrix)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"message": f"Inference failed: {exc}"}) from exc

    index_values = [int(value) for value in np.asarray(prediction_indices).tolist()]
    class_names = entry.class_names or []
    mapped_predictions: list[str | int] = []
    if class_names:
        for value in index_values:
            if 0 <= value < len(class_names):
                mapped_predictions.append(class_names[value])
            else:
                mapped_predictions.append(value)
    else:
        mapped_predictions = index_values

    response: dict[str, Any] = {
        "job_id": payload.job_id,
        "input_shape": [int(value) for value in matrix.shape],
        "prediction_indices": index_values,
        "predictions": mapped_predictions,
        "classes": class_names,
    }

    if payload.return_probabilities and hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(matrix)
            response["probabilities"] = np.asarray(probabilities, dtype=np.float64).tolist()
        except Exception:
            response["probabilities"] = None

    return response
