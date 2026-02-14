from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from core.graph_compiler import GraphCompileError, compile_graph
from core.job_registry import job_registry
from core.shape_inference import validate_graph
from core.training_engine import run_training_job
from models.graph_schema import GraphSpec
from models.training_config import normalize_training_config


router = APIRouter(prefix="/api/model", tags=["model"])
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


class StopRequest(BaseModel):
    job_id: str | None = None


class InferenceRequest(BaseModel):
    job_id: str
    inputs: Any
    return_probabilities: bool = True


def _extract_shapes_from_summary(summary: dict) -> tuple[list[int] | None, int | None]:
    input_shape: list[int] | None = None
    num_classes: int | None = None
    for layer in summary.get("layers", []):
        layer_type = layer.get("type")
        if layer_type == "Input":
            output_shape = layer.get("output_shape")
            if isinstance(output_shape, list):
                input_shape = [int(v) for v in output_shape]
        elif layer_type == "Output":
            output_shape = layer.get("output_shape")
            if (
                isinstance(output_shape, list)
                and len(output_shape) == 1
                and isinstance(output_shape[0], int)
            ):
                num_classes = int(output_shape[0])
    return input_shape, num_classes


def _to_inference_tensor(raw_inputs: Any, expected_shape: list[int] | None) -> torch.Tensor:
    try:
        tensor = torch.tensor(raw_inputs, dtype=torch.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"message": f"Invalid inputs payload: {exc}"}) from exc

    if tensor.ndim == 0:
        raise HTTPException(status_code=400, detail={"message": "inputs must be at least 1D"})

    if expected_shape is None:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    expected_rank = len(expected_shape)
    expected_elems = math.prod(expected_shape)

    if tensor.ndim == expected_rank:
        tensor = tensor.unsqueeze(0)
        return tensor

    if tensor.ndim == expected_rank + 1:
        return tensor

    if tensor.ndim == 1 and tensor.numel() == expected_elems:
        return tensor.view(1, *expected_shape)

    if tensor.ndim == 2 and tensor.shape[1] == expected_elems:
        return tensor.view(tensor.shape[0], *expected_shape)

    raise HTTPException(
        status_code=400,
        detail={
            "message": "Input tensor shape is incompatible with model input shape",
            "expected_shape": expected_shape,
            "got_shape": list(tensor.shape),
        },
    )


@router.post("/validate")
async def validate_model(graph: GraphSpec):
    result = validate_graph(graph)
    return {
        "valid": result.valid,
        "shapes": result.shapes,
        "errors": result.errors,
        "execution_order": result.execution_order,
        "warnings": result.warnings,
    }


@router.post("/compile")
async def compile_model(graph: GraphSpec):
    training = normalize_training_config(graph.training)
    try:
        compiled = compile_graph(graph, training)
    except GraphCompileError as exc:
        raise HTTPException(status_code=400, detail={"errors": exc.errors}) from exc

    return {
        "valid": True,
        "summary": compiled.summary,
        "python_source": compiled.python_source,
        "warnings": compiled.warnings,
    }


@router.post("/train")
async def train_model(graph: GraphSpec):
    training = normalize_training_config(graph.training)

    if training.dataset != "mnist":
        raise HTTPException(
            status_code=400,
            detail={"errors": [{"message": f"Unsupported dataset for v1: {training.dataset}"}]},
        )

    try:
        compiled = compile_graph(graph, training)
    except GraphCompileError as exc:
        raise HTTPException(status_code=400, detail={"errors": exc.errors}) from exc

    input_shape, num_classes = _extract_shapes_from_summary(compiled.summary)
    entry = job_registry.create_job(
        compiled.python_source,
        model=compiled.model,
        input_shape=input_shape,
        num_classes=num_classes,
    )
    task = asyncio.create_task(run_training_job(entry.job_id, compiled, training, ARTIFACTS_DIR))
    job_registry.set_task(entry.job_id, task)

    return {"job_id": entry.job_id, "status": entry.status}


@router.post("/stop")
async def stop_model(payload: StopRequest | None = Body(default=None)):
    job_id = payload.job_id if payload else None
    if job_id is None:
        job_id = job_registry.latest_job_id()

    if job_id is None:
        raise HTTPException(status_code=400, detail={"message": "No active job to stop"})

    entry = job_registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})

    job_registry.request_stop(job_id)
    return {"job_id": job_id, "status": entry.status}


@router.get("/status")
async def status_model(job_id: str):
    entry = job_registry.get(job_id)
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
async def latest_model_job():
    job_id = job_registry.latest_job_id()
    if job_id is None:
        return {
            "job_id": None,
            "status": None,
            "terminal": None,
            "error": None,
            "has_python_source": False,
            "has_artifact": False,
        }

    entry = job_registry.get(job_id)
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
async def export_model(job_id: str, format: Literal["py", "pt"] = "py"):
    entry = job_registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})

    if format == "py":
        return PlainTextResponse(entry.python_source, media_type="text/x-python")

    artifact_path = entry.artifact_path
    if artifact_path is None or not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"message": "No exported .pt artifact yet for this job"},
        )

    return FileResponse(path=artifact_path, filename=f"{job_id}.pt", media_type="application/octet-stream")


@router.post("/infer")
async def infer_model(payload: InferenceRequest):
    entry = job_registry.get(payload.job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {payload.job_id}"})
    if entry.model is None:
        raise HTTPException(status_code=400, detail={"message": "No compiled model available for this job"})

    model = entry.model.to("cpu")
    model.eval()
    input_tensor = _to_inference_tensor(payload.inputs, entry.input_shape)

    with torch.no_grad():
        output = model(input_tensor)

    if output.ndim == 1:
        output = output.unsqueeze(0)

    logits = output.detach().cpu()
    response: dict[str, object] = {
        "job_id": payload.job_id,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(logits.shape),
        "logits": logits.tolist(),
    }

    if logits.ndim == 2:
        response["predictions"] = logits.argmax(dim=1).tolist()
        if payload.return_probabilities:
            response["probabilities"] = torch.softmax(logits, dim=1).tolist()

    return response
