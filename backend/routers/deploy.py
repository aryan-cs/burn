from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.deployment_registry import DeploymentEntry, deployment_registry
from core.graph_compiler import GraphCompileError, compile_graph
from core.job_registry import job_registry
from core.job_storage import GRAPH_FILENAME, NN_ARTIFACT_FILENAME, TRAINING_FILENAME, model_job_dir
from models.graph_schema import GraphSpec
from models.training_config import normalize_training_config


router = APIRouter(prefix="/api/deploy", tags=["deploy"])
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


class CreateDeploymentRequest(BaseModel):
    job_id: str
    target: str = "local"
    name: str | None = None


class DeploymentStatusResponse(BaseModel):
    deployment_id: str
    job_id: str
    status: str
    target: str
    endpoint_path: str
    created_at: str
    last_used_at: str | None = None
    request_count: int
    name: str | None = None


class DeploymentInferenceRequest(BaseModel):
    inputs: Any
    return_probabilities: bool = True


class DeploymentLogResponse(BaseModel):
    timestamp: str
    level: str
    event: str
    message: str
    details: dict[str, Any] | None = None


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
        return tensor.unsqueeze(0)
    if tensor.ndim == expected_rank + 1:
        return tensor
    if tensor.ndim == 1 and tensor.numel() == expected_elems:
        return tensor.view(1, *expected_shape)
    if tensor.ndim == 2 and tensor.shape[1] == expected_elems:
        return tensor.view(tensor.shape[0], *expected_shape)

    raise HTTPException(
        status_code=400,
        detail={
            "message": "Input tensor shape is incompatible with deployed model input shape",
            "expected_shape": expected_shape,
            "got_shape": list(tensor.shape),
        },
    )


def _artifact_path_for_job(job_id: str) -> Path:
    candidate = model_job_dir(ARTIFACTS_DIR, job_id) / NN_ARTIFACT_FILENAME
    if candidate.exists():
        return candidate

    legacy = ARTIFACTS_DIR / f"{job_id}.pt"
    if legacy.exists():
        return legacy

    raise HTTPException(status_code=404, detail={"message": f"No .pt artifact found for job_id: {job_id}"})


def _load_model_from_job(job_id: str) -> tuple[torch.nn.Module, list[int] | None, int | None]:
    job_dir = model_job_dir(ARTIFACTS_DIR, job_id)
    graph_path = job_dir / GRAPH_FILENAME
    if not graph_path.exists():
        # Backward compatibility for jobs created before bundle persistence
        # or still live in memory during the current process.
        entry = job_registry.get(job_id)
        if entry is not None and entry.model is not None:
            model = copy.deepcopy(entry.model).to("cpu")
            model.eval()
            return model, entry.input_shape, entry.num_classes
        raise HTTPException(
            status_code=404,
            detail={"message": f"No graph bundle found for job_id: {job_id}. Retrain or restart with a recent job."},
        )

    try:
        graph_payload = GraphSpec.model_validate_json(graph_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"message": f"Failed to parse graph bundle: {exc}"}) from exc

    training_path = job_dir / TRAINING_FILENAME
    training_payload: dict[str, Any] | None = None
    if training_path.exists():
        try:
            loaded = json.loads(training_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                training_payload = loaded
        except Exception:
            training_payload = None

    training = normalize_training_config(training_payload)

    try:
        compiled = compile_graph(graph_payload, training)
    except GraphCompileError as exc:
        first = exc.errors[0]["message"] if exc.errors else "unknown compile error"
        raise HTTPException(status_code=500, detail={"message": f"Failed to rebuild model: {first}"}) from exc

    artifact_path = _artifact_path_for_job(job_id)
    try:
        state_dict = torch.load(artifact_path, map_location="cpu")
        compiled.model.load_state_dict(state_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"message": f"Failed to load model artifact: {exc}"}) from exc

    compiled.model.eval()
    input_shape, num_classes = _extract_shapes_from_summary(compiled.summary)
    return compiled.model, input_shape, num_classes


def _entry_to_status(entry: DeploymentEntry) -> DeploymentStatusResponse:
    return DeploymentStatusResponse(
        deployment_id=entry.deployment_id,
        job_id=entry.job_id,
        status=entry.status,
        target=entry.target,
        endpoint_path=entry.endpoint_path,
        created_at=entry.created_at.isoformat(),
        last_used_at=entry.last_used_at.isoformat() if isinstance(entry.last_used_at, datetime) else None,
        request_count=entry.request_count,
        name=entry.name,
    )


def _entry_to_log_response(entry) -> DeploymentLogResponse:
    return DeploymentLogResponse(
        timestamp=entry.timestamp.isoformat(),
        level=entry.level,
        event=entry.event,
        message=entry.message,
        details=entry.details,
    )


@router.post("")
async def create_deployment(payload: CreateDeploymentRequest) -> DeploymentStatusResponse:
    target = payload.target.strip().lower()
    if target != "local":
        raise HTTPException(
            status_code=400,
            detail={"message": "Only local deployment is supported right now. Remote targets are planned next."},
        )

    model, input_shape, num_classes = _load_model_from_job(payload.job_id)
    entry = deployment_registry.create_deployment(
        job_id=payload.job_id,
        target=target,
        name=payload.name,
        model=model,
        input_shape=input_shape,
        num_classes=num_classes,
    )
    return _entry_to_status(entry)


@router.get("/list")
async def list_deployments() -> dict[str, list[DeploymentStatusResponse]]:
    return {"deployments": [_entry_to_status(entry) for entry in deployment_registry.list()]}


@router.get("/status")
async def deployment_status(deployment_id: str) -> DeploymentStatusResponse:
    entry = deployment_registry.get(deployment_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown deployment_id: {deployment_id}"})
    return _entry_to_status(entry)


@router.get("/logs")
async def deployment_logs(deployment_id: str, limit: int = 200) -> dict[str, Any]:
    entry = deployment_registry.get(deployment_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown deployment_id: {deployment_id}"})

    safe_limit = min(max(limit, 1), 500)
    logs = deployment_registry.logs(deployment_id, limit=safe_limit)
    return {
        "deployment_id": deployment_id,
        "logs": [_entry_to_log_response(log).model_dump(mode="json") for log in logs],
    }


@router.delete("/{deployment_id}")
async def stop_deployment(deployment_id: str) -> DeploymentStatusResponse:
    entry = deployment_registry.get(deployment_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown deployment_id: {deployment_id}"})

    deployment_registry.mark_stopped(deployment_id)
    return _entry_to_status(entry)


@router.post("/{deployment_id}/start")
async def start_deployment(deployment_id: str) -> DeploymentStatusResponse:
    entry = deployment_registry.get(deployment_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown deployment_id: {deployment_id}"})

    if entry.status == "running":
        return _entry_to_status(entry)

    if entry.model is None:
        model, input_shape, num_classes = _load_model_from_job(entry.job_id)
        entry.model = model
        entry.input_shape = input_shape
        entry.num_classes = num_classes

    deployment_registry.mark_running(deployment_id)
    return _entry_to_status(entry)


@router.post("/{deployment_id}/infer")
async def infer_deployment(deployment_id: str, payload: DeploymentInferenceRequest) -> dict[str, Any]:
    entry = deployment_registry.get(deployment_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown deployment_id: {deployment_id}"})
    if entry.status != "running":
        raise HTTPException(status_code=400, detail={"message": "Deployment is not running"})
    if entry.model is None:
        raise HTTPException(status_code=500, detail={"message": "Deployment model is unavailable"})

    model = entry.model.to("cpu")
    model.eval()
    try:
        input_tensor = _to_inference_tensor(payload.inputs, entry.input_shape)
    except HTTPException as exc:
        deployment_registry.add_log(
            deployment_id,
            level="error",
            event="inference_request_invalid",
            message="Inference request rejected due to incompatible input.",
            details={"status_code": exc.status_code, "detail": exc.detail},
        )
        raise

    with torch.no_grad():
        output = model(input_tensor)

    if output.ndim == 1:
        output = output.unsqueeze(0)

    logits = output.detach().cpu()
    deployment_registry.mark_request(deployment_id)
    response: dict[str, Any] = {
        "deployment_id": deployment_id,
        "job_id": entry.job_id,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(logits.shape),
        "logits": logits.tolist(),
    }
    if logits.ndim == 2:
        response["predictions"] = logits.argmax(dim=1).tolist()
        if payload.return_probabilities:
            response["probabilities"] = torch.softmax(logits, dim=1).tolist()

    deployment_registry.add_log(
        deployment_id,
        level="info",
        event="inference_request",
        message="Inference request handled successfully.",
        details={
            "request_count": entry.request_count,
            "input_shape": response["input_shape"],
            "output_shape": response["output_shape"],
            "prediction_count": len(response.get("predictions", [])),
        },
    )
    return response
