from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from core.graph_compiler import GraphCompileError, compile_graph
from core.job_storage import (
    NN_ARTIFACT_FILENAME,
    load_job_metadata,
    load_python_source,
    model_job_dir,
    persist_job_bundle,
    update_job_metadata,
)
from core.job_registry import job_registry
from core.shape_inference import validate_graph
from core.training_engine import run_training_job
from datasets.registry import get_dataset_meta
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


def _resolve_persisted_latest_job_id() -> str | None:
    jobs_root = ARTIFACTS_DIR / "jobs"
    if not jobs_root.exists():
        return None

    latest_job_id: str | None = None
    latest_mtime = -1.0
    for candidate in jobs_root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_job_id = candidate.name

    return latest_job_id


def _resolve_artifact_path(job_id: str, entry_artifact: Path | None) -> Path | None:
    if entry_artifact is not None and entry_artifact.exists():
        return entry_artifact

    bundled = model_job_dir(ARTIFACTS_DIR, job_id) / NN_ARTIFACT_FILENAME
    if bundled.exists():
        return bundled

    legacy = ARTIFACTS_DIR / f"{job_id}.pt"
    if legacy.exists():
        return legacy

    return None


def _status_from_metadata(job_id: str) -> dict[str, Any] | None:
    job_dir = model_job_dir(ARTIFACTS_DIR, job_id)
    metadata = load_job_metadata(job_dir)
    if metadata is None:
        return None

    return {
        "job_id": job_id,
        "status": metadata.get("status"),
        "terminal": bool(metadata.get("terminal")),
        "error": metadata.get("error"),
        "final_metrics": metadata.get("final_metrics"),
        "has_python_source": bool(load_python_source(job_dir)),
        "has_artifact": _resolve_artifact_path(job_id, None) is not None,
    }


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


def _validate_dataset_contract(
    dataset_id: str,
    input_shape: list[int] | None,
    num_classes: int | None,
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    meta = get_dataset_meta(dataset_id)
    if meta is None:
        return [{"message": f"Unsupported dataset for v1: {dataset_id}"}]

    expected_input = meta.get("input_shape")
    expected_classes = meta.get("num_classes")
    dataset_name = str(meta.get("name", dataset_id))

    if isinstance(expected_input, list) and input_shape != expected_input:
        errors.append(
            {
                "message": f"{dataset_name} requires Input.shape={expected_input}",
                "expected": expected_input,
                "got": input_shape,
            }
        )

    if isinstance(expected_classes, int) and num_classes != expected_classes:
        errors.append(
            {
                "message": f"{dataset_name} requires Output.num_classes={expected_classes}",
                "expected": expected_classes,
                "got": num_classes,
            }
        )

    return errors


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

    if get_dataset_meta(training.dataset) is None:
        raise HTTPException(
            status_code=400,
            detail={"errors": [{"message": f"Unsupported dataset for v1: {training.dataset}"}]},
        )

    try:
        compiled = compile_graph(graph, training)
    except GraphCompileError as exc:
        raise HTTPException(status_code=400, detail={"errors": exc.errors}) from exc

    input_shape, num_classes = _extract_shapes_from_summary(compiled.summary)
    dataset_contract_errors = _validate_dataset_contract(training.dataset, input_shape, num_classes)
    if dataset_contract_errors:
        raise HTTPException(status_code=400, detail={"errors": dataset_contract_errors})

    entry = job_registry.create_job(
        compiled.python_source,
        model=compiled.model,
        input_shape=input_shape,
        num_classes=num_classes,
    )
    job_dir = model_job_dir(ARTIFACTS_DIR, entry.job_id)
    try:
        persist_job_bundle(
            job_dir,
            model_family="nn",
            python_source=compiled.python_source,
            graph_payload=graph.model_dump(mode="json", by_alias=True, exclude_none=True),
            training_payload=training.model_dump(mode="json"),
            summary_payload=compiled.summary,
            warnings=compiled.warnings,
        )
        job_registry.set_job_dir(entry.job_id, job_dir)
    except OSError as exc:
        await job_registry.mark_terminal(entry.job_id, "failed", error=f"Failed to persist job bundle: {exc}")
        raise HTTPException(status_code=500, detail={"message": f"Failed to persist job bundle: {exc}"}) from exc

    task = asyncio.create_task(run_training_job(entry.job_id, compiled, training, ARTIFACTS_DIR))
    job_registry.set_task(entry.job_id, task)

    return {
        "job_id": entry.job_id,
        "status": entry.status,
        "training_backend": os.getenv("TRAINING_BACKEND", "modal").strip().lower(),
        "modal_app_name": os.getenv("MODAL_APP_NAME", "burn-training"),
        "modal_function_name": os.getenv("MODAL_FUNCTION_NAME", "train_job_remote"),
    }


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
    if entry.job_dir is not None:
        update_job_metadata(entry.job_dir, status=entry.status, terminal=False)
    return {"job_id": job_id, "status": entry.status}


@router.get("/status")
async def status_model(job_id: str):
    entry = job_registry.get(job_id)
    if entry is None:
        persisted = _status_from_metadata(job_id)
        if persisted is None:
            raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})
        return persisted

    return {
        "job_id": entry.job_id,
        "status": entry.status,
        "terminal": entry.terminal,
        "error": entry.error,
        "final_metrics": entry.final_metrics,
        "has_python_source": bool(entry.python_source or load_python_source(model_job_dir(ARTIFACTS_DIR, job_id))),
        "has_artifact": _resolve_artifact_path(job_id, entry.artifact_path) is not None,
    }


@router.get("/latest")
async def latest_model_job():
    job_id = job_registry.latest_job_id()
    if job_id is None:
        job_id = _resolve_persisted_latest_job_id()
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
        persisted = _status_from_metadata(job_id)
        if persisted is None:
            return {
                "job_id": None,
                "status": None,
                "terminal": None,
                "error": None,
                "has_python_source": False,
                "has_artifact": False,
            }
        return persisted

    return {
        "job_id": entry.job_id,
        "status": entry.status,
        "terminal": entry.terminal,
        "error": entry.error,
        "has_python_source": bool(entry.python_source or load_python_source(model_job_dir(ARTIFACTS_DIR, job_id))),
        "has_artifact": _resolve_artifact_path(job_id, entry.artifact_path) is not None,
    }


@router.get("/export")
async def export_model(job_id: str, format: Literal["py", "pt"] = "py"):
    entry = job_registry.get(job_id)
    job_dir = model_job_dir(ARTIFACTS_DIR, job_id)

    if format == "py":
        source = entry.python_source if entry and entry.python_source else load_python_source(job_dir)
        if not source:
            raise HTTPException(status_code=404, detail={"message": f"Unknown job_id: {job_id}"})
        return PlainTextResponse(source, media_type="text/x-python")

    artifact_path = _resolve_artifact_path(job_id, entry.artifact_path if entry else None)
    if artifact_path is None:
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
