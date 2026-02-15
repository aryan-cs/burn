from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.deployment_registry import DeploymentEntry, deployment_registry
from core.graph_compiler import GraphCompileError, compile_graph
from core.job_registry import job_registry
from core.job_storage import (
    GRAPH_FILENAME,
    NN_ARTIFACT_FILENAME,
    SUMMARY_FILENAME,
    TRAINING_FILENAME,
    model_job_dir,
)
from models.graph_schema import GraphSpec
from models.training_config import normalize_training_config


router = APIRouter(prefix="/api/deploy", tags=["deploy"])
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "burn-training")
MODAL_ENVIRONMENT_NAME = os.getenv("MODAL_ENVIRONMENT_NAME")
MODAL_DEPLOY_REGISTER_FUNCTION = os.getenv("MODAL_DEPLOY_REGISTER_FUNCTION", "register_deployment_remote")
MODAL_DEPLOY_UNREGISTER_FUNCTION = os.getenv("MODAL_DEPLOY_UNREGISTER_FUNCTION", "unregister_deployment_remote")
MODAL_DEPLOY_INFER_FUNCTION = os.getenv("MODAL_DEPLOY_INFER_FUNCTION", "infer_deployment_remote")
MODAL_SANDBOX_DEPLOY_REGISTER_FUNCTION = os.getenv(
    "MODAL_SANDBOX_DEPLOY_REGISTER_FUNCTION",
    "register_sandbox_deployment_remote",
)
MODAL_SANDBOX_DEPLOY_UNREGISTER_FUNCTION = os.getenv(
    "MODAL_SANDBOX_DEPLOY_UNREGISTER_FUNCTION",
    "unregister_sandbox_deployment_remote",
)
MODAL_SANDBOX_DEPLOY_INFER_FUNCTION = os.getenv(
    "MODAL_SANDBOX_DEPLOY_INFER_FUNCTION",
    "infer_sandbox_deployment_remote",
)


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
    model_family: str = "nn"
    created_at: str
    last_used_at: str | None = None
    request_count: int
    name: str | None = None


class DeploymentInferenceRequest(BaseModel):
    inputs: Any
    return_probabilities: bool = True


class CreateExternalDeploymentRequest(BaseModel):
    model_family: str
    target: str = "local"
    endpoint_path: str | None = None
    name: str | None = None
    job_id: str | None = None
    runtime_config: dict[str, Any] | None = None


class DeploymentTouchRequest(BaseModel):
    event: str = "external_inference_request"
    message: str = "External inference request handled."
    details: dict[str, Any] | None = None


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

    # Accept a common shorthand for single-channel images:
    # [H, W] -> [1, H, W] before adding batch.
    if (
        expected_rank >= 2
        and expected_shape[0] == 1
        and tensor.ndim == expected_rank - 1
        and list(tensor.shape) == expected_shape[1:]
    ):
        tensor = tensor.unsqueeze(0)

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


def _load_modal_deployment_bundle(job_id: str) -> tuple[dict[str, Any], dict[str, Any], str, list[int] | None]:
    job_dir = model_job_dir(ARTIFACTS_DIR, job_id)
    graph_path = job_dir / GRAPH_FILENAME
    if not graph_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"message": f"No graph bundle found for job_id: {job_id}. Retrain first."},
        )
    training_path = job_dir / TRAINING_FILENAME
    if not training_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"message": f"No training bundle found for job_id: {job_id}. Retrain first."},
        )

    try:
        graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
        training_payload = json.loads(training_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"message": f"Failed to read job bundle: {exc}"}) from exc
    if not isinstance(graph_payload, dict) or not isinstance(training_payload, dict):
        raise HTTPException(status_code=500, detail={"message": "Invalid graph/training bundle payload"})

    artifact_path = _artifact_path_for_job(job_id)
    try:
        encoded_state = base64.b64encode(artifact_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise HTTPException(status_code=500, detail={"message": f"Failed to read model artifact: {exc}"}) from exc

    input_shape: list[int] | None = None
    summary_path = job_dir / SUMMARY_FILENAME
    if summary_path.exists():
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary_payload, dict):
                input_shape, _ = _extract_shapes_from_summary(summary_payload)
        except Exception:
            input_shape = None
    return graph_payload, training_payload, encoded_state, input_shape


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
        model_family=entry.model_family,
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


def _normalize_model_family(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized in {"nn", "linreg"}:
        return normalized
    raise HTTPException(
        status_code=400,
        detail={"message": f"Unsupported model_family: {raw}. Supported: nn, linreg"},
    )


def _to_linreg_samples(raw_inputs: Any, expected_features: int) -> list[list[float]]:
    if not isinstance(raw_inputs, list) or len(raw_inputs) == 0:
        raise HTTPException(status_code=400, detail={"message": "inputs must be a non-empty list"})

    # Single sample: [x1, x2, ...]
    if all(not isinstance(item, (list, tuple)) for item in raw_inputs):
        candidates = [raw_inputs]
    elif all(isinstance(item, (list, tuple)) for item in raw_inputs):
        candidates = raw_inputs
    else:
        raise HTTPException(
            status_code=400,
            detail={"message": "inputs must be either a feature vector or a batch of feature vectors"},
        )

    samples: list[list[float]] = []
    for sample in candidates:
        if not isinstance(sample, (list, tuple)):
            raise HTTPException(status_code=400, detail={"message": "Invalid sample format"})
        if len(sample) != expected_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Input vector size is incompatible with deployed linear regression model",
                    "expected_features": expected_features,
                    "got_features": len(sample),
                },
            )
        try:
            samples.append([float(value) for value in sample])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail={"message": f"Non-numeric input value: {exc}"}) from exc

    return samples


def _resolve_modal_function_names(target: str) -> tuple[str, str, str]:
    if target == "sandbox":
        return (
            MODAL_SANDBOX_DEPLOY_REGISTER_FUNCTION,
            MODAL_SANDBOX_DEPLOY_UNREGISTER_FUNCTION,
            MODAL_SANDBOX_DEPLOY_INFER_FUNCTION,
        )
    return (
        MODAL_DEPLOY_REGISTER_FUNCTION,
        MODAL_DEPLOY_UNREGISTER_FUNCTION,
        MODAL_DEPLOY_INFER_FUNCTION,
    )


def _infer_linreg(deployment_id: str, entry: DeploymentEntry, payload: DeploymentInferenceRequest) -> dict[str, Any]:
    runtime = entry.runtime_config
    if not isinstance(runtime, dict):
        raise HTTPException(
            status_code=500,
            detail={"message": "Linear regression deployment runtime config is unavailable"},
        )

    raw_weights = runtime.get("weights")
    raw_means = runtime.get("means")
    raw_stds = runtime.get("stds")
    raw_bias = runtime.get("bias", 0.0)
    if not isinstance(raw_weights, list) or len(raw_weights) == 0:
        raise HTTPException(status_code=500, detail={"message": "Invalid linear regression runtime weights"})
    if not isinstance(raw_means, list) or len(raw_means) != len(raw_weights):
        raise HTTPException(status_code=500, detail={"message": "Invalid linear regression runtime means"})
    if not isinstance(raw_stds, list) or len(raw_stds) != len(raw_weights):
        raise HTTPException(status_code=500, detail={"message": "Invalid linear regression runtime stds"})

    try:
        weights = [float(value) for value in raw_weights]
        means = [float(value) for value in raw_means]
        stds = [float(value) if abs(float(value)) > 1e-12 else 1.0 for value in raw_stds]
        bias = float(raw_bias)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail={"message": f"Invalid linear regression runtime payload: {exc}"}) from exc

    samples = _to_linreg_samples(payload.inputs, len(weights))

    predictions: list[float] = []
    for sample in samples:
        normalized = [
            (sample[index] - means[index]) / stds[index]
            for index in range(len(weights))
        ]
        prediction = sum(normalized[index] * weights[index] for index in range(len(weights))) + bias
        predictions.append(float(prediction))

    deployment_registry.mark_request(deployment_id)
    deployment_registry.add_log(
        deployment_id,
        level="info",
        event="inference_request",
        message="Inference request handled successfully.",
        details={
            "model_family": "linreg",
            "request_count": entry.request_count,
            "batch_size": len(samples),
            "feature_count": len(weights),
        },
    )

    return {
        "deployment_id": deployment_id,
        "job_id": entry.job_id,
        "model_family": "linreg",
        "input_shape": [len(samples), len(weights)],
        "output_shape": [len(samples), 1],
        "predictions": predictions,
    }


@router.post("")
async def create_deployment(payload: CreateDeploymentRequest) -> DeploymentStatusResponse:
    target = payload.target.strip().lower()
    if target == "cloud":
        target = "modal"
    if target not in {"local", "modal", "sandbox"}:
        raise HTTPException(
            status_code=400,
            detail={"message": "Supported deployment targets are: local, modal, sandbox"},
        )

    if target == "local":
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

    graph_payload, training_payload, encoded_state, input_shape = _load_modal_deployment_bundle(payload.job_id)
    register_fn_name, _, infer_fn_name = _resolve_modal_function_names(target)
    try:
        import modal

        register_fn = modal.Function.from_name(
            MODAL_APP_NAME,
            register_fn_name,
            environment_name=MODAL_ENVIRONMENT_NAME,
        )
        infer_fn = modal.Function.from_name(
            MODAL_APP_NAME,
            infer_fn_name,
            environment_name=MODAL_ENVIRONMENT_NAME,
        )
        endpoint_url = await asyncio.to_thread(infer_fn.get_web_url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"message": f"Failed to resolve Modal deployment functions: {exc}"}) from exc

    if not isinstance(endpoint_url, str) or endpoint_url.strip() == "":
        raise HTTPException(status_code=500, detail={"message": "Modal web endpoint URL is unavailable"})

    entry = deployment_registry.create_deployment(
        job_id=payload.job_id,
        target=target,
        name=payload.name,
        model=None,
        input_shape=input_shape,
        num_classes=None,
        endpoint_path=endpoint_url,
    )
    deployment_registry.add_log(
        entry.deployment_id,
        level="info",
        event="modal_endpoint_registered",
        message=f"Modal {target} web endpoint registered for this deployment.",
        details={"modal_web_url": endpoint_url, "target": target},
    )
    try:
        await asyncio.to_thread(
            register_fn.remote,
            entry.deployment_id,
            graph_payload,
            training_payload,
            encoded_state,
            input_shape,
        )
    except Exception as exc:
        deployment_registry.mark_stopped(entry.deployment_id)
        raise HTTPException(status_code=500, detail={"message": f"Failed to register Modal deployment: {exc}"}) from exc
    return _entry_to_status(entry)


@router.post("/external")
async def create_external_deployment(payload: CreateExternalDeploymentRequest) -> DeploymentStatusResponse:
    target = payload.target.strip().lower()
    if target != "local":
        raise HTTPException(
            status_code=400,
            detail={"message": "Only local deployment is supported right now. Remote targets are planned next."},
        )

    model_family = _normalize_model_family(payload.model_family)
    endpoint_path = payload.endpoint_path.strip() if isinstance(payload.endpoint_path, str) else ""
    if endpoint_path and not endpoint_path.startswith("/"):
        raise HTTPException(
            status_code=400,
            detail={"message": "endpoint_path must start with '/' when provided."},
        )

    job_id = (payload.job_id or "").strip()
    if not job_id:
        job_id = f"{model_family}_{uuid4().hex[:12]}"

    entry = deployment_registry.create_deployment(
        job_id=job_id,
        target=target,
        name=payload.name,
        model=None,
        input_shape=None,
        num_classes=None,
        model_family=model_family,
        endpoint_path=endpoint_path or None,
        runtime_config=payload.runtime_config,
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

    if entry.target in {"modal", "sandbox"}:
        _, unregister_fn_name, _ = _resolve_modal_function_names(entry.target)
        try:
            import modal

            unregister_fn = modal.Function.from_name(
                MODAL_APP_NAME,
                unregister_fn_name,
                environment_name=MODAL_ENVIRONMENT_NAME,
            )
            await asyncio.to_thread(unregister_fn.remote, deployment_id)
        except Exception:
            # Best-effort cleanup; local status should still transition.
            pass

    deployment_registry.mark_stopped(deployment_id)
    return _entry_to_status(entry)


@router.post("/{deployment_id}/start")
async def start_deployment(deployment_id: str) -> DeploymentStatusResponse:
    entry = deployment_registry.get(deployment_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown deployment_id: {deployment_id}"})

    if entry.status == "running":
        return _entry_to_status(entry)

    if entry.target in {"modal", "sandbox"}:
        register_fn_name, _, _ = _resolve_modal_function_names(entry.target)
        graph_payload, training_payload, encoded_state, input_shape = _load_modal_deployment_bundle(entry.job_id)
        try:
            import modal

            register_fn = modal.Function.from_name(
                MODAL_APP_NAME,
                register_fn_name,
                environment_name=MODAL_ENVIRONMENT_NAME,
            )
            await asyncio.to_thread(
                register_fn.remote,
                deployment_id,
                graph_payload,
                training_payload,
                encoded_state,
                input_shape,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail={"message": f"Failed to restart Modal deployment: {exc}"}) from exc
        deployment_registry.mark_running(deployment_id)
        return _entry_to_status(entry)

    if entry.model_family == "nn" and entry.model is None:
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
    if entry.target in {"modal", "sandbox"}:
        _, _, infer_fn_name = _resolve_modal_function_names(entry.target)
        try:
            import modal

            infer_fn = modal.Function.from_name(
                MODAL_APP_NAME,
                infer_fn_name,
                environment_name=MODAL_ENVIRONMENT_NAME,
            )
            modal_response = await asyncio.to_thread(
                infer_fn.remote,
                {
                    "deployment_id": deployment_id,
                    "inputs": payload.inputs,
                    "return_probabilities": payload.return_probabilities,
                },
            )
        except Exception as exc:
            deployment_registry.add_log(
                deployment_id,
                level="error",
                event="inference_request_failed",
                message=f"Modal {entry.target} inference request failed.",
                details={"error": str(exc), "target": entry.target},
            )
            raise HTTPException(status_code=500, detail={"message": f"Modal inference failed: {exc}"}) from exc
        if not isinstance(modal_response, dict):
            raise HTTPException(status_code=500, detail={"message": "Modal inference returned invalid payload"})
        modal_response["deployment_id"] = deployment_id
        modal_response["job_id"] = entry.job_id
        deployment_registry.mark_request(deployment_id)
        deployment_registry.add_log(
            deployment_id,
            level="info",
            event="inference_request",
            message=f"Inference request handled successfully via Modal {entry.target}.",
            details={
                "request_count": entry.request_count,
                "target": entry.target,
                "input_shape": modal_response.get("input_shape"),
                "output_shape": modal_response.get("output_shape"),
            },
        )
        return modal_response

    if entry.model_family == "linreg":
        return _infer_linreg(deployment_id, entry, payload)

    if entry.model_family != "nn":
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Inference for model_family '{entry.model_family}' is not supported through /api/deploy/{deployment_id}/infer",
            },
        )

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


@router.post("/{deployment_id}/touch")
async def touch_deployment(deployment_id: str, payload: DeploymentTouchRequest) -> DeploymentStatusResponse:
    entry = deployment_registry.get(deployment_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"message": f"Unknown deployment_id: {deployment_id}"})

    deployment_registry.mark_request(deployment_id)
    deployment_registry.add_log(
        deployment_id,
        level="info",
        event=payload.event,
        message=payload.message,
        details=payload.details,
    )
    return _entry_to_status(entry)
