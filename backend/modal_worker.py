from __future__ import annotations

import base64
import io
import math
import os
import time
from typing import Any

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.graph_compiler import compile_graph
from datasets.loader import get_dataset_dataloaders
from models.graph_schema import GraphSpec
from models.training_config import normalize_training_config


app = modal.App("burn-training")
PROGRESS_DICT_NAME = "burn-training-progress"
progress_store = modal.Dict.from_name(PROGRESS_DICT_NAME, create_if_missing=True)
DEPLOYMENT_DICT_NAME = "burn-modal-deployments"
deployment_store = modal.Dict.from_name(DEPLOYMENT_DICT_NAME, create_if_missing=True)
SANDBOX_DEPLOYMENT_DICT_NAME = "burn-modal-sandbox-deployments"
sandbox_deployment_store = modal.Dict.from_name(SANDBOX_DEPLOYMENT_DICT_NAME, create_if_missing=True)
image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]>=0.115.0",
    "kaggle>=2.0.0",
    "torch>=2.9.0",
    "torchvision>=0.24.0",
    "pydantic>=2.11.0",
    "scikit-learn>=1.7.0",
).add_local_python_source(
    "core",
    "datasets",
    "models",
)


def _build_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float) -> torch.optim.Optimizer:
    params = model.parameters()
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _normalize_loss_name(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"cross_entropy", "crossentropy", "ce"}:
        return "cross_entropy"
    if key in {"mse", "mse_loss", "mean_squared_error"}:
        return "mse"
    raise ValueError(f"Unsupported loss function: {value}")


def _build_loss(loss_name: str) -> tuple[str, nn.Module]:
    canonical = _normalize_loss_name(loss_name)
    if canonical == "cross_entropy":
        return canonical, nn.CrossEntropyLoss()
    if canonical == "mse":
        return canonical, nn.MSELoss()
    raise ValueError(f"Unsupported loss function: {loss_name}")


def _compute_loss(
    loss_name: str,
    loss_fn: nn.Module,
    output: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if loss_name == "mse":
        if output.ndim < 2:
            raise ValueError("MSE loss requires model outputs with a class dimension")
        target_one_hot = F.one_hot(target.long(), num_classes=output.shape[-1]).to(
            dtype=output.dtype
        )
        return loss_fn(output, target_one_hot)
    return loss_fn(output, target)


def _state_dict_for_export(model: nn.Module) -> dict[str, torch.Tensor]:
    # torch.compile wraps the original module as `_orig_mod`; exporting the wrapped
    # state dict would prefix keys and break load_state_dict on the API side.
    inner_model = getattr(model, "_orig_mod", model)
    return inner_model.state_dict()


def _publish_progress(job_id: str | None, payload: dict[str, Any]) -> None:
    if not job_id:
        return
    progress_store[job_id] = payload


def _to_inference_tensor(raw_inputs: Any, expected_shape: list[int] | None) -> torch.Tensor:
    tensor = torch.tensor(raw_inputs, dtype=torch.float32)
    if tensor.ndim == 0:
        raise ValueError("inputs must be at least 1D")
    if expected_shape is None:
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
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
    raise ValueError(
        f"input tensor shape incompatible; expected={expected_shape} got={list(tensor.shape)}"
    )


def _load_deployed_model_from_store(deployment_id: str, store: modal.Dict):
    payload = store.get(deployment_id)
    if not isinstance(payload, dict):
        raise ValueError(f"Unknown modal deployment_id: {deployment_id}")

    graph_payload = payload.get("graph_payload")
    training_payload = payload.get("training_payload")
    state_dict_b64 = payload.get("state_dict_b64")
    input_shape = payload.get("input_shape")
    if not isinstance(graph_payload, dict):
        raise ValueError("Invalid deployment graph payload")
    if not isinstance(training_payload, dict):
        raise ValueError("Invalid deployment training payload")
    if not isinstance(state_dict_b64, str):
        raise ValueError("Invalid deployment model payload")

    graph = GraphSpec.model_validate(graph_payload)
    training = normalize_training_config(training_payload)
    compiled = compile_graph(graph, training)
    state_dict = torch.load(io.BytesIO(base64.b64decode(state_dict_b64.encode("ascii"))), map_location="cpu")
    inner_state_dict = {
        (k[len("_orig_mod.") :] if isinstance(k, str) and k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    compiled.model.load_state_dict(inner_state_dict)
    compiled.model.to("cpu")
    compiled.model.eval()
    normalized_input_shape: list[int] | None = None
    if isinstance(input_shape, list):
        normalized_input_shape = [int(v) for v in input_shape]
    return compiled.model, normalized_input_shape


def _evaluate_model(
    model: nn.Module,
    dataloader,
    loss_name: str,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = _compute_loss(loss_name, loss_fn, output, batch_y)
            total_loss += float(loss.item())
            correct += int((output.argmax(1) == batch_y).sum().item())
            total += int(batch_y.size(0))
            steps += 1
            if max_batches > 0 and steps >= max_batches:
                break

    if steps == 0:
        return 0.0, 0.0

    return total_loss / steps, correct / max(total, 1)


@app.function(image=image, gpu="H100", timeout=60 * 20)
def train_job_remote(
    graph_payload: dict[str, Any],
    training_payload: dict[str, Any],
    job_id: str | None = None,
) -> dict[str, Any]:
    graph = GraphSpec.model_validate(graph_payload)
    training = normalize_training_config(training_payload)
    print(
        (
            "[modal][train_job_remote] start "
            f"job_id={job_id or 'unknown'} "
            f"dataset={training.dataset} epochs={training.epochs} "
            f"batch_size={training.batch_size} optimizer={training.optimizer} "
            f"lr={training.learning_rate} loss={training.loss}"
        ),
        flush=True,
    )
    _publish_progress(
        job_id,
        {
            "status": "running",
            "epoch_metrics": [],
            "final_metrics": None,
            "error": None,
        },
    )

    compiled = compile_graph(graph, training)
    model = compiled.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[modal][train_job_remote] device={device}", flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_compile = os.getenv("MODAL_USE_TORCH_COMPILE", "1").strip().lower() in {"1", "true", "yes"}
    eval_every = max(1, int(os.getenv("MODAL_EVAL_EVERY", "1")))
    max_train_batches = max(0, int(os.getenv("MODAL_MAX_TRAIN_BATCHES", "0")))
    max_test_batches = max(0, int(os.getenv("MODAL_MAX_TEST_BATCHES", "0")))
    batch_multiplier = max(1, int(os.getenv("MODAL_BATCH_SIZE_MULTIPLIER", "1")))
    effective_batch_size = int(training.batch_size) * batch_multiplier
    if use_compile:
        try:
            model = torch.compile(model)
            print("[modal][train_job_remote] torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"[modal][train_job_remote] torch.compile skipped: {exc}", flush=True)

    train_loader, test_loader = get_dataset_dataloaders(training.dataset, effective_batch_size)
    print(
        (
            "[modal][train_job_remote] dataloaders_ready "
            f"train_batches={len(train_loader)} test_batches={len(test_loader)} "
            f"effective_batch_size={effective_batch_size} eval_every={eval_every} "
            f"max_train_batches={max_train_batches} max_test_batches={max_test_batches}"
        ),
        flush=True,
    )
    optimizer = _build_optimizer(model, training.optimizer, training.learning_rate)
    loss_name, loss_fn = _build_loss(training.loss)

    metrics: list[dict[str, float]] = []
    last_train_loss = 0.0
    last_train_accuracy = 0.0
    last_test_loss = 0.0
    last_test_accuracy = 0.0

    for epoch in range(training.epochs):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        steps = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch_x)
                loss = _compute_loss(loss_name, loss_fn, output, batch_y)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            correct += int((output.argmax(1) == batch_y).sum().item())
            total += int(batch_y.size(0))
            steps += 1
            if max_train_batches > 0 and steps >= max_train_batches:
                break

        if steps == 0:
            print("[modal][train_job_remote] no training steps; stopping early", flush=True)
            break

        last_train_loss = running_loss / max(steps, 1)
        last_train_accuracy = correct / max(total, 1)
        do_eval = (epoch + 1) % eval_every == 0 or (epoch + 1) == training.epochs
        if do_eval:
            last_test_loss, last_test_accuracy = _evaluate_model(
                model=model,
                dataloader=test_loader,
                loss_name=loss_name,
                loss_fn=loss_fn,
                device=device,
                max_batches=max_test_batches,
            )
        metrics.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": last_train_loss,
                "train_accuracy": last_train_accuracy,
                "test_loss": last_test_loss,
                "test_accuracy": last_test_accuracy,
            }
        )
        print(
            (
                f"[modal][train_job_remote] epoch={epoch + 1}/{training.epochs} "
                f"train_loss={last_train_loss:.6f} train_acc={last_train_accuracy:.4f} "
                f"test_loss={last_test_loss:.6f} test_acc={last_test_accuracy:.4f} "
                f"sec={time.perf_counter() - epoch_start:.2f}"
            ),
            flush=True,
        )
        _publish_progress(
            job_id,
            {
                "status": "running",
                "epoch_metrics": metrics,
                "final_metrics": None,
                "error": None,
            },
        )

    buffer = io.BytesIO()
    state_dict_cpu = {k: v.detach().to("cpu") for k, v in _state_dict_for_export(model).items()}
    torch.save(state_dict_cpu, buffer)
    encoded_state = base64.b64encode(buffer.getvalue()).decode("ascii")
    print(
        (
            "[modal][train_job_remote] complete "
            f"job_id={job_id or 'unknown'} "
            f"epochs_ran={len(metrics)} final_test_loss={last_test_loss:.6f} "
            f"final_test_acc={last_test_accuracy:.4f}"
        ),
        flush=True,
    )
    final_metrics = {
        "final_train_loss": last_train_loss,
        "final_train_accuracy": last_train_accuracy,
        "final_test_loss": last_test_loss,
        "final_test_accuracy": last_test_accuracy,
        "final_loss": last_test_loss,
        "final_accuracy": last_test_accuracy,
    }
    _publish_progress(
        job_id,
        {
            "status": "completed",
            "epoch_metrics": metrics,
            "final_metrics": final_metrics,
            "error": None,
        },
    )

    return {
        "epoch_metrics": metrics,
        "final_metrics": final_metrics,
        "state_dict_b64": encoded_state,
        "resolved_training": training.model_dump(),
    }


@app.function(image=image, timeout=60 * 5)
def register_deployment_remote(
    deployment_id: str,
    graph_payload: dict[str, Any],
    training_payload: dict[str, Any],
    state_dict_b64: str,
    input_shape: list[int] | None = None,
) -> dict[str, Any]:
    if not deployment_id:
        raise ValueError("deployment_id is required")
    deployment_store[deployment_id] = {
        "graph_payload": graph_payload,
        "training_payload": training_payload,
        "state_dict_b64": state_dict_b64,
        "input_shape": input_shape,
    }
    return {"deployment_id": deployment_id, "status": "ready"}


@app.function(image=image, timeout=60)
def unregister_deployment_remote(deployment_id: str) -> dict[str, Any]:
    try:
        del deployment_store[deployment_id]
    except KeyError:
        pass
    return {"deployment_id": deployment_id, "status": "deleted"}


def _infer_deployment_payload(payload: dict[str, Any], store: modal.Dict) -> dict[str, Any]:
    deployment_id = str(payload.get("deployment_id", "")).strip()
    if not deployment_id:
        raise ValueError("deployment_id is required")
    raw_inputs = payload.get("inputs")
    return_probabilities = bool(payload.get("return_probabilities", True))
    model, input_shape = _load_deployed_model_from_store(deployment_id, store)
    input_tensor = _to_inference_tensor(raw_inputs, input_shape)
    with torch.no_grad():
        output = model(input_tensor)
    if output.ndim == 1:
        output = output.unsqueeze(0)
    logits = output.detach().cpu()
    response: dict[str, Any] = {
        "deployment_id": deployment_id,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(logits.shape),
        "logits": logits.tolist(),
    }
    if logits.ndim == 2:
        response["predictions"] = logits.argmax(dim=1).tolist()
        if return_probabilities:
            response["probabilities"] = torch.softmax(logits, dim=1).tolist()
    return response


@app.function(image=image, timeout=60)
@modal.fastapi_endpoint(method="POST")
def infer_deployment_remote(payload: dict[str, Any]) -> dict[str, Any]:
    return _infer_deployment_payload(payload, deployment_store)


@app.function(image=image, timeout=60 * 5)
def register_sandbox_deployment_remote(
    deployment_id: str,
    graph_payload: dict[str, Any],
    training_payload: dict[str, Any],
    state_dict_b64: str,
    input_shape: list[int] | None = None,
) -> dict[str, Any]:
    if not deployment_id:
        raise ValueError("deployment_id is required")
    sandbox_deployment_store[deployment_id] = {
        "graph_payload": graph_payload,
        "training_payload": training_payload,
        "state_dict_b64": state_dict_b64,
        "input_shape": input_shape,
    }
    return {"deployment_id": deployment_id, "status": "ready"}


@app.function(image=image, timeout=60)
def unregister_sandbox_deployment_remote(deployment_id: str) -> dict[str, Any]:
    try:
        del sandbox_deployment_store[deployment_id]
    except KeyError:
        pass
    return {"deployment_id": deployment_id, "status": "deleted"}


@app.function(image=image, timeout=60)
@modal.fastapi_endpoint(method="POST")
def infer_sandbox_deployment_remote(payload: dict[str, Any]) -> dict[str, Any]:
    return _infer_deployment_payload(payload, sandbox_deployment_store)
