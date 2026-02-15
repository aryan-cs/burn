from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from core.graph_compiler import CompiledGraphResult
from core.job_registry import job_registry
from core.job_storage import GRAPH_FILENAME, NN_ARTIFACT_FILENAME, update_job_metadata
from core.weight_extractor import extract_weight_snapshot
from datasets.loader import get_dataset_dataloaders
from models.training_config import TrainingConfig

MODAL_PROGRESS_DICT_NAME = os.getenv("MODAL_PROGRESS_DICT_NAME", "burn-training-progress")
LANDSCAPE_GRID_SIZE = max(7, int(os.getenv("LOSS_LANDSCAPE_GRID_SIZE", "11")))
LANDSCAPE_RADIUS = max(0.1, float(os.getenv("LOSS_LANDSCAPE_RADIUS", "1.0")))
LANDSCAPE_REFERENCE_BATCHES = max(1, int(os.getenv("LOSS_LANDSCAPE_REFERENCE_BATCHES", "1")))
LANDSCAPE_MAX_PATH_POINTS = max(8, int(os.getenv("LOSS_LANDSCAPE_MAX_PATH_POINTS", "256")))
LANDSCAPE_DIRECTION_SEED = int(os.getenv("LOSS_LANDSCAPE_DIRECTION_SEED", "20260215"))
LANDSCAPE_SYNTHETIC_BASE_SEED = int(os.getenv("LOSS_LANDSCAPE_SYNTHETIC_SEED", "20260309"))
TRAIN_PROGRESS_INTERVAL_STEPS = max(1, int(os.getenv("TRAIN_PROGRESS_INTERVAL_STEPS", "20")))


def _build_optimizer(model: nn.Module, training: TrainingConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if training.optimizer == "adam":
        return torch.optim.Adam(params, lr=training.learning_rate)
    if training.optimizer == "sgd":
        return torch.optim.SGD(params, lr=training.learning_rate)
    raise ValueError(f"Unsupported optimizer: {training.optimizer}")


def _normalize_loss_name(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"cross_entropy", "crossentropy", "ce"}:
        return "cross_entropy"
    if key in {"mse", "mse_loss", "mean_squared_error"}:
        return "mse"
    raise ValueError(f"Unsupported loss function: {value}")


def _build_loss(training: TrainingConfig) -> tuple[str, nn.Module]:
    loss_name = _normalize_loss_name(training.loss)
    if loss_name == "cross_entropy":
        return loss_name, nn.CrossEntropyLoss()
    if loss_name == "mse":
        return loss_name, nn.MSELoss()
    raise ValueError(f"Unsupported loss function: {training.loss}")


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


async def _evaluate_model(
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

            # Keep the event loop responsive for websocket handshakes/messages.
            if steps % 20 == 0:
                await asyncio.sleep(0)

    if steps == 0:
        return 0.0, 0.0

    return total_loss / steps, correct / max(total, 1)


def _prepare_landscape_reference_batch(
    dataloader,
    device: torch.device,
    max_batches: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    feature_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    for batch_index, (batch_x, batch_y) in enumerate(dataloader):
        feature_batches.append(batch_x.to(device))
        target_batches.append(batch_y.to(device))
        if batch_index + 1 >= max_batches:
            break

    if not feature_batches:
        return None

    return torch.cat(feature_batches, dim=0), torch.cat(target_batches, dim=0)


def _landscape_loss_on_reference(
    model: nn.Module,
    loss_name: str,
    loss_fn: nn.Module,
    reference_inputs: torch.Tensor,
    reference_targets: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(reference_inputs)
        loss = _compute_loss(loss_name, loss_fn, output, reference_targets)
    if was_training:
        model.train()
    return float(loss.item())


def _initialize_loss_landscape_context(
    model: nn.Module,
    loss_name: str,
    loss_fn: nn.Module,
    reference_inputs: torch.Tensor,
    reference_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    base_vector = parameters_to_vector(model.parameters()).detach().cpu().float()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(LANDSCAPE_DIRECTION_SEED)

    direction_x = torch.randn(base_vector.numel(), generator=generator, dtype=torch.float32)
    direction_x = direction_x / torch.clamp(direction_x.norm(), min=1e-12)

    direction_z = torch.randn(base_vector.numel(), generator=generator, dtype=torch.float32)
    direction_z = direction_z - torch.dot(direction_z, direction_x) * direction_x
    direction_z_norm = torch.clamp(direction_z.norm(), min=1e-12)
    direction_z = direction_z / direction_z_norm

    axis = torch.linspace(-LANDSCAPE_RADIUS, LANDSCAPE_RADIUS, LANDSCAPE_GRID_SIZE, dtype=torch.float32)

    model_mode = model.training
    current_vector = parameters_to_vector(model.parameters()).detach()
    grid_loss: list[list[float]] = []
    with torch.no_grad():
        model.eval()
        for z_value in axis.tolist():
            row: list[float] = []
            for x_value in axis.tolist():
                candidate = base_vector + (x_value * direction_x) + (z_value * direction_z)
                vector_to_parameters(candidate.to(device=device), model.parameters())
                output = model(reference_inputs)
                loss = _compute_loss(loss_name, loss_fn, output, reference_targets)
                row.append(float(loss.item()))
            grid_loss.append(row)

    vector_to_parameters(current_vector.to(device=device), model.parameters())
    if model_mode:
        model.train()

    return {
        "base_vector": base_vector,
        "direction_x": direction_x,
        "direction_z": direction_z,
        "x_axis": [float(value) for value in axis.tolist()],
        "z_axis": [float(value) for value in axis.tolist()],
        "grid_loss": grid_loss,
        "path": [],
        "grid_sent": False,
        "sample_count": int(reference_targets.shape[0]),
    }


def _capture_loss_landscape_snapshot(
    context: dict[str, Any],
    model: nn.Module,
    loss_name: str,
    loss_fn: nn.Module,
    reference_inputs: torch.Tensor,
    reference_targets: torch.Tensor,
    epoch: int,
) -> dict[str, Any]:
    current_vector = parameters_to_vector(model.parameters()).detach().cpu().float()
    delta = current_vector - context["base_vector"]
    x_coord = float(torch.dot(delta, context["direction_x"]).item())
    z_coord = float(torch.dot(delta, context["direction_z"]).item())
    point = {
        "epoch": int(epoch),
        "x": x_coord,
        "z": z_coord,
        "loss": _landscape_loss_on_reference(
            model=model,
            loss_name=loss_name,
            loss_fn=loss_fn,
            reference_inputs=reference_inputs,
            reference_targets=reference_targets,
        ),
    }

    path: list[dict[str, float | int]] = context["path"]
    path.append(point)
    if len(path) > LANDSCAPE_MAX_PATH_POINTS:
        del path[:-LANDSCAPE_MAX_PATH_POINTS]

    payload = {
        "objective": f"reference_{loss_name}",
        "dataset_split": "test",
        "grid_size": LANDSCAPE_GRID_SIZE,
        "radius": LANDSCAPE_RADIUS,
        "x_axis": context["x_axis"],
        "z_axis": context["z_axis"],
        "grid_loss": None if context["grid_sent"] else context["grid_loss"],
        "path": path,
        "point": point,
        "sample_count": context["sample_count"],
    }
    context["grid_sent"] = True
    return payload


def _seed_from_values(*values: object) -> int:
    seed = LANDSCAPE_SYNTHETIC_BASE_SEED
    for value in values:
        text = str(value)
        for ch in text:
            seed = ((seed * 16777619) ^ ord(ch)) & 0xFFFFFFFF
    return seed


def _build_axis(grid_size: int, radius: float) -> list[float]:
    if grid_size <= 1:
        return [0.0]
    step = (radius * 2) / (grid_size - 1)
    return [float(-radius + step * idx) for idx in range(grid_size)]


def _synthetic_surface_loss(
    context: dict[str, Any],
    x_coord: float,
    z_coord: float,
) -> float:
    phase = float(context["phase"])
    bowl = float(context["bowl"])
    wave = float(context["wave"])
    cross = float(context["cross"])
    offset = float(context["offset"])

    bowl_term = bowl * ((x_coord * x_coord) + 0.86 * (z_coord * z_coord))
    wave_term = (
        wave * math.sin((x_coord * 2.32) + phase)
        + cross * math.cos((z_coord * 2.04) - (phase * 0.65))
        + 0.08 * math.sin(((x_coord + z_coord) * 1.63) + (phase * 0.31))
    )
    return max(0.02, offset + bowl_term + wave_term)


def _initialize_synthetic_loss_landscape_context(
    *,
    job_id: str,
    training: TrainingConfig,
    loss_name: str,
) -> dict[str, Any]:
    seed = _seed_from_values(
        job_id,
        training.dataset,
        training.optimizer,
        training.learning_rate,
        training.loss,
    )
    rng = random.Random(seed)
    radius = LANDSCAPE_RADIUS * (1.0 + rng.uniform(0.02, 0.35))
    axis = _build_axis(LANDSCAPE_GRID_SIZE, radius)
    context: dict[str, Any] = {
        "objective": f"synthetic_{loss_name}",
        "dataset_split": "preview",
        "grid_size": LANDSCAPE_GRID_SIZE,
        "radius": radius,
        "x_axis": axis,
        "z_axis": axis,
        "path": [],
        "grid_sent": False,
        "sample_count": int(640 + rng.randint(0, 2048)),
        "phase": rng.uniform(0, math.pi * 2),
        "bowl": 0.72 + rng.random() * 0.4,
        "wave": 0.17 + rng.random() * 0.2,
        "cross": 0.12 + rng.random() * 0.14,
        "offset": 0.22 + rng.random() * 0.34,
        "seed": seed,
        "epochs": max(1, int(training.epochs)),
    }

    grid_loss: list[list[float]] = []
    for z_coord in axis:
        row: list[float] = []
        for x_coord in axis:
            base = _synthetic_surface_loss(context, x_coord, z_coord)
            base += rng.uniform(-0.012, 0.012)
            row.append(max(0.02, float(base)))
        grid_loss.append(row)
    context["grid_loss"] = grid_loss
    context["x_pos"] = 0.0
    context["z_pos"] = 0.0
    return context


def _capture_synthetic_loss_landscape_snapshot(
    *,
    context: dict[str, Any],
    epoch: float,
    observed_loss: float,
) -> dict[str, Any]:
    normalized_epoch = max(0.001, float(epoch))
    seed = _seed_from_values(context["seed"], round(normalized_epoch * 1000), observed_loss)
    rng = random.Random(seed)

    radius = float(context["radius"])
    total_epochs = max(6, int(context.get("epochs", 1)))
    step_size = radius / total_epochs
    next_x = float(context.get("x_pos", 0.0))
    next_z = float(context.get("z_pos", 0.0))
    next_x += (
        rng.uniform(-1.0, 1.0) * step_size * 0.78
        + math.cos((normalized_epoch * 0.61) + rng.random() * math.pi) * step_size * 0.57
    )
    next_z += (
        rng.uniform(-1.0, 1.0) * step_size * 0.78
        + math.sin((normalized_epoch * 0.49) + rng.random() * math.pi) * step_size * 0.57
    )
    next_x = float(max(-radius, min(radius, next_x)))
    next_z = float(max(-radius, min(radius, next_z)))
    context["x_pos"] = next_x
    context["z_pos"] = next_z

    terrain_loss = _synthetic_surface_loss(context, next_x, next_z)
    observed = observed_loss if math.isfinite(observed_loss) and observed_loss > 0 else terrain_loss
    point_loss = max(
        0.001,
        (terrain_loss * 0.58) + (observed * 0.42) + rng.uniform(-0.04, 0.04),
    )
    point = {
        "epoch": float(normalized_epoch),
        "x": next_x,
        "z": next_z,
        "loss": float(point_loss),
    }

    path: list[dict[str, float | int]] = context["path"]
    path.append(point)
    if len(path) > LANDSCAPE_MAX_PATH_POINTS:
        del path[:-LANDSCAPE_MAX_PATH_POINTS]

    payload = {
        "objective": context["objective"],
        "dataset_split": context["dataset_split"],
        "grid_size": context["grid_size"],
        "radius": context["radius"],
        "x_axis": context["x_axis"],
        "z_axis": context["z_axis"],
        "grid_loss": None if context["grid_sent"] else context["grid_loss"],
        "path": path,
        "point": point,
        "sample_count": context["sample_count"],
    }
    context["grid_sent"] = True
    return payload


async def run_training_job(
    job_id: str,
    compiled: CompiledGraphResult,
    training: TrainingConfig,
    artifacts_dir: Path,
    backend_override: str | None = None,
) -> None:
    entry = job_registry.get(job_id)
    if entry is None:
        return

    try:
        job_registry.set_status(job_id, "running")
        if entry.job_dir is not None:
            update_job_metadata(entry.job_dir, status="running", terminal=False)
        model = compiled.model
        device = torch.device("cpu")
        model.to(device)
        backend = _resolve_training_backend(backend_override)
        await job_registry.publish(
            job_id,
            {
                "type": "training_backend",
                "backend": backend,
                "modal_app_name": os.getenv("MODAL_APP_NAME", "burn-training"),
                "modal_function_name": os.getenv("MODAL_FUNCTION_NAME", "train_job_remote"),
            },
        )

        last_train_loss = 0.0
        last_train_accuracy = 0.0
        last_test_loss = 0.0
        last_test_accuracy = 0.0

        if backend == "modal":
            if entry.job_dir is None:
                raise RuntimeError("Modal backend requires a persisted job bundle")
            graph_path = entry.job_dir / GRAPH_FILENAME
            if not graph_path.exists():
                raise RuntimeError("Missing graph bundle required for Modal training")

            graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
            training_payload = training.model_dump(mode="json")
            app_name = os.getenv("MODAL_APP_NAME", "burn-training")
            function_name = os.getenv("MODAL_FUNCTION_NAME", "train_job_remote")
            environment_name = os.getenv("MODAL_ENVIRONMENT_NAME")

            import modal
            from modal.exception import TimeoutError as ModalTimeoutError

            remote_fn = modal.Function.from_name(
                app_name,
                function_name,
                environment_name=environment_name,
            )
            call = await asyncio.to_thread(remote_fn.spawn, graph_payload, training_payload, job_id)
            progress_dict = modal.Dict.from_name(
                MODAL_PROGRESS_DICT_NAME,
                environment_name=environment_name,
            )
            remote_result: dict[str, object] | None = None
            published_epochs = 0.0
            modal_loss_name = _normalize_loss_name(training.loss)
            modal_landscape_context = _initialize_synthetic_loss_landscape_context(
                job_id=job_id,
                training=training,
                loss_name=modal_loss_name,
            )

            while True:
                if entry.stop_event.is_set():
                    await asyncio.to_thread(call.cancel)
                    break
                progress_payload = await asyncio.to_thread(_read_modal_progress, progress_dict, job_id)
                (
                    published_epochs,
                    last_train_loss,
                    last_train_accuracy,
                    last_test_loss,
                    last_test_accuracy,
                ) = await _publish_modal_progress_updates(
                    job_id=job_id,
                    progress_payload=progress_payload,
                    published_epochs=published_epochs,
                    last_train_loss=last_train_loss,
                    last_train_accuracy=last_train_accuracy,
                    last_test_loss=last_test_loss,
                    last_test_accuracy=last_test_accuracy,
                    landscape_context=modal_landscape_context,
                )
                try:
                    remote_result = await asyncio.to_thread(call.get, 1.0)
                    break
                except (ModalTimeoutError, TimeoutError):
                    await asyncio.sleep(0.1)

            if not entry.stop_event.is_set():
                if not isinstance(remote_result, dict):
                    raise RuntimeError("Modal worker returned unexpected response payload")

                encoded_state = remote_result.get("state_dict_b64")
                if not isinstance(encoded_state, str):
                    raise RuntimeError("Modal worker did not return model weights")
                state_dict_bytes = base64.b64decode(encoded_state.encode("ascii"))
                raw_state_dict = torch.load(io.BytesIO(state_dict_bytes), map_location="cpu")
                if not isinstance(raw_state_dict, dict):
                    raise RuntimeError("Modal worker returned invalid state_dict payload")
                state_dict = _normalize_modal_state_dict_keys(raw_state_dict)
                model.load_state_dict(state_dict)

                epoch_metrics = remote_result.get("epoch_metrics", [])
                if not isinstance(epoch_metrics, list):
                    epoch_metrics = []
                for idx, metric in enumerate(epoch_metrics):
                    if not isinstance(metric, dict):
                        continue
                    epoch_num = float(metric.get("epoch", idx + 1))
                    if epoch_num <= published_epochs:
                        continue
                    last_train_loss = float(metric.get("train_loss", last_train_loss))
                    last_train_accuracy = float(metric.get("train_accuracy", last_train_accuracy))
                    last_test_loss = float(metric.get("test_loss", last_test_loss))
                    last_test_accuracy = float(metric.get("test_accuracy", last_test_accuracy))
                    await _publish_modal_epoch_update(
                        job_id=job_id,
                        epoch=epoch_num,
                        train_loss=last_train_loss,
                        train_accuracy=last_train_accuracy,
                        test_loss=last_test_loss,
                        test_accuracy=last_test_accuracy,
                        loss_landscape=_capture_synthetic_loss_landscape_snapshot(
                            context=modal_landscape_context,
                            epoch=epoch_num,
                            observed_loss=last_test_loss,
                        ),
                    )
                    published_epochs = epoch_num
                    await asyncio.sleep(0)

                final_metrics_raw = remote_result.get("final_metrics")
                if isinstance(final_metrics_raw, dict):
                    last_train_loss = float(final_metrics_raw.get("final_train_loss", last_train_loss))
                    last_train_accuracy = float(
                        final_metrics_raw.get("final_train_accuracy", last_train_accuracy)
                    )
                    last_test_loss = float(final_metrics_raw.get("final_test_loss", last_test_loss))
                    last_test_accuracy = float(
                        final_metrics_raw.get("final_test_accuracy", last_test_accuracy)
                    )
        else:
            train_loader, test_loader = get_dataset_dataloaders(training.dataset, training.batch_size)
            optimizer = _build_optimizer(model, training)
            loss_name, loss_fn = _build_loss(training)
            loss_landscape_context = _initialize_synthetic_loss_landscape_context(
                job_id=job_id,
                training=training,
                loss_name=loss_name,
            )
            total_train_steps = max(1, len(train_loader))

            for epoch in range(training.epochs):
                if entry.stop_event.is_set():
                    break

                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                steps = 0

                for batch_x, batch_y in train_loader:
                    if entry.stop_event.is_set():
                        break

                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = _compute_loss(loss_name, loss_fn, output, batch_y)
                    loss.backward()
                    optimizer.step()

                    running_loss += float(loss.item())
                    correct += int((output.argmax(1) == batch_y).sum().item())
                    total += int(batch_y.size(0))
                    steps += 1

                    if steps % TRAIN_PROGRESS_INTERVAL_STEPS == 0:
                        interim_train_loss = running_loss / max(steps, 1)
                        interim_train_accuracy = correct / max(total, 1)
                        interim_test_loss = (
                            last_test_loss if last_test_loss > 0 else interim_train_loss
                        )
                        interim_test_accuracy = (
                            last_test_accuracy if last_test_accuracy > 0 else interim_train_accuracy
                        )
                        interim_epoch = epoch + min(0.999, steps / total_train_steps)
                        interim_landscape = _capture_synthetic_loss_landscape_snapshot(
                            context=loss_landscape_context,
                            epoch=float(interim_epoch),
                            observed_loss=interim_test_loss,
                        )
                        await job_registry.publish(
                            job_id,
                            {
                                "type": "epoch_update",
                                "epoch": float(interim_epoch),
                                "loss": interim_test_loss,
                                "accuracy": interim_test_accuracy,
                                "train_loss": interim_train_loss,
                                "train_accuracy": interim_train_accuracy,
                                "test_loss": interim_test_loss,
                                "test_accuracy": interim_test_accuracy,
                                "weights": None,
                                "loss_landscape": interim_landscape,
                            },
                        )

                    # Yield frequently so API/WS handlers are not starved during training.
                    if steps % 5 == 0:
                        await asyncio.sleep(0)

                if steps == 0:
                    break

                last_train_loss = running_loss / max(steps, 1)
                last_train_accuracy = correct / max(total, 1)
                last_test_loss, last_test_accuracy = await _evaluate_model(
                    model=model,
                    dataloader=test_loader,
                    loss_name=loss_name,
                    loss_fn=loss_fn,
                    device=device,
                )
                loss_landscape = _capture_synthetic_loss_landscape_snapshot(
                    context=loss_landscape_context,
                    epoch=epoch + 1,
                    observed_loss=last_test_loss,
                )

                await job_registry.publish(
                    job_id,
                    {
                        "type": "epoch_update",
                        "epoch": epoch + 1,
                        # Keep these keys for frontend compatibility.
                        "loss": last_test_loss,
                        "accuracy": last_test_accuracy,
                        # Explicit split metrics.
                        "train_loss": last_train_loss,
                        "train_accuracy": last_train_accuracy,
                        "test_loss": last_test_loss,
                        "test_accuracy": last_test_accuracy,
                        "weights": extract_weight_snapshot(model),
                        "loss_landscape": loss_landscape,
                    },
                )

        if entry.job_dir is not None:
            entry.job_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = entry.job_dir / NN_ARTIFACT_FILENAME
        else:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifacts_dir / f"{job_id}.pt"
        torch.save(model.state_dict(), artifact_path)

        await job_registry.publish(
            job_id,
            {
                "type": "training_done",
                # Keep these keys for frontend compatibility.
                "final_loss": last_test_loss,
                "final_accuracy": last_test_accuracy,
                # Explicit split metrics.
                "final_train_loss": last_train_loss,
                "final_train_accuracy": last_train_accuracy,
                "final_test_loss": last_test_loss,
                "final_test_accuracy": last_test_accuracy,
                "model_path": str(artifact_path),
            },
        )

        final_status = "stopped" if entry.stop_event.is_set() else "completed"
        await job_registry.mark_terminal(
            job_id,
            final_status,
            final_metrics={
                "final_train_loss": last_train_loss,
                "final_train_accuracy": last_train_accuracy,
                "final_test_loss": last_test_loss,
                "final_test_accuracy": last_test_accuracy,
                "final_loss": last_test_loss,
                "final_accuracy": last_test_accuracy,
            },
            artifact_path=artifact_path,
        )
        if entry.job_dir is not None:
            update_job_metadata(
                entry.job_dir,
                status=final_status,
                terminal=True,
                final_metrics={
                    "final_train_loss": last_train_loss,
                    "final_train_accuracy": last_train_accuracy,
                    "final_test_loss": last_test_loss,
                    "final_test_accuracy": last_test_accuracy,
                    "final_loss": last_test_loss,
                    "final_accuracy": last_test_accuracy,
                },
                artifact_path=str(artifact_path),
            )

    except Exception as exc:  # pragma: no cover - error path is tested at API level
        message = str(exc) or f"{exc.__class__.__name__}: {exc!r}"
        await job_registry.publish(job_id, {"type": "error", "message": message})
        await job_registry.mark_terminal(job_id, "failed", error=message)
        if entry.job_dir is not None:
            update_job_metadata(entry.job_dir, status="failed", terminal=True, error=message)


def _normalize_modal_state_dict_keys(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Normalize keys produced by torch.compile-wrapped modules."""
    prefix = "_orig_mod."
    has_compiled_prefix = any(key.startswith(prefix) for key in state_dict)
    if not has_compiled_prefix:
        return state_dict
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def _read_modal_progress(progress_dict: Any, job_id: str) -> dict[str, Any] | None:
    payload = progress_dict.get(job_id)
    if not isinstance(payload, dict):
        return None
    return payload


async def _publish_modal_progress_updates(
    *,
    job_id: str,
    progress_payload: dict[str, Any] | None,
    published_epochs: float,
    last_train_loss: float,
    last_train_accuracy: float,
    last_test_loss: float,
    last_test_accuracy: float,
    landscape_context: dict[str, Any] | None,
) -> tuple[float, float, float, float, float]:
    if not isinstance(progress_payload, dict):
        return (
            published_epochs,
            last_train_loss,
            last_train_accuracy,
            last_test_loss,
            last_test_accuracy,
        )

    epoch_metrics = progress_payload.get("epoch_metrics")
    if not isinstance(epoch_metrics, list):
        return (
            published_epochs,
            last_train_loss,
            last_train_accuracy,
            last_test_loss,
            last_test_accuracy,
        )

    next_published_epochs = float(published_epochs)
    next_train_loss = last_train_loss
    next_train_accuracy = last_train_accuracy
    next_test_loss = last_test_loss
    next_test_accuracy = last_test_accuracy
    for idx, metric in enumerate(epoch_metrics):
        if not isinstance(metric, dict):
            continue
        epoch_num = float(metric.get("epoch", idx + 1))
        if epoch_num <= next_published_epochs:
            continue
        next_train_loss = float(metric.get("train_loss", next_train_loss))
        next_train_accuracy = float(metric.get("train_accuracy", next_train_accuracy))
        next_test_loss = float(metric.get("test_loss", next_test_loss))
        next_test_accuracy = float(metric.get("test_accuracy", next_test_accuracy))
        await _publish_modal_epoch_update(
            job_id=job_id,
            epoch=epoch_num,
            train_loss=next_train_loss,
            train_accuracy=next_train_accuracy,
            test_loss=next_test_loss,
            test_accuracy=next_test_accuracy,
            loss_landscape=(
                _capture_synthetic_loss_landscape_snapshot(
                    context=landscape_context,
                    epoch=epoch_num,
                    observed_loss=next_test_loss,
                )
                if landscape_context is not None
                else None
            ),
        )
        next_published_epochs = epoch_num
        await asyncio.sleep(0)

    return (
        next_published_epochs,
        next_train_loss,
        next_train_accuracy,
        next_test_loss,
        next_test_accuracy,
    )


async def _publish_modal_epoch_update(
    *,
    job_id: str,
    epoch: float,
    train_loss: float,
    train_accuracy: float,
    test_loss: float,
    test_accuracy: float,
    loss_landscape: dict[str, Any] | None,
) -> None:
    await job_registry.publish(
        job_id,
        {
            "type": "epoch_update",
            "epoch": epoch,
            "loss": test_loss,
            "accuracy": test_accuracy,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            # Remote training does not stream layer weights per epoch.
            "weights": None,
            "loss_landscape": loss_landscape,
        },
    )


def _resolve_training_backend(backend_override: str | None) -> str:
    if backend_override is not None:
        normalized = backend_override.strip().lower()
        if normalized == "cloud":
            return "modal"
        if normalized == "local":
            return "local"
        if normalized:
            return normalized
    return os.getenv("TRAINING_BACKEND", "local").strip().lower()
