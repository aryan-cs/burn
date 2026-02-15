from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.graph_compiler import CompiledGraphResult
from core.job_registry import job_registry
from core.job_storage import GRAPH_FILENAME, NN_ARTIFACT_FILENAME, update_job_metadata
from core.weight_extractor import extract_weight_snapshot
from datasets.loader import get_dataset_dataloaders
from models.training_config import TrainingConfig


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

            # Keep the event loop responsive for websocket handshakes/messages.
            if steps % 20 == 0:
                await asyncio.sleep(0)

    if steps == 0:
        return 0.0, 0.0

    return total_loss / steps, correct / max(total, 1)


async def run_training_job(
    job_id: str,
    compiled: CompiledGraphResult,
    training: TrainingConfig,
    artifacts_dir: Path,
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
        backend = os.getenv("TRAINING_BACKEND", "modal").strip().lower()
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
            call = await asyncio.to_thread(remote_fn.spawn, graph_payload, training_payload)
            remote_result: dict[str, object] | None = None

            while True:
                if entry.stop_event.is_set():
                    await asyncio.to_thread(call.cancel)
                    break
                try:
                    remote_result = await asyncio.to_thread(call.get, 1.0)
                    break
                except ModalTimeoutError:
                    await asyncio.sleep(0)

            if not entry.stop_event.is_set():
                if not isinstance(remote_result, dict):
                    raise RuntimeError("Modal worker returned unexpected response payload")

                encoded_state = remote_result.get("state_dict_b64")
                if not isinstance(encoded_state, str):
                    raise RuntimeError("Modal worker did not return model weights")
                state_dict_bytes = base64.b64decode(encoded_state.encode("ascii"))
                state_dict = torch.load(io.BytesIO(state_dict_bytes), map_location="cpu")
                model.load_state_dict(state_dict)

                epoch_metrics = remote_result.get("epoch_metrics", [])
                if not isinstance(epoch_metrics, list):
                    epoch_metrics = []
                for idx, metric in enumerate(epoch_metrics):
                    if not isinstance(metric, dict):
                        continue
                    last_train_loss = float(metric.get("train_loss", last_train_loss))
                    last_train_accuracy = float(metric.get("train_accuracy", last_train_accuracy))
                    last_test_loss = float(metric.get("test_loss", last_test_loss))
                    last_test_accuracy = float(metric.get("test_accuracy", last_test_accuracy))
                    await job_registry.publish(
                        job_id,
                        {
                            "type": "epoch_update",
                            "epoch": int(metric.get("epoch", idx + 1)),
                            "loss": last_test_loss,
                            "accuracy": last_test_accuracy,
                            "train_loss": last_train_loss,
                            "train_accuracy": last_train_accuracy,
                            "test_loss": last_test_loss,
                            "test_accuracy": last_test_accuracy,
                            # Remote training does not stream layer weights per epoch.
                            "weights": None,
                        },
                    )
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
        message = str(exc)
        await job_registry.publish(job_id, {"type": "error", "message": message})
        await job_registry.mark_terminal(job_id, "failed", error=message)
        if entry.job_dir is not None:
            update_job_metadata(entry.job_dir, status="failed", terminal=True, error=message)
