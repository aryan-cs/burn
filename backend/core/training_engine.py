from __future__ import annotations

import asyncio
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.graph_compiler import CompiledGraphResult
from core.job_registry import job_registry
from core.weight_extractor import extract_weight_snapshot
from datasets.loader import get_mnist_dataloaders
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

    if training.dataset != "mnist":
        message = f"Unsupported dataset for v1: {training.dataset}"
        await job_registry.publish(job_id, {"type": "error", "message": message})
        await job_registry.mark_terminal(job_id, "failed", error=message)
        return

    try:
        job_registry.set_status(job_id, "running")
        model = compiled.model
        device = torch.device("cpu")
        model.to(device)

        train_loader, test_loader = get_mnist_dataloaders(training.batch_size)
        optimizer = _build_optimizer(model, training)
        loss_name, loss_fn = _build_loss(training)

        last_train_loss = 0.0
        last_train_accuracy = 0.0
        last_test_loss = 0.0
        last_test_accuracy = 0.0

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

    except Exception as exc:  # pragma: no cover - error path is tested at API level
        message = str(exc)
        await job_registry.publish(job_id, {"type": "error", "message": message})
        await job_registry.mark_terminal(job_id, "failed", error=message)
