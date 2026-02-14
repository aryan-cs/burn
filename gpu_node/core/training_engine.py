from __future__ import annotations

import asyncio
from pathlib import Path

import torch
import torch.nn as nn

from gpu_node.bootstrap import ensure_backend_path
from gpu_node.core.job_registry import remote_job_registry

ensure_backend_path()

from core.graph_compiler import CompiledGraphResult
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


def _build_loss(training: TrainingConfig) -> nn.Module:
    if training.loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss function: {training.loss}")


async def _evaluate_model(model: nn.Module, dataloader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
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
            loss = loss_fn(output, batch_y)
            total_loss += float(loss.item())
            correct += int((output.argmax(1) == batch_y).sum().item())
            total += int(batch_y.size(0))
            steps += 1
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
    entry = remote_job_registry.get(job_id)
    if entry is None:
        return

    if training.dataset != "mnist":
        message = f"Unsupported dataset for v1: {training.dataset}"
        await remote_job_registry.publish(job_id, {"type": "error", "message": message})
        await remote_job_registry.mark_terminal(job_id, "failed", error=message)
        return

    try:
        remote_job_registry.set_status(job_id, "running")
        model = compiled.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_loader, test_loader = get_mnist_dataloaders(training.batch_size)
        optimizer = _build_optimizer(model, training)
        loss_fn = _build_loss(training)

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
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                correct += int((output.argmax(1) == batch_y).sum().item())
                total += int(batch_y.size(0))
                steps += 1
                if steps % 5 == 0:
                    await asyncio.sleep(0)

            if steps == 0:
                break

            last_train_loss = running_loss / max(steps, 1)
            last_train_accuracy = correct / max(total, 1)
            last_test_loss, last_test_accuracy = await _evaluate_model(
                model=model,
                dataloader=test_loader,
                loss_fn=loss_fn,
                device=device,
            )

            await remote_job_registry.publish(
                job_id,
                {
                    "type": "epoch_update",
                    "epoch": epoch + 1,
                    "loss": last_test_loss,
                    "accuracy": last_test_accuracy,
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

        await remote_job_registry.publish(
            job_id,
            {
                "type": "training_done",
                "final_loss": last_test_loss,
                "final_accuracy": last_test_accuracy,
                "final_train_loss": last_train_loss,
                "final_train_accuracy": last_train_accuracy,
                "final_test_loss": last_test_loss,
                "final_test_accuracy": last_test_accuracy,
                "model_path": str(artifact_path),
            },
        )

        final_status = "stopped" if entry.stop_event.is_set() else "completed"
        await remote_job_registry.mark_terminal(job_id, final_status, artifact_path=artifact_path)
    except Exception as exc:  # pragma: no cover - HTTP layer assertions cover this path
        message = str(exc)
        await remote_job_registry.publish(job_id, {"type": "error", "message": message})
        await remote_job_registry.mark_terminal(job_id, "failed", error=message)
