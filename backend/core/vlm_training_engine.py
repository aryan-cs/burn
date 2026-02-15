from __future__ import annotations

import asyncio
import random
from pathlib import Path

import torch

from core.vlm_registry import vlm_job_registry
from core.vlm_runtime import vlm_runtime


async def run_vlm_training_job(
    job_id: str,
    *,
    artifacts_root: Path,
) -> None:
    entry = vlm_job_registry.get(job_id)
    if entry is None:
        return

    try:
        vlm_job_registry.set_status(job_id, "running")
        vlm_runtime.ensure_model(entry.model_id)
        model = vlm_runtime.create_trainable_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=entry.learning_rate)
        image_size, min_objects, max_objects = _dataset_profile(entry.dataset)

        epoch_loss = 0.0
        for epoch_index in range(entry.epochs):
            if entry.stop_event.is_set():
                break

            if vlm_runtime.backend == "huggingface" and vlm_runtime.hf_processor is not None:
                epoch_loss = await _run_hf_epoch(
                    job_id=job_id,
                    epoch_index=epoch_index,
                    total_epochs=entry.epochs,
                    model=model,
                    optimizer=optimizer,
                    batch_size=entry.batch_size,
                    steps_per_epoch=entry.steps_per_epoch,
                    stop_event=entry.stop_event,
                    image_size=image_size,
                    min_objects=min_objects,
                    max_objects=max_objects,
                )
            else:
                # Fallback mode: keep this path fast and deterministic.
                epoch_loss = max(0.01, 0.3 / float(epoch_index + 1))
                for _ in range(entry.steps_per_epoch):
                    if entry.stop_event.is_set():
                        break
                    await asyncio.sleep(0.02)

            vlm_job_registry.update_progress(
                job_id,
                epoch=epoch_index + 1,
                loss=epoch_loss,
            )
            await vlm_job_registry.publish(
                job_id,
                {
                    "type": "vlm_progress",
                    "epoch": epoch_index + 1,
                    "epochs": entry.epochs,
                    "loss": epoch_loss,
                    "status": "running",
                    "dataset": entry.dataset,
                    "model_id": entry.model_id,
                },
            )
            await asyncio.sleep(0)

        artifact_path = artifacts_root / "vlm" / "jobs" / job_id / "model.pt"
        vlm_runtime.save_state_dict(model, artifact_path)

        final_status = "stopped" if entry.stop_event.is_set() else "completed"
        await vlm_job_registry.publish(
            job_id,
            {
                "type": "vlm_done",
                "status": final_status,
                "final_loss": epoch_loss,
                "epochs_ran": entry.current_epoch,
                "dataset": entry.dataset,
                "model_id": entry.model_id,
                "model_path": str(artifact_path),
            },
        )
        vlm_job_registry.mark_terminal(
            job_id,
            status=final_status,
            final_metrics={
                "final_loss": epoch_loss,
                "epochs_ran": entry.current_epoch,
                "dataset": entry.dataset,
                "model_id": entry.model_id,
            },
            artifact_path=artifact_path,
        )
    except Exception as exc:
        message = str(exc)
        await vlm_job_registry.publish(
            job_id,
            {
                "type": "vlm_error",
                "message": message,
            },
        )
        vlm_job_registry.mark_terminal(job_id, status="failed", error=message)


async def _run_hf_epoch(
    *,
    job_id: str,
    epoch_index: int,
    total_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    steps_per_epoch: int,
    stop_event: asyncio.Event,
    image_size: int,
    min_objects: int,
    max_objects: int,
) -> float:
    processor = vlm_runtime.hf_processor
    assert processor is not None

    losses: list[float] = []
    device = vlm_runtime.device

    for _ in range(steps_per_epoch):
        if stop_event.is_set():
            break

        images, labels = _build_synthetic_hf_batch(
            batch_size=batch_size,
            image_size=image_size,
            max_label_id=max(vlm_runtime.label_map.keys(), default=10),
            min_objects=min_objects,
            max_objects=max_objects,
        )
        encoded = processor(images=images, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)
        pixel_mask = encoded.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)
        labels = [{key: value.to(device) for key, value in sample.items()} for sample in labels]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        await vlm_job_registry.publish(
            job_id,
            {
                "type": "vlm_progress",
                "phase": "step",
                "epoch": epoch_index + 1,
                "epochs": total_epochs,
                "step": len(losses),
                "steps_per_epoch": steps_per_epoch,
                "loss": float(sum(losses) / len(losses)),
                "status": "running",
            },
        )
        await asyncio.sleep(0)

    if not losses:
        return 0.0
    return float(sum(losses) / len(losses))


def _build_synthetic_hf_batch(
    *,
    batch_size: int,
    image_size: int,
    max_label_id: int,
    min_objects: int,
    max_objects: int,
) -> tuple[list["Image.Image"], list[dict[str, torch.Tensor]]]:
    from PIL import Image

    safe_label_max = max(1, min(max_label_id, 90))
    images: list[Image.Image] = []
    labels: list[dict[str, torch.Tensor]] = []

    for _ in range(batch_size):
        # Build a synthetic RGB image.
        pixel_tensor = torch.randint(
            low=0,
            high=255,
            size=(image_size, image_size, 3),
            dtype=torch.uint8,
        )
        images.append(Image.fromarray(pixel_tensor.numpy(), mode="RGB"))

        target_count = random.randint(max(1, min_objects), max(max(1, min_objects), max_objects))
        class_labels: list[int] = []
        boxes: list[list[float]] = []
        for _target_index in range(target_count):
            class_labels.append(random.randint(1, safe_label_max))

            width = random.uniform(0.08, 0.45)
            height = random.uniform(0.08, 0.45)
            center_x = random.uniform(width / 2.0, 1.0 - width / 2.0)
            center_y = random.uniform(height / 2.0, 1.0 - height / 2.0)
            boxes.append([center_x, center_y, width, height])

        labels.append(
            {
                "class_labels": torch.tensor(class_labels, dtype=torch.int64),
                "boxes": torch.tensor(boxes, dtype=torch.float32),
            }
        )

    return images, labels


def _dataset_profile(dataset_id: str) -> tuple[int, int, int]:
    if dataset_id == "synthetic_boxes_tiny":
        return 160, 1, 2
    return 320, 1, 4
