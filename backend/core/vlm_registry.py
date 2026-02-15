from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import asyncio
from typing import Any
from uuid import uuid4


@dataclass
class VLMJobEntry:
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    dataset: str
    epochs: int
    learning_rate: float
    batch_size: int
    steps_per_epoch: int
    model_id: str = "hustvl/yolos-tiny"
    current_epoch: int = 0
    latest_loss: float | None = None
    terminal: bool = False
    error: str | None = None
    final_metrics: dict[str, Any] | None = None
    artifact_path: Path | None = None
    task: asyncio.Task | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    messages: list[dict[str, Any]] = field(default_factory=list)


class VLMJobRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, VLMJobEntry] = {}
        self._latest_job_id: str | None = None

    def create_job(
        self,
        *,
        dataset: str,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        steps_per_epoch: int,
        model_id: str,
    ) -> VLMJobEntry:
        now = datetime.now(timezone.utc)
        job_id = uuid4().hex
        entry = VLMJobEntry(
            job_id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            dataset=dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            model_id=model_id,
        )
        self._entries[job_id] = entry
        self._latest_job_id = job_id
        return entry

    def get(self, job_id: str) -> VLMJobEntry | None:
        return self._entries.get(job_id)

    def latest_job_id(self) -> str | None:
        return self._latest_job_id

    def set_task(self, job_id: str, task: asyncio.Task) -> None:
        entry = self._entries[job_id]
        entry.task = task
        entry.updated_at = datetime.now(timezone.utc)

    def set_status(self, job_id: str, status: str) -> None:
        entry = self._entries[job_id]
        entry.status = status
        entry.updated_at = datetime.now(timezone.utc)

    def request_stop(self, job_id: str) -> None:
        entry = self._entries[job_id]
        entry.stop_event.set()
        if entry.status not in {"completed", "failed", "stopped"}:
            entry.status = "stopping"
        entry.updated_at = datetime.now(timezone.utc)

    async def publish(self, job_id: str, message: dict[str, Any]) -> None:
        entry = self._entries[job_id]
        entry.messages.append(message)
        entry.updated_at = datetime.now(timezone.utc)

    async def wait_for_update(
        self,
        job_id: str,
        last_index: int,
        timeout: float = 30.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        entry = self._entries[job_id]
        if last_index < len(entry.messages) or entry.terminal:
            return entry.messages[last_index:], entry.terminal

        elapsed = 0.0
        interval = 0.05
        while elapsed < timeout:
            await asyncio.sleep(interval)
            elapsed += interval
            if last_index < len(entry.messages) or entry.terminal:
                return entry.messages[last_index:], entry.terminal

        return entry.messages[last_index:], entry.terminal

    def mark_terminal(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
        final_metrics: dict[str, Any] | None = None,
        artifact_path: Path | None = None,
    ) -> None:
        entry = self._entries[job_id]
        entry.status = status
        entry.terminal = True
        entry.error = error
        entry.final_metrics = final_metrics
        entry.artifact_path = artifact_path
        entry.updated_at = datetime.now(timezone.utc)

    def update_progress(self, job_id: str, *, epoch: int, loss: float | None) -> None:
        entry = self._entries[job_id]
        entry.current_epoch = max(0, epoch)
        entry.latest_loss = loss
        entry.updated_at = datetime.now(timezone.utc)

    def clear(self) -> None:
        self._entries.clear()
        self._latest_job_id = None


vlm_job_registry = VLMJobRegistry()
