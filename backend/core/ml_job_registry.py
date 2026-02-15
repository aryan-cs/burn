"""Job registry for classical ML training jobs.

Follows the same pub/sub pattern as the NN job registry but adapted
for scikit-learn models (no PyTorch, no weight snapshots).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


ML_TERMINAL_STATUSES = {"completed", "failed", "stopped"}


@dataclass
class MLJobEntry:
    job_id: str
    status: str
    model_type: str
    dataset: str
    created_at: datetime
    stop_event: asyncio.Event
    messages: list[dict[str, Any]] = field(default_factory=list)
    task: asyncio.Task | None = None
    terminal: bool = False
    artifact_path: Path | None = None
    final_metrics: dict[str, float] | None = None
    error: str | None = None
    model: Any | None = None
    scaler: Any | None = None
    feature_names: list[str] | None = None
    target_names: list[str] | None = None
    n_classes: int = 0
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class MLJobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, MLJobEntry] = {}
        self._latest_job_id: str | None = None

    def create_job(
        self,
        model_type: str,
        dataset: str,
        hyperparameters: dict[str, Any] | None = None,
    ) -> MLJobEntry:
        job_id = uuid4().hex
        entry = MLJobEntry(
            job_id=job_id,
            status="queued",
            model_type=model_type,
            dataset=dataset,
            created_at=datetime.now(timezone.utc),
            stop_event=asyncio.Event(),
            hyperparameters=hyperparameters or {},
        )
        self._jobs[job_id] = entry
        self._latest_job_id = job_id
        return entry

    def get(self, job_id: str) -> MLJobEntry | None:
        return self._jobs.get(job_id)

    def latest_job_id(self) -> str | None:
        return self._latest_job_id

    def set_task(self, job_id: str, task: asyncio.Task) -> None:
        self._jobs[job_id].task = task

    def request_stop(self, job_id: str) -> bool:
        entry = self._jobs.get(job_id)
        if entry is None or entry.terminal:
            return False
        entry.stop_event.set()
        entry.status = "stopping"
        return True

    async def publish(self, job_id: str, message: dict[str, Any]) -> None:
        entry = self._jobs[job_id]
        async with entry.condition:
            entry.messages.append(message)
            entry.condition.notify_all()

    async def mark_terminal(
        self,
        job_id: str,
        status: str,
        *,
        final_metrics: dict[str, float] | None = None,
        error: str | None = None,
        artifact_path: Path | None = None,
    ) -> None:
        entry = self._jobs[job_id]
        entry.status = status
        entry.terminal = True
        entry.final_metrics = final_metrics
        entry.error = error
        if artifact_path is not None:
            entry.artifact_path = artifact_path
        async with entry.condition:
            entry.condition.notify_all()

    async def wait_for_update(
        self,
        job_id: str,
        last_index: int,
        timeout: float = 30.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        entry = self._jobs[job_id]
        async with entry.condition:
            if last_index < len(entry.messages) or entry.terminal:
                return entry.messages[last_index:], entry.terminal
            try:
                await asyncio.wait_for(entry.condition.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
            return entry.messages[last_index:], entry.terminal


ml_job_registry = MLJobRegistry()
