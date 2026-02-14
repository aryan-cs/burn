from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


TERMINAL_STATUSES = {"completed", "failed", "stopped"}


@dataclass
class JobEntry:
    job_id: str
    status: str
    created_at: datetime
    stop_event: asyncio.Event
    messages: list[dict[str, Any]] = field(default_factory=list)
    task: asyncio.Task | None = None
    terminal: bool = False
    python_source: str = ""
    artifact_path: Path | None = None
    final_metrics: dict[str, float] | None = None
    error: str | None = None
    model: Any | None = None
    input_shape: list[int] | None = None
    num_classes: int | None = None
    job_dir: Path | None = None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, JobEntry] = {}
        self._latest_job_id: str | None = None

    def create_job(
        self,
        python_source: str,
        *,
        model: Any | None = None,
        input_shape: list[int] | None = None,
        num_classes: int | None = None,
    ) -> JobEntry:
        job_id = uuid4().hex
        entry = JobEntry(
            job_id=job_id,
            status="queued",
            created_at=datetime.now(timezone.utc),
            stop_event=asyncio.Event(),
            python_source=python_source,
            model=model,
            input_shape=input_shape,
            num_classes=num_classes,
        )
        self._jobs[job_id] = entry
        self._latest_job_id = job_id
        return entry

    def clear(self) -> None:
        self._jobs.clear()
        self._latest_job_id = None

    def get(self, job_id: str) -> JobEntry | None:
        return self._jobs.get(job_id)

    def latest_job_id(self) -> str | None:
        return self._latest_job_id

    def set_task(self, job_id: str, task: asyncio.Task) -> None:
        entry = self._jobs[job_id]
        entry.task = task

    def set_job_dir(self, job_id: str, job_dir: Path) -> None:
        self._jobs[job_id].job_dir = job_dir

    def set_status(self, job_id: str, status: str) -> None:
        entry = self._jobs[job_id]
        entry.status = status

    def request_stop(self, job_id: str) -> bool:
        entry = self._jobs.get(job_id)
        if entry is None:
            return False
        if entry.terminal:
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


job_registry = JobRegistry()
