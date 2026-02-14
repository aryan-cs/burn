from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class RemoteJobEntry:
    job_id: str
    status: str
    stop_event: asyncio.Event
    messages: list[dict[str, Any]] = field(default_factory=list)
    terminal: bool = False
    error: str | None = None
    artifact_path: Path | None = None
    task: asyncio.Task | None = None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class RemoteJobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, RemoteJobEntry] = {}

    def clear(self) -> None:
        self._jobs.clear()

    def create_job(self) -> RemoteJobEntry:
        job_id = uuid4().hex
        entry = RemoteJobEntry(job_id=job_id, status="queued", stop_event=asyncio.Event())
        self._jobs[job_id] = entry
        return entry

    def get(self, job_id: str) -> RemoteJobEntry | None:
        return self._jobs.get(job_id)

    def set_task(self, job_id: str, task: asyncio.Task) -> None:
        entry = self._jobs[job_id]
        entry.task = task

    def set_status(self, job_id: str, status: str) -> None:
        entry = self._jobs[job_id]
        entry.status = status

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
        error: str | None = None,
        artifact_path: Path | None = None,
    ) -> None:
        entry = self._jobs[job_id]
        entry.status = status
        entry.error = error
        entry.terminal = True
        if artifact_path is not None:
            entry.artifact_path = artifact_path
        async with entry.condition:
            entry.condition.notify_all()

    async def wait_for_update(
        self,
        job_id: str,
        last_index: int,
        timeout: float = 20.0,
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


remote_job_registry = RemoteJobRegistry()
