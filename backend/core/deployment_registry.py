from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

MAX_LOG_ITEMS = 500


@dataclass
class DeploymentLogEntry:
    timestamp: datetime
    level: str
    event: str
    message: str
    details: dict[str, Any] | None = None


@dataclass
class DeploymentEntry:
    deployment_id: str
    job_id: str
    status: str
    target: str
    endpoint_path: str
    created_at: datetime
    model_family: str = "nn"
    name: str | None = None
    last_used_at: datetime | None = None
    request_count: int = 0
    model: Any | None = None
    input_shape: list[int] | None = None
    num_classes: int | None = None
    runtime_config: dict[str, Any] | None = None
    logs: list[DeploymentLogEntry] | None = None


class DeploymentRegistry:
    def __init__(self, state_path: Path | None = None) -> None:
        self._entries: dict[str, DeploymentEntry] = {}
        self._state_path = state_path or _default_registry_path()
        self._load_from_disk()

    def create_deployment(
        self,
        *,
        job_id: str,
        target: str,
        name: str | None,
        model: Any | None,
        input_shape: list[int] | None,
        num_classes: int | None,
        model_family: str = "nn",
        endpoint_path: str | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> DeploymentEntry:
        deployment_id = uuid4().hex
        resolved_endpoint_path = endpoint_path or f"/api/deploy/{deployment_id}/infer"
        entry = DeploymentEntry(
            deployment_id=deployment_id,
            job_id=job_id,
            status="running",
            target=target,
            endpoint_path=resolved_endpoint_path,
            created_at=datetime.now(timezone.utc),
            model_family=model_family,
            name=name,
            model=model,
            input_shape=input_shape,
            num_classes=num_classes,
            runtime_config=runtime_config,
            logs=[],
        )
        self._entries[deployment_id] = entry
        self.add_log(
            deployment_id,
            level="info",
            event="deployment_created",
            message="Deployment created and running.",
            details={
                "job_id": job_id,
                "target": target,
                "endpoint_path": resolved_endpoint_path,
                "model_family": model_family,
            },
            persist=False,
        )
        self._persist()
        return entry

    def get(self, deployment_id: str) -> DeploymentEntry | None:
        return self._entries.get(deployment_id)

    def list(self) -> list[DeploymentEntry]:
        return sorted(
            self._entries.values(),
            key=lambda entry: entry.created_at,
            reverse=True,
        )

    def mark_stopped(self, deployment_id: str) -> bool:
        entry = self._entries.get(deployment_id)
        if entry is None:
            return False
        entry.status = "stopped"
        self.add_log(
            deployment_id,
            level="info",
            event="deployment_stopped",
            message="Deployment stopped.",
            details=None,
            persist=False,
        )
        self._persist()
        return True

    def mark_running(self, deployment_id: str) -> bool:
        entry = self._entries.get(deployment_id)
        if entry is None:
            return False
        entry.status = "running"
        self.add_log(
            deployment_id,
            level="info",
            event="deployment_started",
            message="Deployment started.",
            details=None,
            persist=False,
        )
        self._persist()
        return True

    def mark_request(self, deployment_id: str) -> None:
        entry = self._entries[deployment_id]
        entry.request_count += 1
        entry.last_used_at = datetime.now(timezone.utc)
        self._persist()

    def add_log(
        self,
        deployment_id: str,
        *,
        level: str,
        event: str,
        message: str,
        details: dict[str, Any] | None,
        persist: bool = True,
    ) -> None:
        entry = self._entries.get(deployment_id)
        if entry is None:
            return
        if entry.logs is None:
            entry.logs = []

        entry.logs.append(
            DeploymentLogEntry(
                timestamp=datetime.now(timezone.utc),
                level=level,
                event=event,
                message=message,
                details=details,
            )
        )
        if len(entry.logs) > MAX_LOG_ITEMS:
            entry.logs = entry.logs[-MAX_LOG_ITEMS:]
        if persist:
            self._persist()

    def logs(self, deployment_id: str, *, limit: int = 200) -> list[DeploymentLogEntry]:
        entry = self._entries.get(deployment_id)
        if entry is None:
            return []
        log_items = entry.logs or []
        if limit <= 0:
            return []
        return log_items[-limit:]

    def clear(self) -> None:
        self._entries.clear()
        try:
            if self._state_path.exists():
                self._state_path.unlink()
        except OSError:
            pass

    def _load_from_disk(self) -> None:
        if not self._state_path.exists():
            return

        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return

        raw_entries = payload.get("deployments")
        if not isinstance(raw_entries, list):
            return

        did_mutate_on_restore = False

        for item in raw_entries:
            if not isinstance(item, dict):
                continue

            deployment_id = str(item.get("deployment_id", "")).strip()
            job_id = str(item.get("job_id", "")).strip()
            target = str(item.get("target", "local")).strip() or "local"
            model_family = str(item.get("model_family", "nn")).strip().lower() or "nn"
            endpoint_path = str(item.get("endpoint_path", "")).strip()
            if not deployment_id or not job_id:
                continue
            if not endpoint_path:
                endpoint_path = f"/api/deploy/{deployment_id}/infer"

            created_at = _parse_datetime(item.get("created_at")) or datetime.now(timezone.utc)
            last_used_at = _parse_datetime(item.get("last_used_at"))

            status = str(item.get("status", "stopped")).strip().lower() or "stopped"
            logs = _deserialize_logs(item.get("logs"))

            # Deployed model objects live in memory only. On restart we restore metadata
            # and mark previously-running endpoints as stopped until the user starts them.
            if status == "running":
                status = "stopped"
                logs.append(
                    DeploymentLogEntry(
                        timestamp=datetime.now(timezone.utc),
                        level="info",
                        event="deployment_restored_stopped",
                        message="Deployment restored after backend restart and marked stopped.",
                        details={"reason": "in_memory_runtime_lost"},
                    )
                )
                did_mutate_on_restore = True

            entry = DeploymentEntry(
                deployment_id=deployment_id,
                job_id=job_id,
                status=status,
                target=target,
                endpoint_path=endpoint_path,
                created_at=created_at,
                model_family=model_family,
                name=_normalize_optional_str(item.get("name")),
                last_used_at=last_used_at,
                request_count=_coerce_non_negative_int(item.get("request_count")),
                model=None,
                input_shape=_deserialize_int_list(item.get("input_shape")),
                num_classes=_coerce_optional_int(item.get("num_classes")),
                runtime_config=_deserialize_dict(item.get("runtime_config")),
                logs=logs[-MAX_LOG_ITEMS:],
            )
            self._entries[deployment_id] = entry

        if did_mutate_on_restore:
            self._persist()

    def _persist(self) -> None:
        data = {
            "deployments": [
                {
                    "deployment_id": entry.deployment_id,
                    "job_id": entry.job_id,
                    "status": entry.status,
                    "target": entry.target,
                    "endpoint_path": entry.endpoint_path,
                    "model_family": entry.model_family,
                    "created_at": entry.created_at.isoformat(),
                    "name": entry.name,
                    "last_used_at": entry.last_used_at.isoformat() if entry.last_used_at else None,
                    "request_count": entry.request_count,
                    "input_shape": entry.input_shape,
                    "num_classes": entry.num_classes,
                    "runtime_config": _make_json_safe(entry.runtime_config),
                    "logs": [
                        {
                            "timestamp": log.timestamp.isoformat(),
                            "level": log.level,
                            "event": log.event,
                            "message": log.message,
                            "details": _make_json_safe(log.details),
                        }
                        for log in (entry.logs or [])
                    ],
                }
                for entry in self._entries.values()
            ]
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(f"{self._state_path.suffix}.tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            tmp.replace(self._state_path)
        except (OSError, TypeError, ValueError):
            # Persistence is best-effort; runtime operations should continue even if
            # state writes fail due to temporary filesystem/serialization issues.
            return


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "deployments" / "registry.json"


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or value.strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _coerce_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _deserialize_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return out


def _deserialize_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def _deserialize_logs(value: Any) -> list[DeploymentLogEntry]:
    if not isinstance(value, list):
        return []
    logs: list[DeploymentLogEntry] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        timestamp = _parse_datetime(item.get("timestamp")) or datetime.now(timezone.utc)
        logs.append(
            DeploymentLogEntry(
                timestamp=timestamp,
                level=str(item.get("level", "info")),
                event=str(item.get("event", "event")),
                message=str(item.get("message", "")),
                details=item.get("details") if isinstance(item.get("details"), dict) else None,
            )
        )
    return logs


def _make_json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


deployment_registry = DeploymentRegistry()
