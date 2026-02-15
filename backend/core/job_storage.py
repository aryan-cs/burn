from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_FILENAME = "model.py"
GRAPH_FILENAME = "graph.json"
TRAINING_FILENAME = "training.json"
SUMMARY_FILENAME = "summary.json"
METADATA_FILENAME = "metadata.json"
NN_ARTIFACT_FILENAME = "model.pt"
RF_ARTIFACT_FILENAME = "model.pkl"


def model_job_dir(artifacts_dir: Path, job_id: str) -> Path:
    return artifacts_dir / "jobs" / job_id


def rf_job_dir(artifacts_dir: Path, job_id: str) -> Path:
    return artifacts_dir / "rf" / "jobs" / job_id


def load_job_metadata(job_dir: Path) -> dict[str, Any] | None:
    path = job_dir / METADATA_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def load_python_source(job_dir: Path) -> str | None:
    source_path = job_dir / SOURCE_FILENAME
    if not source_path.exists():
        return None
    try:
        return source_path.read_text(encoding="utf-8")
    except OSError:
        return None


def persist_job_bundle(
    job_dir: Path,
    *,
    model_family: str,
    python_source: str,
    graph_payload: dict[str, Any],
    training_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    warnings: list[str],
) -> None:
    created_at = _utc_now()
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_text(job_dir / SOURCE_FILENAME, python_source)
    _write_json(job_dir / GRAPH_FILENAME, graph_payload)
    _write_json(job_dir / TRAINING_FILENAME, training_payload)
    _write_json(job_dir / SUMMARY_FILENAME, summary_payload)
    _write_json(
        job_dir / METADATA_FILENAME,
        {
            "job_id": job_dir.name,
            "model_family": model_family,
            "status": "queued",
            "terminal": False,
            "error": None,
            "warnings": warnings,
            "artifact_path": None,
            "final_metrics": None,
            "created_at": created_at,
            "updated_at": created_at,
            "files": {
                "python_source": SOURCE_FILENAME,
                "graph": GRAPH_FILENAME,
                "training": TRAINING_FILENAME,
                "summary": SUMMARY_FILENAME,
                "metadata": METADATA_FILENAME,
            },
        },
    )


def update_job_metadata(
    job_dir: Path,
    *,
    status: str | None = None,
    terminal: bool | None = None,
    error: str | None = None,
    final_metrics: dict[str, Any] | None = None,
    artifact_path: str | None = None,
) -> None:
    payload = load_job_metadata(job_dir) or {"job_id": job_dir.name}
    payload["updated_at"] = _utc_now()

    if status is not None:
        payload["status"] = status
    if terminal is not None:
        payload["terminal"] = terminal
    if error is not None:
        payload["error"] = error
    if final_metrics is not None:
        payload["final_metrics"] = final_metrics
    if artifact_path is not None:
        payload["artifact_path"] = artifact_path

    _write_json(job_dir / METADATA_FILENAME, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
