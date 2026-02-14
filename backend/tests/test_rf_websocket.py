from __future__ import annotations

import asyncio
import time

from core.rf_job_registry import rf_job_registry
from tests.conftest import build_rf_graph_payload


def test_rf_websocket_streams_progress_and_done(monkeypatch, client) -> None:
    async def fake_run_rf_training_job(job_id, compiled, training, artifacts_dir):
        rf_job_registry.set_status(job_id, "running")
        await rf_job_registry.publish(
            job_id,
            {
                "type": "rf_progress",
                "stage": "training",
                "trees_built": 10,
                "total_trees": 20,
                "train_accuracy": 0.8,
                "test_accuracy": 0.75,
                "oob_score": None,
                "elapsed_ms": 42,
            },
        )
        await rf_job_registry.publish(
            job_id,
            {
                "type": "rf_done",
                "final_train_accuracy": 0.9,
                "final_test_accuracy": 0.85,
                "confusion_matrix": [[2, 0], [0, 2]],
                "classes": ["a", "b"],
                "feature_importances": [0.5, 0.5],
                "feature_names": ["f1", "f2"],
                "model_path": "dummy.pkl",
            },
        )
        await rf_job_registry.mark_terminal(
            job_id,
            "completed",
            final_metrics={"final_train_accuracy": 0.9, "final_test_accuracy": 0.85},
        )

    monkeypatch.setattr("routers.rf_model.run_rf_training_job", fake_run_rf_training_job)

    train_res = client.post("/api/rf/train", json=build_rf_graph_payload())
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]

    time.sleep(0.05)

    with client.websocket_connect(f"/ws/rf/training/{job_id}") as ws:
        msg1 = ws.receive_json()
        msg2 = ws.receive_json()

    assert msg1["type"] == "rf_progress"
    assert msg2["type"] == "rf_done"


def test_rf_websocket_streams_error(monkeypatch, client) -> None:
    async def fake_run_rf_training_job(job_id, compiled, training, artifacts_dir):
        rf_job_registry.set_status(job_id, "running")
        await rf_job_registry.publish(job_id, {"type": "rf_error", "message": "rf-boom"})
        await rf_job_registry.mark_terminal(job_id, "failed", error="rf-boom")

    monkeypatch.setattr("routers.rf_model.run_rf_training_job", fake_run_rf_training_job)

    train_res = client.post("/api/rf/train", json=build_rf_graph_payload())
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]

    time.sleep(0.05)

    with client.websocket_connect(f"/ws/rf/training/{job_id}") as ws:
        msg = ws.receive_json()

    assert msg["type"] == "rf_error"
    assert msg["message"] == "rf-boom"
