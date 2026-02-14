from __future__ import annotations

import asyncio
import time

from core.job_registry import job_registry
from tests.conftest import build_graph_payload


def test_websocket_streams_epoch_and_done(monkeypatch, client) -> None:
    async def fake_run_training_job(job_id, compiled, training, artifacts_dir):
        job_registry.set_status(job_id, "running")
        await job_registry.publish(
            job_id,
            {
                "type": "epoch_update",
                "epoch": 1,
                "loss": 0.5,
                "accuracy": 0.75,
                "weights": {"layer": {"mean": 0.1, "std": 0.2, "min": -1.0, "max": 1.0, "histogram": [1, 2]}},
            },
        )
        await job_registry.publish(
            job_id,
            {
                "type": "training_done",
                "final_loss": 0.5,
                "final_accuracy": 0.75,
                "model_path": "dummy.pt",
            },
        )
        await job_registry.mark_terminal(
            job_id,
            "completed",
            final_metrics={"final_loss": 0.5, "final_accuracy": 0.75},
        )

    monkeypatch.setattr("routers.model.run_training_job", fake_run_training_job)

    train_res = client.post("/api/model/train", json=build_graph_payload())
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]

    # Let the background task publish messages first.
    time.sleep(0.05)

    with client.websocket_connect(f"/ws/training/{job_id}") as ws:
        msg1 = ws.receive_json()
        msg2 = ws.receive_json()

    assert msg1["type"] == "epoch_update"
    assert msg2["type"] == "training_done"


def test_websocket_streams_error(monkeypatch, client) -> None:
    async def fake_run_training_job(job_id, compiled, training, artifacts_dir):
        job_registry.set_status(job_id, "running")
        await job_registry.publish(job_id, {"type": "error", "message": "boom"})
        await job_registry.mark_terminal(job_id, "failed", error="boom")

    monkeypatch.setattr("routers.model.run_training_job", fake_run_training_job)

    train_res = client.post("/api/model/train", json=build_graph_payload())
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]

    time.sleep(0.05)

    with client.websocket_connect(f"/ws/training/{job_id}") as ws:
        msg = ws.receive_json()

    assert msg["type"] == "error"
    assert msg["message"] == "boom"
