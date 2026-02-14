from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpu_node.core.job_registry import remote_job_registry
from gpu_node.main import app
from tests.conftest import build_graph_payload


def test_gpu_node_requires_token(monkeypatch) -> None:
    remote_job_registry.clear()
    monkeypatch.setenv("JETSON_PASS", "jetson")
    with TestClient(app) as client:
        res = client.post("/train", json=build_graph_payload())
    assert res.status_code == 401
    remote_job_registry.clear()


def test_gpu_node_train_events_and_artifact(monkeypatch, tmp_path) -> None:
    remote_job_registry.clear()
    monkeypatch.setenv("JETSON_PASS", "jetson")

    async def fake_run_training_job(job_id, compiled, training, artifacts_dir):
        entry = remote_job_registry.get(job_id)
        assert entry is not None
        remote_job_registry.set_status(job_id, "running")
        artifact = Path(tmp_path) / f"{job_id}.pt"
        artifact.write_bytes(b"gpu-node-model")
        await remote_job_registry.publish(
            job_id,
            {
                "type": "epoch_update",
                "epoch": 1,
                "loss": 0.4,
                "accuracy": 0.82,
                "train_loss": 0.35,
                "train_accuracy": 0.84,
                "test_loss": 0.4,
                "test_accuracy": 0.82,
                "weights": {},
            },
        )
        await remote_job_registry.publish(
            job_id,
            {
                "type": "training_done",
                "final_loss": 0.4,
                "final_accuracy": 0.82,
                "model_path": str(artifact),
            },
        )
        await remote_job_registry.mark_terminal(job_id, "completed", artifact_path=artifact)

    def fake_lazy_stack():
        class DummyCompileError(Exception):
            pass

        def fake_compile_graph(graph, training):
            return SimpleNamespace()

        return DummyCompileError, fake_compile_graph, fake_run_training_job

    monkeypatch.setattr("gpu_node.routers.training._lazy_load_training_stack", fake_lazy_stack)

    with TestClient(app) as client:
        headers = {"X-Jetson-Token": "jetson"}
        train_res = client.post("/train", json=build_graph_payload(), headers=headers)
        assert train_res.status_code == 200
        job_id = train_res.json()["job_id"]

        time.sleep(0.05)

        events_res = client.get(f"/jobs/{job_id}/events", params={"after": 0}, headers=headers)
        assert events_res.status_code == 200
        payload = events_res.json()
        assert payload["done"] is True
        assert [event["type"] for event in payload["events"]] == ["epoch_update", "training_done"]

        artifact_res = client.get(f"/jobs/{job_id}/artifact", headers=headers)
        assert artifact_res.status_code == 200
        assert artifact_res.content == b"gpu-node-model"

    remote_job_registry.clear()


def test_gpu_node_stop(monkeypatch) -> None:
    remote_job_registry.clear()
    monkeypatch.setenv("JETSON_PASS", "")

    with TestClient(app) as client:
        async def fake_run_training_job(job_id, compiled, training, artifacts_dir):
            remote_job_registry.set_status(job_id, "running")
            await asyncio.sleep(0.2)
            await remote_job_registry.mark_terminal(job_id, "stopped")

        def fake_lazy_stack():
            class DummyCompileError(Exception):
                pass

            def fake_compile_graph(graph, training):
                return SimpleNamespace()

            return DummyCompileError, fake_compile_graph, fake_run_training_job

        monkeypatch.setattr("gpu_node.routers.training._lazy_load_training_stack", fake_lazy_stack)
        train_res = client.post("/train", json=build_graph_payload())
        assert train_res.status_code == 200
        job_id = train_res.json()["job_id"]

        stop_res = client.post(f"/jobs/{job_id}/stop")
        assert stop_res.status_code == 200
        assert stop_res.json()["status"] == "stopping"

    remote_job_registry.clear()
