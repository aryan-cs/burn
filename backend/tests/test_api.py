from __future__ import annotations

import asyncio
import time

from core.job_registry import job_registry
from tests.conftest import build_graph_payload


def test_model_validate_and_compile(client) -> None:
    payload = build_graph_payload()

    validate_res = client.post("/api/model/validate", json=payload)
    assert validate_res.status_code == 200
    validate_data = validate_res.json()
    assert validate_data["valid"] is True
    assert validate_data["errors"] == []

    compile_res = client.post("/api/model/compile", json=payload)
    assert compile_res.status_code == 200
    compile_data = compile_res.json()
    assert compile_data["valid"] is True
    assert compile_data["summary"]["param_count"] > 0
    assert "class GeneratedModel" in compile_data["python_source"]


def test_model_latest_endpoint(client) -> None:
    latest_before = client.get("/api/model/latest")
    assert latest_before.status_code == 200
    assert latest_before.json()["job_id"] is None

    train_res = client.post("/api/model/train", json=build_graph_payload())
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]

    latest_after = client.get("/api/model/latest")
    assert latest_after.status_code == 200
    assert latest_after.json()["job_id"] == job_id


def test_model_validate_error(client) -> None:
    payload = build_graph_payload()
    payload["edges"].append({"id": "broken", "source": "node_999", "target": "node_4"})

    res = client.post("/api/model/validate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is False
    assert any("unknown" in err["message"].lower() for err in data["errors"])


def test_train_stop_and_export(monkeypatch, client, tmp_path) -> None:
    async def fake_run_training_job(job_id, compiled, training, artifacts_dir):
        entry = job_registry.get(job_id)
        assert entry is not None
        job_registry.set_status(job_id, "running")

        # Simulate minimal train lifecycle and artifact creation.
        artifact = tmp_path / f"{job_id}.pt"
        artifact.write_bytes(b"fake-model")

        await asyncio.sleep(0.05)
        await job_registry.publish(
            job_id,
            {
                "type": "training_done",
                "final_loss": 0.1,
                "final_accuracy": 0.9,
                "model_path": str(artifact),
            },
        )
        await job_registry.mark_terminal(
            job_id,
            "completed",
            final_metrics={"final_loss": 0.1, "final_accuracy": 0.9},
            artifact_path=artifact,
        )

    monkeypatch.setattr("routers.model.run_training_job", fake_run_training_job)

    payload = build_graph_payload(
        training={
            "dataset": "mnist",
            "epochs": 1,
            "batchSize": 8,
            "optimizer": "adam",
            "learningRate": 0.001,
            "loss": "cross_entropy",
        }
    )

    train_res = client.post("/api/model/train", json=payload)
    assert train_res.status_code == 200
    train_data = train_res.json()
    job_id = train_data["job_id"]

    status_res = client.get(f"/api/model/status?job_id={job_id}")
    assert status_res.status_code == 200
    assert status_res.json()["job_id"] == job_id

    stop_res = client.post("/api/model/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["job_id"] == job_id

    py_export = client.get(f"/api/model/export?job_id={job_id}&format=py")
    assert py_export.status_code == 200
    assert "class GeneratedModel" in py_export.text

    infer_payload = {
        "job_id": job_id,
        "inputs": [[[0.0 for _ in range(28)] for _ in range(28)]],
        "return_probabilities": True,
    }
    infer_res = client.post("/api/model/infer", json=infer_payload)
    assert infer_res.status_code == 200
    infer_data = infer_res.json()
    assert infer_data["job_id"] == job_id
    assert len(infer_data["predictions"]) == 1
    assert len(infer_data["probabilities"][0]) == 10

    time.sleep(0.1)

    pt_export = client.get(f"/api/model/export?job_id={job_id}&format=pt")
    assert pt_export.status_code == 200
    assert pt_export.content == b"fake-model"
