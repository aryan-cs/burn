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


def test_datasets_endpoint_lists_digits(client) -> None:
    res = client.get("/api/datasets")
    assert res.status_code == 200
    data = res.json()
    dataset_ids = {entry["id"] for entry in data["datasets"]}
    assert {"mnist", "digits"}.issubset(dataset_ids)


def test_model_inference_samples_for_cats_vs_dogs(monkeypatch, client) -> None:
    def fake_samples(limit: int, split: str):
        assert split == "test"
        assert limit == 3
        return [
            {
                "id": "test:0:cat.jpg",
                "filename": "cat.jpg",
                "label": 0,
                "label_name": "cat",
                "image_data_url": "data:image/jpeg;base64,AAA",
                "inputs": [[[0.0]]],
            }
        ]

    monkeypatch.setattr("routers.model.get_cats_vs_dogs_inference_samples", fake_samples)

    response = client.get("/api/model/inference-samples?dataset=cats_vs_dogs&split=test&limit=3")
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "cats_vs_dogs"
    assert payload["split"] == "test"
    assert payload["classes"] == [{"index": 0, "name": "cat"}, {"index": 1, "name": "dog"}]
    assert payload["samples"][0]["label_name"] == "cat"


def test_model_inference_samples_rejects_unsupported_dataset(client) -> None:
    response = client.get("/api/model/inference-samples?dataset=mnist")
    assert response.status_code == 400
    assert "cats_vs_dogs" in response.json()["detail"]["message"]


def test_model_latest_endpoint(monkeypatch, client, tmp_path) -> None:
    monkeypatch.setattr("routers.model.ARTIFACTS_DIR", tmp_path)

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


def test_train_rejects_shape_for_selected_dataset(client) -> None:
    payload = build_graph_payload(
        training={
            "dataset": "digits",
            "epochs": 1,
            "batchSize": 8,
            "optimizer": "adam",
            "learningRate": 0.001,
            "loss": "cross_entropy",
        }
    )
    # Graph is still [1, 28, 28] / 10-class MNIST-style, so digits contract should fail.
    res = client.post("/api/model/train", json=payload)
    assert res.status_code == 400
    error_messages = [item["message"] for item in res.json()["detail"]["errors"]]
    assert any("Digits" in message for message in error_messages)
    assert any("Input.shape=[1, 8, 8]" in message for message in error_messages)


def test_train_stop_and_export(monkeypatch, client, tmp_path) -> None:
    async def fake_run_training_job(job_id, compiled, training, artifacts_dir, backend_override=None):
        entry = job_registry.get(job_id)
        assert entry is not None
        job_registry.set_status(job_id, "running")

        # Simulate minimal train lifecycle and artifact creation.
        artifact_dir = entry.job_dir or tmp_path
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact = artifact_dir / "model.pt"
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
    monkeypatch.setattr("routers.model.ARTIFACTS_DIR", tmp_path)

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

    # Ensure export works from persisted bundle even after in-memory registry reset.
    job_registry.clear()
    py_export_after_clear = client.get(f"/api/model/export?job_id={job_id}&format=py")
    assert py_export_after_clear.status_code == 200
    assert "class GeneratedModel" in py_export_after_clear.text
    pt_export_after_clear = client.get(f"/api/model/export?job_id={job_id}&format=pt")
    assert pt_export_after_clear.status_code == 200
    assert pt_export_after_clear.content == b"fake-model"
