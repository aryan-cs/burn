from __future__ import annotations

import asyncio
import time

import joblib
import numpy as np

from core.rf_job_registry import rf_job_registry
from tests.conftest import build_rf_graph_payload


class _FakeRFModel:
    def predict(self, matrix):
        rows = np.asarray(matrix).shape[0]
        return np.zeros(rows, dtype=np.int64)

    def predict_proba(self, matrix):
        rows = np.asarray(matrix).shape[0]
        return np.tile(np.array([[0.75, 0.25]], dtype=np.float64), (rows, 1))


def test_rf_validate_compile_and_datasets(client) -> None:
    payload = build_rf_graph_payload()

    validate_res = client.post("/api/rf/validate", json=payload)
    assert validate_res.status_code == 200
    validate_data = validate_res.json()
    assert validate_data["valid"] is True
    assert validate_data["errors"] == []

    compile_res = client.post("/api/rf/compile", json=payload)
    assert compile_res.status_code == 200
    compile_data = compile_res.json()
    assert compile_data["valid"] is True
    assert "GeneratedRandomForestModel" in compile_data["python_source"]

    datasets_res = client.get("/api/rf/datasets")
    assert datasets_res.status_code == 200
    datasets_data = datasets_res.json()
    dataset_ids = {dataset["id"] for dataset in datasets_data["datasets"]}
    assert {"iris", "wine", "breast_cancer"}.issubset(dataset_ids)


def test_rf_train_stop_export_and_infer(monkeypatch, client, tmp_path) -> None:
    async def fake_run_rf_training_job(job_id, compiled, training, artifacts_dir):
        rf_job_registry.set_status(job_id, "running")
        model = _FakeRFModel()
        artifact_path = tmp_path / f"{job_id}.pkl"
        payload = {
            "model": model,
            "dataset": "iris",
            "feature_names": ["f1", "f2", "f3", "f4"],
            "class_names": ["setosa", "versicolor"],
            "expected_feature_count": 4,
            "hyperparameters": compiled.hyperparams.model_dump(),
        }
        joblib.dump(payload, artifact_path)
        rf_job_registry.set_model_data(
            job_id,
            model=model,
            feature_names=["f1", "f2", "f3", "f4"],
            class_names=["setosa", "versicolor"],
            expected_feature_count=4,
        )
        await asyncio.sleep(0.02)
        await rf_job_registry.publish(
            job_id,
            {
                "type": "rf_done",
                "final_train_accuracy": 0.95,
                "final_test_accuracy": 0.9,
                "confusion_matrix": [[5, 1], [0, 4]],
                "classes": ["setosa", "versicolor"],
                "feature_importances": [0.25, 0.25, 0.25, 0.25],
                "feature_names": ["f1", "f2", "f3", "f4"],
                "model_path": str(artifact_path),
            },
        )
        await rf_job_registry.mark_terminal(
            job_id,
            "completed",
            final_metrics={"final_train_accuracy": 0.95, "final_test_accuracy": 0.9},
            artifact_path=artifact_path,
        )

    monkeypatch.setattr("routers.rf_model.run_rf_training_job", fake_run_rf_training_job)

    payload = build_rf_graph_payload(training={"dataset": "iris", "testSize": 0.2, "randomState": 7})
    train_res = client.post("/api/rf/train", json=payload)
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]

    status_res = client.get(f"/api/rf/status?job_id={job_id}")
    assert status_res.status_code == 200

    stop_res = client.post("/api/rf/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["job_id"] == job_id

    py_export = client.get(f"/api/rf/export?job_id={job_id}&format=py")
    assert py_export.status_code == 200
    assert "GeneratedRandomForestModel" in py_export.text

    infer_res = client.post(
        "/api/rf/infer",
        json={"job_id": job_id, "inputs": [5.1, 3.5, 1.4, 0.2], "return_probabilities": True},
    )
    assert infer_res.status_code == 200
    infer_data = infer_res.json()
    assert infer_data["job_id"] == job_id
    assert infer_data["prediction_indices"] == [0]
    assert infer_data["predictions"] == ["setosa"]
    assert infer_data["probabilities"][0] == [0.75, 0.25]

    time.sleep(0.05)

    artifact_export = client.get(f"/api/rf/export?job_id={job_id}&format=pkl")
    assert artifact_export.status_code == 200
