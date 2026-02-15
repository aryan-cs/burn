from __future__ import annotations

import asyncio
import time
import types

import torch

from core.job_registry import job_registry
from tests.conftest import build_graph_payload


def test_create_local_deployment_and_infer(monkeypatch, client, tmp_path) -> None:
    async def fake_run_training_job(job_id, compiled, training, artifacts_dir, backend_override=None):
        entry = job_registry.get(job_id)
        assert entry is not None
        job_registry.set_status(job_id, "running")

        assert entry.model is not None
        artifact_dir = entry.job_dir or tmp_path
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model.pt"
        torch.save(entry.model.state_dict(), artifact_path)

        await asyncio.sleep(0.02)
        await job_registry.mark_terminal(
            job_id,
            "completed",
            final_metrics={"final_loss": 0.1, "final_accuracy": 0.9},
            artifact_path=artifact_path,
        )

    monkeypatch.setattr("routers.model.run_training_job", fake_run_training_job)
    monkeypatch.setattr("routers.model.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("routers.deploy.ARTIFACTS_DIR", tmp_path)

    train_res = client.post("/api/model/train", json=build_graph_payload())
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]
    time.sleep(0.05)

    deploy_res = client.post(
        "/api/deploy",
        json={"job_id": job_id, "target": "local", "name": "mnist-local-v1"},
    )
    assert deploy_res.status_code == 200
    deploy_data = deploy_res.json()
    deployment_id = deploy_data["deployment_id"]
    assert deploy_data["status"] == "running"
    assert deploy_data["target"] == "local"
    assert deploy_data["job_id"] == job_id
    assert deploy_data["endpoint_path"] == f"/api/deploy/{deployment_id}/infer"

    status_res = client.get(f"/api/deploy/status?deployment_id={deployment_id}")
    assert status_res.status_code == 200
    assert status_res.json()["request_count"] == 0

    infer_res = client.post(
        f"/api/deploy/{deployment_id}/infer",
        json={
            "inputs": [[[0.0 for _ in range(28)] for _ in range(28)]],
            "return_probabilities": True,
        },
    )
    assert infer_res.status_code == 200
    infer_data = infer_res.json()
    assert infer_data["deployment_id"] == deployment_id
    assert infer_data["job_id"] == job_id
    assert infer_data["input_shape"] == [1, 1, 28, 28]
    assert len(infer_data["predictions"]) == 1
    assert len(infer_data["probabilities"][0]) == 10

    logs_res = client.get(f"/api/deploy/logs?deployment_id={deployment_id}")
    assert logs_res.status_code == 200
    logs = logs_res.json()["logs"]
    events = [entry["event"] for entry in logs]
    assert "deployment_created" in events
    assert "inference_request" in events

    list_res = client.get("/api/deploy/list")
    assert list_res.status_code == 200
    deployments = list_res.json()["deployments"]
    assert any(item["deployment_id"] == deployment_id for item in deployments)

    stop_res = client.delete(f"/api/deploy/{deployment_id}")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "stopped"

    infer_after_stop = client.post(
        f"/api/deploy/{deployment_id}/infer",
        json={"inputs": [[[0.0 for _ in range(28)] for _ in range(28)]]},
    )
    assert infer_after_stop.status_code == 400

    start_res = client.post(f"/api/deploy/{deployment_id}/start")
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "running"

    infer_after_start = client.post(
        f"/api/deploy/{deployment_id}/infer",
        json={"inputs": [[[0.0 for _ in range(28)] for _ in range(28)]]},
    )
    assert infer_after_start.status_code == 200

    logs_after_stop = client.get(f"/api/deploy/logs?deployment_id={deployment_id}")
    assert logs_after_stop.status_code == 200
    events_after_stop = [entry["event"] for entry in logs_after_stop.json()["logs"]]
    assert "deployment_stopped" in events_after_stop
    assert "deployment_started" in events_after_stop


def test_remote_target_rejected(monkeypatch, client, tmp_path) -> None:
    monkeypatch.setattr("routers.deploy.ARTIFACTS_DIR", tmp_path)
    res = client.post("/api/deploy", json={"job_id": "abc", "target": "remote"})
    assert res.status_code == 400
    assert "supported deployment targets" in res.text.lower()


def test_external_linreg_deployment_visible_and_start_stop(client) -> None:
    create_res = client.post(
        "/api/deploy/external",
        json={
            "model_family": "linreg",
            "target": "local",
            "job_id": "linreg_study_hours_demo",
            "name": "Study Hours Endpoint",
            "runtime_config": {
                "weights": [2.0],
                "bias": 1.0,
                "means": [0.0],
                "stds": [1.0],
            },
        },
    )
    assert create_res.status_code == 200
    created = create_res.json()
    deployment_id = created["deployment_id"]
    assert created["model_family"] == "linreg"
    assert created["status"] == "running"

    list_res = client.get("/api/deploy/list")
    assert list_res.status_code == 200
    deployments = list_res.json()["deployments"]
    assert any(item["deployment_id"] == deployment_id for item in deployments)

    infer_res = client.post(
        f"/api/deploy/{deployment_id}/infer",
        json={"inputs": [3.5], "return_probabilities": False},
    )
    assert infer_res.status_code == 200
    infer_data = infer_res.json()
    assert infer_data["model_family"] == "linreg"
    assert abs(float(infer_data["predictions"][0]) - 8.0) < 1e-6

    touch_res = client.post(
        f"/api/deploy/{deployment_id}/touch",
        json={
            "event": "external_inference_request",
            "message": "External linreg inference completed.",
            "details": {"source": "linreg-ui"},
        },
    )
    assert touch_res.status_code == 200
    assert touch_res.json()["request_count"] >= 2

    stop_res = client.delete(f"/api/deploy/{deployment_id}")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "stopped"

    start_res = client.post(f"/api/deploy/{deployment_id}/start")
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "running"


def test_deploy_can_use_in_memory_job_without_bundle(monkeypatch, client, tmp_path) -> None:
    async def fake_run_training_job(job_id, compiled, training, artifacts_dir, backend_override=None):
        job_registry.set_status(job_id, "running")
        await asyncio.sleep(0.01)

    monkeypatch.setattr("routers.model.run_training_job", fake_run_training_job)
    # Simulate a pre-persistence job or a transient in-memory job.
    monkeypatch.setattr("routers.deploy.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("routers.model.ARTIFACTS_DIR", tmp_path)
    train_res = client.post("/api/model/train", json=build_graph_payload())
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]

    # Remove bundle to force in-memory fallback path.
    bundle_dir = tmp_path / "jobs" / job_id
    if bundle_dir.exists():
        for child in bundle_dir.iterdir():
            child.unlink()
        bundle_dir.rmdir()

    deploy_res = client.post("/api/deploy", json={"job_id": job_id, "target": "local"})
    assert deploy_res.status_code == 200
    deploy_data = deploy_res.json()
    assert deploy_data["job_id"] == job_id


def test_create_modal_deployment_and_proxy_infer(monkeypatch, client, tmp_path) -> None:
    async def fake_run_training_job(job_id, compiled, training, artifacts_dir, backend_override=None):
        entry = job_registry.get(job_id)
        assert entry is not None
        job_registry.set_status(job_id, "running")
        artifact_dir = entry.job_dir or tmp_path
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model.pt"
        torch.save(entry.model.state_dict(), artifact_path)
        await asyncio.sleep(0.02)
        await job_registry.mark_terminal(
            job_id,
            "completed",
            final_metrics={"final_loss": 0.1, "final_accuracy": 0.9},
            artifact_path=artifact_path,
        )

    monkeypatch.setattr("routers.model.run_training_job", fake_run_training_job)
    monkeypatch.setattr("routers.model.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("routers.deploy.ARTIFACTS_DIR", tmp_path)

    class FakeRegisterFunction:
        def __init__(self):
            self.calls = []

        def remote(self, deployment_id, graph_payload, training_payload, state_dict_b64, input_shape):
            self.calls.append(deployment_id)
            return {"deployment_id": deployment_id, "status": "ready"}

    class FakeInferFunction:
        def get_web_url(self):
            return "https://burn-example.modal.run/infer"

        def remote(self, payload):
            return {
                "deployment_id": payload["deployment_id"],
                "input_shape": [1, 1, 28, 28],
                "output_shape": [1, 10],
                "logits": [[0.0 for _ in range(10)]],
                "predictions": [1],
                "probabilities": [[0.1 for _ in range(10)]],
            }

    class FakeUnregisterFunction:
        def remote(self, deployment_id):
            return {"deployment_id": deployment_id, "status": "deleted"}

    register_fn = FakeRegisterFunction()
    infer_fn = FakeInferFunction()
    unregister_fn = FakeUnregisterFunction()

    class FakeFunctionNamespace:
        @staticmethod
        def from_name(app_name: str, fn_name: str, environment_name=None):
            assert app_name == "burn-training"
            assert environment_name is None
            if fn_name == "register_deployment_remote":
                return register_fn
            if fn_name == "infer_deployment_remote":
                return infer_fn
            if fn_name == "unregister_deployment_remote":
                return unregister_fn
            raise AssertionError(f"Unexpected function name: {fn_name}")

    fake_modal = types.ModuleType("modal")
    fake_modal.Function = FakeFunctionNamespace
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)

    train_res = client.post("/api/model/train", json=build_graph_payload())
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]
    time.sleep(0.05)

    deploy_res = client.post(
        "/api/deploy",
        json={"job_id": job_id, "target": "modal", "name": "mnist-modal-v1"},
    )
    assert deploy_res.status_code == 200
    deploy_data = deploy_res.json()
    deployment_id = deploy_data["deployment_id"]
    assert deploy_data["target"] == "modal"
    assert deploy_data["endpoint_path"] == "https://burn-example.modal.run/infer"
    assert deployment_id in register_fn.calls

    infer_res = client.post(
        f"/api/deploy/{deployment_id}/infer",
        json={"inputs": [[[0.0 for _ in range(28)] for _ in range(28)]], "return_probabilities": True},
    )
    assert infer_res.status_code == 200
    infer_data = infer_res.json()
    assert infer_data["deployment_id"] == deployment_id
    assert infer_data["job_id"] == job_id
    assert infer_data["predictions"] == [1]

    logs_res = client.get(f"/api/deploy/logs?deployment_id={deployment_id}")
    assert logs_res.status_code == 200
    events = [entry["event"] for entry in logs_res.json()["logs"]]
    assert "modal_endpoint_registered" in events
