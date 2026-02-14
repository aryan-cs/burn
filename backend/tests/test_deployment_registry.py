from __future__ import annotations

from core.deployment_registry import DeploymentRegistry


def test_registry_persists_and_reloads_stopped_deployments(tmp_path) -> None:
    state_path = tmp_path / "deployments" / "registry.json"
    registry = DeploymentRegistry(state_path=state_path)

    created = registry.create_deployment(
        job_id="job_abc123",
        target="local",
        name="demo",
        model=object(),
        input_shape=[1, 28, 28],
        num_classes=10,
    )
    registry.mark_request(created.deployment_id)
    registry.mark_stopped(created.deployment_id)

    reloaded = DeploymentRegistry(state_path=state_path)
    loaded = reloaded.get(created.deployment_id)

    assert loaded is not None
    assert loaded.status == "stopped"
    assert loaded.job_id == "job_abc123"
    assert loaded.request_count == 1
    assert loaded.model is None
    events = [entry.event for entry in (loaded.logs or [])]
    assert "deployment_created" in events
    assert "deployment_stopped" in events


def test_running_deployments_restore_as_stopped_after_restart(tmp_path) -> None:
    state_path = tmp_path / "deployments" / "registry.json"
    registry = DeploymentRegistry(state_path=state_path)

    created = registry.create_deployment(
        job_id="job_xyz",
        target="local",
        name=None,
        model=object(),
        input_shape=[1, 28, 28],
        num_classes=10,
    )
    assert created.status == "running"

    reloaded = DeploymentRegistry(state_path=state_path)
    loaded = reloaded.get(created.deployment_id)

    assert loaded is not None
    assert loaded.status == "stopped"
    events = [entry.event for entry in (loaded.logs or [])]
    assert "deployment_restored_stopped" in events
