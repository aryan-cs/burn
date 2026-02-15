from __future__ import annotations

import asyncio
import base64
import io
import json
import types

import torch

from core.graph_compiler import CompiledGraphResult, compile_graph
from core.job_registry import job_registry
from core.job_storage import GRAPH_FILENAME, NN_ARTIFACT_FILENAME
from core.training_engine import run_training_job
from models.graph_schema import GraphSpec
from models.training_config import normalize_training_config
from tests.conftest import build_graph_payload


def _encode_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_run_training_job_uses_modal_backend(monkeypatch, tmp_path) -> None:
    job_registry.clear()
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
    graph = GraphSpec.model_validate(payload)
    training = normalize_training_config(payload["training"])
    compiled_graph = compile_graph(graph, training)
    compiled = CompiledGraphResult(
        model=compiled_graph.model,
        python_source=compiled_graph.python_source,
        execution_order=compiled_graph.execution_order,
        warnings=compiled_graph.warnings,
        summary=compiled_graph.summary,
    )

    entry = job_registry.create_job(
        compiled.python_source,
        model=compiled.model,
        input_shape=[1, 8, 8],
        num_classes=10,
    )
    job_dir = tmp_path / "jobs" / entry.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / GRAPH_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    job_registry.set_job_dir(entry.job_id, job_dir)

    remote_state = {name: torch.ones_like(tensor) for name, tensor in compiled.model.state_dict().items()}
    fake_response = {
        "epoch_metrics": [
            {
                "epoch": 1,
                "train_loss": 0.4,
                "train_accuracy": 0.8,
                "test_loss": 0.3,
                "test_accuracy": 0.85,
            }
        ],
        "final_metrics": {
            "final_train_loss": 0.4,
            "final_train_accuracy": 0.8,
            "final_test_loss": 0.3,
            "final_test_accuracy": 0.85,
            "final_loss": 0.3,
            "final_accuracy": 0.85,
        },
        "state_dict_b64": _encode_state_dict(remote_state),
    }

    class FakeCall:
        def __init__(self):
            self.cancelled = False

        def get(self, timeout: float = 0) -> dict:
            return fake_response

        def cancel(self) -> None:
            self.cancelled = True

    class FakeRemoteFunction:
        def spawn(self, graph_payload: dict, training_payload: dict) -> FakeCall:
            assert graph_payload["nodes"][0]["type"] == "Input"
            assert training_payload["dataset"] == "digits"
            return FakeCall()

    class FakeFunctionNamespace:
        @staticmethod
        def from_name(app_name: str, fn_name: str, environment_name=None):
            assert app_name == "burn-training"
            assert fn_name == "train_job_remote"
            assert environment_name is None
            return FakeRemoteFunction()

    fake_modal = types.ModuleType("modal")
    fake_modal.Function = FakeFunctionNamespace
    fake_modal_exception = types.ModuleType("modal.exception")
    fake_modal_exception.TimeoutError = TimeoutError

    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    monkeypatch.setitem(__import__("sys").modules, "modal.exception", fake_modal_exception)
    monkeypatch.setenv("TRAINING_BACKEND", "modal")
    monkeypatch.setenv("MODAL_APP_NAME", "burn-training")
    monkeypatch.setenv("MODAL_FUNCTION_NAME", "train_job_remote")
    monkeypatch.delenv("MODAL_ENVIRONMENT_NAME", raising=False)

    asyncio.run(run_training_job(entry.job_id, compiled, training, tmp_path))

    trained_entry = job_registry.get(entry.job_id)
    assert trained_entry is not None
    assert trained_entry.terminal is True
    assert trained_entry.status == "completed"
    assert trained_entry.final_metrics is not None
    assert trained_entry.final_metrics["final_accuracy"] == 0.85
    assert (job_dir / NN_ARTIFACT_FILENAME).exists()

    saved_state = torch.load(job_dir / NN_ARTIFACT_FILENAME, map_location="cpu")
    for name, tensor in remote_state.items():
        assert torch.equal(saved_state[name], tensor)
    job_registry.clear()


def test_run_training_job_loads_compiled_modal_state_dict(monkeypatch, tmp_path) -> None:
    job_registry.clear()
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
    graph = GraphSpec.model_validate(payload)
    training = normalize_training_config(payload["training"])
    compiled_graph = compile_graph(graph, training)
    compiled = CompiledGraphResult(
        model=compiled_graph.model,
        python_source=compiled_graph.python_source,
        execution_order=compiled_graph.execution_order,
        warnings=compiled_graph.warnings,
        summary=compiled_graph.summary,
    )

    entry = job_registry.create_job(
        compiled.python_source,
        model=compiled.model,
        input_shape=[1, 8, 8],
        num_classes=10,
    )
    job_dir = tmp_path / "jobs" / entry.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / GRAPH_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    job_registry.set_job_dir(entry.job_id, job_dir)

    base_state = compiled.model.state_dict()
    compiled_keyed_state = {f"_orig_mod.{name}": torch.ones_like(tensor) for name, tensor in base_state.items()}
    fake_response = {
        "epoch_metrics": [
            {
                "epoch": 1,
                "train_loss": 0.4,
                "train_accuracy": 0.8,
                "test_loss": 0.3,
                "test_accuracy": 0.85,
            }
        ],
        "final_metrics": {
            "final_train_loss": 0.4,
            "final_train_accuracy": 0.8,
            "final_test_loss": 0.3,
            "final_test_accuracy": 0.85,
            "final_loss": 0.3,
            "final_accuracy": 0.85,
        },
        "state_dict_b64": _encode_state_dict(compiled_keyed_state),
    }

    class FakeCall:
        def __init__(self):
            self.cancelled = False

        def get(self, timeout: float = 0) -> dict:
            return fake_response

        def cancel(self) -> None:
            self.cancelled = True

    class FakeRemoteFunction:
        def spawn(self, graph_payload: dict, training_payload: dict) -> FakeCall:
            assert graph_payload["nodes"][0]["type"] == "Input"
            assert training_payload["dataset"] == "digits"
            return FakeCall()

    class FakeFunctionNamespace:
        @staticmethod
        def from_name(app_name: str, fn_name: str, environment_name=None):
            assert app_name == "burn-training"
            assert fn_name == "train_job_remote"
            assert environment_name is None
            return FakeRemoteFunction()

    fake_modal = types.ModuleType("modal")
    fake_modal.Function = FakeFunctionNamespace
    fake_modal_exception = types.ModuleType("modal.exception")
    fake_modal_exception.TimeoutError = TimeoutError

    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    monkeypatch.setitem(__import__("sys").modules, "modal.exception", fake_modal_exception)
    monkeypatch.setenv("TRAINING_BACKEND", "modal")
    monkeypatch.setenv("MODAL_APP_NAME", "burn-training")
    monkeypatch.setenv("MODAL_FUNCTION_NAME", "train_job_remote")
    monkeypatch.delenv("MODAL_ENVIRONMENT_NAME", raising=False)

    asyncio.run(run_training_job(entry.job_id, compiled, training, tmp_path))

    trained_entry = job_registry.get(entry.job_id)
    assert trained_entry is not None
    assert trained_entry.terminal is True
    assert trained_entry.status == "completed"
    assert trained_entry.final_metrics is not None
    assert trained_entry.final_metrics["final_accuracy"] == 0.85
    assert (job_dir / NN_ARTIFACT_FILENAME).exists()

    saved_state = torch.load(job_dir / NN_ARTIFACT_FILENAME, map_location="cpu")
    for name in base_state:
        assert torch.equal(saved_state[name], compiled_keyed_state[f"_orig_mod.{name}"])
    job_registry.clear()


def test_run_training_job_tolerates_builtin_timeouterror(monkeypatch, tmp_path) -> None:
    job_registry.clear()
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
    graph = GraphSpec.model_validate(payload)
    training = normalize_training_config(payload["training"])
    compiled_graph = compile_graph(graph, training)
    compiled = CompiledGraphResult(
        model=compiled_graph.model,
        python_source=compiled_graph.python_source,
        execution_order=compiled_graph.execution_order,
        warnings=compiled_graph.warnings,
        summary=compiled_graph.summary,
    )

    entry = job_registry.create_job(
        compiled.python_source,
        model=compiled.model,
        input_shape=[1, 8, 8],
        num_classes=10,
    )
    job_dir = tmp_path / "jobs" / entry.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / GRAPH_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    job_registry.set_job_dir(entry.job_id, job_dir)

    remote_state = {name: torch.ones_like(tensor) for name, tensor in compiled.model.state_dict().items()}
    fake_response = {
        "epoch_metrics": [
            {
                "epoch": 1,
                "train_loss": 0.4,
                "train_accuracy": 0.8,
                "test_loss": 0.3,
                "test_accuracy": 0.85,
            }
        ],
        "final_metrics": {
            "final_train_loss": 0.4,
            "final_train_accuracy": 0.8,
            "final_test_loss": 0.3,
            "final_test_accuracy": 0.85,
            "final_loss": 0.3,
            "final_accuracy": 0.85,
        },
        "state_dict_b64": _encode_state_dict(remote_state),
    }

    class FakeCall:
        def __init__(self):
            self.polls = 0
            self.cancelled = False

        def get(self, timeout: float = 0) -> dict:
            self.polls += 1
            if self.polls == 1:
                raise TimeoutError()
            return fake_response

        def cancel(self) -> None:
            self.cancelled = True

    class FakeRemoteFunction:
        def spawn(self, graph_payload: dict, training_payload: dict) -> FakeCall:
            assert graph_payload["nodes"][0]["type"] == "Input"
            assert training_payload["dataset"] == "digits"
            return FakeCall()

    class FakeFunctionNamespace:
        @staticmethod
        def from_name(app_name: str, fn_name: str, environment_name=None):
            assert app_name == "burn-training"
            assert fn_name == "train_job_remote"
            assert environment_name is None
            return FakeRemoteFunction()

    fake_modal = types.ModuleType("modal")
    fake_modal.Function = FakeFunctionNamespace
    fake_modal_exception = types.ModuleType("modal.exception")
    fake_modal_exception.TimeoutError = type("FakeModalTimeoutError", (Exception,), {})

    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    monkeypatch.setitem(__import__("sys").modules, "modal.exception", fake_modal_exception)
    monkeypatch.setenv("TRAINING_BACKEND", "modal")
    monkeypatch.setenv("MODAL_APP_NAME", "burn-training")
    monkeypatch.setenv("MODAL_FUNCTION_NAME", "train_job_remote")
    monkeypatch.delenv("MODAL_ENVIRONMENT_NAME", raising=False)

    asyncio.run(run_training_job(entry.job_id, compiled, training, tmp_path))

    trained_entry = job_registry.get(entry.job_id)
    assert trained_entry is not None
    assert trained_entry.terminal is True
    assert trained_entry.status == "completed"
    assert trained_entry.error is None
    assert (job_dir / NN_ARTIFACT_FILENAME).exists()
    job_registry.clear()
