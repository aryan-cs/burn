from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core.job_registry import job_registry
from main import app


def build_graph_payload(training: dict | None = None) -> dict:
    payload = {
        "nodes": [
            {"id": "node_1", "type": "Input", "config": {"shape": [1, 28, 28]}},
            {"id": "node_2", "type": "Flatten", "config": {}},
            {"id": "node_3", "type": "Dense", "config": {"units": 64, "activation": "relu"}},
            {
                "id": "node_4",
                "type": "Output",
                "config": {"num_classes": 10, "activation": "softmax"},
            },
        ],
        "edges": [
            {"id": "edge_1", "source": "node_1", "target": "node_2"},
            {"id": "edge_2", "source": "node_2", "target": "node_3"},
            {"id": "edge_3", "source": "node_3", "target": "node_4"},
        ],
    }
    if training is not None:
        payload["training"] = training
    return payload


def build_rf_graph_payload(training: dict | None = None, *, feature_count: int = 4, num_classes: int = 3) -> dict:
    payload = {
        "nodes": [
            {"id": "rf_node_1", "type": "RFInput", "config": {"shape": [feature_count]}},
            {"id": "rf_node_2", "type": "RFFlatten", "config": {}},
            {
                "id": "rf_node_3",
                "type": "RandomForestClassifier",
                "config": {
                    "n_estimators": 12,
                    "max_depth": 6,
                    "criterion": "gini",
                    "max_features": "sqrt",
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "bootstrap": True,
                    "random_state": 42,
                },
            },
            {"id": "rf_node_4", "type": "RFOutput", "config": {"num_classes": num_classes}},
        ],
        "edges": [
            {"id": "rf_edge_1", "source": "rf_node_1", "target": "rf_node_2"},
            {"id": "rf_edge_2", "source": "rf_node_2", "target": "rf_node_3"},
            {"id": "rf_edge_3", "source": "rf_node_3", "target": "rf_node_4"},
        ],
    }
    if training is not None:
        payload["training"] = training
    return payload


@pytest.fixture
def client():
    job_registry.clear()
    with TestClient(app) as test_client:
        yield test_client
    job_registry.clear()


@pytest.fixture(autouse=True)
def _default_local_training_env(monkeypatch):
    monkeypatch.delenv("JETSON_HOST", raising=False)
    monkeypatch.delenv("JETSON_PORT", raising=False)
