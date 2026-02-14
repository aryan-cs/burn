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


@pytest.fixture
def client():
    job_registry.clear()
    with TestClient(app) as test_client:
        yield test_client
    job_registry.clear()
