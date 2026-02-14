from __future__ import annotations

from models.graph_schema import GraphSpec
from core.shape_inference import validate_graph
from tests.conftest import build_graph_payload


def test_validate_graph_success() -> None:
    graph = GraphSpec.model_validate(build_graph_payload())
    result = validate_graph(graph)

    assert result.valid is True
    assert result.errors == []
    assert result.execution_order == ["node_1", "node_2", "node_3", "node_4"]
    assert result.shapes["node_1"]["output"] == [1, 28, 28]
    assert result.shapes["node_2"]["output"] == [784]
    assert result.shapes["node_3"]["output"] == [64]
    assert result.shapes["node_4"]["output"] == [10]


def test_validate_graph_rejects_branching() -> None:
    payload = build_graph_payload()
    payload["nodes"].insert(3, {"id": "node_x", "type": "Dropout", "config": {"rate": 0.2}})
    payload["edges"].append({"id": "edge_x", "source": "node_2", "target": "node_x"})
    payload["edges"].append({"id": "edge_y", "source": "node_x", "target": "node_4"})

    graph = GraphSpec.model_validate(payload)
    result = validate_graph(graph)

    assert result.valid is False
    assert any("Intermediate nodes must have in-degree 1 and out-degree 1" in e["message"] or "Output node must have" in e["message"] for e in result.errors)


def test_validate_graph_rejects_cycle() -> None:
    payload = build_graph_payload()
    payload["edges"].append({"id": "edge_cycle", "source": "node_4", "target": "node_2"})

    graph = GraphSpec.model_validate(payload)
    result = validate_graph(graph)

    assert result.valid is False
    assert any("cycle" in e["message"].lower() for e in result.errors)


def test_validate_graph_rejects_dense_rank_mismatch() -> None:
    payload = {
        "nodes": [
            {"id": "in", "type": "Input", "config": {"shape": [1, 28, 28]}},
            {"id": "dense", "type": "Dense", "config": {"units": 32, "activation": "relu"}},
            {"id": "out", "type": "Output", "config": {"num_classes": 10, "activation": "softmax"}},
        ],
        "edges": [
            {"id": "e1", "source": "in", "target": "dense"},
            {"id": "e2", "source": "dense", "target": "out"},
        ],
    }

    graph = GraphSpec.model_validate(payload)
    result = validate_graph(graph)

    assert result.valid is False
    assert any("Dense requires rank-1 input" in e["message"] for e in result.errors)
