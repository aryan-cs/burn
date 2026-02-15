from __future__ import annotations

from core.rf_shape_inference import validate_rf_graph
from models.rf_graph_schema import RFGraphSpec
from tests.conftest import build_rf_graph_payload


def test_rf_shape_inference_valid_chain() -> None:
    graph = RFGraphSpec.model_validate(build_rf_graph_payload())
    result = validate_rf_graph(graph)
    assert result.valid is True
    assert result.errors == []
    assert result.execution_order == ["rf_node_1", "rf_node_2", "rf_node_3", "rf_node_4"]
    assert result.shapes["rf_node_1"]["output"] == [4]
    assert result.shapes["rf_node_3"]["input"] == [4]
    assert result.shapes["rf_node_4"]["output"] == [3]


def test_rf_shape_inference_rejects_branching() -> None:
    payload = build_rf_graph_payload()
    payload["nodes"].append({"id": "rf_node_5", "type": "RFFlatten", "config": {}})
    payload["edges"].append({"id": "rf_edge_4", "source": "rf_node_1", "target": "rf_node_5"})

    graph = RFGraphSpec.model_validate(payload)
    result = validate_rf_graph(graph)
    assert result.valid is False
    assert any("in-degree" in error["message"] or "single" in error["message"].lower() for error in result.errors)
