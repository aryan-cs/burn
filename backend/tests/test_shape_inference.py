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


def test_validate_graph_supports_alexnet_shape_pipeline() -> None:
    payload = {
        "nodes": [
            {"id": "in", "type": "Input", "config": {"shape": [3, 224, 224]}},
            {"id": "conv1", "type": "Conv2D", "config": {"filters": 64, "kernel_size": 11, "stride": 4, "padding": 2}},
            {"id": "pool1", "type": "MaxPool2D", "config": {"kernel_size": 3, "stride": 2}},
            {"id": "conv2", "type": "Conv2D", "config": {"filters": 192, "kernel_size": 5, "padding": 2}},
            {"id": "pool2", "type": "MaxPool2D", "config": {"kernel_size": 3, "stride": 2}},
            {"id": "conv3", "type": "Conv2D", "config": {"filters": 384, "kernel_size": 3, "padding": 1}},
            {"id": "conv4", "type": "Conv2D", "config": {"filters": 256, "kernel_size": 3, "padding": 1}},
            {"id": "conv5", "type": "Conv2D", "config": {"filters": 256, "kernel_size": 3, "padding": 1}},
            {"id": "pool5", "type": "MaxPool2D", "config": {"kernel_size": 3, "stride": 2}},
            {"id": "flat", "type": "Flatten", "config": {}},
            {"id": "fc6", "type": "Dense", "config": {"units": 4096, "activation": "relu"}},
            {"id": "fc7", "type": "Dense", "config": {"units": 4096, "activation": "relu"}},
            {"id": "out", "type": "Output", "config": {"num_classes": 2, "activation": "softmax"}},
        ],
        "edges": [
            {"id": "e1", "source": "in", "target": "conv1"},
            {"id": "e2", "source": "conv1", "target": "pool1"},
            {"id": "e3", "source": "pool1", "target": "conv2"},
            {"id": "e4", "source": "conv2", "target": "pool2"},
            {"id": "e5", "source": "pool2", "target": "conv3"},
            {"id": "e6", "source": "conv3", "target": "conv4"},
            {"id": "e7", "source": "conv4", "target": "conv5"},
            {"id": "e8", "source": "conv5", "target": "pool5"},
            {"id": "e9", "source": "pool5", "target": "flat"},
            {"id": "e10", "source": "flat", "target": "fc6"},
            {"id": "e11", "source": "fc6", "target": "fc7"},
            {"id": "e12", "source": "fc7", "target": "out"},
        ],
    }

    graph = GraphSpec.model_validate(payload)
    result = validate_graph(graph)

    assert result.valid is True
    assert result.shapes["conv1"]["output"] == [64, 55, 55]
    assert result.shapes["pool1"]["output"] == [64, 27, 27]
    assert result.shapes["conv2"]["output"] == [192, 27, 27]
    assert result.shapes["pool2"]["output"] == [192, 13, 13]
    assert result.shapes["pool5"]["output"] == [256, 6, 6]
    assert result.shapes["flat"]["output"] == [9216]
    assert result.shapes["out"]["output"] == [2]
