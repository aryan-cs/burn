from __future__ import annotations

import torch

from core.graph_compiler import compile_graph
from models.graph_schema import GraphSpec
from models.training_config import TrainingConfig
from tests.conftest import build_graph_payload


def test_compile_graph_builds_model_and_source() -> None:
    graph = GraphSpec.model_validate(build_graph_payload())
    training = TrainingConfig()

    compiled = compile_graph(graph, training)
    assert "class GeneratedModel(nn.Module):" in compiled.python_source
    assert compiled.summary["param_count"] > 0
    assert compiled.execution_order == ["node_1", "node_2", "node_3", "node_4"]

    x = torch.randn(4, 1, 28, 28)
    y = compiled.model(x)
    assert list(y.shape) == [4, 10]


def test_compile_graph_softmax_warning_for_cross_entropy() -> None:
    graph = GraphSpec.model_validate(build_graph_payload())
    training = TrainingConfig(loss="cross_entropy")

    compiled = compile_graph(graph, training)
    assert any("softmax suppressed" in warning.lower() for warning in compiled.warnings)


def test_compile_graph_supports_conv2d_and_maxpool2d() -> None:
    payload = {
        "nodes": [
            {"id": "in", "type": "Input", "config": {"shape": [3, 32, 32]}},
            {"id": "conv", "type": "Conv2D", "config": {"filters": 16, "kernel_size": 3, "stride": 1, "padding": 1, "activation": "relu"}},
            {"id": "pool", "type": "MaxPool2D", "config": {"kernel_size": 2, "stride": 2}},
            {"id": "flat", "type": "Flatten", "config": {}},
            {"id": "dense", "type": "Dense", "config": {"units": 32, "activation": "relu"}},
            {"id": "out", "type": "Output", "config": {"num_classes": 2, "activation": "softmax"}},
        ],
        "edges": [
            {"id": "e1", "source": "in", "target": "conv"},
            {"id": "e2", "source": "conv", "target": "pool"},
            {"id": "e3", "source": "pool", "target": "flat"},
            {"id": "e4", "source": "flat", "target": "dense"},
            {"id": "e5", "source": "dense", "target": "out"},
        ],
    }
    graph = GraphSpec.model_validate(payload)
    training = TrainingConfig(dataset="cats_vs_dogs")

    compiled = compile_graph(graph, training)
    assert "nn.Conv2d(" in compiled.python_source
    assert "nn.MaxPool2d(" in compiled.python_source

    x = torch.randn(2, 3, 32, 32)
    y = compiled.model(x)
    assert list(y.shape) == [2, 2]
