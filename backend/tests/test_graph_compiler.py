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
