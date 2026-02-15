from __future__ import annotations

from core.rf_compiler import compile_rf_graph
from models.rf_graph_schema import RFGraphSpec
from models.rf_training_config import RFTrainingConfig
from tests.conftest import build_rf_graph_payload


def test_compile_rf_graph_builds_summary_and_source() -> None:
    graph = RFGraphSpec.model_validate(build_rf_graph_payload())
    training = RFTrainingConfig(dataset="iris")

    compiled = compile_rf_graph(graph, training)
    assert "class GeneratedRandomForestModel" in compiled.python_source
    assert "RandomForestClassifier(" in compiled.python_source
    assert compiled.summary["model_family"] == "random_forest"
    assert compiled.summary["expected_feature_count"] == 4
    assert compiled.summary["num_classes"] == 3
    assert compiled.hyperparams.n_estimators == 12


def test_compile_rf_graph_source_is_deterministic() -> None:
    graph = RFGraphSpec.model_validate(build_rf_graph_payload())
    training = RFTrainingConfig(dataset="iris")

    first = compile_rf_graph(graph, training).python_source
    second = compile_rf_graph(graph, training).python_source
    assert first == second
