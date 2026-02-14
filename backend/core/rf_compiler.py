from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.rf_shape_inference import validate_rf_graph
from models.rf_graph_schema import RFGraphSpec, RFNodeType
from models.rf_training_config import RFHyperParams, RFTrainingConfig, normalize_rf_hyperparams


class RFCompileError(Exception):
    def __init__(self, errors: list[dict[str, Any]]):
        super().__init__("Random forest graph compilation failed")
        self.errors = errors


@dataclass
class CompiledRFResult:
    python_source: str
    execution_order: list[str]
    warnings: list[str]
    summary: dict[str, Any]
    hyperparams: RFHyperParams
    expected_feature_count: int
    num_classes: int


def _python_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def _model_source(
    hyperparams: RFHyperParams,
    *,
    dataset: str,
    expected_feature_count: int,
    num_classes: int,
) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "from pathlib import Path",
        "from typing import Any",
        "",
        "import joblib",
        "import numpy as np",
        "from sklearn.ensemble import RandomForestClassifier",
        "",
        "",
        "class GeneratedRandomForestModel:",
        "    def __init__(self) -> None:",
        "        self.dataset = " + repr(dataset),
        f"        self.expected_feature_count = {expected_feature_count}",
        f"        self.num_classes = {num_classes}",
        "        self.model = RandomForestClassifier(",
    ]

    ordered_keys = [
        "n_estimators",
        "max_depth",
        "criterion",
        "max_features",
        "min_samples_split",
        "min_samples_leaf",
        "bootstrap",
        "random_state",
    ]
    for key in ordered_keys:
        value = getattr(hyperparams, key)
        lines.append(f"            {key}={_python_value(value)},")

    lines.extend(
        [
            "        )",
            "",
            "    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:",
            "        self.model.fit(features, labels)",
            "",
            "    def predict(self, features: np.ndarray) -> np.ndarray:",
            "        features = np.asarray(features, dtype=np.float32)",
            "        if features.ndim == 1:",
            "            features = features.reshape(1, -1)",
            "        if features.shape[1] != self.expected_feature_count:",
            "            raise ValueError(",
            "                f'Expected {self.expected_feature_count} features, got {features.shape[1]}'",
            "            )",
            "        return self.model.predict(features)",
            "",
            "    def predict_proba(self, features: np.ndarray) -> np.ndarray:",
            "        features = np.asarray(features, dtype=np.float32)",
            "        if features.ndim == 1:",
            "            features = features.reshape(1, -1)",
            "        if features.shape[1] != self.expected_feature_count:",
            "            raise ValueError(",
            "                f'Expected {self.expected_feature_count} features, got {features.shape[1]}'",
            "            )",
            "        return self.model.predict_proba(features)",
            "",
            "    def save(self, path: str | Path) -> None:",
            "        payload: dict[str, Any] = {",
            "            'model': self.model,",
            "            'dataset': self.dataset,",
            "            'expected_feature_count': self.expected_feature_count,",
            "            'num_classes': self.num_classes,",
            "        }",
            "        joblib.dump(payload, str(path))",
            "",
            "    @classmethod",
            "    def load(cls, path: str | Path) -> GeneratedRandomForestModel:",
            "        payload = joblib.load(str(path))",
            "        instance = cls()",
            "        instance.model = payload['model']",
            "        instance.dataset = payload.get('dataset', instance.dataset)",
            "        instance.expected_feature_count = payload.get(",
            "            'expected_feature_count', instance.expected_feature_count",
            "        )",
            "        instance.num_classes = payload.get('num_classes', instance.num_classes)",
            "        return instance",
            "",
        ]
    )
    return "\n".join(lines)


def compile_rf_graph(graph: RFGraphSpec, training: RFTrainingConfig) -> CompiledRFResult:
    validation = validate_rf_graph(graph)
    if not validation.valid:
        raise RFCompileError(validation.errors)

    node_lookup = {node.id: node for node in graph.nodes}
    warnings = list(validation.warnings)

    model_node_id = next(
        (node_id for node_id in validation.execution_order if node_lookup[node_id].type == RFNodeType.RANDOM_FOREST),
        None,
    )
    output_node_id = next(
        (node_id for node_id in validation.execution_order if node_lookup[node_id].type == RFNodeType.OUTPUT),
        None,
    )
    input_node_id = next(
        (node_id for node_id in validation.execution_order if node_lookup[node_id].type == RFNodeType.INPUT),
        None,
    )

    if model_node_id is None or output_node_id is None or input_node_id is None:
        raise RFCompileError(
            [
                {
                    "node_id": None,
                    "message": "Missing required RFInput/RandomForestClassifier/RFOutput nodes",
                }
            ]
        )

    model_node = node_lookup[model_node_id]
    output_node = node_lookup[output_node_id]
    input_node = node_lookup[input_node_id]

    model_input_shape = validation.shapes[model_node_id]["input"]
    if not model_input_shape or len(model_input_shape) != 1:
        raise RFCompileError(
            [
                {
                    "node_id": model_node_id,
                    "message": "RandomForestClassifier input shape must resolve to rank-1",
                    "expected": "[feature_count]",
                    "got": model_input_shape,
                }
            ]
        )

    expected_feature_count = int(model_input_shape[0])
    output_classes = output_node.config.get("num_classes")
    if not isinstance(output_classes, int) or output_classes <= 0:
        raise RFCompileError(
            [
                {
                    "node_id": output_node_id,
                    "message": "RFOutput.num_classes must be a positive integer",
                }
            ]
        )

    hyperparams = normalize_rf_hyperparams(model_node.config, training)

    layers_summary: list[dict[str, Any]] = []
    for node_id in validation.execution_order:
        node = node_lookup[node_id]
        layers_summary.append(
            {
                "node_id": node_id,
                "type": node.type.value,
                "input_shape": validation.shapes[node_id]["input"],
                "output_shape": validation.shapes[node_id]["output"],
                "config": node.config,
            }
        )

    python_source = _model_source(
        hyperparams,
        dataset=training.dataset,
        expected_feature_count=expected_feature_count,
        num_classes=output_classes,
    )

    summary = {
        "model_family": "random_forest",
        "layers": layers_summary,
        "resolved_training": training.model_dump(),
        "resolved_hyperparameters": hyperparams.model_dump(),
        "dataset": training.dataset,
        "expected_feature_count": expected_feature_count,
        "num_classes": output_classes,
        "input_shape": input_node.config.get("shape"),
    }

    return CompiledRFResult(
        python_source=python_source,
        execution_order=validation.execution_order,
        warnings=warnings,
        summary=summary,
        hyperparams=hyperparams,
        expected_feature_count=expected_feature_count,
        num_classes=output_classes,
    )
