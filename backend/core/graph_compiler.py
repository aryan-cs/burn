from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from core.shape_inference import validate_graph
from models.graph_schema import GraphSpec, LayerType
from models.training_config import TrainingConfig


class GraphCompileError(Exception):
    def __init__(self, errors: list[dict[str, Any]]):
        super().__init__("Graph compilation failed")
        self.errors = errors


@dataclass
class CompiledStep:
    node_id: str
    node_type: str
    attr_name: str
    expression: str
    module: nn.Module


@dataclass
class CompiledGraphResult:
    model: nn.Module
    python_source: str
    execution_order: list[str]
    warnings: list[str]
    summary: dict[str, Any]


class GeneratedSequentialModel(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):  # type: ignore[override]
        for layer in self.layers:
            x = layer(x)
        return x


def _sanitize_identifier(value: str) -> str:
    cleaned = [ch if ch.isalnum() else "_" for ch in value]
    out = "".join(cleaned)
    if not out:
        return "node"
    if out[0].isdigit():
        return f"n_{out}"
    return out


def _activation_module(name: str | None) -> tuple[nn.Module | None, str | None]:
    if name is None:
        return None, None

    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"none", "identity", "linear", ""}:
        return None, None
    if key == "relu":
        return nn.ReLU(), "nn.ReLU()"
    if key in {"leaky_relu", "leakyrelu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.01), "nn.LeakyReLU(negative_slope=0.01)"
    if key == "sigmoid":
        return nn.Sigmoid(), "nn.Sigmoid()"
    if key == "tanh":
        return nn.Tanh(), "nn.Tanh()"
    if key == "gelu":
        return nn.GELU(), "nn.GELU()"
    if key == "softmax":
        return nn.Softmax(dim=1), "nn.Softmax(dim=1)"

    raise ValueError(f"Unsupported activation: {name}")


def _model_source(steps: list[CompiledStep]) -> str:
    lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        "class GeneratedModel(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]

    if steps:
        for step in steps:
            lines.append(f"        self.{step.attr_name} = {step.expression}")
        joined = ", ".join([f"self.{step.attr_name}" for step in steps])
        lines.append(f"        self.layers = nn.ModuleList([{joined}])")
    else:
        lines.append("        self.layers = nn.ModuleList()")

    lines.extend(
        [
            "",
            "    def forward(self, x):",
            "        for layer in self.layers:",
            "            x = layer(x)",
            "        return x",
            "",
        ]
    )
    return "\n".join(lines)


def compile_graph(graph: GraphSpec, training: TrainingConfig) -> CompiledGraphResult:
    validation = validate_graph(graph)
    if not validation.valid:
        raise GraphCompileError(validation.errors)

    node_lookup = {node.id: node for node in graph.nodes}
    steps: list[CompiledStep] = []
    warnings = list(validation.warnings)
    summary_layers: list[dict[str, Any]] = []

    step_counter = 0

    for node_id in validation.execution_order:
        node = node_lookup[node_id]
        in_shape = validation.shapes[node_id]["input"]
        out_shape = validation.shapes[node_id]["output"]

        if node.type == LayerType.INPUT:
            summary_layers.append(
                {
                    "node_id": node_id,
                    "type": node.type.value,
                    "input_shape": in_shape,
                    "output_shape": out_shape,
                    "module": "input",
                }
            )
            continue

        if node.type == LayerType.FLATTEN:
            attr = f"layer_{step_counter}_{_sanitize_identifier(node_id)}"
            step = CompiledStep(
                node_id=node_id,
                node_type=node.type.value,
                attr_name=attr,
                expression="nn.Flatten()",
                module=nn.Flatten(),
            )
            steps.append(step)
            step_counter += 1
            summary_layers.append(
                {
                    "node_id": node_id,
                    "type": node.type.value,
                    "input_shape": in_shape,
                    "output_shape": out_shape,
                    "module": "nn.Flatten()",
                }
            )
            continue

        if node.type == LayerType.DROPOUT:
            rate = float(node.config.get("rate", 0.5))
            attr = f"layer_{step_counter}_{_sanitize_identifier(node_id)}"
            expr = f"nn.Dropout(p={rate})"
            step = CompiledStep(
                node_id=node_id,
                node_type=node.type.value,
                attr_name=attr,
                expression=expr,
                module=nn.Dropout(p=rate),
            )
            steps.append(step)
            step_counter += 1
            summary_layers.append(
                {
                    "node_id": node_id,
                    "type": node.type.value,
                    "input_shape": in_shape,
                    "output_shape": out_shape,
                    "module": expr,
                }
            )
            continue

        if node.type in {LayerType.DENSE, LayerType.OUTPUT}:
            if not in_shape:
                raise GraphCompileError(
                    [
                        {
                            "node_id": node_id,
                            "message": "Missing input shape during compilation",
                            "expected": "non-empty input shape",
                            "got": in_shape,
                        }
                    ]
                )
            in_features = int(in_shape[0])
            if node.type == LayerType.DENSE:
                out_features = int(node.config["units"])
            else:
                out_features = int(node.config["num_classes"])

            linear_attr = f"layer_{step_counter}_{_sanitize_identifier(node_id)}_linear"
            linear_expr = f"nn.Linear({in_features}, {out_features})"
            linear_step = CompiledStep(
                node_id=node_id,
                node_type=node.type.value,
                attr_name=linear_attr,
                expression=linear_expr,
                module=nn.Linear(in_features, out_features),
            )
            steps.append(linear_step)
            step_counter += 1

            module_desc = [linear_expr]
            activation = node.config.get("activation")
            if activation is None and node.type == LayerType.DENSE:
                activation = "relu"

            if (
                node.type == LayerType.OUTPUT
                and isinstance(activation, str)
                and activation.strip().lower() == "softmax"
                and training.loss == "cross_entropy"
            ):
                warnings.append(
                    "Output softmax suppressed because loss is cross_entropy (expects logits)."
                )
                activation = "none"

            act_module, act_expr = _activation_module(str(activation) if activation is not None else None)
            if act_module is not None and act_expr is not None:
                act_attr = f"layer_{step_counter}_{_sanitize_identifier(node_id)}_act"
                act_step = CompiledStep(
                    node_id=node_id,
                    node_type=node.type.value,
                    attr_name=act_attr,
                    expression=act_expr,
                    module=act_module,
                )
                steps.append(act_step)
                step_counter += 1
                module_desc.append(act_expr)

            summary_layers.append(
                {
                    "node_id": node_id,
                    "type": node.type.value,
                    "input_shape": in_shape,
                    "output_shape": out_shape,
                    "module": " -> ".join(module_desc),
                }
            )
            continue

        raise GraphCompileError(
            [
                {
                    "node_id": node_id,
                    "message": f"Unsupported layer type for v1: {node.type.value}",
                    "expected": [
                        LayerType.INPUT.value,
                        LayerType.DENSE.value,
                        LayerType.DROPOUT.value,
                        LayerType.FLATTEN.value,
                        LayerType.OUTPUT.value,
                    ],
                    "got": node.type.value,
                }
            ]
        )

    model = GeneratedSequentialModel([step.module for step in steps])
    param_count = sum(param.numel() for param in model.parameters())

    summary = {
        "param_count": int(param_count),
        "layers": summary_layers,
        "resolved_training": training.model_dump(),
    }

    return CompiledGraphResult(
        model=model,
        python_source=_model_source(steps),
        execution_order=validation.execution_order,
        warnings=warnings,
        summary=summary,
    )
