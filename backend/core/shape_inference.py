from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.graph_schema import GraphSpec, LayerType


@dataclass
class GraphValidationResult:
    valid: bool
    shapes: dict[str, dict[str, list[int] | None]]
    errors: list[dict[str, Any]]
    execution_order: list[str]
    warnings: list[str]


def _safe_int_list(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty integer list")

    output: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError(f"{field_name} must contain only integers")
        if item <= 0:
            raise ValueError(f"{field_name} values must be > 0")
        output.append(item)
    return output


def _topological_sort(node_ids: list[str], outgoing: dict[str, list[str]], indegree: dict[str, int]) -> list[str]:
    queue: list[str] = sorted([nid for nid in node_ids if indegree[nid] == 0])
    order: list[str] = []

    while queue:
        current = queue.pop(0)
        order.append(current)
        for nxt in sorted(outgoing[current]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    return order


def _build_graph_index(graph: GraphSpec) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, list[str]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    node_by_id: dict[str, Any] = {}

    for node in graph.nodes:
        if node.id in node_by_id:
            errors.append(
                {
                    "node_id": node.id,
                    "message": "Duplicate node ID",
                    "expected": "unique node id",
                    "got": node.id,
                }
            )
            continue
        node_by_id[node.id] = node

    outgoing = {nid: [] for nid in node_by_id}
    incoming = {nid: [] for nid in node_by_id}

    edge_ids: set[str] = set()
    for edge in graph.edges:
        if edge.id in edge_ids:
            errors.append(
                {
                    "node_id": None,
                    "message": "Duplicate edge ID",
                    "expected": "unique edge id",
                    "got": edge.id,
                }
            )
        edge_ids.add(edge.id)

        if edge.source not in node_by_id or edge.target not in node_by_id:
            errors.append(
                {
                    "node_id": edge.target,
                    "message": "Edge references unknown node",
                    "expected": "existing source and target node IDs",
                    "got": {"source": edge.source, "target": edge.target},
                }
            )
            continue

        outgoing[edge.source].append(edge.target)
        incoming[edge.target].append(edge.source)

    return node_by_id, outgoing, incoming, errors


def _infer_output_shape(node_type: LayerType, config: dict[str, Any], input_shape: list[int] | None) -> list[int]:
    if node_type == LayerType.INPUT:
        return _safe_int_list(config.get("shape"), "Input.shape")

    if input_shape is None:
        raise ValueError("Missing input shape")

    if node_type == LayerType.FLATTEN:
        total = 1
        for dim in input_shape:
            total *= dim
        return [total]

    if node_type == LayerType.DENSE:
        if len(input_shape) != 1:
            raise ValueError("Dense requires rank-1 input; add Flatten before Dense")
        units = config.get("units")
        if not isinstance(units, int) or units <= 0:
            raise ValueError("Dense.units must be a positive integer")
        return [units]

    if node_type == LayerType.DROPOUT:
        rate = config.get("rate", 0.5)
        if not isinstance(rate, (int, float)):
            raise ValueError("Dropout.rate must be numeric")
        if rate < 0 or rate >= 1:
            raise ValueError("Dropout.rate must be in [0, 1)")
        return input_shape

    if node_type == LayerType.OUTPUT:
        if len(input_shape) != 1:
            raise ValueError("Output requires rank-1 input")
        classes = config.get("num_classes")
        if not isinstance(classes, int) or classes <= 0:
            raise ValueError("Output.num_classes must be a positive integer")
        return [classes]

    raise ValueError(f"Unsupported layer type for v1: {node_type.value}")


def validate_graph(graph: GraphSpec) -> GraphValidationResult:
    node_by_id, outgoing, incoming, errors = _build_graph_index(graph)
    warnings: list[str] = []

    if not node_by_id:
        errors.append(
            {
                "node_id": None,
                "message": "Graph has no nodes",
                "expected": "at least one node",
                "got": 0,
            }
        )
        return GraphValidationResult(
            valid=False,
            shapes={},
            errors=errors,
            execution_order=[],
            warnings=warnings,
        )

    input_nodes = [n for n in node_by_id.values() if n.type == LayerType.INPUT]
    output_nodes = [n for n in node_by_id.values() if n.type == LayerType.OUTPUT]

    if len(input_nodes) != 1:
        errors.append(
            {
                "node_id": None,
                "message": "Graph must contain exactly one Input node",
                "expected": 1,
                "got": len(input_nodes),
            }
        )

    if len(output_nodes) != 1:
        errors.append(
            {
                "node_id": None,
                "message": "Graph must contain exactly one Output node",
                "expected": 1,
                "got": len(output_nodes),
            }
        )

    node_ids = sorted(node_by_id.keys())
    indegree = {nid: len(incoming[nid]) for nid in node_ids}
    topo_order = _topological_sort(node_ids, outgoing, indegree.copy())
    if len(topo_order) != len(node_ids):
        errors.append(
            {
                "node_id": None,
                "message": "Graph contains a cycle",
                "expected": "acyclic DAG",
                "got": "cycle detected",
            }
        )

    if len(input_nodes) == 1 and len(output_nodes) == 1:
        input_id = input_nodes[0].id
        output_id = output_nodes[0].id

        for node_id, node in node_by_id.items():
            in_d = len(incoming[node_id])
            out_d = len(outgoing[node_id])

            if node.type == LayerType.INPUT:
                if in_d != 0 or out_d != 1:
                    errors.append(
                        {
                            "node_id": node_id,
                            "message": "Input node must have in-degree 0 and out-degree 1",
                            "expected": {"in_degree": 0, "out_degree": 1},
                            "got": {"in_degree": in_d, "out_degree": out_d},
                        }
                    )
            elif node.type == LayerType.OUTPUT:
                if in_d != 1 or out_d != 0:
                    errors.append(
                        {
                            "node_id": node_id,
                            "message": "Output node must have in-degree 1 and out-degree 0",
                            "expected": {"in_degree": 1, "out_degree": 0},
                            "got": {"in_degree": in_d, "out_degree": out_d},
                        }
                    )
            else:
                if in_d != 1 or out_d != 1:
                    errors.append(
                        {
                            "node_id": node_id,
                            "message": "Intermediate nodes must have in-degree 1 and out-degree 1",
                            "expected": {"in_degree": 1, "out_degree": 1},
                            "got": {"in_degree": in_d, "out_degree": out_d},
                        }
                    )

        path_nodes: list[str] = []
        visited: set[str] = set()
        current = input_id

        while True:
            path_nodes.append(current)
            visited.add(current)
            if current == output_id:
                break

            next_nodes = outgoing.get(current, [])
            if len(next_nodes) != 1:
                break

            current = next_nodes[0]
            if current in visited:
                break

        if not path_nodes or path_nodes[-1] != output_id:
            errors.append(
                {
                    "node_id": output_id,
                    "message": "Could not build a single Input->Output path",
                    "expected": "connected sequential path",
                    "got": path_nodes,
                }
            )

        if set(path_nodes) != set(node_ids):
            errors.append(
                {
                    "node_id": None,
                    "message": "All nodes must be on the single Input->Output path",
                    "expected": sorted(node_ids),
                    "got": sorted(path_nodes),
                }
            )

        execution_order = path_nodes if path_nodes else topo_order
    else:
        execution_order = topo_order

    shapes: dict[str, dict[str, list[int] | None]] = {
        nid: {"input": None, "output": None} for nid in node_ids
    }

    if not errors:
        for idx, node_id in enumerate(execution_order):
            node = node_by_id[node_id]
            if idx == 0:
                input_shape = None
            else:
                prev_node_id = execution_order[idx - 1]
                input_shape = shapes[prev_node_id]["output"]

            shapes[node_id]["input"] = input_shape
            try:
                out_shape = _infer_output_shape(node.type, node.config, input_shape)
                shapes[node_id]["output"] = out_shape
            except ValueError as exc:
                errors.append(
                    {
                        "node_id": node_id,
                        "message": str(exc),
                        "expected": "compatible shape/config",
                        "got": {
                            "input_shape": input_shape,
                            "config": node.config,
                            "type": node.type.value,
                        },
                    }
                )
                shapes[node_id]["output"] = None
                break

    return GraphValidationResult(
        valid=len(errors) == 0,
        shapes=shapes,
        errors=errors,
        execution_order=execution_order,
        warnings=warnings,
    )
