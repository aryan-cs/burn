from __future__ import annotations

from typing import Any


DATASET_REGISTRY: list[dict[str, Any]] = [
    {
        "id": "mnist",
        "name": "MNIST",
        "task": "classification",
        "input_shape": [1, 28, 28],
        "num_classes": 10,
        "source": "kaggle",
        "kaggle_dataset": "oddrationale/mnist-in-csv",
    }
]


def list_datasets() -> list[dict[str, Any]]:
    return DATASET_REGISTRY
