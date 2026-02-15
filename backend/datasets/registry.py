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
        "loader": "kaggle_mnist_csv",
    },
    {
        "id": "digits",
        "name": "Digits (sklearn)",
        "task": "classification",
        "input_shape": [1, 8, 8],
        "num_classes": 10,
        "source": "sklearn",
        "loader": "sklearn_digits",
    }
]


def list_datasets() -> list[dict[str, Any]]:
    return DATASET_REGISTRY


def get_dataset_meta(dataset_id: str) -> dict[str, Any] | None:
    normalized = dataset_id.strip().lower()
    for item in DATASET_REGISTRY:
        if item["id"] == normalized:
            return item
    return None
