from __future__ import annotations

from typing import Any


RF_DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "iris": {
        "id": "iris",
        "name": "Iris",
        "task": "classification",
        "source": "kaggle",
        "kaggle_dataset": "uciml/iris",
        "csv_filename": "Iris.csv",
        "target_column": "Species",
        "drop_columns": ["Id"],
        "delimiter": ",",
    },
    "wine": {
        "id": "wine",
        "name": "Wine Quality (Red)",
        "task": "classification",
        "source": "kaggle",
        "kaggle_dataset": "uciml/red-wine-quality-cortez-et-al-2009",
        "csv_filename": "winequality-red.csv",
        "target_column": "quality",
        "drop_columns": [],
        "delimiter": ";",
    },
    "breast_cancer": {
        "id": "breast_cancer",
        "name": "Breast Cancer Wisconsin",
        "task": "classification",
        "source": "kaggle",
        "kaggle_dataset": "uciml/breast-cancer-wisconsin-data",
        "csv_filename": "data.csv",
        "target_column": "diagnosis",
        "drop_columns": ["id", "Unnamed: 32"],
        "delimiter": ",",
    },
}


def get_rf_dataset_meta(dataset_id: str) -> dict[str, Any] | None:
    return RF_DATASET_REGISTRY.get(dataset_id)


def list_rf_datasets() -> list[dict[str, Any]]:
    return [RF_DATASET_REGISTRY[key] for key in sorted(RF_DATASET_REGISTRY.keys())]
