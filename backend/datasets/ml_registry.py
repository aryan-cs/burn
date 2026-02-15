"""Registry of datasets available for classical ML models."""

from __future__ import annotations

from typing import Any


ML_DATASET_REGISTRY: list[dict[str, Any]] = [
    {
        "id": "iris",
        "name": "Iris",
        "task": "classification",
        "n_features": 4,
        "n_classes": 3,
        "n_samples": 150,
        "feature_names": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
        "target_names": ["setosa", "versicolor", "virginica"],
        "description": "Classic flower classification dataset with 4 features and 3 species.",
    },
    {
        "id": "wine",
        "name": "Wine",
        "task": "classification",
        "n_features": 13,
        "n_classes": 3,
        "n_samples": 178,
        "feature_names": [
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280_od315",
            "proline",
        ],
        "target_names": ["class_0", "class_1", "class_2"],
        "description": "Wine recognition dataset from chemical analysis of 3 Italian cultivars.",
    },
    {
        "id": "breast_cancer",
        "name": "Breast Cancer",
        "task": "classification",
        "n_features": 30,
        "n_classes": 2,
        "n_samples": 569,
        "feature_names": [],  # 30 features — populated at load time
        "target_names": ["malignant", "benign"],
        "description": "Wisconsin breast cancer diagnostic dataset. Binary classification from cell nuclei features.",
    },
    {
        "id": "digits",
        "name": "Digits",
        "task": "classification",
        "n_features": 64,
        "n_classes": 10,
        "n_samples": 1797,
        "feature_names": [],  # pixel features — populated at load time
        "target_names": [str(i) for i in range(10)],
        "description": "Hand-written digits (8×8 images). 10-class classification.",
    },
    {
        "id": "california_housing",
        "name": "California Housing",
        "task": "regression",
        "n_features": 8,
        "n_classes": 0,
        "n_samples": 20640,
        "feature_names": [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
        "target_names": ["price"],
        "description": "California median house values. Regression target in 100k USD.",
    },
    {
        "id": "diabetes",
        "name": "Diabetes",
        "task": "regression",
        "n_features": 10,
        "n_classes": 0,
        "n_samples": 442,
        "feature_names": [
            "age",
            "sex",
            "bmi",
            "bp",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6",
        ],
        "target_names": ["progression"],
        "description": "Diabetes progression prediction from 10 baseline variables.",
    },
]


def list_ml_datasets(task: str | None = None) -> list[dict[str, Any]]:
    """Return available datasets, optionally filtered by task type."""
    if task is None:
        return ML_DATASET_REGISTRY
    return [d for d in ML_DATASET_REGISTRY if d["task"] == task]


def get_ml_dataset_info(dataset_id: str) -> dict[str, Any] | None:
    """Look up a single dataset entry by ID."""
    for d in ML_DATASET_REGISTRY:
        if d["id"] == dataset_id:
            return d
    return None
