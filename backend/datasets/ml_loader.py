"""Load scikit-learn datasets and split into train/test arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.datasets import fetch_california_housing as _fetch_california
except ImportError:  # very old sklearn
    _fetch_california = None  # type: ignore[assignment]


_LOADERS: dict[str, Any] = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
    "diabetes": load_diabetes,
    "california_housing": _fetch_california,
}


class MLDataSplit:
    """Holds the train/test split plus metadata."""

    __slots__ = (
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "feature_names",
        "target_names",
        "task",
        "n_classes",
    )

    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
        target_names: list[str],
        task: str,
        n_classes: int,
    ) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.target_names = target_names
        self.task = task
        self.n_classes = n_classes


_TASK_MAP: dict[str, str] = {
    "iris": "classification",
    "wine": "classification",
    "breast_cancer": "classification",
    "digits": "classification",
    "diabetes": "regression",
    "california_housing": "regression",
}


def load_ml_dataset(
    dataset_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> MLDataSplit:
    """Load and split a scikit-learn dataset.

    Returns an ``MLDataSplit`` with scaled features, train/test arrays, and metadata.
    """
    loader = _LOADERS.get(dataset_id)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset_id!r}")

    bunch = loader()

    X: np.ndarray = bunch.data.astype(np.float64)
    y: np.ndarray = bunch.target

    feature_names: list[str] = list(bunch.feature_names) if hasattr(bunch, "feature_names") else [f"f{i}" for i in range(X.shape[1])]
    target_names: list[str] = (
        [str(n) for n in bunch.target_names]
        if hasattr(bunch, "target_names")
        else ["target"]
    )

    task = _TASK_MAP[dataset_id]
    n_classes = len(set(y.tolist())) if task == "classification" else 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task == "classification" else None,
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return MLDataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
        task=task,
        n_classes=n_classes,
    )
