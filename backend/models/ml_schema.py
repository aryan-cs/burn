"""Pydantic schemas for classical ML model configuration and requests."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MLModelType(str, Enum):
    linear_regression = "linear_regression"
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"


# ── Hyperparameter schemas ────────────────────────────


class LinearRegressionConfig(BaseModel):
    fit_intercept: bool = True
    normalize: bool = False  # deprecated in newer sklearn, but kept for API clarity


class LogisticRegressionConfig(BaseModel):
    C: float = Field(default=1.0, gt=0, description="Inverse regularisation strength")
    max_iter: int = Field(default=200, ge=1)
    penalty: str = Field(default="l2", pattern=r"^(l1|l2|elasticnet|none)$")
    solver: str = Field(
        default="lbfgs",
        pattern=r"^(lbfgs|liblinear|newton-cg|sag|saga)$",
    )


class RandomForestConfig(BaseModel):
    n_estimators: int = Field(default=100, ge=1, description="Number of trees")
    max_depth: int | None = Field(default=None, ge=1, description="Max tree depth (None = unlimited)")
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    criterion: str = Field(default="gini", pattern=r"^(gini|entropy|log_loss)$")
    max_features: str = Field(default="sqrt", pattern=r"^(sqrt|log2|auto)$")


# ── Request / response schemas ────────────────────────


class MLTrainRequest(BaseModel):
    model_type: MLModelType
    dataset: str = "iris"
    test_size: float = Field(default=0.2, gt=0, lt=1)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class MLPredictRequest(BaseModel):
    job_id: str
    features: list[list[float]]


class MLTrainResponse(BaseModel):
    job_id: str
    status: str


class MLStatusResponse(BaseModel):
    job_id: str
    status: str
    model_type: str | None = None
    dataset: str | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None
    feature_names: list[str] | None = None
    target_names: list[str] | None = None


class MLPredictResponse(BaseModel):
    predictions: list[Any]
    probabilities: list[list[float]] | None = None
    feature_importances: dict[str, float] | None = None
