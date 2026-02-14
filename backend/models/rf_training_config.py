from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .rf_graph_schema import RFTrainingConfigIn


class RFTrainingConfig(BaseModel):
    dataset: str = "iris"
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    log_every_trees: int = 10

    @field_validator("dataset")
    @classmethod
    def normalize_dataset(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, value: float) -> float:
        if value <= 0.0 or value >= 1.0:
            raise ValueError("test_size must be in (0, 1)")
        return value

    @field_validator("log_every_trees")
    @classmethod
    def validate_log_every_trees(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("log_every_trees must be > 0")
        return value


class RFHyperParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_estimators: int = Field(default=100)
    max_depth: int | None = None
    criterion: str = "gini"
    max_features: str | int | float | None = "sqrt"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = True
    random_state: int | None = None

    @field_validator("n_estimators")
    @classmethod
    def validate_n_estimators(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n_estimators must be > 0")
        return value

    @field_validator("max_depth")
    @classmethod
    def validate_max_depth(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_depth must be > 0 when provided")
        return value

    @field_validator("criterion")
    @classmethod
    def validate_criterion(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"gini", "entropy", "log_loss"}
        if normalized not in allowed:
            raise ValueError(f"criterion must be one of {sorted(allowed)}")
        return normalized

    @field_validator("max_features")
    @classmethod
    def validate_max_features(cls, value: str | int | float | None) -> str | int | float | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            allowed = {"sqrt", "log2"}
            if normalized not in allowed:
                raise ValueError(f"max_features string must be one of {sorted(allowed)}")
            return normalized
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("max_features integer must be > 0")
            return value
        if isinstance(value, float):
            if value <= 0.0:
                raise ValueError("max_features float must be > 0")
            return value
        raise ValueError("max_features must be string, int, float, or null")

    @field_validator("min_samples_split")
    @classmethod
    def validate_min_samples_split(cls, value: int) -> int:
        if value < 2:
            raise ValueError("min_samples_split must be >= 2")
        return value

    @field_validator("min_samples_leaf")
    @classmethod
    def validate_min_samples_leaf(cls, value: int) -> int:
        if value < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        return value


def normalize_rf_training_config(training: RFTrainingConfigIn | dict | None) -> RFTrainingConfig:
    if training is None:
        return RFTrainingConfig()

    if isinstance(training, RFTrainingConfigIn):
        payload = training.model_dump(by_alias=False)
    else:
        payload = RFTrainingConfigIn.model_validate(training).model_dump(by_alias=False)

    return RFTrainingConfig.model_validate(payload)


def normalize_rf_hyperparams(node_config: dict[str, Any], training: RFTrainingConfig) -> RFHyperParams:
    payload = dict(node_config)
    if payload.get("random_state") is None:
        payload["random_state"] = training.random_state
    return RFHyperParams.model_validate(payload)
