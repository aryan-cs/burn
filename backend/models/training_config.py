from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .graph_schema import TrainingConfigIn

_LOSS_ALIASES = {
    "cross_entropy": "cross_entropy",
    "crossentropy": "cross_entropy",
    "ce": "cross_entropy",
    "mse": "mse",
    "mse_loss": "mse",
    "mean_squared_error": "mse",
}


class TrainingConfig(BaseModel):
    dataset: str = "mnist"
    epochs: int = 20
    batch_size: int = 64
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    loss: str = "cross_entropy"

    @field_validator("dataset")
    @classmethod
    def normalize_dataset(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("optimizer")
    @classmethod
    def normalize_optimizer(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("loss")
    @classmethod
    def normalize_loss(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        canonical = _LOSS_ALIASES.get(normalized)
        if canonical is None:
            raise ValueError("loss must be one of: cross_entropy, mse")
        return canonical

    @field_validator("epochs")
    @classmethod
    def positive_epochs(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("epochs must be > 0")
        return value

    @field_validator("batch_size")
    @classmethod
    def positive_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size must be > 0")
        return value

    @field_validator("learning_rate")
    @classmethod
    def positive_learning_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("learning_rate must be > 0")
        return value


def normalize_training_config(training: TrainingConfigIn | dict | None) -> TrainingConfig:
    if training is None:
        return TrainingConfig()

    if isinstance(training, TrainingConfigIn):
        payload = training.model_dump(by_alias=False)
    else:
        payload = TrainingConfigIn.model_validate(training).model_dump(by_alias=False)

    return TrainingConfig.model_validate(payload)
