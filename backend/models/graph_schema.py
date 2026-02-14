from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class LayerType(str, Enum):
    INPUT = "Input"
    DENSE = "Dense"
    DROPOUT = "Dropout"
    FLATTEN = "Flatten"
    OUTPUT = "Output"


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: LayerType
    config: dict[str, Any] = Field(default_factory=dict)


class EdgeSpec(BaseModel):
    id: str
    source: str
    target: str


class TrainingConfigIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    dataset: str = "mnist"
    epochs: int = 20
    batch_size: int = Field(
        default=64,
        validation_alias=AliasChoices("batch_size", "batchSize"),
        serialization_alias="batch_size",
    )
    optimizer: str = "adam"
    learning_rate: float = Field(
        default=1e-3,
        validation_alias=AliasChoices("learning_rate", "learningRate"),
        serialization_alias="learning_rate",
    )
    loss: str = "cross_entropy"


class GraphSpec(BaseModel):
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    training: TrainingConfigIn | None = None
