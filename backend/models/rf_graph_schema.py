from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class RFNodeType(str, Enum):
    INPUT = "RFInput"
    FLATTEN = "RFFlatten"
    RANDOM_FOREST = "RandomForestClassifier"
    OUTPUT = "RFOutput"


class RFNodeSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: RFNodeType
    config: dict[str, Any] = Field(default_factory=dict)


class RFEdgeSpec(BaseModel):
    id: str
    source: str
    target: str


class RFTrainingConfigIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    dataset: str = "iris"
    test_size: float = Field(
        default=0.2,
        validation_alias=AliasChoices("test_size", "testSize"),
        serialization_alias="test_size",
    )
    random_state: int = Field(
        default=42,
        validation_alias=AliasChoices("random_state", "randomState"),
        serialization_alias="random_state",
    )
    stratify: bool = True
    log_every_trees: int = Field(
        default=10,
        validation_alias=AliasChoices("log_every_trees", "logEveryTrees"),
        serialization_alias="log_every_trees",
    )


class RFGraphSpec(BaseModel):
    nodes: list[RFNodeSpec]
    edges: list[RFEdgeSpec]
    training: RFTrainingConfigIn | None = None
