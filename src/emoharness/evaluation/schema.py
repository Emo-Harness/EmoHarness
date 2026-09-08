"""Prediction and score records. One row = one method; one record = one item."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from emoharness.data.tasks import Task


class PredictionRecord(BaseModel):
    """What a method produced for one item, before any judging."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    example_id: str
    row: str = Field(min_length=1, description="ladder rung or method name, e.g. 'L5'")
    task: Task | None = None
    labels: tuple[str, ...] = Field(min_length=1)
    response_text: str = Field(min_length=1)
    target_calls: int = Field(ge=0)
    parse_failures: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_s: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class StepScores(BaseModel):
    """Per-step ratings as the official judge emits them, already parsed to fractions."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    example_id: str
    scores: tuple[float, ...] = Field(min_length=1)

    @property
    def recognition(self) -> float:
        return self.scores[-1]

    @property
    def reasoning(self) -> float:
        head = self.scores[:-1]
        return sum(head) / len(head) if head else 0.0

    @property
    def num_steps(self) -> int:
        return len(self.scores)


class MethodScores(BaseModel):
    """Aggregate scores for one row on one item set."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    row: str
    item_set: str
    count: int = Field(ge=0)
    recognition: float = Field(ge=0.0, le=1.0)
    reasoning: float = Field(ge=0.0, le=1.0)
    cot: float = Field(ge=0.0, le=1.0)
    avg_steps: float = Field(ge=0.0)
