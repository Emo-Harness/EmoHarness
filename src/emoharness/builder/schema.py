"""Builder proposal contract (ADR-0006). The Builder returns JSON matching these models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from emoharness.common.constants import MAX_CANDIDATES_PER_ROUND


class PatchFile(BaseModel):
    """Full-file replacement. ``path`` is relative to the harness bundle root."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    path: str = Field(min_length=1)
    content: str


class CandidatePatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str = Field(min_length=1, max_length=64)
    targeted_cluster: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    files: tuple[PatchFile, ...] = Field(min_length=1)


class BuilderProposal(BaseModel):
    """One round's response."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    candidates: tuple[CandidatePatch, ...] = Field(
        default=(), max_length=MAX_CANDIDATES_PER_ROUND
    )
    summary: str = ''
    continue_search: bool = Field(
        default=True,
        description='whether the Builder has further directions worth a round; '
        'honoured only when BuilderConfig.autonomous_stop is set (ADR-0006 §10)',
    )
    stop_reason: str = Field(
        default='', description='why the Builder would stop, in its own words'
    )

    @model_validator(mode='after')
    def _empty_only_when_stopping(self) -> BuilderProposal:
        if not self.candidates and self.continue_search:
            raise ValueError('an empty proposal must set continue_search: false')
        return self


class SelectionMetrics(BaseModel):
    """Aggregate selection-dev diagnostics. Never carries item IDs or labels."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    macro: float = Field(ge=0.0, le=1.0)
    pooled: float = Field(ge=0.0, le=1.0)
    per_task: dict[str, float]
    scaffold_per_task: dict[str, float]
    incumbent_per_task: dict[str, float] = Field(default_factory=dict)
    per_task_delta: dict[str, float] = Field(default_factory=dict)
    macro_delta: float = 0.0
    pooled_delta: float = 0.0
    improved_count: int = Field(ge=0)
    regressed_count: int = Field(ge=0)
    comparison: str = 'current_incumbent'


class SelectionFeedback(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    metrics: SelectionMetrics
    mean_calls: float
    p95_latency_s: float
    accepted: bool
    gate_passed: bool
    gate_reasons: tuple[str, ...] = ()


class CandidateScores(BaseModel):
    """Everything the gates need about one evaluated candidate."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str
    recognition: float = Field(ge=0.0, le=1.0, description='Rec-S on D_val')
    reasoning: float | None = Field(
        default=None, ge=0.0, le=1.0, description='proxy Rea-S on the gate sample'
    )
    regression_rate: float = Field(ge=0.0, le=1.0)
    mean_calls: float = Field(ge=0.0)
    p95_latency_s: float = Field(ge=0.0)
    selection: SelectionMetrics | None = None


class BuildAssessment(BaseModel):
    """Candidate effects on D_build only, safe to return to the Builder."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    before: float = Field(ge=0.0, le=1.0)
    after: float = Field(ge=0.0, le=1.0)
    improved_count: int = Field(ge=0)
    regressed_count: int = Field(ge=0)
    improved_examples: tuple[str, ...] = ()
    regressed_examples: tuple[str, ...] = ()


class CandidateFeedback(BaseModel):
    """Search memory without validation scores, examples, or gate decisions."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    round_index: int = Field(ge=0)
    name: str
    patch_hash: str
    targeted_cluster: str
    rationale: str
    files: tuple[str, ...]
    status: str
    review_issues: tuple[str, ...] = ()
    build: BuildAssessment | None = None
    selection: SelectionFeedback | None = None
