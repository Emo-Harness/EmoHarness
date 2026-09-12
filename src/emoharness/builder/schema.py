"""Builder proposal contract (ADR-0006). The Builder returns JSON matching these models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

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
        min_length=1, max_length=MAX_CANDIDATES_PER_ROUND
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
