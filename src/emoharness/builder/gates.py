"""Promotion gates. A candidate is promoted or it is not.

Recognition is the objective; reasoning, the regression set, calls, and latency are
non-regression gates. Every rejection carries a reason so provenance can explain it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from emoharness.builder.schema import CandidateScores
from emoharness.common.constants import (
    CALL_BUDGET_EXPECTED,
    POINTS_PER_UNIT,
    REASONING_GATE_FLOOR_POINTS,
    REGRESSION_RATE_LIMIT,
)


class GateConfig(BaseModel):
    """Thresholds one round applies. ``max_p95_latency_s`` None means unbounded."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    selection_protocol: Literal['current', 'macro_feedback', 'context_quality'] = (
        'current'
    )
    max_task_drop: float = Field(default=0.03, ge=0.0, le=1.0)
    regression_rate_limit: float = Field(default=REGRESSION_RATE_LIMIT, ge=0.0, le=1.0)
    max_accuracy_drop: float = Field(default=0.01, ge=0.0, le=1.0)
    reasoning_margin_points: float = Field(default=REASONING_GATE_FLOOR_POINTS, ge=0.0)
    min_recognition_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    max_mean_calls: float = Field(default=float(CALL_BUDGET_EXPECTED), ge=0.0)
    max_p95_latency_s: float | None = Field(default=None, gt=0.0)


class GateResult(BaseModel):
    """Why one candidate was promoted or rejected."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str
    passed: bool
    reasons: tuple[str, ...] = ()


def evaluate_gates(
    candidate: CandidateScores, incumbent: CandidateScores, cfg: GateConfig
) -> GateResult:
    """Apply every gate to one candidate against the current incumbent."""
    reasons = tuple(
        reason
        for reason in (
            _recognition_reason(candidate, incumbent, cfg),
            _reasoning_reason(candidate, incumbent, cfg),
            _regression_reason(candidate, cfg),
            _accuracy_reason(candidate, incumbent, cfg),
            *_budget_reasons(candidate, cfg),
        )
        if reason is not None
    )
    return GateResult(name=candidate.name, passed=not reasons, reasons=reasons)


def _recognition_reason(
    candidate: CandidateScores, incumbent: CandidateScores, cfg: GateConfig
) -> str | None:
    gain = candidate.recognition - incumbent.recognition
    if gain > cfg.min_recognition_gain:
        return None
    return (
        f'recognition {candidate.recognition:.4f} does not exceed incumbent '
        f'{incumbent.recognition:.4f} by more than {cfg.min_recognition_gain:.4f}'
    )


def _reasoning_reason(
    candidate: CandidateScores, incumbent: CandidateScores, cfg: GateConfig
) -> str | None:
    if candidate.reasoning is None or incumbent.reasoning is None:
        return None
    floor = incumbent.reasoning - cfg.reasoning_margin_points / POINTS_PER_UNIT
    if candidate.reasoning >= floor:
        return None
    return (
        f'reasoning {candidate.reasoning:.4f} falls below the incumbent floor '
        f'{floor:.4f} ({cfg.reasoning_margin_points:.1f} points)'
    )


def _regression_reason(candidate: CandidateScores, cfg: GateConfig) -> str | None:
    if cfg.selection_protocol == 'macro_feedback':
        metrics = candidate.selection
        if metrics is None or set(metrics.per_task) != {'ER_Lab', 'ER_Wild', 'SA'}:
            return 'selection-dev requires all three seen-task aggregates'
        if set(metrics.scaffold_per_task) != set(metrics.per_task):
            return 'selection-dev missing frozen scaffold task reference'
        drops = tuple(
            f'{task} drop {metrics.scaffold_per_task[task] - score:.4f}'
            for task, score in sorted(metrics.per_task.items())
            if metrics.scaffold_per_task[task] - score > cfg.max_task_drop + 1e-12
        )
        return (
            f'task harm exceeds scaffold-relative limit {cfg.max_task_drop:.4f}: '
            + ', '.join(drops)
            if drops
            else None
        )
    if candidate.regression_rate <= cfg.regression_rate_limit:
        return None
    return (
        f'regression rate {candidate.regression_rate:.4f} exceeds '
        f'{cfg.regression_rate_limit:.4f}'
    )


def _accuracy_reason(
    candidate: CandidateScores, incumbent: CandidateScores, cfg: GateConfig
) -> str | None:
    """context_quality: the objective may not rise on plain accuracy's back.

    Moving items off an absorbing label necessarily breaks some it used to get
    right, so the regression limit has to be looser under this protocol; this
    keeps the overall hit rate from paying for it.
    """
    if cfg.selection_protocol != 'context_quality':
        return None
    if candidate.quality is None or incumbent.quality is None:
        return 'context_quality requires quality metrics for both bundles'
    floor = incumbent.quality.recognition - cfg.max_accuracy_drop
    if candidate.quality.recognition >= floor - 1e-12:
        return None
    return (
        f'plain accuracy {candidate.quality.recognition:.4f} falls below the '
        f'incumbent floor {floor:.4f}'
    )


def _budget_reasons(
    candidate: CandidateScores, cfg: GateConfig
) -> tuple[str | None, ...]:
    calls = (
        f'mean calls {candidate.mean_calls:.3f} exceeds {cfg.max_mean_calls:.3f}'
        if candidate.mean_calls > cfg.max_mean_calls
        else None
    )
    limit = cfg.max_p95_latency_s
    latency = (
        f'p95 latency {candidate.p95_latency_s:.3f}s exceeds {limit:.3f}s'
        if limit is not None and candidate.p95_latency_s > limit
        else None
    )
    return (calls, latency)


def select_candidate(
    candidates: Iterable[CandidateScores], incumbent: CandidateScores, cfg: GateConfig
) -> tuple[CandidateScores | None, tuple[GateResult, ...]]:
    """Pick the best promotable candidate: recognition, then fewer calls, then name."""
    rows = tuple(candidates)
    results = tuple(evaluate_gates(row, incumbent, cfg) for row in rows)
    passing = tuple(
        row for row, result in zip(rows, results, strict=True) if result.passed
    )
    if not passing:
        return None, results
    best = min(passing, key=lambda row: (-row.recognition, row.mean_calls, row.name))
    return best, results
