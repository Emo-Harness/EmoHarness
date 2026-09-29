"""Calibrating the proxy judge against the reference judge.

The Reasoning gate is only allowed to block a Builder candidate if the proxy
that scores it agrees with the official judge often enough. This module scores
one fixed sample with both, reports the agreement, and derives the margin the
gate should use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from emoharness.common.constants import POINTS_PER_UNIT, REASONING_GATE_FLOOR_POINTS
from emoharness.common.errors import ValidationError
from emoharness.common.io import write_json_atomic
from emoharness.common.logging import get_logger
from emoharness.evaluation.judge import (
    CalibrationReport,
    JudgeInput,
    StepScorer,
    calibrate,
    judge_input_for,
)
from emoharness.evaluation.schema import PredictionRecord
from emoharness.runner.context import RunContext
from emoharness.runner.items import by_id, examples_for
from emoharness.runner.judging import ScoreSpec, read_predictions, score_specs
from emoharness.runner.paths import guard_write
from emoharness.runner.perception import judge_frames_by_id

CALIBRATION_FILE: Final = 'proxy_judge.json'
PROXY_SALT: Final = '#proxy'

_LOGGER = get_logger(__name__)


def judge_inputs(
    predictions: Sequence[PredictionRecord], ctx: RunContext, item_set: str
) -> tuple[JudgeInput, ...]:
    """One judge call's worth of input per prediction, with its cached frames."""
    lookup = by_id(examples_for(item_set, ctx.paths))
    frames = judge_frames_by_id(lookup.values(), ctx.clip_cache(), ctx.preprocess)
    return tuple(
        judge_input_for(
            prediction.example_id,
            frames.get(prediction.example_id, ()),
            lookup[prediction.example_id].labels,
            prediction.response_text,
        )
        for prediction in predictions
        if prediction.example_id in lookup
    )


def _specs(
    inputs: Sequence[JudgeInput],
    predictions: Sequence[PredictionRecord],
    ctx: RunContext,
    item_set: str,
) -> Mapping[str, ScoreSpec]:
    """The scored sample, in the shape ``RunContext.step_score_table`` reads."""
    lookup = by_id(examples_for(item_set, ctx.paths))
    chosen = frozenset(item.example_id for item in inputs)
    return {
        example_id: spec
        for example_id, spec in score_specs(predictions, lookup).items()
        if example_id in chosen
    }


def _scorers(
    inputs: Sequence[JudgeInput],
    predictions: Sequence[PredictionRecord],
    ctx: RunContext,
    item_set: str,
) -> tuple[StepScorer, StepScorer]:
    specs = _specs(inputs, predictions, ctx, item_set)
    proxy = ctx.step_scorer(ctx.step_score_table(specs, PROXY_SALT))
    if ctx.fake:
        return proxy, ctx.step_scorer(ctx.step_score_table(specs))
    from emoharness.evaluation.judge import OpenAIStepScorer

    spec = ctx.proxy_judge().calibration
    reference = OpenAIStepScorer(
        model_id=spec.reference_model,
        api_key_env='OPENAI_API_KEY',
        endpoint=None,
        ledger_path=ctx.ledger,
        experiment=ctx.experiment,
        pricing=ctx.proxy_judge().pricing,
    )
    return proxy, reference


def gate_margin_points(report: CalibrationReport) -> float:
    """The Reasoning gate margin: the measured spread, never below the floor."""
    return max(REASONING_GATE_FLOOR_POINTS, report.reasoning_std * POINTS_PER_UNIT)


def calibrate_proxy_judge(
    row: str, item_set: str, ctx: RunContext, sample: int | None = None
) -> Mapping[str, object]:
    """Score one sample with both judges and report their agreement."""
    spec = ctx.proxy_judge()
    predictions = read_predictions(row, item_set, ctx.paths)
    size = sample if sample is not None else spec.calibration.sample_size
    chosen = predictions[: max(1, size)]
    inputs = judge_inputs(chosen, ctx, item_set)
    if not inputs:
        raise ValidationError(f'{row}/{item_set}: nothing to calibrate on')
    proxy_scorer, reference_scorer = _scorers(inputs, chosen, ctx, item_set)
    report = calibrate(
        tuple(proxy_scorer.score(item) for item in inputs),
        tuple(reference_scorer.score(item) for item in inputs),
    )
    _LOGGER.info(
        'proxy judge agreement: step %.3f, recognition %.3f over %d items',
        report.step_agreement,
        report.recognition_agreement,
        report.n,
    )
    return {
        'row': row,
        'item_set': item_set,
        'proxy_model': spec.model_id,
        'reference_model': spec.calibration.reference_model,
        'sample_size': len(inputs),
        'min_agreement': spec.calibration.min_agreement,
        'passes': report.recognition_agreement >= spec.calibration.min_agreement,
        'gate_margin_points': gate_margin_points(report),
        'report': report.model_dump(mode='json'),
    }


def write_calibration(payload: Mapping[str, object], ctx: RunContext) -> Path:
    """Write ``results/calibration/proxy_judge.json``."""
    path = guard_write(ctx.paths.calibration / CALIBRATION_FILE, ctx.force)
    write_json_atomic(path, dict(payload))
    return path
