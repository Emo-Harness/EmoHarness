"""Running one ladder row over one item set and writing its raw predictions.

Every row goes through the same Target wrapper and the same preprocessing cache,
so the adjacent differences the ladder claims to isolate really are the only
differences. Harness rows additionally persist their ``HarnessOutput``
records beside the predictions, because the Builder loop diagnoses failures from
the evidence and stage traces those records carry, not from the labels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from emoharness.baselines.best_of_n import run_best_of_n
from emoharness.baselines.direct import (
    AUDIO_BLOCK_HEADER,
    DIRECT_AUDIO_TAG,
    direct_request,
    direct_with_audio_request,
    handcrafted_request,
    run_single_call,
)
from emoharness.common.errors import ValidationError, assert_never
from emoharness.common.io import write_jsonl_atomic
from emoharness.common.logging import get_logger
from emoharness.common.parallel import map_ordered
from emoharness.data.manifest import Example
from emoharness.evaluation.ladder import MethodName, RowSpec
from emoharness.evaluation.schema import PredictionRecord
from emoharness.harness.loader import load_bundle
from emoharness.harness.pipeline import HarnessPipeline, PipelineConfig
from emoharness.harness.schema import HarnessOutput, PerceivedInputs
from emoharness.runner.context import GEMINI_PROVIDER, RunContext, TeacherMode
from emoharness.runner.paths import guard_write
from emoharness.runner.perception import perceived_inputs
from emoharness.target.base import TargetModel, TargetRequest, TargetResponse
from emoharness.target.pricing import Pricing, estimate_cost

CostFn = Callable[[TargetResponse], float]
Requests = tuple[tuple[Example, TargetRequest], ...]
RequestsFn = Callable[[Sequence[Example], 'RunContext'], Requests]
PerceivedRequestFn = Callable[[Example, PerceivedInputs], TargetRequest]
STAGE_ONE: Final = 1
FINAL_BUNDLE_DIR: Final = 'final'
NATIVE_VIDEO: Final[TeacherMode] = 'native_video'
MATCHED_PROTOCOL: Final[TeacherMode] = 'matched_protocol'

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class RowResult:
    """One row's predictions and, for harness rows, the records behind them."""

    row: str
    item_set: str
    predictions: tuple[PredictionRecord, ...]
    outputs: tuple[HarnessOutput, ...] = ()


def cost_fn(ctx: RunContext, pricing: Pricing | None = None) -> CostFn:
    """Nominal USD cost of one response at ``pricing`` (the student's by default)."""
    if pricing is None:
        pricing = ctx.pricing()

    def _cost(response: TargetResponse) -> float:
        return estimate_cost(pricing, response.input_tokens, response.output_tokens)

    return _cost


def pipeline_config(ctx: RunContext) -> PipelineConfig:
    """Decoding and budget settings for the harness rows."""
    if ctx.fake:
        return PipelineConfig()
    config = ctx.target_model_config()
    return PipelineConfig(
        call_hard_cap=config.call_budget.hard_cap,
        retries_per_stage=config.call_budget.retries_per_stage,
        max_tokens=config.decoding.max_tokens,
        temperature=config.decoding.temperature,
        seed=config.decoding.seed,
    )


def bundle_dir_for(stage: int, ctx: RunContext) -> Path:
    """Where a stage's frozen bundle lives: the repo scaffold, or a loop's output."""
    if ctx.bundle_override is not None:
        return ctx.bundle_override
    if stage == STAGE_ONE:
        return ctx.paths.bundle
    return ctx.paths.stage_dir(stage, ctx.replicate) / FINAL_BUNDLE_DIR


def inputs_for(
    examples: Iterable[Example], ctx: RunContext
) -> Mapping[str, PerceivedInputs]:
    """Perceived inputs for a set of examples, built once and reused by every row."""
    cache, preprocess = ctx.clip_cache(), ctx.clip_preprocessor()
    return {
        example.example_id: perceived_inputs(example, cache, ctx.preprocess, preprocess)
        for example in examples
    }


def _direct_requests(examples: Sequence[Example], ctx: RunContext) -> Requests:
    tokens = pipeline_config(ctx).max_tokens
    return tuple((e, direct_request(e, e.media_path, tokens)) for e in examples)


def _perceived_requests(
    examples: Sequence[Example], ctx: RunContext, build: PerceivedRequestFn
) -> Requests:
    perceived = inputs_for(examples, ctx)
    return tuple((e, build(e, perceived[e.example_id])) for e in examples)


def _audio_requests(examples: Sequence[Example], ctx: RunContext) -> Requests:
    return _perceived_requests(examples, ctx, direct_with_audio_request)


def _handcrafted_requests(examples: Sequence[Example], ctx: RunContext) -> Requests:
    return _perceived_requests(examples, ctx, handcrafted_request)


#: The single-call methods and the one request each of them builds. A method that
#: is not here is not something ``best_of_n`` can resample either.
SINGLE_CALL_REQUESTS: Final[Mapping[MethodName, RequestsFn]] = {
    'direct': _direct_requests,
    'direct_audio': _audio_requests,
    'handcrafted': _handcrafted_requests,
}


def _run_single_call(
    row: RowSpec, requests: Requests, ctx: RunContext
) -> tuple[PredictionRecord, ...]:
    target, cost = ctx.target(), cost_fn(ctx)
    return map_ordered(
        lambda pair: run_single_call(target, pair[1], pair[0], row.name, cost),
        requests,
    )


def _run_best_of_n(
    row: RowSpec, examples: Sequence[Example], ctx: RunContext
) -> tuple[PredictionRecord, ...]:
    base = ctx.ladder.rows.get(row.base or '')
    build = SINGLE_CALL_REQUESTS.get(base.method) if base is not None else None
    if build is None or row.n is None:
        raise ValidationError(f'{row.name}: best_of_n needs a known base row and n')
    target, cost = ctx.target(), cost_fn(ctx)
    requests = build(examples, ctx)
    n, temperature, top_p = row.n, row.temperature or 0.0, row.top_p or 1.0

    def one(pair: tuple[Example, TargetRequest]) -> PredictionRecord:
        example, request = pair
        return run_best_of_n(
            target, request, example, n, temperature, top_p, None, row.name, cost
        )

    return map_ordered(one, requests)


def prediction_from_output(
    output: HarnessOutput, example: Example, row: str, pricing: Pricing
) -> PredictionRecord:
    """One harness item's prediction, with its own call, token, and latency totals."""
    input_tokens = sum(trace.input_tokens for trace in output.traces)
    output_tokens = sum(trace.output_tokens for trace in output.traces)
    return PredictionRecord(
        example_id=output.example_id,
        row=row,
        task=example.task,
        labels=output.labels,
        response_text=output.response_text,
        target_calls=output.target_calls,
        parse_failures=output.parse_failures,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=sum(trace.latency_s for trace in output.traces),
        cost_usd=estimate_cost(pricing, input_tokens, output_tokens),
    )


def run_harness(
    examples: Sequence[Example],
    bundle_dir: Path,
    target: TargetModel,
    ctx: RunContext,
    perceived: Mapping[str, PerceivedInputs] | None = None,
) -> tuple[HarnessOutput, ...]:
    """Run one bundle over a set of items; the shared caches do the rest.

    ``perceived`` is accepted so a caller that evaluates many bundles over the
    same items builds the inputs once.
    """
    pipeline = HarnessPipeline(load_bundle(bundle_dir), target, pipeline_config(ctx))
    inputs = perceived if perceived is not None else inputs_for(examples, ctx)
    return map_ordered(
        lambda example: pipeline.run(example, inputs[example.example_id]), examples
    )


def _run_harness_row(
    row: RowSpec, examples: Sequence[Example], ctx: RunContext
) -> RowResult:
    stage = row.stage or STAGE_ONE
    bundle_dir = bundle_dir_for(stage, ctx)
    if not bundle_dir.is_dir():
        raise ValidationError(
            f'{row.name}: no harness bundle at {bundle_dir}; '
            'build the stage first or pass --bundle'
        )
    outputs = run_harness(examples, bundle_dir, ctx.target(), ctx)
    pricing = ctx.pricing()
    by_id = {example.example_id: example for example in examples}
    predictions = tuple(
        prediction_from_output(output, by_id[output.example_id], row.name, pricing)
        for output in outputs
    )
    return RowResult(row.name, row.item_set, predictions, outputs)


def _matched_request(example: Example, inputs: PerceivedInputs) -> TargetRequest:
    prompt = f'{example.prompt}\n\n{AUDIO_BLOCK_HEADER}{inputs.audio_text()}'
    return TargetRequest(
        prompt=prompt, tag=DIRECT_AUDIO_TAG, image_paths=inputs.frame_paths
    )


def _teacher_mode(row: RowSpec, ctx: RunContext) -> TeacherMode:
    """How the teacher sees a clip.

    Native video is a Gemini-only variant, because the OpenAI API takes images
    and not video; every other teacher runs the matched protocol, which sends
    the shared cache's sampled frames and the audio evidence as text.
    """
    if row.inputs == NATIVE_VIDEO and ctx.teacher_provider() == GEMINI_PROVIDER:
        return NATIVE_VIDEO
    return MATCHED_PROTOCOL


def _run_teacher_direct(
    row: RowSpec, examples: Sequence[Example], ctx: RunContext
) -> tuple[PredictionRecord, ...]:
    mode = _teacher_mode(row, ctx)
    target = ctx.teacher_target(mode)
    cost = cost_fn(ctx, ctx.teacher_pricing())
    if mode == NATIVE_VIDEO:
        pairs = tuple((e, direct_request(e, e.media_path)) for e in examples)
    else:
        perceived = inputs_for(examples, ctx)
        pairs = tuple(
            (e, _matched_request(e, perceived[e.example_id])) for e in examples
        )
    return map_ordered(
        lambda pair: run_single_call(target, pair[1], pair[0], row.name, cost), pairs
    )


def run_row(
    row: RowSpec,
    examples: Iterable[Example],
    ctx: RunContext,
    item_set: str | None = None,
) -> tuple[PredictionRecord, ...]:
    """Run one ladder row and write its artifacts under ``results/raw_predictions``."""
    rows = tuple(examples)
    if not rows:
        raise ValidationError(f'{row.name}: no items selected')
    result = _dispatch(row, rows, ctx)
    chosen = item_set or row.item_set
    _LOGGER.info('row %s: %d items on %s', row.name, len(rows), chosen)
    write_row(result, chosen, ctx)
    return result.predictions


def _dispatch(row: RowSpec, rows: Sequence[Example], ctx: RunContext) -> RowResult:
    """One row's result. Every method name is handled; the ladder admits no others."""
    method = row.method
    match method:
        case 'harness':
            return _run_harness_row(row, rows, ctx)
        case 'direct' | 'direct_audio' | 'handcrafted':
            requests = SINGLE_CALL_REQUESTS[method](rows, ctx)
            predictions = _run_single_call(row, requests, ctx)
        case 'best_of_n':
            predictions = _run_best_of_n(row, rows, ctx)
        case 'teacher_direct':
            predictions = _run_teacher_direct(row, rows, ctx)
        case _:
            assert_never(method)
    return RowResult(row.name, row.item_set, predictions)


def write_row(result: RowResult, item_set: str, ctx: RunContext) -> tuple[Path, ...]:
    """Write the predictions, and the harness outputs when there are any."""
    written = [
        guard_write(ctx.paths.predictions(result.row, item_set), ctx.force),
    ]
    write_jsonl_atomic(
        written[0], (record.model_dump(mode='json') for record in result.predictions)
    )
    if result.outputs:
        path = guard_write(ctx.paths.harness_outputs(result.row, item_set), ctx.force)
        write_jsonl_atomic(
            path, (output.model_dump(mode='json') for output in result.outputs)
        )
        written.append(path)
    return tuple(written)
