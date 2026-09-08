"""Calling the Target for one stage and turning its answer into a record.

Implementation detail of the harness; ``HarnessPipeline.run`` is the only caller.
Every Target answer is untrusted text: it is stripped of code fences, parsed as
one JSON object, and coerced into the frozen models. A stage that fails to parse
is retried once with a stricter reminder (ADR-0011 §4-5); the failure count ends
up in the stage trace so that parse-failure rates appear in the results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple

from emoharness.common.jsonish import extract_json_object
from emoharness.data.label_set import LabelSet
from emoharness.harness._controller import MODEL_WRITTEN_STEPS
from emoharness.harness.schema import (
    Hypothesis,
    HypothesisSet,
    StageTrace,
    VisualEvidence,
)
from emoharness.target.base import TargetModel, TargetRequest

STRICT_JSON_REMINDER: Final = (
    'Your previous answer could not be read as JSON. Answer again with one JSON '
    'object and nothing else: no prose, no code fence, no comments.'
)

_DEFAULT_COARSE: Final = 'unspecified'
_DEFAULT_CONFIDENCE: Final = 0.5


class StageOutcome(NamedTuple):
    """What one stage produced and what it cost."""

    payload: dict[str, Any] | None
    trace: StageTrace


def run_json_stage(
    stage: str, target: TargetModel, request: TargetRequest, max_calls: int
) -> StageOutcome:
    """Call the Target until its answer parses, the retries run out, or the budget does."""
    payload: dict[str, Any] | None = None
    attempt = request
    calls = failures = input_tokens = output_tokens = 0
    latency = 0.0
    while calls < max_calls and payload is None:
        response = target.generate(attempt)
        calls += 1
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        latency += response.latency_s
        payload = extract_json_object(response.text)
        if payload is None:
            failures += 1
            attempt = with_reminder(attempt)
    trace = StageTrace(
        stage=stage,
        calls=calls,
        parse_failures=failures,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency,
    )
    return StageOutcome(payload=payload, trace=trace)


def with_reminder(request: TargetRequest) -> TargetRequest:
    """The same request with the stricter format reminder appended."""
    return request.model_copy(
        update={'prompt': f'{request.prompt}\n\n{STRICT_JSON_REMINDER}'}
    )


def visual_from_payload(payload: Mapping[str, Any]) -> VisualEvidence:
    """A VisualEvidence record from whatever shape the Target returned."""
    return VisualEvidence(
        face=_string_tuple(payload.get('face')),
        body=_string_tuple(payload.get('body')),
        scene=_string_tuple(payload.get('scene')),
        temporal_change=_string_or_none(payload.get('temporal_change')),
    )


def hypothesis_set_from_payload(
    payload: Mapping[str, Any], label_set: LabelSet
) -> HypothesisSet | None:
    """A HypothesisSet, or None when no candidate label survived."""
    hypotheses = _hypotheses_from_payload(payload, label_set)
    if not hypotheses:
        return None
    coarse = _string_or_none(payload.get('coarse')) or _DEFAULT_COARSE
    return HypothesisSet(coarse=coarse, hypotheses=hypotheses)


def uniform_hypothesis_set(label_set: LabelSet) -> HypothesisSet:
    """The fallback when the Target names no candidate: every candidate, no support."""
    confidence = round(1.0 / len(label_set.labels), 6)
    return HypothesisSet(
        coarse=_DEFAULT_COARSE,
        hypotheses=tuple(
            Hypothesis(label=label, confidence=confidence) for label in label_set.labels
        ),
    )


def step_texts_from_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """The model-written step texts, with its justification folded into the last one."""
    steps = _string_tuple(payload.get('steps'))[:MODEL_WRITTEN_STEPS]
    justification = _string_or_none(payload.get('justification'))
    if justification and steps:
        return steps[:-1] + (f'{steps[-1]} {justification}',)
    return steps


def agreement_from_payload(payload: Mapping[str, Any]) -> bool | None:
    """The Target's own agreement flag, when it returned a boolean."""
    value = payload.get('modalities_agree')
    return value if isinstance(value, bool) else None


def _hypotheses_from_payload(
    payload: Mapping[str, Any], label_set: LabelSet
) -> tuple[Hypothesis, ...]:
    """Hypotheses whose labels are candidates, spelled as the prompt spells them."""
    rows = payload.get('hypotheses')
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return ()
    found = (_hypothesis_from_row(row, label_set) for row in rows)
    return tuple(item for item in found if item is not None)


def _hypothesis_from_row(row: Any, label_set: LabelSet) -> Hypothesis | None:
    if not isinstance(row, Mapping):
        return None
    raw_label = _string_or_none(row.get('label'))
    label = label_set.canonical(raw_label) if raw_label else None
    if label is None:
        return None
    return Hypothesis(
        label=label,
        support=_string_tuple(row.get('support')),
        contradiction=_string_tuple(row.get('contradiction')),
        confidence=_confidence(row.get('confidence')),
    )


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_CONFIDENCE
    return min(1.0, max(0.0, number))


def _string_tuple(value: Any) -> tuple[str, ...]:
    """A tuple of non-empty strings from a string, a sequence, or anything else."""
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _string_or_none(value: Any) -> str | None:
    """A non-empty string, or None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
