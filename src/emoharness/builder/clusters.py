"""Deterministic failure clustering for the diagnosis packet (ADR-0006 §3).

Failures are grouped by the (gold, predicted) pair and an evidence-conflict flag.
Nothing here reads a task name, a split, or a benchmark id: the input is whatever
the harness produced on the construction set.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.errors import ValidationError
from emoharness.harness.schema import HarnessOutput

EMPTY_LABEL: Final = '(none)'
LABEL_JOINER: Final = ', '
KEY_JOINER: Final = ' | '
CONFLICT_TAG: Final = 'conflict'
AGREE_TAG: Final = 'agree'


class FailureRecord(BaseModel):
    """One item the harness got wrong, with the evidence the Builder may read."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    example_id: str = Field(min_length=1)
    gold: tuple[str, ...] = ()
    predicted: tuple[str, ...] = ()
    conflict: bool = False
    failure_stage: str = 'unknown'
    evidence_json: str = ''
    stage_outputs: Mapping[str, str] = Field(default_factory=dict)
    keyframe_paths: tuple[str, ...] = ()


class FailureCluster(BaseModel):
    """A group of failures that share a gold label, a prediction, and a flag."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    key: str = Field(min_length=1)
    gold: str
    predicted: str
    conflict: bool
    count: int = Field(ge=1)
    samples: tuple[FailureRecord, ...] = ()


def join_labels(labels: Iterable[str]) -> str:
    """Render a label tuple for a cluster key; empty means the harness emitted none."""
    joined = LABEL_JOINER.join(labels)
    return joined if joined else EMPTY_LABEL


def cluster_key(gold: Iterable[str], predicted: Iterable[str], conflict: bool) -> str:
    """The stable identifier a candidate patch targets."""
    tag = CONFLICT_TAG if conflict else AGREE_TAG
    return KEY_JOINER.join((join_labels(gold), join_labels(predicted), tag))


def cluster_failures(
    records: Iterable[FailureRecord], max_samples_per_cluster: int
) -> tuple[FailureCluster, ...]:
    """Group failures by (gold, predicted, conflict), largest cluster first."""
    if max_samples_per_cluster < 1:
        raise ValidationError('max_samples_per_cluster must be at least 1')
    grouped: dict[str, list[FailureRecord]] = {}
    for record in records:
        key = cluster_key(record.gold, record.predicted, record.conflict)
        grouped.setdefault(key, []).append(record)
    clusters = tuple(
        FailureCluster(
            key=key,
            gold=join_labels(rows[0].gold),
            predicted=join_labels(rows[0].predicted),
            conflict=rows[0].conflict,
            count=len(rows),
            samples=tuple(rows[:max_samples_per_cluster]),
        )
        for key, rows in grouped.items()
    )
    return tuple(sorted(clusters, key=lambda c: (-c.count, c.key)))


def confusion_matrix(
    records: Iterable[FailureRecord],
) -> Mapping[str, Mapping[str, int]]:
    """Gold label to predicted label to count, using the first label of each."""
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        gold = _first_label(record.gold)
        predicted = _first_label(record.predicted)
        row = counts.setdefault(gold, {})
        row[predicted] = row.get(predicted, 0) + 1
    return {gold: dict(sorted(row.items())) for gold, row in sorted(counts.items())}


def failure_records_from_outputs(
    outputs: Iterable[HarnessOutput],
    gold_by_id: Mapping[str, tuple[str, ...]],
    keyframes_by_id: Mapping[str, tuple[str, ...]],
    normalize: Callable[[str], str],
    *,
    correct: bool = False,
) -> tuple[FailureRecord, ...]:
    """Turn harness outputs into failure records, keeping only the wrong items.

    ``normalize`` is injected so that this module never decides what counts as the
    same label; the caller passes ``emoharness.data.label_set.normalize_label``.
    """
    records = []
    for output in outputs:
        gold = _gold_for(output.example_id, gold_by_id)
        solved = _normalized(gold, normalize) == _normalized(output.labels, normalize)
        if solved != correct:
            continue
        records.append(
            FailureRecord(
                example_id=output.example_id,
                gold=gold,
                predicted=output.labels,
                conflict=not output.verification.modalities_agree,
                failure_stage='correct'
                if solved
                else _failure_stage(output, gold, normalize),
                evidence_json=compact_json(output.evidence.model_dump(mode='json')),
                stage_outputs=_stage_outputs(output),
                keyframe_paths=keyframes_by_id.get(output.example_id, ()),
            )
        )
    return tuple(records)


def compact_json(obj: object) -> str:
    """Canonical, whitespace-free JSON so packets are byte-stable across runs."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _stage_outputs(output: HarnessOutput) -> dict[str, str]:
    return {
        'hypotheses': compact_json(output.hypotheses.model_dump(mode='json')),
        'verification': compact_json(output.verification.model_dump(mode='json')),
        'emitted_labels': compact_json(output.labels),
        'parse_failures': compact_json(
            {t.stage: t.parse_failures for t in output.traces}
        ),
    }


def _gold_for(
    example_id: str, gold_by_id: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    if example_id not in gold_by_id:
        raise ValidationError(f'no gold labels for example {example_id}')
    return gold_by_id[example_id]


def _normalized(
    labels: Iterable[str], normalize: Callable[[str], str]
) -> frozenset[str]:
    return frozenset(normalize(label) for label in labels)


def _first_label(labels: tuple[str, ...]) -> str:
    return labels[0] if labels else EMPTY_LABEL


def _failure_stage(
    output: HarnessOutput, gold: tuple[str, ...], normalize: Callable[[str], str]
) -> str:
    """Observable trace symptoms, not a claim about unobserved perception errors."""
    if output.parse_failures:
        return 'parse_failure'
    expected = _normalized(gold, normalize)
    proposed = _normalized((h.label for h in output.hypotheses.hypotheses), normalize)
    verified = _normalized((h.label for h in output.verification.ranked), normalize)
    if not expected <= proposed:
        return 'missing_hypothesis'
    if not expected <= verified:
        return 'verification_removed_hypothesis'
    known = output.evidence.decision_ids()
    supported = _normalized(
        (
            h.label
            for h in output.verification.ranked
            if len(set(h.support) & known) > len(set(h.contradiction) & known)
        ),
        normalize,
    )
    if not expected <= supported:
        return 'unsupported_hypothesis'
    return 'ranking_or_cardinality'
