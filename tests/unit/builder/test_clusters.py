from __future__ import annotations

from collections.abc import Callable

import pytest

from emoharness.builder.clusters import (
    EMPTY_LABEL,
    FailureRecord,
    cluster_failures,
    cluster_key,
    confusion_matrix,
    failure_records_from_outputs,
)
from emoharness.common.errors import ValidationError
from emoharness.data.label_set import normalize_label
from emoharness.harness.schema import HarnessOutput

Record = Callable[..., FailureRecord]
Output = Callable[..., HarnessOutput]


def _records(make_record: Record) -> tuple[FailureRecord, ...]:
    return (
        make_record('a1', ('anger',), ('sadness',)),
        make_record('a2', ('anger',), ('sadness',)),
        make_record('a3', ('anger',), ('sadness',)),
        make_record('b1', ('happiness',), ('neutral',)),
        make_record('b2', ('happiness',), ('neutral',)),
        make_record('c1', ('anger',), ('sadness',), conflict=True),
    )


def test_clusters_are_ordered_by_count_then_key(make_record: Record) -> None:
    clusters = cluster_failures(_records(make_record), 5)
    assert [c.count for c in clusters] == [3, 2, 1]
    assert clusters[0].key == cluster_key(('anger',), ('sadness',), False)
    assert clusters[0].gold == 'anger'
    assert clusters[0].predicted == 'sadness'
    assert clusters[2].conflict is True


def test_clustering_is_independent_of_input_order(make_record: Record) -> None:
    rows = _records(make_record)
    forward = cluster_failures(rows, 5)
    backward = cluster_failures(tuple(reversed(rows)), 5)
    assert [c.key for c in forward] == [c.key for c in backward]
    assert [c.count for c in forward] == [c.count for c in backward]


def test_samples_are_capped_per_cluster(make_record: Record) -> None:
    clusters = cluster_failures(_records(make_record), 2)
    assert [len(c.samples) for c in clusters] == [2, 2, 1]
    assert [s.example_id for s in clusters[0].samples] == ['a1', 'a2']


def test_ties_break_on_key(make_record: Record) -> None:
    rows = (
        make_record('z', ('sadness',), ('anger',)),
        make_record('a', ('anger',), ('happiness',)),
    )
    keys = [c.key for c in cluster_failures(rows, 5)]
    assert keys == sorted(keys)


def test_empty_labels_render_as_a_placeholder(make_record: Record) -> None:
    clusters = cluster_failures((make_record('x', ('anger',), ()),), 5)
    assert clusters[0].predicted == EMPTY_LABEL


def test_cluster_failures_rejects_a_zero_sample_cap(make_record: Record) -> None:
    with pytest.raises(ValidationError, match='at least 1'):
        cluster_failures((make_record('x'),), 0)


def test_confusion_matrix_counts_first_labels(make_record: Record) -> None:
    rows = (
        *_records(make_record),
        make_record('d1', ('anger', 'disgust'), ('neutral', 'sadness')),
    )
    matrix = confusion_matrix(rows)
    assert matrix == {
        'anger': {'neutral': 1, 'sadness': 4},
        'happiness': {'neutral': 2},
    }
    assert list(matrix) == sorted(matrix)


def test_confusion_matrix_of_nothing_is_empty() -> None:
    assert confusion_matrix(()) == {}


def test_failure_records_keep_only_wrong_items(make_output: Output) -> None:
    outputs = (
        make_output('good', ('anger',)),
        make_output('bad', ('sadness',), agree=False),
    )
    records = failure_records_from_outputs(
        outputs,
        {'good': ('anger',), 'bad': ('anger',)},
        {'bad': ('f0.jpg', 'f1.jpg')},
        normalize_label,
    )
    assert [r.example_id for r in records] == ['bad']
    assert records[0].gold == ('anger',)
    assert records[0].predicted == ('sadness',)
    assert records[0].conflict is True
    assert records[0].keyframe_paths == ('f0.jpg', 'f1.jpg')
    assert sorted(records[0].stage_outputs) == [
        'emitted_labels',
        'hypotheses',
        'parse_failures',
        'verification',
    ]
    assert 'negative' in records[0].stage_outputs['hypotheses']
    assert 'brow lowered' in records[0].evidence_json


def test_failure_records_use_the_injected_normalizer(make_output: Output) -> None:
    outputs = (make_output('syn', ('happy',)),)
    records = failure_records_from_outputs(
        outputs, {'syn': ('happiness',)}, {}, normalize_label
    )
    assert records == ()


def test_failure_records_ignore_label_order(make_output: Output) -> None:
    outputs = (make_output('multi', ('anger', 'sadness')),)
    records = failure_records_from_outputs(
        outputs, {'multi': ('sadness', 'anger')}, {}, normalize_label
    )
    assert records == ()


def test_failure_records_require_gold_labels(make_output: Output) -> None:
    with pytest.raises(ValidationError, match='no gold labels'):
        failure_records_from_outputs((make_output('x'),), {}, {}, normalize_label)
