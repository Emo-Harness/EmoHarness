from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emoharness.common.errors import ValidationError
from emoharness.data.manifest import Example
from emoharness.data.splits import (
    SubsetManifest,
    assign_external_splits,
    draw_subset,
    read_subset,
    subset_hash,
    write_subset,
)
from emoharness.data.tasks import Kind, Split, Task

_LABEL_POOL = ('positive', 'negative', 'neutral')
_KIND_POOL = tuple(Kind)
_TASK_POOL = tuple(Task)
_PER_KIND = {Split.BUILD: 3, Split.VAL: 2, Split.EXT_TEST: 1}


def _external_example(source_id: str, kind: Kind, label: str, index: int) -> Example:
    return Example(
        example_id=f'ex_{index}',
        source_id=source_id,
        origin='synthetic',
        kind=kind,
        split=Split.BUILD,
        media_path=f'media/{index}.mp4',
        prompt='dummy prompt',
        label_set=_LABEL_POOL,
        labels=(label,),
    )


def _benchmark_example(task: Task, label: str, index: int) -> Example:
    return Example(
        example_id=f'mme_{task.value.lower()}_{index:06d}',
        source_id=f'src_{index}',
        origin='mme_emotion',
        task=task,
        split=Split.TEST,
        media_path=f'media/b{index}.mp4',
        prompt='dummy prompt',
        label_set=_LABEL_POOL,
        labels=(label,),
    )


@st.composite
def _external_examples(draw: st.DrawFn) -> tuple[Example, ...]:
    source_ids = draw(
        st.lists(
            st.text(alphabet='abcdef', min_size=1, max_size=4),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    examples: list[Example] = []
    index = 0
    for source_id in source_ids:
        kind = draw(st.sampled_from(_KIND_POOL))
        count = draw(st.integers(min_value=1, max_value=4))
        for _ in range(count):
            label = draw(st.sampled_from(_LABEL_POOL))
            examples.append(_external_example(source_id, kind, label, index))
            index += 1
    return tuple(examples)


@st.composite
def _benchmark_examples(draw: st.DrawFn) -> tuple[Example, ...]:
    tasks = draw(
        st.lists(st.sampled_from(_TASK_POOL), min_size=1, max_size=3, unique=True)
    )
    examples: list[Example] = []
    index = 0
    for task in tasks:
        count = draw(st.integers(min_value=0, max_value=12))
        for _ in range(count):
            label = draw(st.sampled_from(_LABEL_POOL))
            examples.append(_benchmark_example(task, label, index))
            index += 1
    return tuple(examples)


@given(_external_examples())
def test_assign_external_splits_never_crosses_source_ids(
    examples: tuple[Example, ...],
) -> None:
    assigned = assign_external_splits(examples, _PER_KIND, seed=7)
    by_source: dict[str, set[Split]] = {}
    for example in assigned:
        by_source.setdefault(example.source_id, set()).add(example.split)
    assert all(len(splits) == 1 for splits in by_source.values())


@given(_external_examples())
def test_assign_external_splits_is_deterministic(examples: tuple[Example, ...]) -> None:
    first = assign_external_splits(examples, _PER_KIND, seed=11)
    second = assign_external_splits(examples, _PER_KIND, seed=11)
    assert first == second


@given(_external_examples())
def test_assign_external_splits_only_uses_requested_splits(
    examples: tuple[Example, ...],
) -> None:
    assigned = assign_external_splits(examples, _PER_KIND, seed=3)
    assert all(e.split in _PER_KIND for e in assigned)
    assert all(e.example_id in {x.example_id for x in examples} for e in assigned)


def test_assign_external_splits_fills_only_requested_splits() -> None:
    examples = (
        _external_example('s1', Kind.LAB_ER, 'positive', 0),
        _external_example('s2', Kind.LAB_ER, 'negative', 1),
    )
    assigned = assign_external_splits(examples, {Split.BUILD: 1}, seed=1)
    assert len(assigned) == 1
    assert assigned[0].split is Split.BUILD


def test_assign_external_splits_drops_examples_once_every_split_is_full() -> None:
    examples = tuple(
        _external_example(f's{i}', Kind.LAB_ER, 'positive', i) for i in range(5)
    )
    assigned = assign_external_splits(examples, {Split.BUILD: 2}, seed=1)
    assert len(assigned) == 2


def test_split_label_proportions_roughly_match_overall() -> None:
    examples = tuple(
        _external_example(
            f'src_{i}', Kind.SENTIMENT, 'positive' if i < 90 else 'negative', i
        )
        for i in range(100)
    )
    per_kind = {Split.BUILD: 50, Split.VAL: 30, Split.EXT_TEST: 20}
    assigned = assign_external_splits(examples, per_kind, seed=99)
    for split, target in per_kind.items():
        split_examples = [e for e in assigned if e.split is split]
        assert len(split_examples) == target
        positive_share = (
            sum(1 for e in split_examples if e.labels == ('positive',)) / target
        )
        assert abs(positive_share - 0.9) < 0.15


@given(_benchmark_examples(), st.integers(min_value=0, max_value=10))
def test_draw_subset_counts_match_per_task(
    examples: tuple[Example, ...], size_per_task: int
) -> None:
    ids = draw_subset(examples, size_per_task, seed=5)
    id_to_task = {e.example_id: e.task for e in examples}
    task_totals: Counter[Task] = Counter(e.task for e in examples if e.task is not None)
    sampled_counts: Counter[Task | None] = Counter(id_to_task[i] for i in ids)
    for task, total in task_totals.items():
        assert sampled_counts.get(task, 0) == min(size_per_task, total)
    assert len(set(ids)) == len(ids)
    assert set(ids) <= set(id_to_task)


@given(_benchmark_examples())
def test_draw_subset_is_deterministic(examples: tuple[Example, ...]) -> None:
    first = draw_subset(examples, size_per_task=5, seed=42)
    second = draw_subset(examples, size_per_task=5, seed=42)
    assert first == second
    assert first == tuple(sorted(first))


def test_subset_hash_is_order_independent() -> None:
    assert subset_hash(('a', 'b', 'c')) == subset_hash(('c', 'b', 'a'))


def test_write_then_read_subset_round_trips(tmp_path: Path) -> None:
    path = tmp_path / 'subset.json'
    ids = ('mme_sa_000001', 'mme_sa_000002')
    write_subset(path, ids, seed=20260907, size_per_task=125)
    manifest = read_subset(path)
    assert manifest.ids == ids
    assert manifest.seed == 20260907
    assert manifest.size_per_task == 125
    assert manifest.subset_hash == subset_hash(ids)


def test_read_subset_rejects_a_tampered_file(tmp_path: Path) -> None:
    path = tmp_path / 'subset.json'
    write_subset(path, ('a', 'b'), seed=1, size_per_task=1)
    tampered = SubsetManifest.model_validate_json(path.read_text()).model_copy(
        update={'ids': ('a', 'b', 'c')}
    )
    path.write_text(json.dumps(tampered.model_dump(mode='json')), encoding='utf-8')
    with pytest.raises(ValidationError, match='subset hash'):
        read_subset(path)
