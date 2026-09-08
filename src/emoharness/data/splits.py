"""External split assignment and the pre-registered benchmark subset (ADR-0003, ADR-0013)."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.errors import ValidationError
from emoharness.common.hashing import stable_hash
from emoharness.common.io import read_json, write_json_atomic
from emoharness.data.manifest import Example
from emoharness.data.tasks import Kind, Split, Task

_TARGET_SPLITS: Final[tuple[Split, ...]] = (Split.BUILD, Split.VAL, Split.EXT_TEST)


class SubsetManifest(BaseModel):
    """The pre-registered subset id list, its draw parameters, and its hash."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    ids: tuple[str, ...] = Field(min_length=1)
    seed: int
    size_per_task: int = Field(ge=1)
    subset_hash: str = Field(min_length=1)


def assign_external_splits(
    examples: tuple[Example, ...], per_kind: Mapping[Split, int], seed: int
) -> tuple[Example, ...]:
    """Assign BUILD/VAL/EXT_TEST to external examples; drop what does not fit.

    Examples are grouped by ``source_id`` within each ``kind`` so that no
    source ever crosses a split boundary. Groups are visited in an order
    shuffled deterministically from ``seed`` (after a canonical sort, so the
    result does not depend on the input's own ordering), and each group is
    assigned, whole, to whichever split's running label distribution is
    currently furthest below its target share for that group's labels. Each
    kind is filled independently against the same ``per_kind`` target counts
    (ADR-0003: 150/150/100 per kind). Examples of a kind not present in
    ``per_kind``, or left over once every split for their kind is full, are
    dropped rather than returned with a stale split.
    """
    rng = random.Random(seed)  # noqa: S311 -- reproducible assignment, not crypto
    assigned: list[Example] = []
    for kind in Kind:
        kind_examples = tuple(e for e in examples if e.kind is kind)
        if kind_examples:
            assigned.extend(_assign_kind_groups(kind_examples, per_kind, rng))
    return tuple(assigned)


def _assign_kind_groups(
    kind_examples: tuple[Example, ...],
    per_kind: Mapping[Split, int],
    rng: random.Random,
) -> tuple[Example, ...]:
    groups = _group_by_source(kind_examples)
    shuffled_ids = sorted(groups)
    rng.shuffle(shuffled_ids)
    overall = Counter(label for e in kind_examples for label in e.labels)
    targets = {split: per_kind.get(split, 0) for split in _TARGET_SPLITS}
    split_counts = dict.fromkeys(_TARGET_SPLITS, 0)
    split_label_counts: dict[Split, Counter[str]] = {
        s: Counter() for s in _TARGET_SPLITS
    }
    result: list[Example] = []
    for source_id in shuffled_ids:
        group = groups[source_id]
        split = _best_split(group, overall, targets, split_counts, split_label_counts)
        if split is None:
            continue
        result.extend(_with_split(e, split) for e in group)
        split_counts[split] += len(group)
        split_label_counts[split].update(label for e in group for label in e.labels)
    return tuple(result)


def _group_by_source(examples: tuple[Example, ...]) -> dict[str, tuple[Example, ...]]:
    groups: dict[str, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.source_id, []).append(example)
    return {source_id: tuple(items) for source_id, items in groups.items()}


def _best_split(
    group: tuple[Example, ...],
    overall: Counter[str],
    targets: Mapping[Split, int],
    split_counts: Mapping[Split, int],
    split_label_counts: Mapping[Split, Counter[str]],
) -> Split | None:
    candidates = [s for s in _TARGET_SPLITS if split_counts[s] < targets[s]]
    if not candidates:
        return None
    total = sum(overall.values()) or 1
    group_labels = Counter(label for e in group for label in e.labels)

    def deficit(split: Split) -> float:
        return sum(
            count
            * (
                targets[split] * overall[label] / total
                - split_label_counts[split][label]
            )
            for label, count in group_labels.items()
        )

    return max(candidates, key=deficit)


def _with_split(example: Example, split: Split) -> Example:
    return example.model_copy(update={'split': split})


def draw_subset(
    examples: tuple[Example, ...], size_per_task: int, seed: int
) -> tuple[str, ...]:
    """Proportional-by-label stratified sample of benchmark ids, per task.

    Multi-label items are stratified by their first gold label. Within each
    task, target counts per label bucket are rounded by largest remainder so
    the task contributes exactly ``min(size_per_task, task_size)`` ids; the
    returned tuple is sorted for a stable, order-independent result.
    """
    rng = random.Random(seed)  # noqa: S311 -- reproducible sampling, not crypto
    benchmark = tuple(e for e in examples if e.task is not None)
    tasks = sorted({cast(Task, e.task) for e in benchmark}, key=lambda t: t.value)
    ids: list[str] = []
    for task in tasks:
        task_examples = tuple(e for e in benchmark if e.task is task)
        ids.extend(_sample_task(task_examples, size_per_task, rng))
    return tuple(sorted(ids))


def _sample_task(
    task_examples: tuple[Example, ...], size_per_task: int, rng: random.Random
) -> tuple[str, ...]:
    buckets = _group_by_first_label(task_examples)
    target = min(size_per_task, len(task_examples))
    bucket_sizes = {label: len(items) for label, items in buckets.items()}
    allocation = _largest_remainder(bucket_sizes, target)
    sampled: list[str] = []
    for label in sorted(buckets):
        pool = sorted(e.example_id for e in buckets[label])
        sampled.extend(rng.sample(pool, allocation[label]))
    return tuple(sampled)


def _group_by_first_label(
    examples: tuple[Example, ...],
) -> dict[str, tuple[Example, ...]]:
    groups: dict[str, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.labels[0], []).append(example)
    return {label: tuple(items) for label, items in groups.items()}


def _largest_remainder(counts: Mapping[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values()) or 1
    shares = {label: target * count / total for label, count in counts.items()}
    floors = {label: math.floor(share) for label, share in shares.items()}
    leftover = target - sum(floors.values())
    ranked = sorted(counts, key=lambda label: (-(shares[label] - floors[label]), label))
    for label in ranked[:leftover]:
        floors[label] += 1
    return floors


def subset_hash(ids: tuple[str, ...]) -> str:
    """Order-independent hash of a subset id list."""
    return stable_hash(sorted(ids))


def write_subset(
    path: Path, ids: tuple[str, ...], seed: int, size_per_task: int
) -> None:
    """Write the subset id list, its draw parameters, and its hash atomically."""
    manifest = SubsetManifest(
        ids=ids, seed=seed, size_per_task=size_per_task, subset_hash=subset_hash(ids)
    )
    write_json_atomic(path, manifest.model_dump(mode='json'))


def read_subset(path: Path) -> SubsetManifest:
    """Read a subset manifest, verifying its hash still matches its id list."""
    manifest = SubsetManifest.model_validate(read_json(path))
    if manifest.subset_hash != subset_hash(manifest.ids):
        raise ValidationError(f'{path}: subset hash does not match its id list')
    return manifest
