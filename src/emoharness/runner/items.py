"""Loading manifests and resolving the item-set names rows are declared with.

An item set is one of ``full`` (every benchmark item), ``subset`` (the
pre-registered 1,000 ids), or ``external:<split>`` where the split is one of
``build``, ``val``, or ``ext_test``. Nothing else is accepted: a typo must fail
at the boundary rather than silently score a different set of items.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from emoharness.common.errors import ValidationError
from emoharness.data.manifest import Example, read_manifest
from emoharness.data.splits import read_subset
from emoharness.data.tasks import Split, Task
from emoharness.runner.paths import ProjectPaths

FULL: Final = 'full'
SUBSET: Final = 'subset'
EXTERNAL_PREFIX: Final = 'external:'
EXTERNAL_SPLITS: Final[Mapping[str, Split]] = {
    'build': Split.BUILD,
    'val': Split.VAL,
    'ext_test': Split.EXT_TEST,
}


def is_external(item_set: str) -> bool:
    """True when the item set names a construction split rather than the benchmark."""
    return item_set.startswith(EXTERNAL_PREFIX)


def external_split(item_set: str) -> Split:
    """The split an ``external:<split>`` item set names, or raise."""
    name = item_set[len(EXTERNAL_PREFIX) :]
    split = EXTERNAL_SPLITS.get(name)
    if split is None:
        raise ValidationError(
            f'unknown external split {name!r}; expected one of '
            f'{", ".join(sorted(EXTERNAL_SPLITS))}'
        )
    return split


def load_benchmark(paths: ProjectPaths) -> tuple[Example, ...]:
    """Every benchmark item from ``data/manifests/mme_emotion.jsonl``."""
    return _require(paths.benchmark_manifest, 'benchmark')


def load_external(paths: ProjectPaths) -> tuple[Example, ...]:
    """Every construction item from ``data/manifests/external.jsonl``."""
    return _require(paths.external_manifest, 'external')


def _require(manifest: Path, name: str) -> tuple[Example, ...]:
    if not manifest.is_file():
        raise ValidationError(
            f'{name} manifest {manifest} does not exist; run the prepare step first'
        )
    return read_manifest(manifest)


def select_items(
    examples: Iterable[Example], item_set: str, paths: ProjectPaths
) -> tuple[Example, ...]:
    """Restrict ``examples`` to one item set, preserving manifest order."""
    rows = tuple(examples)
    if item_set == FULL:
        return rows
    if item_set == SUBSET:
        ids = frozenset(read_subset(paths.subset).ids)
        return tuple(row for row in rows if row.example_id in ids)
    if is_external(item_set):
        split = external_split(item_set)
        return tuple(row for row in rows if row.split is split)
    raise ValidationError(
        f'unknown item set {item_set!r}; expected {FULL}, {SUBSET}, '
        f'or {EXTERNAL_PREFIX}<split>'
    )


def examples_for(item_set: str, paths: ProjectPaths) -> tuple[Example, ...]:
    """Load the manifest an item set belongs to and select from it."""
    source = load_external(paths) if is_external(item_set) else load_benchmark(paths)
    return select_items(source, item_set, paths)


def by_id(examples: Iterable[Example]) -> Mapping[str, Example]:
    """Examples keyed by id, for joining predictions back to their gold data."""
    return {example.example_id: example for example in examples}


def task_of(examples: Iterable[Example]) -> Mapping[str, Task]:
    """Benchmark task per example id; external items contribute nothing."""
    return {
        example.example_id: example.task
        for example in examples
        if example.task is not None
    }


def gold_of(examples: Iterable[Example]) -> Mapping[str, tuple[str, ...]]:
    """Gold labels per example id, verbatim as the manifest carries them."""
    return {example.example_id: example.labels for example in examples}
