"""Isolate construction splits by connected media and recorded source identities."""

from __future__ import annotations

from collections.abc import Sequence

from emoharness.common.errors import ValidationError
from emoharness.data.manifest import Example
from emoharness.data.tasks import Split

_PRIORITY = {Split.BUILD: 0, Split.VAL: 1, Split.EXT_TEST: 2, Split.TEST: 3}


def identity_groups(examples: Sequence[Example]) -> tuple[tuple[Example, ...], ...]:
    """Transitive components; matching either source or media connects two items."""
    parents = list(range(len(examples)))
    owners: dict[tuple[str, ...], int] = {}

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, example in enumerate(examples):
        keys = [
            ('source', example.origin, example.source_id),
            ('path', example.media_path),
        ]
        if example.media_hash:
            keys.append(('hash', example.media_hash))
        for key in keys:
            if key in owners:
                parents[root(index)] = root(owners[key])
            owners[key] = index
    groups: dict[int, list[Example]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(root(index), []).append(example)
    return tuple(tuple(group) for group in groups.values())


def assert_isolated(examples: Sequence[Example]) -> None:
    """Refuse source/media overlap and duplicate ids before any model calls."""
    if len({example.example_id for example in examples}) != len(examples):
        raise ValidationError('construction manifest contains duplicate example ids')
    overlaps = sum(
        len({item.split for item in group}) > 1 for group in identity_groups(examples)
    )
    if overlaps:
        raise ValidationError(
            f'construction splits share {overlaps} source/media groups; prepare an isolated manifest'
        )


def quarantine_overlaps(
    examples: Sequence[Example],
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    """Keep the more protected split; quarantine lower-priority copies, never move labels."""
    keep: set[str] = set()
    for group in identity_groups(examples):
        protected = max((item.split for item in group), key=_PRIORITY.__getitem__)
        keep.update(item.example_id for item in group if item.split == protected)
    retained = tuple(item for item in examples if item.example_id in keep)
    removed = tuple(item for item in examples if item.example_id not in keep)
    assert_isolated(retained)
    return retained, removed
