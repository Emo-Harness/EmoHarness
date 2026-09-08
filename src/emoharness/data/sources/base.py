"""Adapter protocol, shared CSV/label helpers, and the origin registry (ADR-0003).

Every external construction source is loaded into placeholder-split
``Example``s: ``task`` is always ``None``, ``kind`` is the seen capability kind
the source stands in for, and ``split`` is always ``Split.BUILD`` until
``emoharness.data.splits.assign_external_splits`` reassigns it.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from emoharness.common.errors import ValidationError
from emoharness.data.label_set import LabelSet
from emoharness.data.manifest import Example

SENTIMENT_LABELS: Final[tuple[str, ...]] = ('positive', 'negative', 'neutral')


class SourceAdapter(Protocol):
    """Loads one external corpus into a tuple of placeholder-split examples."""

    def load(self, root: Path, split_names: tuple[str, ...]) -> tuple[Example, ...]:
        """Return every example this source contributes."""
        ...  # pragma: no cover -- Protocol stub, never executed directly


@dataclass(frozen=True)
class FunctionAdapter:
    """Adapts a plain loader function to the ``SourceAdapter`` protocol."""

    loader: Callable[[Path, tuple[str, ...]], tuple[Example, ...]]

    def load(self, root: Path, split_names: tuple[str, ...]) -> tuple[Example, ...]:
        return self.loader(root, split_names)


def read_csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Read a CSV with a header row into a tuple of string-keyed dict rows."""
    try:
        with path.open('r', encoding='utf-8', newline='') as handle:
            return tuple(csv.DictReader(handle))
    except OSError as exc:
        raise ValidationError(f'cannot read CSV {path}: {exc}') from exc


def require_column(row: dict[str, str], column: str, *, context: str) -> str:
    """Return a required, non-blank CSV column value or raise ``ValidationError``."""
    value = row.get(column)
    if value is None or not value.strip():
        raise ValidationError(f'{context}: missing required column {column!r}')
    return value.strip()


def require_canonical(raw: str, label_set: tuple[str, ...], *, context: str) -> str:
    """The candidate ``raw`` names, spelled as ``label_set`` spells it, or raise.

    Returns the canonical entry from ``label_set``, never ``raw`` itself, so
    downstream label sets stay consistent regardless of source file casing.
    """
    label = LabelSet.from_labels(label_set).canonical(raw)
    if label is None:
        raise ValidationError(f'{context}: {raw!r} is not one of {label_set}')
    return label


def collapse_sentiment(score: float) -> str:
    """Collapse a continuous sentiment score to positive/negative/neutral by sign.

    Used by both CMU-MOSEI ([-3, 3]) and CH-SIMS ([-1, 1]): only the sign
    matters, so no corpus-specific magnitude threshold is needed.
    """
    if score > 0:
        return 'positive'
    if score < 0:
        return 'negative'
    return 'neutral'


def parse_int(raw: str, *, context: str) -> int:
    """Parse an int field or raise ``ValidationError`` with context."""
    try:
        return int(raw)
    except ValueError as exc:
        raise ValidationError(f'{context}: {raw!r} is not an integer') from exc


def parse_float(raw: str, *, context: str) -> float:
    """Parse a float field or raise ``ValidationError`` with context."""
    try:
        return float(raw)
    except ValueError as exc:
        raise ValidationError(f'{context}: {raw!r} is not a number') from exc
