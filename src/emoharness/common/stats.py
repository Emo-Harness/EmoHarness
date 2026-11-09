"""Tiny numeric helpers shared by the scoreboard, the tables, and the Builder loop.

They exist so that "the mean of nothing" is answered in exactly one place: it is
0.0 for a budget column that must still print, and None for a score that has no
items behind it. Mixing the two is how an empty run silently reads as a zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

PERCENT: float = 100.0


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean, 0.0 when there is nothing to average."""
    numbers = tuple(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def mean_or_none(values: Iterable[float]) -> float | None:
    """Arithmetic mean, None when there is nothing to average."""
    numbers = tuple(values)
    return sum(numbers) / len(numbers) if numbers else None


def percentile(values: Iterable[float], q: float) -> float:
    """The ``q``-th percentile with linear interpolation, 0.0 when empty.

    Matches numpy's default ``method='linear'`` so that latency budgets recorded
    before and after this helper existed stay comparable.
    """
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q / PERCENT
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
