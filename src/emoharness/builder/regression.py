"""The regression set: items some accepted harness once solved (README §4.1).

The set only grows. A candidate that stops solving items already in it is charged a
regression rate, which Gate 2 caps (ADR-0005).
"""

from __future__ import annotations

from collections.abc import Iterable


def update_regression_set(
    previous: frozenset[str], solved_now: Iterable[str]
) -> frozenset[str]:
    """Add the items solved by the newly accepted harness; never remove any."""
    return previous | frozenset(solved_now)


def regression_rate(regression_set: frozenset[str], solved_now: Iterable[str]) -> float:
    """Share of the regression set this candidate no longer solves; 0 when empty."""
    if not regression_set:
        return 0.0
    lost = regression_set - frozenset(solved_now)
    return len(lost) / len(regression_set)
