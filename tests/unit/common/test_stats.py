"""The one place that decides what the mean of nothing is."""

from __future__ import annotations

import numpy as np
import pytest

from emoharness.common.stats import mean, mean_or_none, percentile


def test_mean_of_nothing_is_zero_so_a_budget_column_still_prints() -> None:
    assert mean(()) == 0.0
    assert mean((1.0, 2.0, 3.0)) == 2.0


def test_mean_or_none_of_nothing_is_none_so_a_score_is_never_invented() -> None:
    assert mean_or_none(()) is None
    assert mean_or_none((0.0, 1.0)) == 0.5


def test_percentile_of_nothing_is_zero() -> None:
    assert percentile((), 95) == 0.0


def test_percentile_hits_an_exact_index_without_interpolating() -> None:
    assert percentile((0.0, 1.0, 2.0, 3.0, 4.0), 50) == 2.0
    assert percentile((5.0,), 95) == 5.0


def test_percentile_interpolates_between_neighbours() -> None:
    assert percentile((0.0, 10.0), 95) == pytest.approx(9.5)


@pytest.mark.parametrize('q', [0, 25, 50, 90, 95, 99, 100])
def test_percentile_matches_numpys_default_method(q: float) -> None:
    values = [0.5, 3.0, 1.25, 7.0, 2.0, 2.0, 9.5, 0.0]
    assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))
