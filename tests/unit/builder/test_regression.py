from __future__ import annotations

import pytest

from emoharness.builder.regression import regression_rate, update_regression_set


def test_regression_set_only_grows() -> None:
    first = update_regression_set(frozenset(), ['a', 'b'])
    second = update_regression_set(first, ['b', 'c'])
    assert first == {'a', 'b'}
    assert second == {'a', 'b', 'c'}


def test_updating_does_not_mutate_the_previous_set() -> None:
    previous = frozenset({'a'})
    update_regression_set(previous, ['b'])
    assert previous == {'a'}


def test_rate_is_zero_for_an_empty_set() -> None:
    assert regression_rate(frozenset(), []) == 0.0


def test_rate_is_the_share_no_longer_solved() -> None:
    regression_set = frozenset({'a', 'b', 'c', 'd'})
    assert regression_rate(regression_set, ['a', 'b', 'c', 'd']) == 0.0
    assert regression_rate(regression_set, ['a', 'b']) == pytest.approx(0.5)
    assert regression_rate(regression_set, []) == 1.0


def test_extra_solved_items_do_not_lower_the_rate() -> None:
    assert regression_rate(frozenset({'a'}), ['a', 'x', 'y']) == 0.0
