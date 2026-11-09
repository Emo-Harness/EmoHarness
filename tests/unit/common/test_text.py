"""Tests for emoharness.common.text.truncate.

Both callers depend on a different half of the contract: the evidence record cuts
a transcript with no marker at all, the diagnosis packet marks what it dropped.
"""

from __future__ import annotations

import pytest

from emoharness.common.text import truncate


def test_no_limit_returns_the_text() -> None:
    assert truncate('abcdef', None) == 'abcdef'


def test_text_within_the_limit_is_untouched() -> None:
    assert truncate('abc', 3) == 'abc'
    assert truncate('abc', 10) == 'abc'


def test_a_cut_text_carries_the_marker() -> None:
    assert truncate('abcdef', 3, ' ...[truncated]') == 'abc ...[truncated]'


def test_the_default_marker_is_an_ellipsis() -> None:
    assert truncate('abcdef', 3) == 'abc…'


def test_an_empty_marker_cuts_silently() -> None:
    assert truncate('abcdef', 3, marker='') == 'abc'


@pytest.mark.parametrize('limit', [0, -1])
def test_a_limit_of_zero_or_less_leaves_nothing(limit: int) -> None:
    assert truncate('abcdef', limit, ' ...[truncated]') == ''


def test_an_empty_text_is_never_marked() -> None:
    assert truncate('', 0) == ''
