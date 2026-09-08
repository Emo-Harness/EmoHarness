"""Tests for emoharness.common.errors."""

from __future__ import annotations

from typing import Any, cast

import pytest

from emoharness.common.errors import (
    EmoHarnessError,
    ValidationError,
    assert_never,
)


def test_every_error_derives_from_the_base() -> None:
    assert issubclass(ValidationError, EmoHarnessError)


def test_assert_never_raises_when_a_new_name_slips_through() -> None:
    # Only reachable if a caller stops being exhaustive, which mypy rejects first.
    with pytest.raises(EmoHarnessError, match='unhandled value'):
        assert_never(cast('Any', 'telepathy'))
