"""Tests for emoharness.common.logging."""

from __future__ import annotations

import logging
import sys

from emoharness.common.logging import get_logger


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger('emoharness.test.one')

    assert logger.name == 'emoharness.test.one'


def test_get_logger_attaches_exactly_one_handler() -> None:
    logger = get_logger('emoharness.test.two')
    handler_count_first = len(logger.handlers)

    get_logger('emoharness.test.two')
    get_logger('emoharness.test.two')

    assert handler_count_first == 1
    assert len(logger.handlers) == 1


def test_get_logger_does_not_propagate_to_root() -> None:
    logger = get_logger('emoharness.test.three')

    assert logger.propagate is False


def test_get_logger_sets_requested_level() -> None:
    logger = get_logger('emoharness.test.four', level=logging.DEBUG)

    assert logger.level == logging.DEBUG

    get_logger('emoharness.test.four', level=logging.WARNING)

    assert logger.level == logging.WARNING
    # Re-configuring the level must not attach a second handler.
    assert len(logger.handlers) == 1


def test_get_logger_handler_writes_to_stderr() -> None:
    logger = get_logger('emoharness.test.five')

    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].stream is sys.stderr
