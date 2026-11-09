"""Structured logging with one consistent format across entry points."""

from __future__ import annotations

import logging
import sys
from typing import Final

_FORMAT: Final = '%(asctime)s %(levelname)s %(name)s: %(message)s'


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stderr; handlers are attached once per name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
