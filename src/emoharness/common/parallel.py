"""Ordered, bounded parallelism over independent items.

Every row and every construction round is a loop of independent Target calls,
and a served Target batches concurrent requests. ``map_ordered`` runs such a
loop on a thread pool and returns results in input order; ``worker_count``
reads the pool size from ``EMOHARNESS_WORKERS`` so the same code runs serially
in tests and in ``--fake`` mode (the default is one worker) and in parallel
against a server.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Final, TypeVar

from emoharness.common.errors import ValidationError

WORKERS_ENV: Final = 'EMOHARNESS_WORKERS'
DEFAULT_WORKERS: Final = 1

T = TypeVar('T')
R = TypeVar('R')


def worker_count() -> int:
    """The pool size ``EMOHARNESS_WORKERS`` asks for; one when it is unset."""
    raw = os.environ.get(WORKERS_ENV, '').strip()
    if not raw:
        return DEFAULT_WORKERS
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValidationError(f'{WORKERS_ENV} must be an integer, got {raw!r}') from exc
    if count < 1:
        raise ValidationError(f'{WORKERS_ENV} must be at least 1, got {count}')
    return count


def map_ordered(
    fn: Callable[[T], R], items: Iterable[T], workers: int | None = None
) -> tuple[R, ...]:
    """``tuple(map(fn, items))``, computed on ``workers`` threads.

    Results keep input order. With one worker (the default when
    ``EMOHARNESS_WORKERS`` is unset) this is a plain loop, so exceptions and
    logging behave exactly as they would without a pool. With more, the first
    exception raised by ``fn`` propagates after the pool drains.
    """
    rows = tuple(items)
    count = workers if workers is not None else worker_count()
    if count <= 1 or len(rows) <= 1:
        return tuple(fn(row) for row in rows)
    with ThreadPoolExecutor(max_workers=min(count, len(rows))) as pool:
        return tuple(pool.map(fn, rows))
