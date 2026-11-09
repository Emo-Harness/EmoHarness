"""Text helpers shared by the evidence record and the Builder's diagnosis packet."""

from __future__ import annotations

from typing import Final

DEFAULT_MARKER: Final = '…'


def truncate(text: str, limit: int | None, marker: str = DEFAULT_MARKER) -> str:
    """``text`` cut to ``limit`` characters, with ``marker`` appended when it was cut.

    ``limit`` None means no limit at all. A limit of zero or less leaves nothing,
    not even the marker: a caller that budgets a field down to zero wants the
    field gone, not a stub of punctuation.
    """
    if limit is None or len(text) <= limit:
        return text
    if limit <= 0:
        return ''
    return text[:limit] + marker
