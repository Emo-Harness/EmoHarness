"""Append-only cost ledger. Every model call in the project writes one entry."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.io import append_jsonl, iter_jsonl


class LedgerEntry(BaseModel):
    """One billable call. ``cost_usd`` is the caller's best estimate at call time."""

    model_config = ConfigDict(frozen=True, extra='forbid', protected_namespaces=())

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    role: str  # 'target' | 'builder' | 'judge_official' | 'judge_proxy' | 'teacher_direct'
    model_id: str
    experiment: str
    example_id: str | None = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)


class LedgerSummary(BaseModel):
    """Totals over a set of entries."""

    model_config = ConfigDict(frozen=True)

    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def record(path: Path, entry: LedgerEntry) -> None:
    """Append one entry to the ledger file."""
    append_jsonl(path, entry.model_dump())


def read_entries(path: Path) -> tuple[LedgerEntry, ...]:
    """Read every entry; a missing file is an empty ledger."""
    if not path.exists():
        return ()
    return tuple(LedgerEntry.model_validate(row) for row in iter_jsonl(path))


def summarize(entries: Iterable[LedgerEntry]) -> LedgerSummary:
    """Sum calls, tokens, and cost."""
    rows = tuple(entries)
    return LedgerSummary(
        calls=len(rows),
        input_tokens=sum(e.input_tokens for e in rows),
        output_tokens=sum(e.output_tokens for e in rows),
        cost_usd=sum(e.cost_usd for e in rows),
    )


def summarize_by(entries: Iterable[LedgerEntry], key: str) -> dict[str, LedgerSummary]:
    """Group entries by an attribute name (``role``, ``model_id``, ``experiment``)."""
    groups: dict[str, list[LedgerEntry]] = {}
    for entry in entries:
        groups.setdefault(str(getattr(entry, key)), []).append(entry)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}
