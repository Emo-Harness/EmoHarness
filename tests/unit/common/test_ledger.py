"""Tests for emoharness.common.ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.common.ledger import (
    LedgerEntry,
    read_entries,
    record,
    summarize,
    summarize_by,
)


def _entry(
    role: str = 'target',
    model_id: str = 'model-a',
    experiment: str = 'exp1',
    example_id: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    latency_s: float = 0.1,
    cost_usd: float = 0.01,
) -> LedgerEntry:
    return LedgerEntry(
        role=role,
        model_id=model_id,
        experiment=experiment,
        example_id=example_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency_s,
        cost_usd=cost_usd,
    )


def test_read_entries_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_entries(tmp_path / 'missing.jsonl') == ()


def test_record_then_read_entries_round_trips(tmp_path: Path) -> None:
    path = tmp_path / 'ledger.jsonl'
    first = _entry()
    second = _entry(role='builder')

    record(path, first)
    record(path, second)

    assert read_entries(path) == (first, second)


def test_record_preserves_example_id(tmp_path: Path) -> None:
    path = tmp_path / 'ledger.jsonl'
    entry = _entry(example_id='mme_er_wild_000001')

    record(path, entry)

    (loaded,) = read_entries(path)
    assert loaded.example_id == 'mme_er_wild_000001'


def test_summarize_sums_calls_tokens_and_cost() -> None:
    entries = [
        _entry(input_tokens=10, output_tokens=5, cost_usd=0.01),
        _entry(input_tokens=20, output_tokens=15, cost_usd=0.02),
    ]

    summary = summarize(entries)

    assert summary.calls == 2
    assert summary.input_tokens == 30
    assert summary.output_tokens == 20
    assert summary.cost_usd == pytest.approx(0.03)


def test_summarize_empty_is_zero() -> None:
    summary = summarize(())

    assert summary.calls == 0
    assert summary.input_tokens == 0
    assert summary.output_tokens == 0
    assert summary.cost_usd == 0.0


def test_summarize_by_groups_and_sorts_by_key() -> None:
    entries = [
        _entry(role='target', input_tokens=10, output_tokens=0, cost_usd=0.1),
        _entry(role='builder', input_tokens=1, output_tokens=0, cost_usd=0.2),
        _entry(role='target', input_tokens=5, output_tokens=0, cost_usd=0.3),
    ]

    grouped = summarize_by(entries, 'role')

    assert list(grouped.keys()) == ['builder', 'target']
    assert grouped['target'].calls == 2
    assert grouped['target'].input_tokens == 15
    assert grouped['builder'].calls == 1


def test_summarize_by_model_id() -> None:
    entries = [_entry(model_id='a'), _entry(model_id='b'), _entry(model_id='a')]

    grouped = summarize_by(entries, 'model_id')

    assert grouped['a'].calls == 2
    assert grouped['b'].calls == 1
