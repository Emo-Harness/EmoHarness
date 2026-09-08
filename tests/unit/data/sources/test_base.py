from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.common.errors import ValidationError
from emoharness.data.manifest import Example
from emoharness.data.sources.base import (
    FunctionAdapter,
    collapse_sentiment,
    parse_float,
    parse_int,
    read_csv_rows,
    require_canonical,
    require_column,
)


def test_require_canonical_matches_case_insensitively() -> None:
    assert require_canonical(' Anger ', ('anger', 'joy'), context='c') == 'anger'
    assert require_canonical('JOY', ('anger', 'joy'), context='c') == 'joy'


def test_require_canonical_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError, match='not one of'):
        require_canonical('confused', ('anger', 'joy'), context='row 3')


@pytest.mark.parametrize(
    ('score', 'expected'), [(1.0, 'positive'), (-0.5, 'negative'), (0.0, 'neutral')]
)
def test_collapse_sentiment_by_sign(score: float, expected: str) -> None:
    assert collapse_sentiment(score) == expected


def test_parse_int_rejects_non_numeric_strings() -> None:
    assert parse_int('42', context='c') == 42
    with pytest.raises(ValidationError, match='not an integer'):
        parse_int('forty-two', context='c')


def test_parse_float_rejects_non_numeric_strings() -> None:
    assert parse_float('1.5', context='c') == 1.5
    with pytest.raises(ValidationError, match='not a number'):
        parse_float('nan-ish', context='c')


def test_require_column_rejects_missing_and_blank() -> None:
    assert require_column({'a': ' x '}, 'a', context='c') == 'x'
    with pytest.raises(ValidationError, match='missing required column'):
        require_column({}, 'a', context='c')
    with pytest.raises(ValidationError, match='missing required column'):
        require_column({'a': '   '}, 'a', context='c')


def test_read_csv_rows_parses_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / 'rows.csv'
    path.write_text('a,b\n1,2\n3,4\n', encoding='utf-8')
    assert read_csv_rows(path) == ({'a': '1', 'b': '2'}, {'a': '3', 'b': '4'})


def test_read_csv_rows_missing_file_raises_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match='cannot read CSV'):
        read_csv_rows(tmp_path / 'missing.csv')


def test_function_adapter_delegates_to_the_wrapped_loader(tmp_path: Path) -> None:
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def loader(root: Path, split_names: tuple[str, ...]) -> tuple[Example, ...]:
        calls.append((root, split_names))
        return ()

    adapter = FunctionAdapter(loader)
    result = adapter.load(tmp_path, ('train',))
    assert result == ()
    assert calls == [(tmp_path, ('train',))]
