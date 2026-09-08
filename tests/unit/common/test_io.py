"""Tests for emoharness.common.io: atomic JSON/JSONL/YAML I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.common.errors import ValidationError
from emoharness.common.io import (
    append_jsonl,
    iter_jsonl,
    read_json,
    read_text,
    read_yaml,
    write_json_atomic,
    write_jsonl_atomic,
)


def test_write_json_atomic_round_trips(tmp_path: Path) -> None:
    path = tmp_path / 'out' / 'doc.json'
    payload = {'a': 1, 'b': [1, 2, 3], 'c': None}

    write_json_atomic(path, payload)

    assert read_json(path) == payload


def test_write_json_atomic_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / 'doc.json'

    write_json_atomic(path, {'x': 1})

    leftovers = list(tmp_path.glob(f'.{path.name}.*.tmp'))
    assert leftovers == []
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_atomic_cleans_up_temp_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / 'doc.json'

    def _boom(_src: str, _dst: str) -> None:
        raise OSError('disk full')

    monkeypatch.setattr('emoharness.common.io.os.replace', _boom)

    with pytest.raises(OSError, match='disk full'):
        write_json_atomic(path, {'x': 1})

    assert list(tmp_path.iterdir()) == []


def test_read_text_reads_utf8(tmp_path: Path) -> None:
    path = tmp_path / 'note.md'
    path.write_text('emotion recognition éèê\n', encoding='utf-8')

    assert read_text(path) == 'emotion recognition éèê\n'


def test_read_text_missing_file_raises_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match='cannot read'):
        read_text(tmp_path / 'missing.md')


def test_read_text_undecodable_file_raises_validation_error(tmp_path: Path) -> None:
    path = tmp_path / 'binary.md'
    path.write_bytes(b'\xff\xfe\x00binary')

    with pytest.raises(ValidationError, match='cannot read'):
        read_text(path)


def test_read_json_missing_file_raises_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        read_json(tmp_path / 'missing.json')


def test_read_json_invalid_content_raises_validation_error(tmp_path: Path) -> None:
    path = tmp_path / 'bad.json'
    path.write_text('{not valid json', encoding='utf-8')

    with pytest.raises(ValidationError):
        read_json(path)


def test_read_yaml_round_trips_via_manual_file(tmp_path: Path) -> None:
    path = tmp_path / 'doc.yaml'
    path.write_text('a: 1\nb:\n  - x\n  - y\n', encoding='utf-8')

    assert read_yaml(path) == {'a': 1, 'b': ['x', 'y']}


def test_read_yaml_missing_file_raises_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        read_yaml(tmp_path / 'missing.yaml')


def test_read_yaml_invalid_content_raises_validation_error(tmp_path: Path) -> None:
    path = tmp_path / 'bad.yaml'
    path.write_text('a: [1, 2\n', encoding='utf-8')

    with pytest.raises(ValidationError):
        read_yaml(path)


def test_write_jsonl_atomic_round_trips(tmp_path: Path) -> None:
    path = tmp_path / 'rows.jsonl'
    rows = [{'i': 0}, {'i': 1}, {'i': 2}]

    write_jsonl_atomic(path, rows)

    assert list(iter_jsonl(path)) == rows


def test_iter_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / 'rows.jsonl'
    path.write_text('{"i": 0}\n\n{"i": 1}\n   \n', encoding='utf-8')

    assert list(iter_jsonl(path)) == [{'i': 0}, {'i': 1}]


def test_iter_jsonl_missing_file_raises_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        list(iter_jsonl(tmp_path / 'missing.jsonl'))


def test_iter_jsonl_invalid_line_raises_validation_error_with_line_number(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'rows.jsonl'
    path.write_text('{"i": 0}\nnot json\n', encoding='utf-8')

    with pytest.raises(ValidationError, match='2'):
        list(iter_jsonl(path))


def test_append_jsonl_appends_and_creates_parents(tmp_path: Path) -> None:
    path = tmp_path / 'nested' / 'ledger.jsonl'

    append_jsonl(path, {'i': 0})
    append_jsonl(path, {'i': 1})

    assert list(iter_jsonl(path)) == [{'i': 0}, {'i': 1}]
