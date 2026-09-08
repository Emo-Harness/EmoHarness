"""Atomic JSON, JSONL, and YAML I/O. Writes go to a temp file and are renamed."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

from emoharness.common.errors import ValidationError


def read_text(path: Path) -> str:
    """Read a UTF-8 text file, rejecting anything unreadable or undecodable."""
    try:
        return path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f'cannot read {path}: {exc}') from exc


def read_json(path: Path) -> Any:
    """Load a JSON document."""
    try:
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f'cannot read JSON {path}: {exc}') from exc


def read_yaml(path: Path) -> Any:
    """Load a YAML document with the safe loader."""
    try:
        with path.open('r', encoding='utf-8') as handle:
            return yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f'cannot read YAML {path}: {exc}') from exc


def iter_jsonl(path: Path) -> Iterator[Any]:
    """Yield one parsed object per non-empty line."""
    try:
        with path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield _parse_line(path, line_number, line)
    except OSError as exc:
        raise ValidationError(f'cannot read JSONL {path}: {exc}') from exc


def _parse_line(path: Path, line_number: int, line: str) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValidationError(f'{path}:{line_number}: invalid JSON: {exc}') from exc


def write_json_atomic(path: Path, obj: Any, *, indent: int = 2) -> None:
    """Write JSON to ``path`` atomically, creating parent directories."""
    _write_atomic(path, json.dumps(obj, indent=indent, ensure_ascii=False) + '\n')


def write_jsonl_atomic(path: Path, rows: Iterable[Any]) -> None:
    """Write one JSON object per line to ``path`` atomically."""
    body = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    _write_atomic(path, body)


def append_jsonl(path: Path, row: Any) -> None:
    """Append one JSON object as a line; used for append-only ledgers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp'
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except OSError:
        Path(tmp_name).unlink(missing_ok=True)
        raise
