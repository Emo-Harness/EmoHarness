"""Stable hashing for files, text, and JSON-serializable objects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

_CHUNK_SIZE = 1 << 20
DEFAULT_IGNORED_DIRS: Final[tuple[str, ...]] = ('__pycache__',)

_FileStamp = tuple[str, int, int]
_file_digests: dict[_FileStamp, str] = {}


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes, streaming in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_cached(path: Path) -> str:
    """:func:`sha256_file`, memoized per process by ``(path, size, mtime_ns)``.

    Media files are hashed once per clip and then again by every row, every
    cache lookup, and every leakage sweep; re-reading gigabytes to learn the
    same digest is the difference between a run and an afternoon.
    """
    stat = path.stat()
    stamp: _FileStamp = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _file_digests.get(stamp)
    if cached is not None:
        return cached
    digest = sha256_file(path)
    _file_digests[stamp] = digest
    return digest


def directory_hash(
    root: Path, ignore_dirs: tuple[str, ...] = DEFAULT_IGNORED_DIRS
) -> str:
    """Content address of a directory: every relative path and its file digest."""
    ignored = frozenset(ignore_dirs)
    digests = {
        relative.as_posix(): sha256_file(path)
        for path, relative in _walk(root)
        if ignored.isdisjoint(relative.parts)
    }
    return stable_hash(digests)


def _walk(root: Path) -> list[tuple[Path, Path]]:
    return [
        (path, path.relative_to(root))
        for path in sorted(root.rglob('*'))
        if path.is_file()
    ]


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of UTF-8 encoded text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def stable_hash(obj: Any) -> str:
    """Return the hex SHA-256 of an object's canonical JSON form (sorted keys)."""
    canonical = json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)
    return sha256_text(canonical)
