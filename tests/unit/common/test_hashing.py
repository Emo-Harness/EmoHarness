"""Tests for emoharness.common.hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emoharness.common.hashing import (
    directory_hash,
    sha256_file,
    sha256_file_cached,
    sha256_text,
    stable_hash,
)


def test_sha256_file_matches_hashlib_for_small_file(tmp_path: Path) -> None:
    path = tmp_path / 'small.bin'
    content = b'the quick brown fox jumps over the lazy dog'
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_sha256_file_streams_across_chunk_boundary(tmp_path: Path) -> None:
    path = tmp_path / 'large.bin'
    content = (b'0123456789abcdef' * ((1 << 20) // 16 + 8))[: (1 << 20) + 137]
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_sha256_file_differs_for_different_content(tmp_path: Path) -> None:
    path_a = tmp_path / 'a.bin'
    path_b = tmp_path / 'b.bin'
    path_a.write_bytes(b'alpha')
    path_b.write_bytes(b'beta')

    assert sha256_file(path_a) != sha256_file(path_b)


def test_sha256_text_matches_hashlib_utf8() -> None:
    text = 'emotion recognition éèê'

    assert sha256_text(text) == hashlib.sha256(text.encode('utf-8')).hexdigest()


def test_stable_hash_is_deterministic() -> None:
    obj = {'b': 2, 'a': [1, 2, 3], 'c': {'nested': True}}

    assert stable_hash(obj) == stable_hash(obj)


def test_stable_hash_ignores_dict_key_order() -> None:
    ordered = {'a': 1, 'b': 2, 'c': 3}
    reordered = {'c': 3, 'a': 1, 'b': 2}

    assert stable_hash(ordered) == stable_hash(reordered)


def test_stable_hash_distinguishes_list_order() -> None:
    assert stable_hash([1, 2, 3]) != stable_hash([3, 2, 1])


def test_stable_hash_differs_for_different_values() -> None:
    assert stable_hash({'x': 1}) != stable_hash({'x': 2})


def test_stable_hash_falls_back_to_str_for_unknown_types(tmp_path: Path) -> None:
    # Path is not JSON-serializable by default; `default=str` must cover it.
    digest = stable_hash({'path': tmp_path})

    assert digest == stable_hash({'path': str(tmp_path)})


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=8), st.integers(), min_size=0, max_size=6
    )
)
def test_stable_hash_key_order_never_changes_hash(mapping: dict[str, int]) -> None:
    shuffled = dict(reversed(list(mapping.items())))

    assert stable_hash(mapping) == stable_hash(shuffled)


def _bundle(root: Path) -> Path:
    (root / 'prompts').mkdir(parents=True)
    (root / 'prompts' / 'stage.md').write_text('one\n', encoding='utf-8')
    (root / 'routing.yaml').write_text('order: [visual]\n', encoding='utf-8')
    return root


def test_directory_hash_is_stable_for_the_same_tree(tmp_path: Path) -> None:
    root = _bundle(tmp_path / 'bundle')

    assert directory_hash(root) == directory_hash(root)


def test_directory_hash_changes_with_any_file(tmp_path: Path) -> None:
    root = _bundle(tmp_path / 'bundle')
    before = directory_hash(root)

    (root / 'routing.yaml').write_text('order: [audio]\n', encoding='utf-8')

    assert directory_hash(root) != before


def test_directory_hash_changes_when_a_file_is_added(tmp_path: Path) -> None:
    root = _bundle(tmp_path / 'bundle')
    before = directory_hash(root)

    (root / 'extra.yaml').write_text('x: 1\n', encoding='utf-8')

    assert directory_hash(root) != before


def test_directory_hash_ignores_caches(tmp_path: Path) -> None:
    root = _bundle(tmp_path / 'bundle')
    before = directory_hash(root)

    cache = root / 'verification' / '__pycache__'
    cache.mkdir(parents=True)
    (cache / 'x.pyc').write_bytes(b'\x00compiled')

    assert directory_hash(root) == before


def test_directory_hash_honours_a_custom_ignore_list(tmp_path: Path) -> None:
    root = _bundle(tmp_path / 'bundle')
    (root / 'scratch').mkdir()
    (root / 'scratch' / 'note.txt').write_text('draft\n', encoding='utf-8')

    assert directory_hash(root, ignore_dirs=('scratch',)) != directory_hash(root)


def test_sha256_file_cached_matches_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / 'clip.mp4'
    path.write_bytes(b'\x00' * 128)

    assert sha256_file_cached(path) == sha256_file(path)


def test_sha256_file_cached_reads_a_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / 'clip.mp4'
    path.write_bytes(b'\x00' * 128)
    reads: list[Path] = []

    def counting(target: Path) -> str:
        reads.append(target)
        return 'digest'

    monkeypatch.setattr('emoharness.common.hashing.sha256_file', counting)

    assert sha256_file_cached(path) == 'digest'
    assert sha256_file_cached(path) == 'digest'
    assert reads == [path]


def test_sha256_file_cached_rehashes_changed_content(tmp_path: Path) -> None:
    path = tmp_path / 'clip.mp4'
    path.write_bytes(b'first')
    before = sha256_file_cached(path)

    path.write_bytes(b'second content')

    assert sha256_file_cached(path) != before


def test_sha256_file_cached_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        sha256_file_cached(tmp_path / 'gone.mp4')
