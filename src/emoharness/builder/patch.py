"""Patch permissions and application.

A candidate patch is a set of full-file replacements. It is checked against the
stage's editable globs, against the files the bundle actually has, and - for
verification code - against an injected static checker before anything is written.
"""

from __future__ import annotations

import fnmatch
import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Final

from emoharness.builder.schema import CandidatePatch
from emoharness.common.constants import STAGE_EDITABLE_GLOBS
from emoharness.common.errors import PatchRejectedError, ValidationError
from emoharness.common.hashing import sha256_text, stable_hash
from emoharness.common.io import read_text

CodeChecker = Callable[[str, str], tuple[str, ...]]

VERIFICATION_GLOB: Final = 'verification/*.py'
IGNORED_DIRS: Final[tuple[str, ...]] = ('__pycache__', '.git', '.ipynb_checkpoints')


def matches_glob(path: str, pattern: str) -> bool:
    """Match one relative path against one glob, segment by segment.

    ``fnmatch`` alone would let ``*.yaml`` match ``verification/evil.yaml`` because
    ``*`` also matches a separator, so each segment is matched on its own.
    """
    path_parts = path.split('/')
    pattern_parts = pattern.split('/')
    if len(path_parts) != len(pattern_parts):
        return False
    return all(
        fnmatch.fnmatchcase(part, glob)
        for part, glob in zip(path_parts, pattern_parts, strict=True)
    )


def stage_globs(stage: int) -> tuple[str, ...]:
    """The globs the Builder may edit at ``stage``."""
    if stage not in STAGE_EDITABLE_GLOBS:
        raise ValidationError(f'unknown builder stage {stage}')
    return STAGE_EDITABLE_GLOBS[stage]


def editable_paths_for(stage: int, paths: Iterable[str]) -> tuple[str, ...]:
    """The subset of ``paths`` the Builder may replace at ``stage``, sorted."""
    globs = stage_globs(stage)
    return tuple(
        sorted(path for path in paths if any(matches_glob(path, g) for g in globs))
    )


def is_unsafe_path(path: str) -> bool:
    """True for absolute paths, drive letters, backslashes, and ``..`` segments."""
    if not path or path.startswith('/') or path.startswith('~') or '\\' in path:
        return True
    if ':' in path:
        return True
    parts = path.split('/')
    return any(part in ('', '.', '..') for part in parts)


def validate_patch(
    patch: CandidatePatch,
    stage: int,
    bundle_files: Iterable[str],
    code_checker: CodeChecker | None = None,
) -> tuple[str, ...]:
    """Return every reason this patch must not be applied; empty means accepted."""
    globs = stage_globs(stage)
    known = frozenset(bundle_files)
    violations: list[str] = []
    seen: set[str] = set()
    for file in patch.files:
        if file.path in seen:
            violations.append(f'{file.path}: listed more than once')
            continue
        seen.add(file.path)
        violations.extend(
            _file_violations(file.path, file.content, stage, globs, known)
        )
        violations.extend(_code_violations(file.path, file.content, code_checker))
    return tuple(violations)


def _file_violations(
    path: str,
    content: str,
    stage: int,
    globs: tuple[str, ...],
    known: frozenset[str],
) -> tuple[str, ...]:
    if is_unsafe_path(path):
        return (f'{path}: absolute or traversing path',)
    reasons: list[str] = []
    if not any(matches_glob(path, glob) for glob in globs):
        reasons.append(f'{path}: not editable at stage {stage}')
    if path not in known:
        reasons.append(f'{path}: not present in the harness bundle')
    if not content.strip():
        reasons.append(f'{path}: empty file content')
    return tuple(reasons)


def _code_violations(
    path: str, content: str, code_checker: CodeChecker | None
) -> tuple[str, ...]:
    if code_checker is None or not matches_glob(path, VERIFICATION_GLOB):
        return ()
    return tuple(code_checker(content, path))


def apply_patch(bundle_root: Path, patch: CandidatePatch, dest_root: Path) -> Path:
    """Copy the bundle to ``dest_root`` and overwrite the patched files."""
    if not bundle_root.is_dir():
        raise ValidationError(f'harness bundle is not a directory: {bundle_root}')
    if dest_root.exists():
        raise ValidationError(f'destination already exists: {dest_root}')
    dest_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        bundle_root, dest_root, ignore=shutil.ignore_patterns(*IGNORED_DIRS)
    )
    for file in patch.files:
        _write_file(dest_root, file.path, file.content)
    return dest_root


def _write_file(dest_root: Path, path: str, content: str) -> None:
    if is_unsafe_path(path):
        raise PatchRejectedError(f'{path}: refusing to write an unsafe path')
    destination = dest_root / path
    if not destination.resolve().is_relative_to(dest_root.resolve()):
        raise PatchRejectedError(f'{path}: escapes the candidate bundle')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding='utf-8')


def patch_hash(patch: CandidatePatch) -> str:
    """Content address of a patch: its name and the digest of every replaced file."""
    files = sorted(
        ({'path': f.path, 'sha256': sha256_text(f.content)} for f in patch.files),
        key=lambda entry: entry['path'],
    )
    return stable_hash({'name': patch.name, 'files': files})


def read_bundle_files(root: Path) -> Mapping[str, str]:
    """Read a harness bundle into ``relative path -> text``, sorted by path."""
    if not root.is_dir():
        raise ValidationError(f'harness bundle is not a directory: {root}')
    files: dict[str, str] = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or _is_ignored(path, root):
            continue
        files[path.relative_to(root).as_posix()] = read_text(path)
    return files


def _is_ignored(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(part in IGNORED_DIRS or part.startswith('.') for part in parts)
