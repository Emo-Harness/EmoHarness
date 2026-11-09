from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from emoharness.builder.patch import (
    _write_file,
    apply_patch,
    editable_paths_for,
    matches_glob,
    patch_hash,
    read_bundle_files,
    validate_patch,
)
from emoharness.builder.schema import CandidatePatch
from emoharness.common.errors import PatchRejectedError, ValidationError

Patch = Callable[..., CandidatePatch]
PROMPT = 'prompts/extract_visual.md'
VERIFICATION = 'verification/cross_modal.py'


def _reasons(patch: CandidatePatch, stage: int, files: Mapping[str, str]) -> str:
    return ' '.join(validate_patch(patch, stage, files.keys()))


def test_a_prompt_edit_is_valid_at_stage_two(
    make_patch: Patch, bundle_files: Mapping[str, str]
) -> None:
    assert validate_patch(make_patch(), 2, bundle_files.keys()) == ()


def test_stage_one_allows_nothing(
    make_patch: Patch, bundle_files: Mapping[str, str]
) -> None:
    assert 'not editable at stage 1' in _reasons(make_patch(), 1, bundle_files)


def test_stage_two_refuses_verification_code(
    make_patch: Patch, bundle_files: Mapping[str, str]
) -> None:
    patch = make_patch(
        path=VERIFICATION, content='def agree() -> bool:\n    return True\n'
    )
    assert 'not editable at stage 2' in _reasons(patch, 2, bundle_files)
    assert validate_patch(patch, 3, bundle_files.keys()) == ()


@pytest.mark.parametrize(
    'path',
    [
        '../evil.md',
        '/etc/passwd',
        'prompts/../../evil.md',
        'prompts\\evil.md',
        '~/x.md',
    ],
)
def test_unsafe_paths_are_rejected(
    path: str, make_patch: Patch, bundle_files: Mapping[str, str]
) -> None:
    assert 'absolute or traversing path' in _reasons(
        make_patch(path=path), 3, bundle_files
    )


def test_unknown_files_are_rejected(
    make_patch: Patch, bundle_files: Mapping[str, str]
) -> None:
    reasons = _reasons(make_patch(path='prompts/brand_new.md'), 2, bundle_files)
    assert 'not present in the harness bundle' in reasons


def test_empty_content_is_rejected(
    make_patch: Patch, bundle_files: Mapping[str, str]
) -> None:
    assert 'empty file content' in _reasons(make_patch(content='  \n'), 2, bundle_files)


def test_a_path_listed_twice_is_rejected(bundle_files: Mapping[str, str]) -> None:
    patch = CandidatePatch(
        name='dup',
        targeted_cluster='k',
        rationale='r',
        files=(
            {'path': PROMPT, 'content': 'one\n'},
            {'path': PROMPT, 'content': 'two\n'},
        ),
    )
    assert 'listed more than once' in _reasons(patch, 2, bundle_files)


def test_unknown_stages_are_rejected(
    make_patch: Patch, bundle_files: Mapping[str, str]
) -> None:
    with pytest.raises(ValidationError, match='unknown builder stage'):
        validate_patch(make_patch(), 9, bundle_files.keys())


def test_the_code_checker_only_sees_verification_code(
    make_patch: Patch, bundle_files: Mapping[str, str]
) -> None:
    seen: list[str] = []

    def checker(source: str, path: str) -> tuple[str, ...]:
        seen.append(path)
        return (
            (f'{path}: import of os is not allowed',) if 'import os' in source else ()
        )

    clean = make_patch(path=VERIFICATION, content='def f() -> int:\n    return 1\n')
    assert validate_patch(clean, 3, bundle_files.keys(), checker) == ()
    dirty = make_patch(path=VERIFICATION, content='import os\n')
    assert validate_patch(dirty, 3, bundle_files.keys(), checker) == (
        'verification/cross_modal.py: import of os is not allowed',
    )
    assert validate_patch(make_patch(), 3, bundle_files.keys(), checker) == ()
    assert seen == [VERIFICATION, VERIFICATION]


def test_globs_do_not_leak_across_directories() -> None:
    assert matches_glob('routing.yaml', '*.yaml') is True
    assert matches_glob('verification/evil.yaml', '*.yaml') is False
    assert matches_glob('prompts/deep/x.md', 'prompts/*.md') is False
    assert matches_glob('prompts/x.md', 'prompts/*.md') is True


def test_editable_paths_are_sorted_per_stage(bundle_files: Mapping[str, str]) -> None:
    assert editable_paths_for(1, bundle_files) == ()
    assert editable_paths_for(2, bundle_files) == (
        'prompts/extract_visual.md',
        'prompts/generate_hypotheses.md',
        'routing.yaml',
        'sampling.yaml',
    )
    assert VERIFICATION in editable_paths_for(3, bundle_files)


def test_apply_patch_writes_an_isolated_copy(
    bundle_root: Path, tmp_path: Path, make_patch: Patch
) -> None:
    patch = make_patch(content='Revised prompt.\n')
    destination = apply_patch(bundle_root, patch, tmp_path / 'cand' / 'c0')
    assert (destination / PROMPT).read_text(encoding='utf-8') == 'Revised prompt.\n'
    assert (destination / 'routing.yaml').exists()
    assert (bundle_root / PROMPT).read_text(encoding='utf-8').startswith('List the')
    assert not (destination / 'verification' / '__pycache__').exists()


def test_apply_patch_refuses_an_existing_destination(
    bundle_root: Path, tmp_path: Path, make_patch: Patch
) -> None:
    destination = tmp_path / 'cand'
    destination.mkdir()
    with pytest.raises(ValidationError, match='already exists'):
        apply_patch(bundle_root, make_patch(), destination)


def test_apply_patch_requires_a_bundle_directory(
    tmp_path: Path, make_patch: Patch
) -> None:
    with pytest.raises(ValidationError, match='not a directory'):
        apply_patch(tmp_path / 'missing', make_patch(), tmp_path / 'out')


def test_patch_hash_depends_on_content_not_wording(make_patch: Patch) -> None:
    first = make_patch(content='same\n')
    same = make_patch(content='same\n')
    other = make_patch(content='different\n')
    assert patch_hash(first) == patch_hash(same)
    assert patch_hash(first) != patch_hash(other)


def test_read_bundle_files_skips_caches(bundle_root: Path) -> None:
    files = read_bundle_files(bundle_root)
    assert set(files) == {
        PROMPT,
        'prompts/generate_hypotheses.md',
        'routing.yaml',
        'sampling.yaml',
        VERIFICATION,
        'manifest.json',
    }
    assert files['routing.yaml'] == 'order: [visual, audio]\n'


def test_read_bundle_files_requires_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match='not a directory'):
        read_bundle_files(tmp_path / 'nope')


def test_windows_style_paths_are_unsafe(
    make_patch: Patch, bundle_files: Mapping[str, str]
) -> None:
    reasons = _reasons(make_patch(path='C:/prompts/x.md'), 3, bundle_files)
    assert 'absolute or traversing path' in reasons


def test_apply_patch_refuses_to_write_outside_the_bundle(
    bundle_root: Path, tmp_path: Path, make_patch: Patch
) -> None:
    patch = make_patch(path='../escaped.md')
    with pytest.raises(PatchRejectedError, match='unsafe path'):
        apply_patch(bundle_root, patch, tmp_path / 'cand')


def test_write_file_refuses_a_symlinked_escape(tmp_path: Path) -> None:
    outside = tmp_path / 'outside'
    outside.mkdir()
    destination = tmp_path / 'cand'
    destination.mkdir()
    (destination / 'link').symlink_to(outside, target_is_directory=True)
    with pytest.raises(PatchRejectedError, match='escapes the candidate bundle'):
        _write_file(destination, 'link/x.md', 'text\n')


def test_read_bundle_files_rejects_undecodable_files(bundle_root: Path) -> None:
    (bundle_root / 'broken.md').write_bytes(b'\xff\xfe\x00binary')
    with pytest.raises(ValidationError, match='cannot read'):
        read_bundle_files(bundle_root)
