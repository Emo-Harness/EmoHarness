"""Static firewall checks (ADR-0004, ADR-0010, ADR-0012).

These tests read source files, never data. They fail when a module could read the
benchmark's judge-side audio clues or branch on a benchmark task name at inference.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / 'src' / 'emoharness'
AUDIO_CLUE_FILE_PATTERN = re.compile(r'audio_clue[s]?[/\\]|qwen2_audio_clues')
AUDIO_CLUE_ALLOWED = {SRC / 'evaluation' / 'official.py'}
TASK_ENUM_PATTERN = re.compile(
    r'\bTask\.[A-Z_]+\b|from emoharness\.data\.tasks import .*\bTask\b'
)
INFERENCE_DIRS = ('harness', 'baselines', 'perceive')


def _python_files(*subdirs: str) -> tuple[Path, ...]:
    roots = [SRC / sub for sub in subdirs] if subdirs else [SRC]
    return tuple(path for root in roots for path in root.rglob('*.py'))


def test_only_the_official_wrapper_mentions_audio_clue_files() -> None:
    offenders = [
        path
        for path in _python_files()
        if path not in AUDIO_CLUE_ALLOWED
        and AUDIO_CLUE_FILE_PATTERN.search(path.read_text())
    ]
    assert (
        offenders == []
    ), f'audio_clue files referenced outside the official wrapper: {offenders}'


def test_inference_code_never_names_a_benchmark_task() -> None:
    offenders = [
        path
        for path in _python_files(*INFERENCE_DIRS)
        if TASK_ENUM_PATTERN.search(path.read_text())
    ]
    assert (
        offenders == []
    ), f'inference-time modules must not branch on Task: {offenders}'
