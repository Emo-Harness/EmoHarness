"""Fixtures for perceive tests. Synthetic media are generated per test in tmp_path."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """A tiny 2-second clip with video and a 440 Hz sine audio track, or a skip."""
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg not installed')
    clip = tmp_path / 'clip.mp4'
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-f',
            'lavfi',
            '-i',
            'testsrc=size=64x64:rate=10',
            '-f',
            'lavfi',
            '-i',
            'sine=frequency=440',
            '-t',
            '2',
            '-pix_fmt',
            'yuv420p',
            '-c:a',
            'aac',
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    return clip


@pytest.fixture
def silent_clip(tmp_path: Path) -> Path:
    """A tiny 2-second video-only clip with no audio stream, or a skip."""
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg not installed')
    clip = tmp_path / 'silent.mp4'
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-f',
            'lavfi',
            '-i',
            'testsrc=size=64x64:rate=10',
            '-t',
            '2',
            '-pix_fmt',
            'yuv420p',
            '-an',
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    return clip
