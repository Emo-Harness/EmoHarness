from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from emoharness.common.constants import JUDGE_FRAME_INTERVAL_SECONDS, JUDGE_MAX_FRAMES
from emoharness.common.errors import ValidationError
from emoharness.perceive.video import (
    _uniform_timestamps,
    extract_audio_track,
    extract_frames,
    frame_timestamps,
    judge_frame_timestamps,
    probe_duration,
)


def test_frame_timestamps_uniform_when_within_bounds() -> None:
    result = frame_timestamps(8.0, fps=2.0, min_frames=4, max_frames=32)
    assert result == tuple(i * 0.5 for i in range(16))


def test_frame_timestamps_subsamples_when_too_many() -> None:
    result = frame_timestamps(60.0, fps=2.0, min_frames=4, max_frames=32)
    assert len(result) == 32
    assert result[0] == 0.0
    assert all(a < b for a, b in zip(result, result[1:], strict=False))
    assert all(t < 60.0 for t in result)


def test_frame_timestamps_pads_when_too_few() -> None:
    result = frame_timestamps(1.0, fps=2.0, min_frames=4, max_frames=32)
    assert len(result) == 4
    assert result[0] == 0.0
    assert all(t < 1.0 for t in result)
    assert result == tuple(i * 0.25 for i in range(4))


def test_frame_timestamps_zero_duration_returns_min_frames_at_zero() -> None:
    result = frame_timestamps(0.0, fps=2.0, min_frames=4, max_frames=32)
    assert result == (0.0, 0.0, 0.0, 0.0)


def test_frame_timestamps_is_deterministic() -> None:
    args = (17.3, 2.0, 4, 32)
    assert frame_timestamps(*args) == frame_timestamps(*args)


@pytest.mark.parametrize(
    ('duration_s', 'fps', 'min_frames', 'max_frames'),
    [
        (-1.0, 2.0, 4, 32),
        (10.0, 0.0, 4, 32),
        (10.0, -1.0, 4, 32),
        (10.0, 2.0, 0, 32),
        (10.0, 2.0, 8, 4),
    ],
)
def test_frame_timestamps_rejects_invalid_params(
    duration_s: float, fps: float, min_frames: int, max_frames: int
) -> None:
    with pytest.raises(ValidationError):
        frame_timestamps(duration_s, fps, min_frames, max_frames)


@settings(max_examples=200)
@given(
    duration_s=st.floats(
        min_value=0.05, max_value=60.0, allow_nan=False, allow_infinity=False
    ),
    fps=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_frames=st.integers(min_value=1, max_value=8),
    extra_frames=st.integers(min_value=0, max_value=56),
)
def test_frame_timestamps_invariants(
    duration_s: float, fps: float, min_frames: int, extra_frames: int
) -> None:
    max_frames = min_frames + extra_frames
    result = frame_timestamps(duration_s, fps, min_frames, max_frames)
    assert result == frame_timestamps(duration_s, fps, min_frames, max_frames)
    assert min_frames <= len(result) <= max_frames
    assert result[0] == 0.0
    assert all(t < duration_s for t in result)
    assert list(result) == sorted(result)
    assert len(set(result)) == len(result)


def test_judge_frame_timestamps_matches_official_rate() -> None:
    result = judge_frame_timestamps(10.5)
    assert result == tuple(float(i) for i in range(JUDGE_MAX_FRAMES))


def test_judge_frame_timestamps_short_clip_is_not_padded() -> None:
    assert judge_frame_timestamps(3.2) == (0.0, 1.0, 2.0)


def test_judge_frame_timestamps_uses_one_second_interval() -> None:
    assert JUDGE_FRAME_INTERVAL_SECONDS == 1.0
    assert judge_frame_timestamps(1.5) == (0.0,)


def test_judge_frame_timestamps_sub_second_clip_keeps_frame_zero() -> None:
    assert judge_frame_timestamps(0.4) == (0.0,)


def test_judge_frame_timestamps_zero_or_negative_duration_is_empty() -> None:
    assert judge_frame_timestamps(0.0) == ()
    assert judge_frame_timestamps(-5.0) == ()


def test_uniform_timestamps_helper_handles_zero_count() -> None:
    assert _uniform_timestamps(5.0, 0) == ()


def test_probe_duration_missing_file_raises() -> None:
    with pytest.raises(ValidationError):
        probe_duration(Path('/no/such/clip.mp4'))


def test_probe_duration_raises_when_ffprobe_output_is_not_a_float(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / 'clip.mp4'
    media.write_bytes(b'fake')

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout='N/A\n', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    with pytest.raises(ValidationError):
        probe_duration(media)


def test_run_wraps_command_failure_as_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / 'clip.mp4'
    media.write_bytes(b'fake')

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr='boom')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    with pytest.raises(ValidationError):
        probe_duration(media)


@pytest.mark.integration
def test_probe_duration_reads_real_clip(synthetic_clip: Path) -> None:
    duration = probe_duration(synthetic_clip)
    assert math.isclose(duration, 2.0, abs_tol=0.2)


@pytest.mark.integration
def test_extract_frames_writes_jpegs(synthetic_clip: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / 'frames'
    timestamps = (0.0, 0.5, 1.0)
    paths = extract_frames(synthetic_clip, timestamps, out_dir, prefix='frame')
    assert len(paths) == 3
    assert [p.name for p in paths] == [
        'frame_0000.jpg',
        'frame_0001.jpg',
        'frame_0002.jpg',
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


@pytest.mark.integration
def test_extract_frames_skips_existing_files(
    synthetic_clip: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / 'frames'
    timestamps = (0.0, 0.5)
    first = extract_frames(synthetic_clip, timestamps, out_dir, prefix='frame')
    mtimes = [p.stat().st_mtime_ns for p in first]
    second = extract_frames(synthetic_clip, timestamps, out_dir, prefix='frame')
    assert second == first
    assert [p.stat().st_mtime_ns for p in second] == mtimes


@pytest.mark.integration
def test_extract_audio_track_real_clip(synthetic_clip: Path, tmp_path: Path) -> None:
    wav_path = tmp_path / 'audio.wav'
    result = extract_audio_track(synthetic_clip, wav_path)
    assert result == wav_path
    assert wav_path.exists()
    assert wav_path.stat().st_size > 0


@pytest.mark.integration
def test_extract_audio_track_returns_none_without_audio_stream(
    silent_clip: Path, tmp_path: Path
) -> None:
    wav_path = tmp_path / 'audio.wav'
    result = extract_audio_track(silent_clip, wav_path)
    assert result is None
    assert not wav_path.exists()
