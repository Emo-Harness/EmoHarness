"""Frame timestamp sampling and ffmpeg/ffprobe wrappers for video and audio extraction.

Timestamp math is pure and deterministic; the ffmpeg/ffprobe calls are the only
side-effecting functions and are isolated here so the pipeline can monkeypatch them.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from emoharness.common.constants import JUDGE_FRAME_INTERVAL_SECONDS, JUDGE_MAX_FRAMES
from emoharness.common.errors import ValidationError

_EPSILON: Final = 1e-9
_PROBE_TIMEOUT_S: Final = 30.0
_EXTRACT_TIMEOUT_S: Final = 60.0
#: Frames are read by the teacher, the Builder, and the proxy judge as images,
#: all of which tokenize by area: a 640-wide frame costs a third of a 720p one
#: and shows the same face. Smaller sources are left alone; height keeps aspect.
_FRAME_SCALE_FILTER: Final = "scale='min(640,iw)':-2"


def frame_timestamps(
    duration_s: float, fps: float, min_frames: int, max_frames: int
) -> tuple[float, ...]:
    """Frame times covering ``[0, duration_s)``, uniform at ``fps``.

    The uniform-at-``fps`` grid is subsampled uniformly down to ``max_frames`` when
    it is longer, or padded to ``min_frames`` by uniform spacing over the whole
    duration when it is shorter. Deterministic; every timestamp is strictly less
    than ``duration_s`` (or exactly ``0.0`` when ``duration_s`` is ``0``).
    """
    if duration_s < 0 or fps <= 0 or min_frames < 1 or max_frames < min_frames:
        raise ValidationError(
            'invalid frame sampling parameters: '
            f'duration_s={duration_s}, fps={fps}, min_frames={min_frames}, '
            f'max_frames={max_frames}'
        )
    uniform = _at_fps(duration_s, fps)
    if len(uniform) > max_frames:
        return _subsample(uniform, max_frames)
    if len(uniform) < min_frames:
        return _uniform_timestamps(duration_s, min_frames)
    return uniform


def judge_frame_timestamps(duration_s: float) -> tuple[float, ...]:
    """Frame times the official judge extracts: ``0, 1, 2, ...`` at 1 fps, capped at 10.

    Unlike :func:`frame_timestamps`, short clips are never padded: a clip shorter
    than ``JUDGE_MAX_FRAMES`` seconds yields fewer than ``JUDGE_MAX_FRAMES`` frames,
    matching the official judge, which only ever looks at the first ten seconds.
    """
    fps = 1.0 / JUDGE_FRAME_INTERVAL_SECONDS
    frames = _at_fps(max(duration_s, 0.0), fps)[:JUDGE_MAX_FRAMES]
    if not frames and duration_s > 0.0:
        return (0.0,)  # the official script always keeps frame 0
    return frames


def _at_fps(duration_s: float, fps: float) -> tuple[float, ...]:
    """Timestamps ``0, 1/fps, 2/fps, ...`` while strictly less than ``duration_s``."""
    step = 1.0 / fps
    count = max(0, math.floor(duration_s * fps + _EPSILON))
    return tuple(i * step for i in range(count))


def _uniform_timestamps(duration_s: float, n: int) -> tuple[float, ...]:
    """``n`` timestamps evenly spaced over ``[0, duration_s)``, starting at 0."""
    if n <= 0:
        return ()
    step = duration_s / n
    return tuple(i * step for i in range(n))


def _subsample(seq: tuple[float, ...], k: int) -> tuple[float, ...]:
    """``k`` uniformly spaced elements of ``seq`` (``k`` strictly less than ``len(seq)``)."""
    if k <= 1:
        return (seq[0],)
    last = len(seq) - 1
    return tuple(seq[round(i * last / (k - 1))] for i in range(k))


def probe_duration(path: Path) -> float:
    """Duration in seconds via ``ffprobe``. Raises ``ValidationError`` on any failure."""
    if not path.exists():
        raise ValidationError(f'media file not found: {path}')
    result = _run(
        [
            'ffprobe',
            '-v',
            'error',
            '-show_entries',
            'format=duration',
            '-of',
            'default=noprint_wrappers=1:nokey=1',
            str(path),
        ],
        timeout=_PROBE_TIMEOUT_S,
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise ValidationError(
            f'ffprobe returned no duration for {path}: {result.stdout!r}'
        ) from exc


def extract_frames(
    path: Path, timestamps: Sequence[float], out_dir: Path, prefix: str
) -> tuple[Path, ...]:
    """Write one JPEG per timestamp to ``out_dir``, skipping files that already exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, timestamp in enumerate(timestamps):
        out_path = out_dir / f'{prefix}_{index:04d}.jpg'
        if not out_path.exists():
            _extract_one_frame(path, timestamp, out_path)
        if not out_path.exists():
            # ffmpeg writes nothing when a seek lands past the last frame of a
            # clip a few frames long; repeat the previous frame so the entry
            # keeps ``min_frames`` and every listed path exists.
            if not outputs:
                raise ValidationError(f'no frame could be extracted from {path}')
            shutil.copyfile(outputs[-1], out_path)
        outputs.append(out_path)
    return tuple(outputs)


def _extract_one_frame(path: Path, timestamp: float, out_path: Path) -> None:
    _run(
        [
            'ffmpeg',
            '-y',
            '-ss',
            f'{timestamp:.6f}',
            '-i',
            str(path),
            '-frames:v',
            '1',
            '-vf',
            _FRAME_SCALE_FILTER,
            '-q:v',
            '2',
            str(out_path),
        ],
        timeout=_EXTRACT_TIMEOUT_S,
    )


def extract_audio_track(path: Path, wav_path: Path) -> Path | None:
    """16 kHz mono PCM WAV at ``wav_path``, or ``None`` when ``path`` has no audio."""
    if not _has_audio_stream(path):
        return None
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            'ffmpeg',
            '-y',
            '-i',
            str(path),
            '-vn',
            '-acodec',
            'pcm_s16le',
            '-ar',
            '16000',
            '-ac',
            '1',
            str(wav_path),
        ],
        timeout=_EXTRACT_TIMEOUT_S,
    )
    return wav_path


def _has_audio_stream(path: Path) -> bool:
    result = _run(
        [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'a',
            '-show_entries',
            'stream=index',
            '-of',
            'csv=p=0',
            str(path),
        ],
        timeout=_PROBE_TIMEOUT_S,
    )
    return bool(result.stdout.strip())


def _run(cmd: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg/ffprobe command, raising ``ValidationError`` on any failure."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValidationError(f'command failed: {" ".join(cmd)}: {exc}') from exc
