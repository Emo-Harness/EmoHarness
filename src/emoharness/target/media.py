"""Media encoding for OpenAI-style chat content parts.

Images and video are read from disk and either inlined as base64 ``data:`` URIs
or referenced by suffix-derived MIME type; ``content_parts`` assembles a user
message's content list in the order every client in this package expects:
text, then the video (if any), then images.
"""

from __future__ import annotations

import base64
import subprocess
import tempfile
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

from emoharness.common.errors import ValidationError
from emoharness.common.hashing import sha256_file_cached
from emoharness.common.logging import get_logger
from emoharness.target.base import TargetRequest

_LOGGER = get_logger(__name__)

_IMAGE_MIME_BY_SUFFIX: Final[Mapping[str, str]] = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
}
_VIDEO_MIME_BY_SUFFIX: Final[Mapping[str, str]] = {
    '.mp4': 'video/mp4',
    '.webm': 'video/webm',
    '.mov': 'video/quicktime',
    # MME-Emotion ships ER_Wild as .avi and a few clips as .mkv; vLLM decodes
    # a data: URI by content, so the type only has to say "video".
    '.avi': 'video/x-msvideo',
    '.mkv': 'video/x-matroska',
}


def image_mime_type(path: str | Path) -> str:
    """Return the MIME type for a supported image suffix, or raise ValidationError."""
    suffix = Path(path).suffix.lower()
    mime = _IMAGE_MIME_BY_SUFFIX.get(suffix)
    if mime is None:
        raise ValidationError(f'unsupported image suffix: {suffix!r}')
    return mime


def video_mime_type(path: str | Path) -> str:
    """Return the MIME type for a supported video suffix, or raise ValidationError."""
    suffix = Path(path).suffix.lower()
    mime = _VIDEO_MIME_BY_SUFFIX.get(suffix)
    if mime is None:
        raise ValidationError(f'unsupported video suffix: {suffix!r}')
    return mime


def encode_image_data_uri(path: str | Path) -> str:
    """Return a base64 ``data:`` URI for a JPEG/PNG image."""
    mime = image_mime_type(path)
    encoded = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    return f'data:{mime};base64,{encoded}'


def encode_video_data_uri(path: str | Path) -> str:
    """Return a base64 ``data:`` URI for a video file."""
    mime = video_mime_type(path)
    encoded = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    return f'data:{mime};base64,{encoded}'


#: Containers the server's OpenCV loader counts frames in reliably. Matroska is
#: not one of them: CAP_PROP_FRAME_COUNT over-reports and vLLM asserts on it.
_PLAYABLE_SUFFIXES: Final = frozenset({'.mp4', '.webm', '.mov'})
#: Widest source the server is handed. Its frontend decodes every sampled frame
#: and tensorises it at source resolution before the processor's smart resize,
#: in host RAM: 32 frames of 1080p is 0.8 GB per request, which is an OOM on an
#: 8 GB box. The processor's own total-pixel budget shrinks 32 frames far below
#: this anyway, so the cap costs no information the model would have seen.
MAX_SOURCE_WIDTH: Final = 640
REMUXED_NAME: Final = 'video.mp4'
_REMUX_TIMEOUT_S: Final = 300.0
_PROBE_TIMEOUT_S: Final = 30.0


def default_remux_dir() -> Path:
    """Where remuxed copies go when a client is not given a media directory."""
    return Path(tempfile.gettempdir()) / 'emoharness-remux'


def playable_video(path: str | Path, media_dir: Path) -> Path:
    """``path`` itself when the server can take it as is, else a cached mp4 copy.

    A clip is copied when its container is not in ``_PLAYABLE_SUFFIXES`` or it
    is wider than ``MAX_SOURCE_WIDTH``. The copy lives at
    ``<media_dir>/<sha256 of the source>/video.mp4``: a stream copy when only
    the container is at fault, a downscale to ``MAX_SOURCE_WIDTH`` otherwise.
    Callers keep hashing the *source* for cache keys, so this never changes
    what a request is remembered as.
    """
    source = Path(path)
    wide = _width(str(source)) > MAX_SOURCE_WIDTH
    if source.suffix.lower() in _PLAYABLE_SUFFIXES and not wide:
        return source
    target = media_dir / sha256_file_cached(source) / REMUXED_NAME
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.partial.mp4')
    # Video stream only: the server never hears the audio, and MME ships a few
    # clips whose 6-channel AAC or data tracks no encoder or muxer will take.
    if wide:
        ok = _ffmpeg(source, partial, _downscale_args())
    else:
        ok = _ffmpeg(source, partial, ('-map', '0:v:0', '-c:v', 'copy', '-an'))
        if not ok or not partial.exists():
            _LOGGER.warning('stream copy failed for %s; re-encoding', source.name)
            ok = _ffmpeg(source, partial, _downscale_args())
    if not ok or not partial.exists():
        raise ValidationError(f'cannot convert {source} to mp4')
    partial.replace(target)
    return target


def _downscale_args() -> tuple[str, ...]:
    return (
        '-map',
        '0:v:0',
        '-vf',
        f"scale='min({MAX_SOURCE_WIDTH},iw)':-2",
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-crf',
        '20',
        '-pix_fmt',
        'yuv420p',
        '-an',
    )


@lru_cache(maxsize=65536)
def _width(path: str) -> int:
    """The first video stream's width in pixels; 0 when ffprobe cannot tell.

    Memoised per path: every Target call would otherwise spawn an ffprobe.
    """
    cmd = [
        'ffprobe',
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=width',
        '-of',
        'csv=p=0',
        path,
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=_PROBE_TIMEOUT_S
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def _ffmpeg(source: Path, target: Path, codec_args: tuple[str, ...]) -> bool:
    cmd = [
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        str(source),
        *codec_args,
        '-movflags',
        '+faststart',
        str(target),
    ]
    try:
        subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=_REMUX_TIMEOUT_S
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        target.unlink(missing_ok=True)
        return False
    return True


def _video_url(path: str, video_mode: Literal['data_uri', 'file_url']) -> str:
    if video_mode == 'data_uri':
        return encode_video_data_uri(path)
    return Path(path).resolve().as_uri()


def content_parts(
    request: TargetRequest,
    video_mode: Literal['data_uri', 'file_url'],
    frames_as_images: bool = False,
) -> list[dict[str, Any]]:
    """Build OpenAI-style user content parts: text, then video, then images.

    ``frames_as_images`` drops the video part, for an endpoint that accepts
    images but no video: such a request carries the sampled frames in
    ``image_paths`` and they are sent as ``image_url`` parts instead.
    """
    parts: list[dict[str, Any]] = [{'type': 'text', 'text': request.prompt}]
    if request.video_path and not frames_as_images:
        parts.append(
            {
                'type': 'video_url',
                'video_url': {'url': _video_url(request.video_path, video_mode)},
            }
        )
    for image_path in request.image_paths:
        parts.append(
            {
                'type': 'image_url',
                'image_url': {'url': encode_image_data_uri(image_path)},
            }
        )
    return parts
