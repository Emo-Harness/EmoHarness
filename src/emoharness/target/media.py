"""Media encoding for OpenAI-style chat content parts.

Images and video are read from disk and either inlined as base64 ``data:`` URIs
or referenced by suffix-derived MIME type; ``content_parts`` assembles a user
message's content list in the order every client in this package expects:
text, then the video (if any), then images.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from emoharness.common.errors import ValidationError
from emoharness.target.base import TargetRequest

_IMAGE_MIME_BY_SUFFIX: Final[Mapping[str, str]] = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
}
_VIDEO_MIME_BY_SUFFIX: Final[Mapping[str, str]] = {
    '.mp4': 'video/mp4',
    '.webm': 'video/webm',
    '.mov': 'video/quicktime',
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
