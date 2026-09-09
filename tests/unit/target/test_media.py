"""Tests for emoharness.target.media."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path

import pytest

from emoharness.common.errors import ValidationError
from emoharness.target.base import TargetRequest
from emoharness.target.media import (
    content_parts,
    encode_image_data_uri,
    encode_video_data_uri,
    image_mime_type,
    video_mime_type,
)


def test_image_mime_type_by_suffix() -> None:
    assert image_mime_type('frame.jpg') == 'image/jpeg'
    assert image_mime_type('frame.jpeg') == 'image/jpeg'
    assert image_mime_type('frame.PNG') == 'image/png'


def test_image_mime_type_rejects_unknown_suffix() -> None:
    with pytest.raises(ValidationError):
        image_mime_type('frame.gif')


def test_video_mime_type_by_suffix() -> None:
    assert video_mime_type('clip.mp4') == 'video/mp4'
    assert video_mime_type('clip.webm') == 'video/webm'
    assert video_mime_type('clip.mov') == 'video/quicktime'


def test_video_mime_type_rejects_unknown_suffix() -> None:
    with pytest.raises(ValidationError):
        video_mime_type('clip.flv')


def test_video_mime_type_knows_the_benchmark_containers() -> None:
    """MME-Emotion ships .mp4, .avi, and .mkv clips; all three must encode."""
    assert video_mime_type('clip.avi') == 'video/x-msvideo'
    assert video_mime_type('clip.mkv') == 'video/x-matroska'


def test_encode_image_data_uri_round_trips_bytes(
    make_image: Callable[..., Path],
) -> None:
    image = make_image()
    original = image.read_bytes()

    uri = encode_image_data_uri(image)

    assert uri.startswith('data:image/jpeg;base64,')
    encoded = uri.split(',', 1)[1]
    assert base64.b64decode(encoded) == original


def test_encode_video_data_uri_round_trips_bytes(
    make_video: Callable[..., Path],
) -> None:
    video = make_video()
    original = video.read_bytes()

    uri = encode_video_data_uri(video)

    assert uri.startswith('data:video/mp4;base64,')
    encoded = uri.split(',', 1)[1]
    assert base64.b64decode(encoded) == original


def test_content_parts_text_only_request() -> None:
    request = TargetRequest(prompt='hello there')

    parts = content_parts(request, video_mode='data_uri')

    assert parts == [{'type': 'text', 'text': 'hello there'}]


def test_content_parts_orders_text_then_video_then_images(
    make_video: Callable[..., Path], make_image: Callable[..., Path]
) -> None:
    video = make_video()
    image_a = make_image('a.jpg')
    image_b = make_image('b.jpg')
    request = TargetRequest(
        prompt='describe',
        video_path=str(video),
        image_paths=(str(image_a), str(image_b)),
    )

    parts = content_parts(request, video_mode='data_uri')

    assert [part['type'] for part in parts] == [
        'text',
        'video_url',
        'image_url',
        'image_url',
    ]
    assert parts[0]['text'] == 'describe'
    assert parts[1]['video_url']['url'].startswith('data:video/mp4;base64,')
    assert parts[2]['image_url']['url'].startswith('data:image/jpeg;base64,')


def test_content_parts_data_uri_mode_inlines_video(
    make_video: Callable[..., Path],
) -> None:
    video = make_video()
    request = TargetRequest(prompt='p', video_path=str(video))

    parts = content_parts(request, video_mode='data_uri')

    assert parts[1]['video_url']['url'].startswith('data:video/')


def test_content_parts_file_url_mode_uses_file_scheme(
    make_video: Callable[..., Path],
) -> None:
    video = make_video()
    request = TargetRequest(prompt='p', video_path=str(video))

    parts = content_parts(request, video_mode='file_url')

    url = parts[1]['video_url']['url']
    assert url.startswith('file://')
    assert url.endswith(video.name)


def test_content_parts_no_video_skips_video_part(
    make_image: Callable[..., Path],
) -> None:
    image = make_image()
    request = TargetRequest(prompt='p', image_paths=(str(image),))

    parts = content_parts(request, video_mode='data_uri')

    assert [part['type'] for part in parts] == ['text', 'image_url']
