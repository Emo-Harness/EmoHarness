"""Tests for emoharness.target.cache."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from emoharness.target.base import TargetRequest, TargetResponse
from emoharness.target.cache import CachedTarget, CallCache, request_key
from emoharness.target.fake import FakeTarget


def test_request_key_is_stable_for_identical_requests(
    make_video: Callable[..., Path],
) -> None:
    video = make_video()
    request = TargetRequest(prompt='describe the clip', video_path=str(video))

    assert request_key('model-a', request) == request_key('model-a', request)


def test_request_key_same_media_different_paths_match(
    tmp_path: Path, make_video: Callable[..., Path]
) -> None:
    original = make_video('clip_a.mp4')
    copy_path = tmp_path / 'clip_b.mp4'
    copy_path.write_bytes(original.read_bytes())

    key_a = request_key('model-a', TargetRequest(prompt='p', video_path=str(original)))
    key_b = request_key('model-a', TargetRequest(prompt='p', video_path=str(copy_path)))

    assert key_a == key_b


def test_request_key_different_media_differ(
    tmp_path: Path, make_video: Callable[..., Path]
) -> None:
    video_a = make_video('a.mp4', extra_bytes=32)
    video_b = make_video('b.mp4', extra_bytes=64)

    key_a = request_key('model-a', TargetRequest(prompt='p', video_path=str(video_a)))
    key_b = request_key('model-a', TargetRequest(prompt='p', video_path=str(video_b)))

    assert key_a != key_b


def test_request_key_differs_for_different_prompt() -> None:
    key_a = request_key('model-a', TargetRequest(prompt='prompt one'))
    key_b = request_key('model-a', TargetRequest(prompt='prompt two'))

    assert key_a != key_b


def test_request_key_differs_for_different_model_id() -> None:
    request = TargetRequest(prompt='same prompt')

    assert request_key('model-a', request) != request_key('model-b', request)


def test_request_key_same_images_different_paths_match(
    tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    original = make_image('img_a.jpg')
    copy_path = tmp_path / 'img_b.jpg'
    copy_path.write_bytes(original.read_bytes())

    request_a = TargetRequest(prompt='p', image_paths=(str(original),))
    request_b = TargetRequest(prompt='p', image_paths=(str(copy_path),))

    assert request_key('model-a', request_a) == request_key('model-a', request_b)


def test_call_cache_get_missing_returns_none(tmp_path: Path) -> None:
    cache = CallCache(tmp_path)

    assert cache.get('deadbeef') is None


def test_call_cache_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = CallCache(tmp_path)
    response = TargetResponse(
        text='<answer>joy</answer>',
        model_id='model-a',
        input_tokens=10,
        output_tokens=5,
        latency_s=0.25,
    )

    cache.put('abcd1234', response)
    loaded = cache.get('abcd1234')

    assert loaded is not None
    assert loaded.text == response.text
    assert loaded.input_tokens == response.input_tokens
    assert loaded.cached is True


def test_call_cache_shards_files_by_key_prefix(tmp_path: Path) -> None:
    cache = CallCache(tmp_path)
    response = TargetResponse(
        text='t', model_id='m', input_tokens=1, output_tokens=1, latency_s=0.0
    )

    cache.put('ab1234', response)

    assert (tmp_path / 'ab' / 'ab1234.json').exists()


def test_cached_target_starts_with_zero_hits_and_misses(tmp_path: Path) -> None:
    cached = CachedTarget(FakeTarget(), CallCache(tmp_path))

    assert cached.hits == 0
    assert cached.misses == 0


def test_cached_target_hit_avoids_calling_inner(tmp_path: Path) -> None:
    fake = FakeTarget(default_text='<answer>neutral</answer>')
    cache = CallCache(tmp_path)
    cached = CachedTarget(fake, cache)
    request = TargetRequest(prompt='hello there')

    first = cached.generate(request)
    second = cached.generate(request)

    assert fake.call_count == 1
    assert fake.requests == (request,)
    assert cached.misses == 1
    assert cached.hits == 1
    assert first.cached is False
    assert second.cached is True
    assert first.text == second.text


def test_cached_target_miss_for_different_requests(tmp_path: Path) -> None:
    fake = FakeTarget(default_text='<answer>neutral</answer>')
    cache = CallCache(tmp_path)
    cached = CachedTarget(fake, cache)

    cached.generate(TargetRequest(prompt='one'))
    cached.generate(TargetRequest(prompt='two'))

    assert fake.call_count == 2
    assert cached.misses == 2
    assert cached.hits == 0


def test_cached_target_model_id_matches_inner(tmp_path: Path) -> None:
    fake = FakeTarget(model_id='fake-123')
    cached = CachedTarget(fake, CallCache(tmp_path))

    assert cached.model_id == 'fake-123'
