"""The visual stage's frame budget scales with clip length, and only when a bundle asks."""

from __future__ import annotations

import pytest

from emoharness.harness.bundle_models import VideoFramesConfig
from emoharness.target.base import TargetRequest
from emoharness.target.cache import request_key


@pytest.mark.parametrize(
    ('duration', 'frames'), [(3.0, 32), (45.2, 46), (60.0, 60), (180.0, 64), (0.0, 32)]
)
def test_frames_follow_duration_within_bounds(duration: float, frames: int) -> None:
    config = VideoFramesConfig(min_fps=1.0, min_frames=32, max_frames=64)
    assert config.frames_for(duration) == frames


def test_bounds_are_checked() -> None:
    with pytest.raises(ValueError, match='max_frames'):
        VideoFramesConfig(min_fps=1.0, min_frames=40, max_frames=32)
    with pytest.raises(ValueError):
        VideoFramesConfig(min_fps=1.0, min_frames=1, max_frames=65)


def test_the_cache_key_moves_only_when_a_frame_cap_is_set() -> None:
    base = TargetRequest(prompt='p')
    legacy = request_key('m', base)
    assert (
        request_key('m', base.model_copy(update={'video_max_frames': None})) == legacy
    )
    capped = request_key('m', base.model_copy(update={'video_max_frames': 60}))
    assert capped != legacy
    assert capped != request_key('m', base.model_copy(update={'video_max_frames': 64}))
