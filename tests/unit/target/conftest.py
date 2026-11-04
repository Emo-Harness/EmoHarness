"""Shared fixtures for emoharness.target unit tests. Media are synthetic bytes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from emoharness.target.config import (
    CallBudget,
    DecodingConfig,
    TargetConfig,
    VideoConfig,
)
from emoharness.target.pricing import Pricing

_JPEG_MAGIC = b'\xff\xd8\xff\xe0'
_MP4_MAGIC = b'\x00\x00\x00\x18ftypmp42'


@pytest.fixture
def target_config() -> TargetConfig:
    """A TargetConfig matching configs/target/qwen3vl_8b_instruct.yaml, priced non-zero."""
    return TargetConfig(
        model_id='Qwen/Qwen3-VL-8B-Instruct',
        endpoint='http://127.0.0.1:8000/v1',
        decoding=DecodingConfig(temperature=0.0, top_p=1.0, seed=0, max_tokens=64),
        video=VideoConfig(fps=2.0, min_frames=4, max_frames=32),
        call_budget=CallBudget(expected=4, hard_cap=6, retries_per_stage=1),
        pricing=Pricing(input_per_million=1.0, output_per_million=2.0),
    )


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    """Factory for a small synthetic .mp4 file; each call may give a distinct name."""

    def _make(name: str = 'clip.mp4', extra_bytes: int = 32) -> Path:
        path = tmp_path / name
        path.write_bytes(_MP4_MAGIC + b'\x00' * extra_bytes)
        return path

    return _make


@pytest.fixture
def make_image(tmp_path: Path) -> Callable[..., Path]:
    """Factory for a small synthetic .jpg file; each call may give a distinct name."""

    def _make(name: str = 'frame.jpg', extra_bytes: int = 16) -> Path:
        path = tmp_path / name
        path.write_bytes(_JPEG_MAGIC + b'\x00' * extra_bytes)
        return path

    return _make
