"""Tests for emoharness.target.config."""

from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from emoharness.common.config import load_model
from emoharness.common.errors import ValidationError
from emoharness.target.config import (
    CallBudget,
    DecodingConfig,
    TargetConfig,
    VideoConfig,
)
from emoharness.target.pricing import Pricing

_MINIMAL_YAML = """
model_id: Qwen/Qwen3-VL-8B-Instruct
endpoint: {endpoint}
decoding: {{temperature: 0.0, top_p: 1.0, seed: 0, max_tokens: 1024}}
video: {{fps: 2.0, min_frames: 4, max_frames: 32}}
call_budget: {{expected: 4, hard_cap: 6, retries_per_stage: 1}}
pricing_usd_per_million: {{input: 0.0, output: 0.0}}
"""


def test_decoding_config_defaults_match_target_request_defaults() -> None:
    decoding = DecodingConfig()

    assert decoding.temperature == 0.0
    assert decoding.top_p == 1.0
    assert decoding.seed == 0
    assert decoding.max_tokens == 1024


def test_video_config_accepts_valid_bounds() -> None:
    video = VideoConfig(fps=2.0, min_frames=4, max_frames=32)

    assert video.min_frames < video.max_frames


def test_video_config_sends_video_unless_frames_are_asked_for() -> None:
    assert VideoConfig(fps=2.0, min_frames=4, max_frames=32).frames_as_images is False
    frames = VideoConfig(
        fps=2.0, min_frames=4, max_frames=32, frames_as_images=True
    )
    assert frames.frames_as_images is True


def test_target_config_defaults_to_a_keyless_openai_compatible_endpoint() -> None:
    config = TargetConfig(
        model_id='m',
        endpoint='http://localhost:8000/v1',
        decoding=DecodingConfig(),
        video=VideoConfig(fps=2.0, min_frames=4, max_frames=32),
        call_budget=CallBudget(expected=4, hard_cap=6, retries_per_stage=1),
    )

    assert config.provider == 'openai_compat'
    assert config.api_key_env is None


def test_video_config_rejects_max_below_min() -> None:
    with pytest.raises(pydantic.ValidationError):
        VideoConfig(fps=2.0, min_frames=32, max_frames=4)


def test_call_budget_rejects_hard_cap_below_expected() -> None:
    with pytest.raises(pydantic.ValidationError):
        CallBudget(expected=6, hard_cap=4, retries_per_stage=1)


def test_target_config_is_frozen() -> None:
    config = TargetConfig(
        model_id='m',
        endpoint='http://localhost:8000/v1',
        decoding=DecodingConfig(),
        video=VideoConfig(fps=2.0, min_frames=4, max_frames=32),
        call_budget=CallBudget(expected=4, hard_cap=6, retries_per_stage=1),
        pricing=Pricing(input_per_million=0.0, output_per_million=0.0),
    )

    with pytest.raises(pydantic.ValidationError):
        config.model_id = 'other'  # type: ignore[misc]


def test_target_config_rejects_non_http_endpoint() -> None:
    with pytest.raises(pydantic.ValidationError):
        TargetConfig(
            model_id='m',
            endpoint='not-a-url',
            decoding=DecodingConfig(),
            video=VideoConfig(fps=2.0, min_frames=4, max_frames=32),
            call_budget=CallBudget(expected=4, hard_cap=6, retries_per_stage=1),
            pricing=Pricing(input_per_million=0.0, output_per_million=0.0),
        )


def test_load_model_reads_repo_qwen3vl_config(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('VLLM_ENDPOINT', raising=False)
    path = repo_root / 'configs' / 'target' / 'qwen3vl_8b_instruct.yaml'

    config = load_model(path, TargetConfig)

    assert config.model_id == 'Qwen/Qwen3-VL-8B-Instruct'
    assert config.endpoint == 'http://127.0.0.1:8000/v1'
    assert config.decoding == DecodingConfig(
        temperature=0.0, top_p=1.0, seed=0, max_tokens=1024
    )
    assert config.video == VideoConfig(fps=2.0, min_frames=4, max_frames=32)
    assert config.call_budget == CallBudget(expected=4, hard_cap=6, retries_per_stage=1)
    assert config.pricing == Pricing(input_per_million=0.0, output_per_million=0.0)


def test_load_model_expands_endpoint_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('EMOHARNESS_TEST_ENDPOINT', 'http://gpu-box:9001/v1')
    path = tmp_path / 'target.yaml'
    path.write_text(
        _MINIMAL_YAML.format(endpoint='${EMOHARNESS_TEST_ENDPOINT:-http://x/v1}'),
        encoding='utf-8',
    )

    config = load_model(path, TargetConfig)

    assert config.endpoint == 'http://gpu-box:9001/v1'


def test_load_model_rejects_invalid_endpoint_after_expansion(tmp_path: Path) -> None:
    path = tmp_path / 'target.yaml'
    path.write_text(_MINIMAL_YAML.format(endpoint='not-a-url'), encoding='utf-8')

    with pytest.raises(ValidationError, match='endpoint'):
        load_model(path, TargetConfig)


def test_load_model_missing_key_raises_validation_error(tmp_path: Path) -> None:
    path = tmp_path / 'target.yaml'
    path.write_text('model_id: m\nendpoint: http://x/v1\n', encoding='utf-8')

    with pytest.raises(ValidationError):
        load_model(path, TargetConfig)


def test_load_model_rejects_non_mapping_document(tmp_path: Path) -> None:
    path = tmp_path / 'target.yaml'
    path.write_text('- just\n- a\n- list\n', encoding='utf-8')

    with pytest.raises(ValidationError):
        load_model(path, TargetConfig)


def test_pricing_accepts_the_yaml_aliases() -> None:
    pricing = Pricing.model_validate({'input': 1.5, 'output': 3.0})

    assert pricing.input_per_million == 1.5
    assert pricing.output_per_million == 3.0


def test_pricing_defaults_to_free_when_the_block_is_absent(tmp_path: Path) -> None:
    path = tmp_path / 'target.yaml'
    path.write_text(
        _MINIMAL_YAML.format(endpoint='http://x/v1').replace(
            'pricing_usd_per_million: {input: 0.0, output: 0.0}', ''
        ),
        encoding='utf-8',
    )

    config = load_model(path, TargetConfig)

    assert config.pricing == Pricing(input_per_million=0.0, output_per_million=0.0)
