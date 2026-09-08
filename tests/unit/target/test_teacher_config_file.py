"""The committed teacher configs: the OpenAI default and the Gemini alternative."""

from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.common.config import load_model
from emoharness.target.config import TargetConfig
from emoharness.target.gemini import gemini_config_from


def test_committed_teacher_config_validates(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OPENAI_ENDPOINT', raising=False)

    config = load_model(
        repo_root / 'configs' / 'target' / 'teacher_gpt_5_6_luna.yaml', TargetConfig
    )

    assert config.provider == 'openai_compat'
    assert config.model_id == 'gpt-5.6-luna'
    assert config.endpoint == 'https://api.openai.com/v1'
    assert config.api_key_env == 'OPENAI_API_KEY'
    assert config.decoding.temperature == 0.0
    assert config.video.frames_as_images is True
    assert config.pricing.input_per_million > 0.0


def test_committed_gemini_teacher_config_maps_onto_the_gemini_client(
    repo_root: Path,
) -> None:
    config = load_model(
        repo_root / 'configs' / 'target' / 'gemini_3_1_pro.yaml', TargetConfig
    )

    assert config.provider == 'gemini'

    gemini = gemini_config_from(config)

    assert gemini.model_id == config.model_id
    assert gemini.model_id.startswith('gemini')
    assert gemini.temperature == config.decoding.temperature
    assert gemini.api_key_env == 'GEMINI_API_KEY'
    assert gemini.pricing == config.pricing


def test_a_gemini_config_without_a_key_env_falls_back_to_the_default(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('VLLM_ENDPOINT', raising=False)
    config = load_model(
        repo_root / 'configs' / 'target' / 'qwen3vl_8b_instruct.yaml', TargetConfig
    )

    assert config.api_key_env is None
    assert gemini_config_from(config).api_key_env == 'GEMINI_API_KEY'


def test_committed_cross_target_config_is_a_qwen3_model(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('VLLM_ENDPOINT', raising=False)

    config = load_model(
        repo_root / 'configs' / 'target' / 'qwen3vl_4b_instruct.yaml', TargetConfig
    )

    assert config.model_id == 'Qwen/Qwen3-VL-4B-Instruct'
    assert config.endpoint == 'http://127.0.0.1:8000/v1'
    assert config.video.frames_as_images is False
