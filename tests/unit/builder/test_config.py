from __future__ import annotations

from pathlib import Path

import pytest

from emoharness.builder.config import BuilderConfig, BuilderPricing
from emoharness.common.config import load_model
from emoharness.common.constants import (
    EARLY_STOP_ROUNDS,
    JUDGE_MAX_FRAMES,
    MAX_CANDIDATES_PER_ROUND,
    MAX_ROUNDS_PER_STAGE,
    REGRESSION_RATE_LIMIT,
)
from emoharness.common.errors import ValidationError


def test_defaults_come_from_constants() -> None:
    config = BuilderConfig()
    assert config.max_candidates_per_round == MAX_CANDIDATES_PER_ROUND
    assert config.max_rounds_per_stage == MAX_ROUNDS_PER_STAGE
    assert config.early_stop_rounds == EARLY_STOP_ROUNDS
    assert config.packet.max_frames_per_sample == JUDGE_MAX_FRAMES
    assert config.gates.regression_rate_limit == REGRESSION_RATE_LIMIT


def test_defaults_name_the_openai_teacher() -> None:
    config = BuilderConfig()
    assert config.provider == 'openai'
    assert config.api_key_env == 'OPENAI_API_KEY'
    assert config.endpoint == 'https://api.openai.com/v1'
    assert config.model_id == 'gpt-5.6-luna'


def test_load_model_reads_the_committed_builder_config(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OPENAI_ENDPOINT', raising=False)
    path = repo_root / 'configs' / 'builder' / 'teacher_gpt_5_6_luna.yaml'
    config = load_model(path, BuilderConfig)
    assert config.provider == 'openai'
    assert config.model_id == 'gpt-5.6-luna'
    assert config.api_key_env == 'OPENAI_API_KEY'
    assert config.endpoint == 'https://api.openai.com/v1'
    assert config.temperature == 0.0
    assert config.max_candidates_per_round == 4
    assert config.packet.max_failures_per_cluster == 5
    assert config.packet.max_keyframe_samples == 20
    assert config.packet.target_input_tokens == 100_000
    assert config.gates.gate_sample_size == 200
    assert config.pricing.input_usd_per_million == 2.0
    assert config.pricing.output_usd_per_million == 12.0


def test_load_model_reads_the_alternative_gemini_builder_config(
    repo_root: Path,
) -> None:
    path = repo_root / 'configs' / 'builder' / 'gemini_3_1_pro.yaml'
    config = load_model(path, BuilderConfig)
    assert config.provider == 'gemini'
    assert config.model_id == 'gemini-3.1-pro'
    assert config.api_key_env == 'GEMINI_API_KEY'


def test_an_unknown_provider_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / 'builder.yaml'
    path.write_text('provider: telepathy\n', encoding='utf-8')
    with pytest.raises(ValidationError, match='provider'):
        load_model(path, BuilderConfig)


def test_pricing_costs_a_call() -> None:
    pricing = BuilderPricing(input_usd_per_million=2.0, output_usd_per_million=12.0)
    assert pricing.cost_usd(1_000_000, 0) == pytest.approx(2.0)
    assert pricing.cost_usd(0, 500_000) == pytest.approx(6.0)
    assert pricing.cost_usd(0, 0) == 0.0


def test_pricing_accepts_the_yaml_aliases() -> None:
    pricing = BuilderPricing.model_validate({'input': 1.5, 'output': 3.0})
    assert pricing.input_usd_per_million == 1.5
    assert pricing.output_usd_per_million == 3.0


def test_load_model_rejects_a_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / 'builder.yaml'
    path.write_text('- one\n- two\n', encoding='utf-8')
    with pytest.raises(ValidationError, match=str(path)):
        load_model(path, BuilderConfig)


def test_load_model_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / 'builder.yaml'
    path.write_text('model_id: m\nunexpected: 1\n', encoding='utf-8')
    with pytest.raises(ValidationError, match='unexpected'):
        load_model(path, BuilderConfig)


def test_load_model_rejects_too_many_candidates(tmp_path: Path) -> None:
    path = tmp_path / 'builder.yaml'
    path.write_text('max_candidates_per_round: 99\n', encoding='utf-8')
    with pytest.raises(ValidationError, match='max_candidates_per_round'):
        load_model(path, BuilderConfig)


def test_config_is_frozen() -> None:
    config = BuilderConfig()
    with pytest.raises(Exception, match='frozen'):
        config.model_id = 'other'  # type: ignore[misc]
