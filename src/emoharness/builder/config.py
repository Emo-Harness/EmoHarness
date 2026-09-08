"""Builder configuration, mirroring ``configs/builder/*.yaml`` (ADR-0005, ADR-0006).

Defaults come from :mod:`emoharness.common.constants` wherever the protocol fixes a
value. Three numbers have no constant because they are budget knobs rather than
protocol facts: the per-cluster failure sample, the packet token target, and the
Builder's price list. Their defaults mirror
``configs/builder/teacher_gpt_5_6_luna.yaml`` and are meant to be overridden by
the committed config, which is read with
:func:`emoharness.common.config.load_model`.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from emoharness.common.constants import (
    EARLY_STOP_ROUNDS,
    GATE_SAMPLE_SIZE,
    JUDGE_MAX_FRAMES,
    MAX_CANDIDATES_PER_ROUND,
    MAX_ROUNDS_PER_STAGE,
    REASONING_GATE_FLOOR_POINTS,
    REGRESSION_RATE_LIMIT,
)

#: Which client answers a packet: the OpenAI chat-completions API or Gemini.
BuilderProvider = Literal['openai', 'gemini']

DEFAULT_MODEL_ID: Final = 'gpt-5.6-luna'
DEFAULT_API_KEY_ENV: Final = 'OPENAI_API_KEY'
DEFAULT_ENDPOINT: Final = 'https://api.openai.com/v1'
DEFAULT_FAILURES_PER_CLUSTER: Final = 5
DEFAULT_KEYFRAME_SAMPLES: Final = 20
DEFAULT_TARGET_INPUT_TOKENS: Final = 100_000
DEFAULT_INPUT_USD_PER_MILLION: Final = 2.0
DEFAULT_OUTPUT_USD_PER_MILLION: Final = 12.0
_TOKENS_PER_MILLION: Final = 1_000_000.0


class PacketLimits(BaseModel):
    """How much of the failure evidence one diagnosis packet may carry."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    max_failures_per_cluster: int = Field(default=DEFAULT_FAILURES_PER_CLUSTER, ge=1)
    max_keyframe_samples: int = Field(default=DEFAULT_KEYFRAME_SAMPLES, ge=0)
    max_frames_per_sample: int = Field(default=JUDGE_MAX_FRAMES, ge=1)
    target_input_tokens: int = Field(default=DEFAULT_TARGET_INPUT_TOKENS, ge=1)


class GateThresholds(BaseModel):
    """The promotion thresholds an experiment may tune (ADR-0005, ADR-0007)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    regression_rate_limit: float = Field(default=REGRESSION_RATE_LIMIT, ge=0.0, le=1.0)
    reasoning_floor_points: float = Field(default=REASONING_GATE_FLOOR_POINTS, ge=0.0)
    gate_sample_size: int = Field(default=GATE_SAMPLE_SIZE, ge=1)


class BuilderPricing(BaseModel):
    """Price list used to fill each Builder call's ledger entry."""

    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True)

    input_usd_per_million: float = Field(
        default=DEFAULT_INPUT_USD_PER_MILLION, ge=0.0, alias='input'
    )
    output_usd_per_million: float = Field(
        default=DEFAULT_OUTPUT_USD_PER_MILLION, ge=0.0, alias='output'
    )

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Cost of one call in US dollars."""
        priced = (
            input_tokens * self.input_usd_per_million
            + output_tokens * self.output_usd_per_million
        )
        return priced / _TOKENS_PER_MILLION


class BuilderConfig(BaseModel):
    """Everything the construction loop needs about the Builder model."""

    model_config = ConfigDict(
        frozen=True, extra='forbid', populate_by_name=True, protected_namespaces=()
    )

    model_id: str = Field(default=DEFAULT_MODEL_ID, min_length=1)
    provider: BuilderProvider = 'openai'
    api_key_env: str = Field(default=DEFAULT_API_KEY_ENV, min_length=1)
    endpoint: str = Field(default=DEFAULT_ENDPOINT, min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_candidates_per_round: int = Field(
        default=MAX_CANDIDATES_PER_ROUND, ge=1, le=MAX_CANDIDATES_PER_ROUND
    )
    max_rounds_per_stage: int = Field(default=MAX_ROUNDS_PER_STAGE, ge=1)
    early_stop_rounds: int = Field(default=EARLY_STOP_ROUNDS, ge=1)
    packet: PacketLimits = PacketLimits()
    gates: GateThresholds = GateThresholds()
    pricing: BuilderPricing = Field(
        default=BuilderPricing(), alias='pricing_usd_per_million'
    )
