"""Target configuration: decoding, video sampling, call budget, and pricing.

Mirrors ``configs/target/*.yaml`` (ADR-0011). The file is read with
:func:`emoharness.common.config.load_model`, which expands ``${VAR}`` /
``${VAR:-default}`` references against the environment so a single committed
config can point at different endpoints per machine.
"""

from __future__ import annotations

from collections.abc import Mapping

from typing import Literal, Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from emoharness.target.pricing import Pricing

#: Which client answers a request for this config: the OpenAI-compatible chat
#: endpoint (vLLM and the OpenAI API alike) or the Gemini SDK.
ProviderName = Literal['openai_compat', 'gemini']

#: How the OpenAI-compatible client spells a request. ``standard`` is what vLLM
#: and the classic OpenAI models take; ``openai_reasoning`` is the GPT-5 family,
#: which rejects ``max_tokens`` (it wants ``max_completion_tokens``), takes a
#: ``reasoning_effort``, and only accepts sampling parameters when that effort
#: is ``none``.
ApiFlavor = Literal['standard', 'openai_reasoning']

_FREE: Pricing = Pricing(input_per_million=0.0, output_per_million=0.0)


class DecodingConfig(BaseModel):
    """Sampling parameters sent with every call. Defaults match ``TargetRequest``."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=1024, ge=1)


class VideoConfig(BaseModel):
    """Frame sampling for the video processor (ADR-0011)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    fps: float = Field(gt=0.0)
    min_frames: int = Field(ge=1)
    max_frames: int = Field(ge=1)
    frames_as_images: bool = Field(
        default=False,
        description='send the sampled frames as images; the endpoint takes no video',
    )

    @model_validator(mode='after')
    def _check_frame_bounds(self) -> VideoConfig:
        if self.max_frames < self.min_frames:
            raise ValueError('max_frames must be >= min_frames')
        return self


class CallBudget(BaseModel):
    """Per-item Target-call budget (ADR-0011)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    expected: int = Field(ge=1)
    hard_cap: int = Field(ge=1)
    retries_per_stage: int = Field(ge=0)

    @model_validator(mode='after')
    def _check_budget_bounds(self) -> CallBudget:
        if self.hard_cap < self.expected:
            raise ValueError('hard_cap must be >= expected')
        return self


class TargetConfig(BaseModel):
    """Everything an inference client needs for one Target model."""

    model_config = ConfigDict(
        frozen=True, extra='forbid', populate_by_name=True, protected_namespaces=()
    )

    model_id: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    decoding: DecodingConfig
    video: VideoConfig
    call_budget: CallBudget
    provider: ProviderName = 'openai_compat'
    api_key_env: str | None = Field(
        default=None,
        description='environment variable holding the bearer token, when the '
        'endpoint needs one; local vLLM serving needs none',
    )
    api_flavor: ApiFlavor = Field(
        default='standard',
        description='request spelling: standard, or openai_reasoning for GPT-5',
    )
    reasoning_effort: str = Field(
        default='none',
        min_length=1,
        description='reasoning_effort sent under the openai_reasoning flavor',
    )
    chat_template_kwargs: Mapping[str, Any] | None = Field(
        default=None,
        description='extra arguments for the served chat template, e.g. '
        "{'enable_thinking': false} for the Qwen3.5 family. Sent verbatim as "
        'the top-level chat_template_kwargs field vLLM reads; omitted when '
        'unset, so every existing Target keeps its exact request spelling.',
    )
    pricing: Pricing = Field(default=_FREE, alias='pricing_usd_per_million')

    @field_validator('endpoint')
    @classmethod
    def _validate_endpoint(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc:
            raise ValueError(f'endpoint must be an http(s) URL, got {value!r}')
        return value
