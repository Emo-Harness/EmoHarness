"""USD cost model for Target and teacher calls, priced per million tokens."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_TOKENS_PER_MILLION = 1_000_000


class Pricing(BaseModel):
    """USD price per million input/output tokens. Zero for locally served models.

    The YAML spells the two fields ``input`` and ``output`` under a
    ``pricing_usd_per_million`` block; both spellings validate.
    """

    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True)

    input_per_million: float = Field(ge=0.0, alias='input')
    output_per_million: float = Field(ge=0.0, alias='output')


def estimate_cost(pricing: Pricing, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of one call at the given pricing and token counts."""
    input_cost = input_tokens * pricing.input_per_million
    output_cost = output_tokens * pricing.output_per_million
    return (input_cost + output_cost) / _TOKENS_PER_MILLION
