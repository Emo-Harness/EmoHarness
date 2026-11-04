"""Tests for emoharness.target.pricing."""

from __future__ import annotations

import pydantic
import pytest

from emoharness.target.pricing import Pricing, estimate_cost


def test_estimate_cost_charges_input_and_output_independently() -> None:
    pricing = Pricing(input_per_million=1.0, output_per_million=2.0)

    cost = estimate_cost(pricing, input_tokens=1_000_000, output_tokens=500_000)

    assert cost == pytest.approx(1.0 + 1.0)


def test_estimate_cost_zero_pricing_is_free() -> None:
    pricing = Pricing(input_per_million=0.0, output_per_million=0.0)

    assert estimate_cost(pricing, input_tokens=10_000, output_tokens=10_000) == 0.0


def test_estimate_cost_zero_tokens_is_free() -> None:
    pricing = Pricing(input_per_million=5.0, output_per_million=5.0)

    assert estimate_cost(pricing, input_tokens=0, output_tokens=0) == 0.0


def test_estimate_cost_scales_linearly() -> None:
    pricing = Pricing(input_per_million=3.0, output_per_million=0.0)

    single = estimate_cost(pricing, input_tokens=1_000_000, output_tokens=0)
    double = estimate_cost(pricing, input_tokens=2_000_000, output_tokens=0)

    assert double == pytest.approx(single * 2)


def test_pricing_rejects_negative_rates() -> None:
    with pytest.raises(pydantic.ValidationError):
        Pricing(input_per_million=-1.0, output_per_million=0.0)


def test_pricing_is_frozen() -> None:
    pricing = Pricing(input_per_million=1.0, output_per_million=1.0)

    with pytest.raises(pydantic.ValidationError):
        pricing.input_per_million = 5.0  # type: ignore[misc]
