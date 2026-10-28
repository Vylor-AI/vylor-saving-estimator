from __future__ import annotations

import pytest

from vylor_estimator.pricing import (
    ClaudePricingCalculator,
    OpenAIPricingCalculator,
    RoutingPricingCalculator,
)


@pytest.fixture
def calculator() -> ClaudePricingCalculator:
    return ClaudePricingCalculator()


@pytest.mark.parametrize("model,expected_input,expected_output", [
    ("claude-haiku-4-5-20251001", 1.0, 5.0),
    ("haiku-4-5-20251001", 1.0, 5.0),
    ("claude-haiku-4.5", 1.0, 5.0),
    ("claude-3-5-haiku-20241022", 0.8, 4.0),
    ("claude-3-haiku-20240307", 0.25, 1.25),
    ("claude-opus-4", 15.0, 75.0),
    ("claude-opus-4-5-20251001", 5.0, 25.0),
    ("claude-opus-4.5", 5.0, 25.0),
    ("claude-opus-4.1", 15.0, 75.0),
    ("claude-sonnet-4-5", 3.0, 15.0),
    ("claude-3-7-sonnet-20250219", 3.0, 15.0),
    ("claude-3-5-sonnet-20241022", 3.0, 15.0),
    ("claude-sonnet-4", 3.0, 15.0),
    ("claude-sonnet-5", 2.0, 10.0),
])
def test_model_pricing_resolution(calculator: ClaudePricingCalculator, model: str, expected_input: float, expected_output: float):
    rates = calculator.get_pricing(model)
    assert rates[0] == expected_input
    assert rates[4] == expected_output


def test_unknown_model_fallback(calculator: ClaudePricingCalculator):
    rates = calculator.get_pricing("unknown-llm-model")
    assert rates == calculator.default_rates


def test_empty_model_fallback(calculator: ClaudePricingCalculator):
    rates = calculator.get_pricing("")
    assert rates == calculator.default_rates


@pytest.fixture
def openai_calc() -> OpenAIPricingCalculator:
    return OpenAIPricingCalculator()


@pytest.mark.parametrize("model,expected_input,expected_output", [
    ("gpt-5.6-terra", 1.25, 10.0),
    ("gpt-5.6-luna", 1.25, 10.0),
    ("gpt-5-codex", 1.25, 10.0),
    ("gpt-5", 1.25, 10.0),
    ("gpt-5-mini", 0.25, 2.0),
    ("gpt-6-luna", 2.0, 15.0),
    ("gpt-6", 2.0, 15.0),
    ("o3", 2.0, 8.0),
    ("o3-mini", 1.1, 4.4),
    ("o4-mini", 1.1, 4.4),
    ("gpt-4o", 2.5, 10.0),
    ("gpt-4o-mini", 0.15, 0.6),
    ("codex-auto-review", 1.25, 10.0),
    ("gpt-5-future-variant", 1.25, 10.0),  # Family fallback
    ("o3-custom", 2.0, 8.0),              # Family fallback
])
def test_openai_model_pricing(openai_calc: OpenAIPricingCalculator, model: str, expected_input: float, expected_output: float):
    rates = openai_calc.get_pricing(model)
    assert rates[0] == expected_input
    assert rates[4] == expected_output


def test_routing_calculator():
    router = RoutingPricingCalculator()
    # Claude model routed to Claude pricing
    claude_rates = router.get_pricing("claude-sonnet-4-5")
    assert claude_rates[0] == 3.0
    assert claude_rates[4] == 15.0

    # OpenAI model routed to OpenAI pricing
    openai_rates = router.get_pricing("gpt-5.6-terra")
    assert openai_rates[0] == 1.25
    assert openai_rates[4] == 10.0

    # Compute turn cost works across routed providers
    cost_claude = router.compute_turn_cost("claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=0)
    assert cost_claude == pytest.approx(3.0)

    cost_openai = router.compute_turn_cost("gpt-5.6-terra", input_tokens=1_000_000, output_tokens=0)
    assert cost_openai == pytest.approx(1.25)

