from __future__ import annotations

import pytest

from vylor_estimator.pricing import ClaudePricingCalculator


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
