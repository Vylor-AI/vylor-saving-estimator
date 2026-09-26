from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

PRICING: dict[str, tuple[float, float, float, float, float]] = {
    "claude-fable-5":    (10.0, 12.50, 20.0, 1.0,  50.0),
    "claude-mythos-5":   (10.0, 12.50, 20.0, 1.0,  50.0),
    "claude-opus-5":     (5.0,  6.25,  10.0, 0.50, 25.0),
    "claude-opus-4.8":   (5.0,  6.25,  10.0, 0.50, 25.0),
    "claude-opus-4.7":   (5.0,  6.25,  10.0, 0.50, 25.0),
    "claude-opus-4.6":   (5.0,  6.25,  10.0, 0.50, 25.0),
    "claude-opus-4.5":   (5.0,  6.25,  10.0, 0.50, 25.0),
    "claude-opus-4.1":   (15.0, 18.75, 30.0, 1.50, 75.0),
    "claude-opus-4":     (15.0, 18.75, 30.0, 1.50, 75.0),
    "claude-sonnet-5":   (2.0,  2.50,  4.0,  0.20, 10.0),
    "claude-sonnet-4.6": (3.0,  3.75,  6.0,  0.30, 15.0),
    "claude-sonnet-4.5": (3.0,  3.75,  6.0,  0.30, 15.0),
    "claude-sonnet-4":   (3.0,  3.75,  6.0,  0.30, 15.0),
    "claude-haiku-4.5":  (1.0,  1.25,  2.0,  0.10,  5.0),
    "claude-haiku-3.5":  (0.80, 1.0,   1.60, 0.08,  4.0),
}

_DEFAULT_RATES: tuple[float, float, float, float, float] = (2.0, 2.50, 4.0, 0.20, 10.0)


class IPricingCalculator(ABC):
    @abstractmethod
    def get_pricing(self, model_name: str) -> tuple[float, float, float, float, float]:
        """Return (input, 5m_cache_write, 1h_cache_write, cache_read, output) per 1M tokens."""

    @abstractmethod
    def compute_turn_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        ephemeral_5m_tokens: int = 0,
        ephemeral_1h_tokens: int = 0,
    ) -> float:
        """Compute the baseline USD cost for a single turn."""


class ClaudePricingCalculator(IPricingCalculator):
    """Pricing calculator for Claude models with customizable rate tables."""

    def __init__(
        self,
        pricing_table: Mapping[str, tuple[float, float, float, float, float]] | None = None,
        default_rates: tuple[float, float, float, float, float] = _DEFAULT_RATES,
    ) -> None:
        self.pricing_table = dict(pricing_table) if pricing_table is not None else dict(PRICING)
        self.default_rates = default_rates

    def get_pricing(self, model_name: str) -> tuple[float, float, float, float, float]:
        """Return rates for the model using longest-substring match; falls back to default rates."""
        normalized = (model_name or "").lower().strip()
        best_key = ""
        for key in self.pricing_table:
            if key in normalized and len(key) > len(best_key):
                best_key = key
        return self.pricing_table[best_key] if best_key else self.default_rates

    def compute_turn_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        ephemeral_5m_tokens: int = 0,
        ephemeral_1h_tokens: int = 0,
    ) -> float:
        """Compute the baseline USD cost for a single turn."""
        r = self.get_pricing(model)
        return (
            input_tokens * r[0]
            + ephemeral_5m_tokens * r[1]
            + ephemeral_1h_tokens * r[2]
            + cache_read_tokens * r[3]
            + output_tokens * r[4]
        ) / 1_000_000.0
