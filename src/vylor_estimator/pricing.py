from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping

PRICING: dict[str, tuple[float, float, float, float, float]] = {
    # 5.x
    "claude-fable-5":    (10.0, 12.50, 20.0, 1.0,  50.0),
    "claude-mythos-5":   (10.0, 12.50, 20.0, 1.0,  50.0),
    "claude-opus-5":     (5.0,  6.25,  10.0, 0.50, 25.0),
    "claude-sonnet-5":   (2.0,  2.50,  4.0,  0.20, 10.0),

    # 4.x
    "claude-opus-4.8":   (5.0,  6.25,  10.0, 0.50, 25.0),
    "claude-opus-4.7":   (5.0,  6.25,  10.0, 0.50, 25.0),
    "claude-opus-4.6":   (5.0,  6.25,  10.0, 0.50, 25.0),
    "claude-opus-4.5":   (5.0,  6.25,  10.0, 0.50, 25.0),
    "claude-opus-4.1":   (15.0, 18.75, 30.0, 1.50, 75.0),
    "claude-opus-4":     (15.0, 18.75, 30.0, 1.50, 75.0),
    "claude-sonnet-4.6": (3.0,  3.75,  6.0,  0.30, 15.0),
    "claude-sonnet-4.5": (3.0,  3.75,  6.0,  0.30, 15.0),
    "claude-sonnet-4":   (3.0,  3.75,  6.0,  0.30, 15.0),
    "claude-haiku-4.5":  (1.0,  1.25,  2.0,  0.10,  5.0),

    # 3.x
    "claude-opus-3":     (15.0, 18.75, 30.0, 1.50, 75.0),
    "claude-sonnet-3.7": (3.0,  3.75,  6.0,  0.30, 15.0),
    "claude-sonnet-3.5": (3.0,  3.75,  6.0,  0.30, 15.0),
    "claude-sonnet-3":   (3.0,  3.75,  6.0,  0.30, 15.0),
    "claude-haiku-3.5":  (0.80, 1.0,   1.60, 0.08,  4.0),
    "claude-haiku-3":    (0.25, 0.30,  0.50, 0.03,  1.25),
}

_MODEL_PATTERNS = (
    # Matches patterns like "3-5-sonnet", "3.5-sonnet", "3-opus"
    re.compile(r"(?:claude-)?(?P<v>\d+(?:[.-]\d+)?)-(?P<family>sonnet|haiku|opus|fable|mythos)", re.IGNORECASE),
    # Matches patterns like "sonnet-4-5", "haiku-4.5", "opus-4"
    re.compile(r"(?:claude-)?(?P<family>sonnet|haiku|opus|fable|mythos)-(?P<v>\d+(?:[.-]\d+)?)", re.IGNORECASE),
)

_FAMILY_DEFAULTS: dict[str, tuple[float, float, float, float, float]] = {
    "haiku":  (1.0,  1.25, 2.0,  0.10, 5.0),
    "sonnet": (3.0,  3.75, 6.0,  0.30, 15.0),
    "opus":   (5.0,  6.25, 10.0, 0.50, 25.0),
    "fable":  (10.0, 12.50, 20.0, 1.0,  50.0),
    "mythos": (10.0, 12.50, 20.0, 1.0,  50.0),
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
        """Return rates for the model using structured family+version matching."""
        if not model_name:
            return self.default_rates

        norm = model_name.lower().strip()
        if norm in self.pricing_table:
            return self.pricing_table[norm]

        # Structured family + version extraction (handles dashes, dots, date suffixes)
        for pat in _MODEL_PATTERNS:
            m = pat.search(norm)
            if m:
                family = m.group("family").lower()
                v = m.group("v").replace("-", ".")
                candidates = [f"claude-{family}-{v}", f"{family}-{v}"]
                if v.endswith(".0"):
                    short_v = v[:-2]
                    candidates.extend([f"claude-{family}-{short_v}", f"{family}-{short_v}"])
                else:
                    candidates.extend([f"claude-{family}-{v}.0", f"{family}-{v}.0"])

                for c in candidates:
                    if c in self.pricing_table:
                        return self.pricing_table[c]

                if family in _FAMILY_DEFAULTS:
                    return _FAMILY_DEFAULTS[family]
                break

        # Fallback to family default if model string contains the family name
        for fam, rates in _FAMILY_DEFAULTS.items():
            if fam in norm:
                return rates

        # Fallback to longest substring match on custom tables
        best_key = ""
        for key in self.pricing_table:
            if key in norm and len(key) > len(best_key):
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
