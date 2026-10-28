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


OPENAI_PRICING: dict[str, tuple[float, float, float, float, float]] = {
    # Rates per 1M tokens: (input, 5m_cache_write, 1h_cache_write, cache_read, output)
    # gpt-5.x / gpt-6
    "gpt-5.6-terra":     (1.25, 1.25, 1.25, 0.125, 10.0),
    "gpt-5.6-luna":      (1.25, 1.25, 1.25, 0.125, 10.0),
    "gpt-5-codex":       (1.25, 1.25, 1.25, 0.125, 10.0),
    "gpt-5":             (1.25, 1.25, 1.25, 0.125, 10.0),
    "gpt-5-mini":        (0.25, 0.25, 0.25, 0.025,  2.0),
    "gpt-6-luna":        (2.00, 2.00, 2.00, 0.20,  15.0),
    "gpt-6":             (2.00, 2.00, 2.00, 0.20,  15.0),
    "gpt-reserve":       (1.25, 1.25, 1.25, 0.125, 10.0),
    "codex-auto-review": (1.25, 1.25, 1.25, 0.125, 10.0),

    # Reasoning models (o3, o4, o1)
    "o3":                (2.00, 2.00, 2.00, 0.50,   8.0),
    "o3-mini":           (1.10, 1.10, 1.10, 0.275,  4.4),
    "o4-mini":           (1.10, 1.10, 1.10, 0.275,  4.4),
    "o1":                (15.0, 15.0, 15.0, 7.50,  60.0),
    "o1-mini":           (1.10, 1.10, 1.10, 0.55,   4.4),

    # gpt-4.x
    "gpt-4.1":           (2.00, 2.00, 2.00, 0.50,   8.0),
    "gpt-4.1-mini":      (0.40, 0.40, 0.40, 0.10,   1.6),
    "gpt-4o":            (2.50, 2.50, 2.50, 1.25,  10.0),
    "gpt-4o-mini":       (0.15, 0.15, 0.15, 0.075,  0.6),
}

_OPENAI_FAMILY_DEFAULTS: dict[str, tuple[float, float, float, float, float]] = {
    "gpt-6": (2.00, 2.00, 2.00, 0.20,  15.0),
    "gpt-5": (1.25, 1.25, 1.25, 0.125, 10.0),
    "gpt-4": (2.50, 2.50, 2.50, 1.25,  10.0),
    "o3":    (2.00, 2.00, 2.00, 0.50,   8.0),
    "o4":    (1.10, 1.10, 1.10, 0.275,  4.4),
    "o1":    (15.0, 15.0, 15.0, 7.50,  60.0),
    "codex": (1.25, 1.25, 1.25, 0.125, 10.0),
}

_OPENAI_DEFAULT_RATES: tuple[float, float, float, float, float] = (1.25, 1.25, 1.25, 0.125, 10.0)


class OpenAIPricingCalculator(IPricingCalculator):
    """Pricing calculator for OpenAI / Codex models."""

    def __init__(
        self,
        pricing_table: Mapping[str, tuple[float, float, float, float, float]] | None = None,
        default_rates: tuple[float, float, float, float, float] = _OPENAI_DEFAULT_RATES,
    ) -> None:
        self.pricing_table = dict(pricing_table) if pricing_table is not None else dict(OPENAI_PRICING)
        self.default_rates = default_rates

    def get_pricing(self, model_name: str) -> tuple[float, float, float, float, float]:
        if not model_name:
            return self.default_rates

        norm = model_name.lower().strip()
        if norm in self.pricing_table:
            return self.pricing_table[norm]

        # Check family prefixes
        for fam, rates in _OPENAI_FAMILY_DEFAULTS.items():
            if norm.startswith(fam) or fam in norm:
                return rates

        # Substring matching on custom table
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
        r = self.get_pricing(model)
        return (
            input_tokens * r[0]
            + ephemeral_5m_tokens * r[1]
            + ephemeral_1h_tokens * r[2]
            + cache_read_tokens * r[3]
            + output_tokens * r[4]
        ) / 1_000_000.0


class RoutingPricingCalculator(IPricingCalculator):
    """Routes pricing calculations to Claude or OpenAI calculators based on model identifier."""

    def __init__(
        self,
        claude_calculator: IPricingCalculator | None = None,
        openai_calculator: IPricingCalculator | None = None,
    ) -> None:
        self.claude_calculator = claude_calculator or ClaudePricingCalculator()
        self.openai_calculator = openai_calculator or OpenAIPricingCalculator()

    def _route(self, model_name: str) -> IPricingCalculator:
        norm = (model_name or "").lower().strip()
        if any(prefix in norm for prefix in ("gpt", "o1", "o3", "o4", "codex", "terra", "luna")):
            return self.openai_calculator
        return self.claude_calculator

    def get_pricing(self, model_name: str) -> tuple[float, float, float, float, float]:
        return self._route(model_name).get_pricing(model_name)

    def compute_turn_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        ephemeral_5m_tokens: int = 0,
        ephemeral_1h_tokens: int = 0,
    ) -> float:
        return self._route(model).compute_turn_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            ephemeral_5m_tokens=ephemeral_5m_tokens,
            ephemeral_1h_tokens=ephemeral_1h_tokens,
        )
