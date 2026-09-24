"""savings.py -- Compounding Cache Savings calculation engine.

Accounts for the session-level cache compounding effect:
1. Intercepted file exploration turns save on new context (input_tokens + cache_write_tokens).
2. The avoided context is tracked chronologically per session.
3. Every subsequent turn in the session (even text-generation or editing turns)
   pays less cache_read_tokens because avoided context was never dumped into the prompt cache.
4. Mathematical invariants ensure saved tokens and costs never exceed actual baseline amounts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from vylor_estimator.classifier import (
    CACHE_READ_SAVINGS_FRACTION,
    TIME_SAVINGS_FRACTION,
    ClassifiedTurn,
    VylorTool,
)
from vylor_estimator.pricing import (
    ClaudePricingCalculator,
    IPricingCalculator,
    compute_turn_cost,
)


@dataclass
class TurnSavings:
    classified: ClassifiedTurn
    saved_input_tokens: int
    saved_cache_tokens: int            # avoided prompt cache read tokens
    saved_cache_write_tokens: int = 0  # avoided cache creation tokens
    saved_cost: float = 0.0
    saved_time_seconds: float | None = None

    @property
    def total_saved_tokens(self) -> int:
        return self.saved_input_tokens + self.saved_cache_tokens + self.saved_cache_write_tokens


@dataclass
class Aggregate:
    """Rolled-up savings metrics for a group of turns."""
    turns_analyzed: int = 0
    turns_intercepted: int = 0
    subagent_turns: int = 0

    # Baseline
    baseline_cost: float = 0.0
    baseline_input_tokens: int = 0
    baseline_output_tokens: int = 0
    baseline_cache_read_tokens: int = 0
    baseline_cache_write_tokens: int = 0
    baseline_time_seconds: float = 0.0

    # Savings
    saved_cost: float = 0.0
    saved_input_tokens: int = 0
    saved_cache_tokens: int = 0        # cache read
    saved_cache_write_tokens: int = 0  # cache creation
    saved_time_seconds: float = 0.0

    # Per-tool breakdown: {VylorTool: {turns, saved_cost, saved_tokens}}
    by_tool: dict[str, dict] = field(default_factory=dict)

    # Per-model breakdown
    by_model: dict[str, dict] = field(default_factory=dict)

    @property
    def estimated_cost(self) -> float:
        return max(0.0, self.baseline_cost - self.saved_cost)

    @property
    def estimated_tokens(self) -> int:
        return max(0, self.baseline_total_tokens - self.total_saved_tokens)

    @property
    def estimated_time_seconds(self) -> float:
        return max(0.0, self.baseline_time_seconds - self.saved_time_seconds)

    @property
    def baseline_total_tokens(self) -> int:
        """Total tokens including cache reads and cache writes (the full billable token footprint)."""
        return (
            self.baseline_input_tokens
            + self.baseline_output_tokens
            + self.baseline_cache_read_tokens
            + self.baseline_cache_write_tokens
        )

    @property
    def total_saved_tokens(self) -> int:
        return self.saved_input_tokens + self.saved_cache_tokens + self.saved_cache_write_tokens

    @property
    def pct_cost_cut(self) -> float:
        if self.baseline_cost == 0:
            return 0.0
        return round((self.saved_cost / self.baseline_cost) * 100, 1)

    @property
    def pct_tokens_cut(self) -> float:
        total = self.baseline_total_tokens
        if total == 0:
            return 0.0
        return round((self.total_saved_tokens / total) * 100, 1)

    @property
    def pct_time_cut(self) -> float:
        if self.baseline_time_seconds == 0:
            return 0.0
        return round((self.saved_time_seconds / self.baseline_time_seconds) * 100, 1)


@dataclass
class SavingsReport:
    """Full report: all-time aggregate + per-session + per-day breakdowns."""
    total: Aggregate
    by_session: dict[str, Aggregate]   # session_id -> Aggregate
    by_day: dict[date, Aggregate]       # date -> Aggregate
    turn_savings: list[TurnSavings]


class ISavingsEngine(ABC):
    """Abstract interface for savings calculation."""

    @abstractmethod
    def calculate(self, classified_turns: list[ClassifiedTurn]) -> SavingsReport:
        """Compute the full savings report from classified turns."""
        pass


class ConservativeSavingsEngine(ISavingsEngine):
    """
    Standard Compounding Cache Savings calculation engine.

    Chronologically models session context growth:
    - Turns that perform heavy file reads or searches have their new tokens
      (input_tokens + cache_write_tokens) reduced by savings_fraction.
    - Avoided context is tracked as accumulated_avoided_context for the session.
    - All downstream turns in that session avoid reading that accumulated context
      from the prompt cache (cache_read_tokens).
    """

    def __init__(
        self,
        pricing_calculator: IPricingCalculator | None = None,
        cache_savings_fraction: float = CACHE_READ_SAVINGS_FRACTION,
        time_savings_fraction: float = TIME_SAVINGS_FRACTION,
    ) -> None:
        self.pricing_calculator = pricing_calculator or ClaudePricingCalculator()
        self.cache_savings_fraction = cache_savings_fraction
        self.time_savings_fraction = time_savings_fraction

    def _ensure_tool_key(self, agg: Aggregate, tool: str) -> None:
        if tool not in agg.by_tool:
            agg.by_tool[tool] = {"turns": 0, "saved_cost": 0.0, "saved_tokens": 0}

    def _ensure_model_key(self, agg: Aggregate, model: str) -> None:
        if model not in agg.by_model:
            agg.by_model[model] = {
                "turns": 0,
                "baseline_cost": 0.0,
                "saved_cost": 0.0,
                "saved_tokens": 0,
            }

    def _add_to_aggregate(self, agg: Aggregate, ts: TurnSavings) -> None:
        ct = ts.classified
        turn = ct.turn

        agg.turns_analyzed += 1
        if turn.is_subagent:
            agg.subagent_turns += 1

        agg.baseline_cost += turn.cost
        agg.baseline_input_tokens += turn.input_tokens
        agg.baseline_output_tokens += turn.output_tokens
        agg.baseline_cache_read_tokens += turn.cache_read_tokens
        agg.baseline_cache_write_tokens += turn.cache_write_tokens
        if turn.duration_seconds is not None:
            agg.baseline_time_seconds += turn.duration_seconds

        if ts.saved_cost > 0 or ts.total_saved_tokens > 0:
            agg.saved_cost += ts.saved_cost
            agg.saved_input_tokens += ts.saved_input_tokens
            agg.saved_cache_tokens += ts.saved_cache_tokens
            agg.saved_cache_write_tokens += ts.saved_cache_write_tokens
            if ts.saved_time_seconds is not None:
                agg.saved_time_seconds += ts.saved_time_seconds

        if ct.vylor_intercept:
            agg.turns_intercepted += 1
            tool_key = ct.vylor_tool.value
            self._ensure_tool_key(agg, tool_key)
            agg.by_tool[tool_key]["turns"] += 1
            agg.by_tool[tool_key]["saved_cost"] += ts.saved_cost
            agg.by_tool[tool_key]["saved_tokens"] += ts.total_saved_tokens

        model_key = turn.model or "unknown"
        self._ensure_model_key(agg, model_key)
        agg.by_model[model_key]["turns"] += 1
        agg.by_model[model_key]["baseline_cost"] += turn.cost
        if ts.saved_cost > 0 or ts.total_saved_tokens > 0:
            agg.by_model[model_key]["saved_cost"] += ts.saved_cost
            agg.by_model[model_key]["saved_tokens"] += ts.total_saved_tokens

    def calculate(self, classified_turns: list[ClassifiedTurn]) -> SavingsReport:
        """Compute the full savings report using session-aware compounding cache math."""
        total = Aggregate()
        by_session: dict[str, Aggregate] = defaultdict(Aggregate)
        by_day: dict[date, Aggregate] = defaultdict(Aggregate)
        turn_savings_list: list[TurnSavings] = []

        # Group turns by session_id to process sessions chronologically
        turns_by_session: dict[str, list[ClassifiedTurn]] = defaultdict(list)
        for ct in classified_turns:
            turns_by_session[ct.turn.session_id].append(ct)

        for session_id, session_cts in turns_by_session.items():
            # Group by context stream (main agent vs subagent) to accurately model prompt caching
            streams: dict[bool, list[ClassifiedTurn]] = defaultdict(list)
            for ct in session_cts:
                streams[ct.turn.is_subagent].append(ct)

            for is_sub, stream_cts in streams.items():
                stream_cts.sort(
                    key=lambda x: x.turn.timestamp or datetime.min.replace(tzinfo=timezone.utc)
                )

                accumulated_avoided_context = 0

                for ct in stream_cts:
                    turn = ct.turn

                    saved_input = 0
                    saved_cache_write = 0
                    saved_cache_read = 0

                    if ct.vylor_intercept:
                        # New context introduced in this turn that Vylor would replace
                        new_tokens = turn.input_tokens + turn.cache_write_tokens
                        saved_new_tokens = int(new_tokens * ct.savings_fraction)

                        saved_input = min(turn.input_tokens, saved_new_tokens)
                        saved_cache_write = min(
                            turn.cache_write_tokens, saved_new_tokens - saved_input
                        )

                        # Avoided cache read from prior turns
                        prior_cache_savings = int(
                            min(turn.cache_read_tokens, accumulated_avoided_context)
                            * self.cache_savings_fraction
                        )
                        # Direct warm index replacement for current turn
                        remaining_cache = max(0, turn.cache_read_tokens - prior_cache_savings)
                        direct_cache_savings = int(remaining_cache * self.cache_savings_fraction)
                        saved_cache_read = min(
                            turn.cache_read_tokens, prior_cache_savings + direct_cache_savings
                        )

                        # Accumulate avoided context for downstream turns
                        accumulated_avoided_context += saved_new_tokens

                    else:
                        # Downstream turns (e.g. reasoning/coding/other tools) benefit from avoided cache bloat
                        if accumulated_avoided_context > 0 and turn.cache_read_tokens > 0:
                            saved_cache_read = int(
                                min(turn.cache_read_tokens, accumulated_avoided_context)
                                * self.cache_savings_fraction
                            )

                    # Split saved_cache_write between 5m and 1h ephemeral tokens
                    saved_5m = 0
                    saved_1h = 0
                    if turn.cache_write_tokens > 0 and saved_cache_write > 0:
                        ratio_5m = turn.ephemeral_5m_tokens / turn.cache_write_tokens
                        saved_5m = int(saved_cache_write * ratio_5m)
                        saved_1h = saved_cache_write - saved_5m

                    # Recompute cost delta
                    original_cost = self.pricing_calculator.compute_turn_cost(
                        model=turn.model,
                        input_tokens=turn.input_tokens,
                        output_tokens=turn.output_tokens,
                        cache_read_tokens=turn.cache_read_tokens,
                        ephemeral_5m_tokens=turn.ephemeral_5m_tokens,
                        ephemeral_1h_tokens=turn.ephemeral_1h_tokens,
                    )
                    reduced_cost = self.pricing_calculator.compute_turn_cost(
                        model=turn.model,
                        input_tokens=max(0, turn.input_tokens - saved_input),
                        output_tokens=turn.output_tokens,
                        cache_read_tokens=max(0, turn.cache_read_tokens - saved_cache_read),
                        ephemeral_5m_tokens=max(0, turn.ephemeral_5m_tokens - saved_5m),
                        ephemeral_1h_tokens=max(0, turn.ephemeral_1h_tokens - saved_1h),
                    )
                    saved_cost = max(0.0, min(original_cost, original_cost - reduced_cost))

                    saved_time: float | None = None
                    if ct.vylor_intercept and turn.duration_seconds is not None:
                        saved_time = turn.duration_seconds * ct.savings_fraction * self.time_savings_fraction

                    ts = TurnSavings(
                        classified=ct,
                        saved_input_tokens=saved_input,
                        saved_cache_tokens=saved_cache_read,
                        saved_cache_write_tokens=saved_cache_write,
                        saved_cost=saved_cost,
                        saved_time_seconds=saved_time,
                    )
                    turn_savings_list.append(ts)

                    self._add_to_aggregate(total, ts)
                    self._add_to_aggregate(by_session[turn.session_id], ts)

                    if turn.timestamp:
                        day = turn.timestamp.astimezone(timezone.utc).date()
                        self._add_to_aggregate(by_day[day], ts)

        return SavingsReport(
            total=total,
            by_session=dict(by_session),
            by_day=dict(sorted(by_day.items())),
            turn_savings=turn_savings_list,
        )


def compute_savings(
    classified_turns: list[ClassifiedTurn],
    pricing_calculator: IPricingCalculator | None = None,
) -> SavingsReport:
    """Backward-compatible functional API."""
    engine = ConservativeSavingsEngine(pricing_calculator=pricing_calculator)
    return engine.calculate(classified_turns)
