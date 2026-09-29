
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timezone

from vylor_estimator.classifier import ClassifiedTurn
from vylor_estimator.pricing import (
    ClaudePricingCalculator,
    IPricingCalculator,
)


@dataclass
class TurnSavings:
    classified: ClassifiedTurn
    saved_input_tokens: int
    saved_cache_tokens: int            # avoided prompt cache read tokens
    saved_output_tokens: int = 0       # avoided completion + reasoning tokens
    saved_cache_write_tokens: int = 0  # avoided cache creation tokens
    saved_cost: float = 0.0

    @property
    def total_saved_tokens(self) -> int:
        return (
            self.saved_input_tokens
            + self.saved_cache_tokens
            + self.saved_output_tokens
            + self.saved_cache_write_tokens
        )


@dataclass
class Aggregate:
    """Rolled-up savings metrics for a group of turns."""
    turns_analyzed: int = 0
    subagent_turns: int = 0

    # Baseline
    baseline_cost: float = 0.0
    baseline_input_tokens: int = 0
    baseline_output_tokens: int = 0
    baseline_cache_read_tokens: int = 0
    baseline_cache_write_tokens: int = 0

    # Savings
    saved_cost: float = 0.0
    saved_input_tokens: int = 0
    saved_output_tokens: int = 0       # completion + reasoning tokens
    saved_cache_tokens: int = 0        # cache read
    saved_cache_write_tokens: int = 0  # cache creation

    # Per-model breakdown
    by_model: dict[str, dict] = field(default_factory=dict)

    @property
    def estimated_cost(self) -> float:
        return max(0.0, self.baseline_cost - self.saved_cost)

    @property
    def estimated_tokens(self) -> int:
        return max(0, self.baseline_total_tokens - self.total_saved_tokens)

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
        return (
            self.saved_input_tokens
            + self.saved_output_tokens
            + self.saved_cache_tokens
            + self.saved_cache_write_tokens
        )

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


@dataclass
class SavingsReport:
    """Full report: all-time aggregate + per-session + per-day breakdowns."""
    total: Aggregate
    by_session: dict[str, Aggregate]   # session_id -> Aggregate
    by_day: dict[date, Aggregate]       # date -> Aggregate
    turn_savings: list[TurnSavings]


class ISavingsEngine(ABC):
    @abstractmethod
    def calculate(self, classified_turns: list[ClassifiedTurn]) -> SavingsReport:
        """Compute the full savings report from classified turns."""


class ConservativeSavingsEngine(ISavingsEngine):
    """
    Two-tier savings engine.

    Tier 1 — Subagent elimination (is_subagent=True):
      Vylor MCP replaces the entire background exploration agent with a single
      lightweight tool call.  The saving is the full cost of the subagent turn
      plus all of its tokens.

    Tier 2 — Shell-exec cache savings (vylor_intercept=True, main agent):
      Main-agent turns that call shell/glob/grep/file-read tools generate
      cache_read_tokens on every subsequent turn.  When Vylor intercepts
      these calls, those cache re-reads never happen.  The saving is exactly
      the cache_read_tokens cost of the intercepted turn.
    """

    def __init__(self, pricing_calculator: IPricingCalculator | None = None) -> None:
        self.pricing_calculator = pricing_calculator or ClaudePricingCalculator()

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

        if ts.saved_cost > 0 or ts.total_saved_tokens > 0:
            agg.saved_cost += ts.saved_cost
            agg.saved_input_tokens += ts.saved_input_tokens
            agg.saved_output_tokens += ts.saved_output_tokens
            agg.saved_cache_tokens += ts.saved_cache_tokens
            agg.saved_cache_write_tokens += ts.saved_cache_write_tokens

        model_key = turn.model or "unknown"
        self._ensure_model_key(agg, model_key)
        agg.by_model[model_key]["turns"] += 1
        agg.by_model[model_key]["baseline_cost"] += turn.cost
        if ts.saved_cost > 0 or ts.total_saved_tokens > 0:
            agg.by_model[model_key]["saved_cost"] += ts.saved_cost
            agg.by_model[model_key]["saved_tokens"] += ts.total_saved_tokens

    def calculate(self, classified_turns: list[ClassifiedTurn]) -> SavingsReport:
        """
        Two-tier savings calculation.

        Tier 1 — Subagent elimination:
            Every turn where is_subagent=True is fully eliminated by Vylor MCP.
            Saving = full cost + all tokens of that turn.

        Tier 2 — Shell-exec cache savings:
            Main-agent turns with vylor_intercept=True called a shell/glob/grep/
            file-read tool.  Vylor intercepts that call; the cache_read_tokens
            those tools would have generated on all later turns are saved.
            Saving = cache_read_tokens cost only (input/output unchanged).

        Example:
            Main turn 1   vylor_intercept=True   cache_read=10k  → save $cache_read cost
            Sub  turn 1   is_subagent=True        cost=$0.05      → save $0.05 + all tokens
            Main turn 2   vylor_intercept=False   (text only)     → $0 saved
        """
        total = Aggregate()
        by_session: dict[str, Aggregate] = defaultdict(Aggregate)
        by_day: dict[date, Aggregate] = defaultdict(Aggregate)
        turn_savings_list: list[TurnSavings] = []

        for ct in classified_turns:
            turn = ct.turn

            if turn.is_subagent:
                # Tier 1: full subagent turn eliminated
                saved_input = turn.input_tokens
                saved_output = turn.output_tokens
                saved_cache = turn.cache_read_tokens
                saved_write = turn.cache_write_tokens
                saved_cost  = turn.cost

            elif ct.vylor_intercept:
                # Tier 2: main-agent shell-exec turn — save cache_read + output cost
                # The Bash/Glob/Grep call is replaced by a concise Vylor tool call;
                # all output tokens (including reasoning) from this turn are gone.
                saved_input = 0
                saved_output = turn.output_tokens
                saved_cache = turn.cache_read_tokens
                saved_write = 0
                saved_cost  = self.pricing_calculator.compute_turn_cost(
                    model=turn.model,
                    input_tokens=0,
                    output_tokens=turn.output_tokens,
                    cache_read_tokens=turn.cache_read_tokens,
                )

            else:
                # No saving
                saved_input = 0
                saved_output = 0
                saved_cache = 0
                saved_write = 0
                saved_cost  = 0.0

            ts = TurnSavings(
                classified=ct,
                saved_input_tokens=saved_input,
                saved_output_tokens=saved_output,
                saved_cache_tokens=saved_cache,
                saved_cache_write_tokens=saved_write,
                saved_cost=saved_cost,
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
