
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timezone

from vylor_estimator.classifier import ClassifiedTurn


@dataclass
class TurnSavings:
    classified: ClassifiedTurn
    saved_input_tokens: int
    saved_cache_tokens: int            # avoided prompt cache read tokens
    saved_cache_write_tokens: int = 0  # avoided cache creation tokens
    saved_cost: float = 0.0

    @property
    def total_saved_tokens(self) -> int:
        return self.saved_input_tokens + self.saved_cache_tokens + self.saved_cache_write_tokens


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
    Subagent-elimination savings engine.

    Vylor MCP replaces background exploration subagents with a single lightweight
    tool call.  Every subagent turn that existed in the baseline session is therefore
    an eliminated cost.  The saving for each subagent turn is its **full** cost and
    all of its tokens (input + output + cache_read + cache_write).

    Main-agent turns are never credited as savings — they represent work Claude
    still needs to do even with Vylor MCP.
    """

    def __init__(self) -> None:
        pass

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
        Compute savings by eliminating exploration subagent turns.

        Vylor MCP replaces background exploration subagents with a single lightweight
        tool call.  Each subagent turn is therefore an eliminated cost; the saving is
        the **full** cost of that turn plus all of its tokens.

        Main-agent turns produce zero savings — those are the turns Claude still
        performs even with Vylor MCP.

        Example:
            Main turn 1   is_subagent=False   cost=$0.02  → no saving
            Sub  turn 1   is_subagent=True    cost=$0.05  → save $0.05 + all tokens
            Sub  turn 2   is_subagent=True    cost=$0.03  → save $0.03 + all tokens
            Main turn 2   is_subagent=False   cost=$0.04  → no saving
        Total saved = $0.08
        """
        total = Aggregate()
        by_session: dict[str, Aggregate] = defaultdict(Aggregate)
        by_day: dict[date, Aggregate] = defaultdict(Aggregate)
        turn_savings_list: list[TurnSavings] = []

        for ct in classified_turns:
            turn = ct.turn

            if turn.is_subagent:
                # Full subagent turn is eliminated by Vylor MCP
                saved_input = turn.input_tokens
                saved_cache = turn.cache_read_tokens
                saved_write = turn.cache_write_tokens
                saved_cost = turn.cost
            else:
                # Main-agent turn: no saving
                saved_input = 0
                saved_cache = 0
                saved_write = 0
                saved_cost = 0.0

            ts = TurnSavings(
                classified=ct,
                saved_input_tokens=saved_input,
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
