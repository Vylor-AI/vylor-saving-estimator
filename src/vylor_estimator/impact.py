from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from vylor_estimator.activity import classify_tool_call
from vylor_estimator.parser import Turn
from vylor_estimator.timing import turn_seconds

# Below this many Vylor sessions the report is labelled "early results".
EARLY_RESULTS_THRESHOLD = 5

__all__ = [
    "EARLY_RESULTS_THRESHOLD",
    "FirstEdit",
    "GroupImpact",
    "IImpactAnalyzer",
    "ImpactReport",
    "SessionImpactAnalyzer",
    "build_impact_report",
    "first_edit_stats",
    "turns_to_first_edit",
]


@dataclass(frozen=True)
class FirstEdit:
    """What a session spent before (and including) the turn of its first file edit."""

    turns: int
    cost: float
    seconds: float


@dataclass
class GroupImpact:
    """Per-session averages for one group of sessions (with or without Vylor)."""

    sessions: int = 0
    total_cost: float = 0.0
    edit_sessions: int = 0
    total_turns_to_first_edit: int = 0
    total_cost_to_first_edit: float = 0.0
    total_seconds_to_first_edit: float = 0.0

    def add_session(self, session_turns: list[Turn]) -> None:
        """Fold one session (its turns, in order) into the group totals."""
        self.sessions += 1
        self.total_cost += sum(t.cost for t in session_turns)
        first_edit = first_edit_stats(session_turns)
        if first_edit is not None:
            self.edit_sessions += 1
            self.total_turns_to_first_edit += first_edit.turns
            self.total_cost_to_first_edit += first_edit.cost
            self.total_seconds_to_first_edit += first_edit.seconds

    @property
    def avg_cost(self) -> float | None:
        """Average cost per session, or None when the group has no sessions."""
        return self.total_cost / self.sessions if self.sessions else None

    @property
    def avg_turns_to_first_edit(self) -> float | None:
        """Average turns before the first edit, over sessions that have an edit."""
        return self._avg_per_edit_session(self.total_turns_to_first_edit)

    @property
    def avg_cost_to_first_edit(self) -> float | None:
        """Average cost spent up to the first edit, over sessions that have an edit."""
        return self._avg_per_edit_session(self.total_cost_to_first_edit)

    @property
    def avg_seconds_to_first_edit(self) -> float | None:
        """Average agent-time up to the first edit, over sessions that have an edit."""
        return self._avg_per_edit_session(self.total_seconds_to_first_edit)

    def _avg_per_edit_session(self, total: float) -> float | None:
        return total / self.edit_sessions if self.edit_sessions else None


@dataclass
class ImpactReport:
    """Sessions without Vylor tools next to sessions that used them."""

    without_vylor: GroupImpact = field(default_factory=GroupImpact)
    with_vylor: GroupImpact = field(default_factory=GroupImpact)

    @property
    def has_vylor_sessions(self) -> bool:
        return self.with_vylor.sessions > 0

    @property
    def is_early_results(self) -> bool:
        return 0 < self.with_vylor.sessions < EARLY_RESULTS_THRESHOLD

    @property
    def cost_improvement_pct(self) -> float | None:
        """Percent the average session cost is lower with Vylor; None unless it improved."""
        return _improvement_pct(self.without_vylor.avg_cost, self.with_vylor.avg_cost)

    @property
    def turns_improvement_pct(self) -> float | None:
        """Percent fewer turns to first edit with Vylor; None unless it improved."""
        return _improvement_pct(
            self.without_vylor.avg_turns_to_first_edit,
            self.with_vylor.avg_turns_to_first_edit,
        )

    @property
    def time_to_first_edit_improvement_pct(self) -> float | None:
        """Percent less time to first edit with Vylor; None unless it improved."""
        return _improvement_pct(
            self.without_vylor.avg_seconds_to_first_edit,
            self.with_vylor.avg_seconds_to_first_edit,
        )

    @property
    def cost_to_first_edit_improvement_pct(self) -> float | None:
        """Percent less cost to first edit with Vylor; None unless it improved."""
        return _improvement_pct(
            self.without_vylor.avg_cost_to_first_edit,
            self.with_vylor.avg_cost_to_first_edit,
        )


def _improvement_pct(without: float | None, with_: float | None) -> float | None:
    if without is None or with_ is None or without <= 0 or with_ >= without:
        return None
    return (without - with_) / without * 100


def first_edit_stats(session_turns: list[Turn]) -> FirstEdit | None:
    """
    Turns, cost and agent-time spent up to and including the first turn that edits
    a file, counting every turn in the session in order, including sub-agent turns.
    Agent-time is the sum of (capped) turn durations, as in the savings estimate.
    None if the session never edits.
    """
    cost = 0.0
    seconds = 0.0
    for index, turn in enumerate(session_turns, start=1):
        cost += turn.cost
        seconds += turn_seconds(turn)
        if any(classify_tool_call(tc.name, tc.args) == "Write" for tc in turn.tool_calls):
            return FirstEdit(turns=index, cost=cost, seconds=seconds)
    return None


def turns_to_first_edit(session_turns: list[Turn]) -> int | None:
    """Position (1-based) of the first editing turn in the session, or None if it never edits."""
    first_edit = first_edit_stats(session_turns)
    return first_edit.turns if first_edit is not None else None


def build_impact_report(turns: Iterable[Turn], vylor_session_ids: set[str]) -> ImpactReport:
    """
    Split sessions into "used Vylor tools" and "did not", then average cost per
    session and the cost of reaching the first edit within each group.

    ``turns`` must be in chronological order. Turns of sub-agents belong to the
    session of their parent, so they count toward the session's cost and turn count.
    """
    return SessionImpactAnalyzer().analyze(turns, vylor_session_ids)


class IImpactAnalyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        turns: Iterable[Turn],
        vylor_session_ids: set[str],
    ) -> ImpactReport:
        """Compare sessions that used Vylor tools with sessions that did not."""


class SessionImpactAnalyzer(IImpactAnalyzer):
    """Computes per-session cost and first-edit metrics across Vylor and non-Vylor sessions."""

    def analyze(
        self,
        turns: Iterable[Turn],
        vylor_session_ids: set[str],
    ) -> ImpactReport:
        by_session: dict[str, list[Turn]] = defaultdict(list)
        for turn in turns:
            by_session[turn.session_id].append(turn)

        report = ImpactReport()
        for session_id, session_turns in by_session.items():
            group = report.with_vylor if session_id in vylor_session_ids else report.without_vylor
            group.add_session(session_turns)
        return report
