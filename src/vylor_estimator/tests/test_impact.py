from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import pytest
from rich.console import Console

from vylor_estimator.classifier import PatternTurnClassifier
from vylor_estimator.impact import (
    EARLY_RESULTS_THRESHOLD,
    ImpactReport,
    build_impact_report,
    first_edit_stats,
    turns_to_first_edit,
)
from vylor_estimator.parser import ToolCall, Turn
from vylor_estimator.report.terminal import TerminalReportRenderer
from vylor_estimator.savings import ConservativeSavingsEngine

_T0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


def _turn(
    session_id: str,
    index: int,
    tools: list[str],
    cost: float = 0.10,
    is_subagent: bool = False,
    seconds: float = 10.0,
) -> Turn:
    return Turn(
        turn_id=f"{session_id}-{index}",
        session_id=session_id,
        model="claude-sonnet-4.5",
        timestamp=_T0 + timedelta(minutes=index),
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        cache_write_tokens=0,
        ephemeral_5m_tokens=0,
        ephemeral_1h_tokens=0,
        reasoning_tokens=0,
        duration_seconds=seconds,
        tool_calls=[ToolCall(name=t) for t in tools],
        cost=cost,
        is_subagent=is_subagent,
    )


def _session(
    session_id: str,
    tool_per_turn: list[list[str]],
    cost: float = 0.10,
    seconds: float = 10.0,
) -> list[Turn]:
    return [
        _turn(session_id, i, tools, cost, seconds=seconds)
        for i, tools in enumerate(tool_per_turn)
    ]


def test_first_edit_stats_sums_cost_and_time_up_to_first_edit():
    turns = _session("s", [["Grep"], ["Read"], ["Edit"], ["Edit"], ["Read"]], cost=0.10, seconds=30)

    stats = first_edit_stats(turns)

    assert stats is not None
    assert stats.turns == 3
    assert stats.cost == pytest.approx(0.30)
    assert stats.seconds == pytest.approx(90.0)


def test_first_edit_stats_caps_turn_time_and_handles_missing_duration():
    turns = [
        _turn("s", 0, ["Grep"], seconds=5000),  # capped to MAX_TURN_SECONDS (600)
        _turn("s", 1, ["Read"]),
        _turn("s", 2, ["Edit"]),
    ]
    turns[1].duration_seconds = None

    stats = first_edit_stats(turns)

    assert stats is not None
    assert stats.seconds == pytest.approx(600.0 + 0.0 + 10.0)


def test_first_edit_stats_is_none_without_edits():
    assert first_edit_stats(_session("s", [["Grep"], ["Read"]])) is None


def test_group_first_edit_averages_use_edit_sessions_only():
    turns = (
        _session("a", [["Grep"], ["Edit"], ["Read"]], cost=0.10, seconds=20)  # 0.20 / 40s
        + _session("b", [["Grep"], ["Grep"], ["Grep"], ["Edit"]], cost=0.10, seconds=20)  # 0.40 / 80s
        + _session("c", [["Grep"], ["Read"]], cost=0.50, seconds=100)  # read-only
    )
    turns.sort(key=lambda t: t.timestamp)

    group = build_impact_report(turns, set()).without_vylor

    assert group.sessions == 3
    assert group.edit_sessions == 2
    assert group.avg_cost_to_first_edit == pytest.approx(0.30)
    assert group.avg_seconds_to_first_edit == pytest.approx(60.0)
    assert group.avg_turns_to_first_edit == pytest.approx(3.0)


def test_first_edit_improvement_percentages():
    turns = (
        _session("n", [["Grep"], ["Grep"], ["Edit"]], cost=0.20, seconds=40)  # 0.60 / 120s
        + _session("v", [["mcp__vylor__find_files"], ["Edit"]], cost=0.15, seconds=30)  # 0.30 / 60s
    )
    turns.sort(key=lambda t: t.timestamp)

    report = build_impact_report(turns, {"v"})

    assert report.cost_to_first_edit_improvement_pct == pytest.approx(50.0)
    assert report.time_to_first_edit_improvement_pct == pytest.approx(50.0)


def test_first_edit_improvement_none_when_not_better():
    turns = (
        _session("n", [["Edit"]], cost=0.10, seconds=10)
        + _session("v", [["mcp__vylor__find_files"], ["Edit"]], cost=0.10, seconds=10)
    )
    turns.sort(key=lambda t: t.timestamp)

    report = build_impact_report(turns, {"v"})

    assert report.cost_to_first_edit_improvement_pct is None
    assert report.time_to_first_edit_improvement_pct is None


def test_turns_to_first_edit_counts_position_of_first_write():
    turns = _session("s", [["Grep"], ["Read"], ["Edit"], ["Edit"]])
    assert turns_to_first_edit(turns) == 3


def test_turns_to_first_edit_is_none_without_edits():
    turns = _session("s", [["Grep"], ["Read"], ["Bash"]])
    assert turns_to_first_edit(turns) is None


def test_turns_to_first_edit_counts_subagent_turns():
    turns = [
        _turn("s", 0, ["Task"]),
        _turn("s", 1, ["Grep"], is_subagent=True),
        _turn("s", 2, ["Read"], is_subagent=True),
        _turn("s", 3, ["Edit"]),
    ]
    assert turns_to_first_edit(turns) == 4


def test_groups_are_split_by_vylor_sessions():
    turns = (
        _session("a", [["Grep"], ["Edit"]], cost=0.20)  # without, edits at turn 2
        + _session("b", [["Grep"], ["Grep"], ["Edit"]], cost=0.30)  # without, edit at 3
        + _session("c", [["mcp__vylor__find_files"], ["Edit"]], cost=0.10)  # with, turn 2
    )
    turns.sort(key=lambda t: t.timestamp)

    report = build_impact_report(turns, {"c"})

    assert report.without_vylor.sessions == 2
    assert report.with_vylor.sessions == 1
    # (2 sessions: 2 turns @0.20 + 3 turns @0.30) / 2
    assert report.without_vylor.avg_cost == pytest.approx((0.40 + 0.90) / 2)
    assert report.with_vylor.avg_cost == pytest.approx(0.20)
    assert report.without_vylor.edit_sessions == 2
    assert report.without_vylor.avg_turns_to_first_edit == pytest.approx(2.5)
    assert report.with_vylor.avg_turns_to_first_edit == pytest.approx(2.0)


def test_read_only_sessions_count_toward_cost_but_not_turns_to_first_edit():
    turns = _session("a", [["Grep"], ["Edit"]]) + _session("b", [["Grep"], ["Read"]])
    turns.sort(key=lambda t: t.timestamp)

    report = build_impact_report(turns, set())

    assert report.without_vylor.sessions == 2
    assert report.without_vylor.edit_sessions == 1
    assert report.without_vylor.avg_turns_to_first_edit == pytest.approx(2.0)


def test_group_without_edit_sessions_has_no_turns_average():
    turns = _session("a", [["mcp__vylor__request_repo_map"], ["Read"]])
    report = build_impact_report(turns, {"a"})

    assert report.with_vylor.avg_turns_to_first_edit is None
    assert report.turns_improvement_pct is None


def test_improvement_percentages_only_when_improved():
    better = build_impact_report(
        _session("a", [["Grep"], ["Grep"], ["Edit"]], cost=0.40)
        + _session("b", [["Edit"]], cost=0.30),
        {"b"},
    )
    # without avg cost 1.20, with avg cost 0.30 -> 75% lower
    assert better.cost_improvement_pct == pytest.approx(75.0)
    # without turns 3, with turns 1 -> 66.7% fewer
    assert better.turns_improvement_pct == pytest.approx(200 / 3)

    worse = build_impact_report(
        _session("a", [["Edit"]], cost=0.10) + _session("b", [["Grep"], ["Edit"]], cost=0.50),
        {"b"},
    )
    assert worse.cost_improvement_pct is None
    assert worse.turns_improvement_pct is None


def test_no_vylor_sessions_means_no_impact_block():
    report = build_impact_report(_session("a", [["Grep"], ["Edit"]]), set())
    assert not report.has_vylor_sessions
    assert not report.is_early_results


def test_early_results_threshold():
    report = ImpactReport()
    report.with_vylor.sessions = EARLY_RESULTS_THRESHOLD - 1
    assert report.is_early_results
    report.with_vylor.sessions = EARLY_RESULTS_THRESHOLD
    assert not report.is_early_results


def _render(turns: list[Turn]) -> str:
    classifier = PatternTurnClassifier()
    classified = classifier.classify(turns)
    savings = ConservativeSavingsEngine().calculate(classified)
    savings.impact = build_impact_report(turns, classifier.skipped_sessions)

    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, highlight=False)
    TerminalReportRenderer(console=console).render(savings)
    return buf.getvalue()


def _mixed_turns() -> list[Turn]:
    turns = (
        _session("n1", [["Grep"], ["Grep"], ["Read"], ["Edit"]], cost=0.20)
        + _session("n2", [["Grep"], ["Grep"], ["Read"], ["Edit"]], cost=0.20)
        + _session("v1", [["mcp__vylor__find_code_definition"], ["Edit"]], cost=0.10)
    )
    turns.sort(key=lambda t: t.timestamp)
    return turns


def test_terminal_renders_impact_block_with_edit_session_counts():
    out = _render(_mixed_turns())

    assert "VYLOR MCP IMPACT" in out
    assert "Estimated savings on sessions that did not use Vylor MCP" in out
    assert "Measured comparison: normal sessions vs. sessions using Vylor MCP" in out
    assert "Without Vylor" in out and "With Vylor" in out
    assert "Avg cost per session" in out
    # 0.80 per no-Vylor session vs 0.20 per Vylor session -> 75% lower
    assert "$0.80" in out and "$0.20" in out
    assert "(75% lower)" in out
    assert "Avg turns to first edit" in out
    assert "4.0 (of 2)" in out
    assert "2.0 (of 1)" in out
    assert "(50% faster)" in out
    assert "First-edit rows average edit sessions only" in out
    assert "(sessions with at least one file edit)" in out


def test_terminal_renders_time_and_cost_to_first_edit_rows():
    turns = (
        _session("n1", [["Grep"], ["Grep"], ["Edit"], ["Read"]], cost=0.20, seconds=60)
        + _session("v1", [["mcp__vylor__find_files"], ["Edit"], ["Read"]], cost=0.10, seconds=30)
    )
    turns.sort(key=lambda t: t.timestamp)

    out = _render(turns)

    # no-Vylor first edit: 3 turns, $0.60, 3m   |   Vylor first edit: 2 turns, $0.20, 1m
    assert "Avg time to first edit" in out
    assert "3m" in out and "1m" in out
    assert "(67% faster)" in out
    assert "Avg cost to first edit" in out
    assert "$0.60" in out and "$0.20" in out
    assert "(67% lower)" in out


def test_terminal_shows_early_results_note_for_single_vylor_session():
    out = _render(_mixed_turns())
    assert "Early results: based on 1 Vylor session." in out


def test_terminal_hides_block_without_vylor_sessions():
    out = _render(_session("n1", [["Grep"], ["Edit"]]))
    assert "VYLOR MCP IMPACT" not in out


def test_terminal_shows_na_when_group_has_no_edit_sessions():
    turns = _session("n1", [["Grep"], ["Read"]]) + _session("v1", [["find_files"], ["Read"]])
    turns.sort(key=lambda t: t.timestamp)
    out = _render(turns)

    assert "VYLOR MCP IMPACT" in out
    assert "n/a (of 0)" in out
    assert "Avg time to first edit" in out and "n/a" in out
    assert "faster" not in out
