from __future__ import annotations

import io

from rich.console import Console

from vylor_estimator.classifier import (
    SOURCE_SEARCH,
    SOURCE_SUBAGENTS,
    ClassifiedTurn,
)
from vylor_estimator.parser import ToolCall, Turn
from vylor_estimator.report.terminal import TerminalReportRenderer, _fmt_time
from vylor_estimator.savings import ConservativeSavingsEngine


def _turn(turn_id: str, *, is_subagent: bool, seconds: float) -> Turn:
    from datetime import datetime, timezone

    return Turn(
        turn_id=turn_id,
        session_id="session-abc",
        model="claude-sonnet-4.5",
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        input_tokens=1_000,
        output_tokens=200,
        cache_read_tokens=500,
        cache_write_tokens=300,
        ephemeral_5m_tokens=0,
        ephemeral_1h_tokens=0,
        reasoning_tokens=0,
        duration_seconds=seconds,
        tool_calls=[ToolCall(name="Grep")],
        cost=0.05,
        is_subagent=is_subagent,
    )


def _report():
    cts = [
        ClassifiedTurn(_turn("a", is_subagent=True, seconds=120.0), False, SOURCE_SUBAGENTS),
        ClassifiedTurn(_turn("b", is_subagent=False, seconds=60.0), True, SOURCE_SEARCH),
        ClassifiedTurn(_turn("c", is_subagent=False, seconds=60.0), False, None),
    ]
    return ConservativeSavingsEngine().calculate(cts)


def _render(debug: bool) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, highlight=False)
    TerminalReportRenderer(console=console, debug=debug).render(
        _report(), date_range_label="Last 30 days"
    )
    return buf.getvalue()


def test_default_output_is_compact():
    out = _render(debug=False)

    assert "VYLOR MCP SAVINGS ESTIMATE" in out
    assert "1 session(s)" in out and "(Last 30 days) 2026-01-01 to 2026-01-01" in out
    assert "Total" in out and "Vylor MCP savings" in out
    assert "Cost" in out and "Tokens" in out and "Time" in out
    assert "Top savings sources: sub-agents" in out
    assert "Tool calls" in out
    assert "Cost breakdown" in out
    assert "-d" in out
    # No detailed breakdowns by default
    assert "Top Sessions by Savings" not in out
    assert "Breakdown by Model" not in out
    assert "Savings by Source" not in out
    assert "Powered by" not in out


def test_debug_output_includes_details():
    out = _render(debug=True)

    assert "Vylor MCP savings" in out
    assert "Savings by Source" in out
    assert "Breakdown by Model" in out
    assert "Top Sessions by Savings" in out
    assert "Session" in out
    assert "Notes: avoidable overhead" not in out


def test_fmt_time():
    assert _fmt_time(0) == "0s"
    assert _fmt_time(45) == "45s"
    assert _fmt_time(300) == "5m"
    assert _fmt_time(4320) == "1h 12m"
    assert _fmt_time(7200) == "2h"


def test_multi_agent_breakdown_rendering():
    from datetime import datetime, timezone
    t_claude = Turn(
        turn_id="c1",
        session_id="s_claude",
        model="claude-sonnet-4.5",
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        input_tokens=1_000,
        output_tokens=200,
        cache_read_tokens=500,
        cache_write_tokens=300,
        ephemeral_5m_tokens=0,
        ephemeral_1h_tokens=0,
        reasoning_tokens=0,
        duration_seconds=30.0,
        tool_calls=[ToolCall(name="Grep")],
        cost=0.05,
        is_subagent=False,
        agent="claude",
    )
    t_codex = Turn(
        turn_id="x1",
        session_id="s_codex",
        model="gpt-5-codex",
        timestamp=datetime(2026, 1, 1, 10, 5, tzinfo=timezone.utc),
        input_tokens=2_000,
        output_tokens=400,
        cache_read_tokens=1000,
        cache_write_tokens=0,
        ephemeral_5m_tokens=0,
        ephemeral_1h_tokens=0,
        reasoning_tokens=0,
        duration_seconds=45.0,
        tool_calls=[ToolCall(name="exec_command", args={"cmd": "rg foo"})],
        cost=0.08,
        is_subagent=False,
        agent="codex",
    )
    cts = [
        ClassifiedTurn(t_claude, True, SOURCE_SEARCH),
        ClassifiedTurn(t_codex, True, SOURCE_SEARCH),
    ]
    report = ConservativeSavingsEngine().calculate(cts)

    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, highlight=False)
    TerminalReportRenderer(console=console, debug=True).render(report)
    out = buf.getvalue()

    assert "Breakdown by Agent" in out
    assert "Claude" in out
    assert "Codex" in out
    assert "Analyzed: 2 session(s) (Claude, Codex)" in out
