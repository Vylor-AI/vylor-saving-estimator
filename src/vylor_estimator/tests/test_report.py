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

    assert "AGENT OVERHEAD ESTIMATE" in out
    assert "1 session(s)" in out and "Last 30 days" in out
    assert "Total" in out and "Avoidable overhead" in out
    assert "Cost" in out and "Tokens" in out and "Time" in out
    assert "Top overhead sources: sub-agents" in out
    assert "-d" in out
    # No detailed breakdowns or Vylor marketing by default
    assert "Top Sessions by Overhead" not in out
    assert "Breakdown by Model" not in out
    assert "Overhead by Source" not in out
    assert "Powered by" not in out
    assert "SAVINGS" not in out.upper()


def test_debug_output_includes_details():
    out = _render(debug=True)

    assert "Avoidable overhead" in out
    assert "Overhead by Source" in out
    assert "Breakdown by Model" in out
    assert "Top Sessions by Overhead" in out
    assert "Session" in out
    assert "agent-time" in out


def test_fmt_time():
    assert _fmt_time(0) == "0s"
    assert _fmt_time(45) == "45s"
    assert _fmt_time(300) == "5m"
    assert _fmt_time(4320) == "1h 12m"
    assert _fmt_time(7200) == "2h"
