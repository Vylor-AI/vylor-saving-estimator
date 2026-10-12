from __future__ import annotations

from datetime import datetime, timezone

from vylor_estimator.activity import (
    ActivityReport,
    ToolActivityAnalyzer,
    classify_tool_call,
)
from vylor_estimator.parser import ToolCall, Turn


def _make_turn(
    name: str,
    args: dict | None = None,
    *,
    input_tokens: int = 100,
    output_tokens: int = 200,
    reasoning_tokens: int = 50,
    cache_read: int = 1_000,
    cache_write: int = 500,
    is_subagent: bool = False,
    subagent_type: str | None = None,
) -> Turn:
    return Turn(
        turn_id="turn-1",
        session_id="session-1",
        model="claude-sonnet-4.5",
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        ephemeral_5m_tokens=cache_write,
        ephemeral_1h_tokens=0,
        reasoning_tokens=reasoning_tokens,
        duration_seconds=5.0,
        tool_calls=[ToolCall(name=name, args=args or {})],
        cost=0.01,
        is_subagent=is_subagent,
        subagent_type=subagent_type,
    )


def test_classify_tool_call():
    # Write tools
    assert classify_tool_call("Write") == "Write"
    assert classify_tool_call("edit") == "Write"
    assert classify_tool_call("multi_replace_file_content") == "Write"

    # Read tools
    assert classify_tool_call("Read") == "Read"
    assert classify_tool_call("view_file") == "Read"

    # Search tools
    assert classify_tool_call("Grep") == "Search"
    assert classify_tool_call("glob") == "Search"
    assert classify_tool_call("list_dir") == "Search"

    # Subagent tools
    assert classify_tool_call("Task") == "Sub-agent"
    assert classify_tool_call("Agent") == "Sub-agent"

    # Web tools
    assert classify_tool_call("WebFetch") == "Web"

    # Shell commands
    assert classify_tool_call("Bash", {"command": "grep -rn 'foo'"}) == "Search"
    assert classify_tool_call("Bash", {"command": "cat file.txt"}) == "Read"
    assert classify_tool_call("Bash", {"command": "npm test"}) == "Commands"
    assert classify_tool_call("run_command", {"CommandLine": "git status"}) == "Commands"


def test_analyzer_counts_categories_and_cost():
    t1 = _make_turn("Write", {"file_path": "a.py"}, output_tokens=1000, reasoning_tokens=200)
    t2 = _make_turn("Read", {"file_path": "b.py"})
    t3 = _make_turn("Bash", {"command": "npm test"})
    t4 = _make_turn("Agent", {"subagent_type": "Explore"})

    analyzer = ToolActivityAnalyzer()
    report: ActivityReport = analyzer.analyze([t1, t2, t3, t4])

    assert report.total_calls == 4
    assert report.calls_by_category["Write"] == 1
    assert report.calls_by_category["Read"] == 1
    assert report.calls_by_category["Commands"] == 1
    assert report.calls_by_category["Sub-agent"] == 1
    assert report.subagent_types == {"Explore": 1}

    # Verify cost percentages are calculated and non-zero
    assert report.cache_read_cost_pct > 0
    assert report.cache_creation_cost_pct > 0
    assert report.output_cost_pct > 0
    assert report.thinking_pct_of_output > 0


def test_analyzer_empty_turns():
    analyzer = ToolActivityAnalyzer()
    report = analyzer.analyze([])
    assert report.total_calls == 0
    assert not report.has_activity


def test_defensive_guards():
    # None or empty name
    assert classify_tool_call("") == "Other"
    assert classify_tool_call(None) == "Other"  # type: ignore

    # Non-dict args or None args
    assert classify_tool_call("Bash", None) == "Commands"
    assert classify_tool_call("Bash", "not a dict") == "Commands"  # type: ignore
    assert classify_tool_call("Bash", [1, 2, 3]) == "Commands"  # type: ignore

    # Subagent with invalid or non-string subagent_type
    t_invalid = _make_turn("Agent", {"subagent_type": 123})
    t_none = _make_turn("Agent", {"subagent_type": None})
    analyzer = ToolActivityAnalyzer()
    report = analyzer.analyze([t_invalid, t_none])
    assert report.total_calls == 2
    assert report.subagent_types == {}

    # Zero tokens / zero cost turn doesn't cause ZeroDivisionError
    t_zero = _make_turn(
        "Read",
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read=0,
        cache_write=0,
    )
    report_zero = analyzer.analyze([t_zero])
    assert report_zero.cache_read_cost_pct == 0.0
    assert report_zero.thinking_pct_of_output == 0.0
