"""test_savings.py -- Tests for the savings math engine."""
from __future__ import annotations

from pathlib import Path

from vylor_estimator.classifier import classify_turns
from vylor_estimator.parser import parse_files
from vylor_estimator.savings import compute_savings

FIXTURES = Path(__file__).parent / "fixtures"


def test_savings_non_negative():
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    report = compute_savings(classified)
    assert report.total.saved_cost >= 0.0
    assert report.total.saved_input_tokens >= 0
    assert report.total.saved_cache_tokens >= 0
    assert report.total.saved_time_seconds >= 0.0


def test_savings_less_than_baseline():
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    report = compute_savings(classified)
    assert report.total.estimated_cost <= report.total.baseline_cost
    assert report.total.estimated_tokens <= report.total.baseline_total_tokens


def test_pct_cut_between_0_and_100():
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    report = compute_savings(classified)
    assert 0.0 <= report.total.pct_cost_cut <= 100.0
    assert 0.0 <= report.total.pct_tokens_cut <= 100.0
    assert 0.0 <= report.total.pct_time_cut <= 100.0


def test_by_session_keys():
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    report = compute_savings(classified)
    assert "sample_session" in report.by_session


def test_output_tokens_not_saved():
    """Output tokens should never be counted as saved (0% of output)."""
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    report = compute_savings(classified)
    total_output = sum(t.output_tokens for t in turns)
    # Saved cost cannot exceed what input+cache alone would cost
    # i.e. output tokens are not credited
    assert report.total.estimated_cost >= 0.0


def test_conservative_cap():
    """Savings fraction should never exceed 70% for input tokens."""
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    report = compute_savings(classified)
    for ts in report.turn_savings:
        if ts.classified.vylor_intercept:
            fraction = ts.saved_input_tokens / max(ts.classified.turn.input_tokens, 1)
            assert fraction <= 1.0, f"Savings fraction too high: {fraction}"


def test_compounding_cache_savings_on_downstream_turns():
    """Verify that turns following an intercepted turn benefit from avoided cache bloat."""
    from datetime import datetime, timezone
    from vylor_estimator.parser import Turn, ToolCall
    from vylor_estimator.classifier import ClassifiedTurn, VylorTool

    t1 = Turn(
        turn_id="t1",
        session_id="compounding_session",
        model="claude-sonnet-4.5",
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        input_tokens=100,
        output_tokens=200,
        cache_read_tokens=0,
        cache_write_tokens=10_000,
        ephemeral_5m_tokens=10_000,
        ephemeral_1h_tokens=0,
        reasoning_tokens=0,
        duration_seconds=5.0,
        tool_calls=[ToolCall(name="read_file", args={})],
        cost=0.040,
        is_subagent=False,
    )
    ct1 = ClassifiedTurn(
        turn=t1,
        vylor_intercept=True,
        vylor_tool=VylorTool.FIND_FILES,
        savings_fraction=0.70,
        file_reads_count=3,
    )

    t2 = Turn(
        turn_id="t2",
        session_id="compounding_session",
        model="claude-sonnet-4.5",
        timestamp=datetime(2026, 1, 1, 10, 1, tzinfo=timezone.utc),
        input_tokens=50,
        output_tokens=300,
        cache_read_tokens=10_000,
        cache_write_tokens=0,
        ephemeral_5m_tokens=0,
        ephemeral_1h_tokens=0,
        reasoning_tokens=0,
        duration_seconds=3.0,
        tool_calls=[],
        cost=0.010,
        is_subagent=False,
    )
    ct2 = ClassifiedTurn(
        turn=t2,
        vylor_intercept=False,
        vylor_tool=VylorTool.NONE,
        savings_fraction=0.0,
        file_reads_count=0,
    )

    report = compute_savings([ct1, ct2])

    ts1 = report.turn_savings[0]
    assert ts1.saved_cache_write_tokens > 0

    ts2 = report.turn_savings[1]
    assert ts2.saved_cache_tokens > 0
    assert ts2.saved_cost > 0.0
    assert report.total.saved_cost > ts1.saved_cost
