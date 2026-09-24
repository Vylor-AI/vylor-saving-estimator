"""test_classifier.py -- Tests for the tool-use classifier."""
from __future__ import annotations

from pathlib import Path

from vylor_estimator.classifier import VylorTool, classify_turns
from vylor_estimator.parser import parse_files

FIXTURES = Path(__file__).parent / "fixtures"


def test_list_dir_classified_as_repo_map():
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    repo_map_turns = [ct for ct in classified if ct.vylor_tool == VylorTool.REPO_MAP]
    assert len(repo_map_turns) >= 1


def test_grep_classified_as_symbol_lookup():
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    symbol_turns = [ct for ct in classified if ct.vylor_tool == VylorTool.FIND_CODE_DEFINITION]
    assert len(symbol_turns) >= 1


def test_multi_read_classified_as_find_files():
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    file_turns = [ct for ct in classified if ct.vylor_tool == VylorTool.FIND_FILES]
    assert len(file_turns) >= 1


def test_text_only_turn_not_intercepted():
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    # Last turn (msg_004) is text-only, should not be intercepted
    text_only = [ct for ct in classified if not ct.turn.tool_calls]
    for ct in text_only:
        assert ct.vylor_intercept is False


def test_savings_fraction_conservative():
    turns = parse_files([FIXTURES / "sample_session.jsonl"])
    classified = classify_turns(turns)
    for ct in classified:
        if ct.vylor_intercept:
            assert ct.savings_fraction <= 1.0, f"Savings fraction too aggressive: {ct.savings_fraction}"
        else:
            assert ct.savings_fraction == 0.0


def test_claude_code_read_and_glob_recognized():
    """Verify that Claude Code tools Read and Glob are recognized."""
    from datetime import datetime, timezone
    from vylor_estimator.parser import Turn, ToolCall
    from vylor_estimator.classifier import classify_turn

    t_read = Turn(
        turn_id="t_read",
        session_id="s",
        model="claude-sonnet-4.5",
        timestamp=datetime.now(timezone.utc),
        input_tokens=100,
        output_tokens=100,
        cache_read_tokens=0,
        cache_write_tokens=0,
        ephemeral_5m_tokens=0,
        ephemeral_1h_tokens=0,
        reasoning_tokens=0,
        duration_seconds=1.0,
        tool_calls=[ToolCall(name="Read"), ToolCall(name="Read"), ToolCall(name="Read")],
        cost=0.01,
        is_subagent=False,
    )
    ct_read = classify_turn(t_read)
    assert ct_read.vylor_intercept is True
    assert ct_read.vylor_tool == VylorTool.FIND_FILES

    t_glob = Turn(
        turn_id="t_glob",
        session_id="s",
        model="claude-sonnet-4.5",
        timestamp=datetime.now(timezone.utc),
        input_tokens=100,
        output_tokens=100,
        cache_read_tokens=0,
        cache_write_tokens=0,
        ephemeral_5m_tokens=0,
        ephemeral_1h_tokens=0,
        reasoning_tokens=0,
        duration_seconds=1.0,
        tool_calls=[ToolCall(name="Glob")],
        cost=0.01,
        is_subagent=False,
    )
    ct_glob = classify_turn(t_glob)
    assert ct_glob.vylor_intercept is True
    assert ct_glob.vylor_tool == VylorTool.REPO_MAP
