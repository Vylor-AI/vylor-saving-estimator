from datetime import datetime, timezone
from pathlib import Path

import pytest

from vylor_estimator.classifier import PatternTurnClassifier
from vylor_estimator.parser import ClaudeJsonlParser, Turn, ToolCall

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def classifier() -> PatternTurnClassifier:
    return PatternTurnClassifier()


@pytest.fixture
def parser() -> ClaudeJsonlParser:
    return ClaudeJsonlParser()


def _turn(name: str, tool_names: list[str], session_id: str = "s1", is_subagent: bool = False) -> Turn:
    return Turn(
        turn_id=name,
        session_id=session_id,
        model="claude-sonnet-4.5",
        timestamp=datetime.now(timezone.utc),
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=5_000,
        cache_write_tokens=0,
        ephemeral_5m_tokens=0,
        ephemeral_1h_tokens=0,
        reasoning_tokens=0,
        duration_seconds=1.0,
        tool_calls=[ToolCall(name=n) for n in tool_names],
        cost=0.01,
        is_subagent=is_subagent,
    )


# ---------------------------------------------------------------------------
# Shell-exec tools → vylor_intercept=True
# ---------------------------------------------------------------------------

def test_bash_call_intercepted(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", ["Bash"]))
    assert ct.vylor_intercept is True


def test_powershell_call_intercepted(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", ["PowerShell"]))
    assert ct.vylor_intercept is True


def test_grep_call_intercepted(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", ["grep_search"]))
    assert ct.vylor_intercept is True


def test_glob_call_intercepted(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", ["Glob"]))
    assert ct.vylor_intercept is True


def test_list_dir_intercepted(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", ["list_dir"]))
    assert ct.vylor_intercept is True


def test_read_file_intercepted(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", ["Read", "Read", "Read"]))
    assert ct.vylor_intercept is True


def test_multi_tool_turn_intercepted(classifier: PatternTurnClassifier):
    """A turn mixing bash + grep should still be intercepted."""
    ct = classifier.classify_turn(_turn("t", ["Bash", "grep_search"]))
    assert ct.vylor_intercept is True


# ---------------------------------------------------------------------------
# Text-only / reasoning turns → vylor_intercept=False
# ---------------------------------------------------------------------------

def test_text_only_turn_not_intercepted(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", []))
    assert ct.vylor_intercept is False


def test_unknown_tool_not_intercepted(classifier: PatternTurnClassifier):
    """An unrecognised tool name should not be intercepted."""
    ct = classifier.classify_turn(_turn("t", ["some_custom_tool_xyz"]))
    assert ct.vylor_intercept is False


# ---------------------------------------------------------------------------
# Already-Vylor turns → vylor_intercept=False (already optimised)
# ---------------------------------------------------------------------------

def test_vylor_tools_not_intercepted_in_classify_turn(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", ["mcp__vylor__find_files"]))
    assert ct.vylor_intercept is False


def test_vylor_prefix_variants_not_intercepted(classifier: PatternTurnClassifier):
    for name in ["mcp__vylor__request_repo_map", "vylor__find_files", "find_code_definition"]:
        ct = classifier.classify_turn(_turn("t", [name]))
        assert ct.vylor_intercept is False, f"Expected False for {name}"


# ---------------------------------------------------------------------------
# Session skipping
# ---------------------------------------------------------------------------

def test_sessions_with_vylor_tools_skipped_entirely(classifier: PatternTurnClassifier):
    """Sessions containing any Vylor tool call are completely excluded."""
    t1 = _turn("t1", ["read_file"], session_id="standard_session")
    t2_a = _turn("t2_a", ["read_file"], session_id="vylor_session")
    t2_b = _turn("t2_b", ["mcp__vylor__request_repo_map"], session_id="vylor_session")

    classified = classifier.classify([t1, t2_a, t2_b])

    assert len(classified) == 1
    assert classified[0].turn.session_id == "standard_session"
    assert "vylor_session" in classifier.skipped_sessions


# ---------------------------------------------------------------------------
# Fixture-based smoke tests
# ---------------------------------------------------------------------------

def test_sample_session_has_intercepted_turns(
    classifier: PatternTurnClassifier,
    parser: ClaudeJsonlParser,
):
    """Sample fixture session should have at least one shell-exec intercepted turn."""
    turns = parser.parse([FIXTURES / "sample_session.jsonl"])
    classified = classifier.classify(turns)
    intercepted = [ct for ct in classified if ct.vylor_intercept]
    assert len(intercepted) >= 1


def test_text_only_turns_never_intercepted(
    classifier: PatternTurnClassifier,
    parser: ClaudeJsonlParser,
):
    turns = parser.parse([FIXTURES / "sample_session.jsonl"])
    classified = classifier.classify(turns)
    for ct in classified:
        if not ct.turn.tool_calls:
            assert ct.vylor_intercept is False
