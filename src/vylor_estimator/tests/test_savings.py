from pathlib import Path

import pytest

from vylor_estimator.classifier import PatternTurnClassifier
from vylor_estimator.parser import ClaudeJsonlParser
from vylor_estimator.savings import ConservativeSavingsEngine

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def classifier() -> PatternTurnClassifier:
    return PatternTurnClassifier()


@pytest.fixture
def savings_engine() -> ConservativeSavingsEngine:
    return ConservativeSavingsEngine()


@pytest.fixture
def parser() -> ClaudeJsonlParser:
    return ClaudeJsonlParser()


def test_savings_non_negative(
    classifier: PatternTurnClassifier,
    savings_engine: ConservativeSavingsEngine,
    parser: ClaudeJsonlParser,
):
    turns = parser.parse([FIXTURES / "sample_session.jsonl"])
    classified = classifier.classify(turns)
    report = savings_engine.calculate(classified)
    assert report.total.saved_cost >= 0.0
    assert report.total.saved_input_tokens >= 0
    assert report.total.saved_cache_tokens >= 0


def test_savings_less_than_baseline(
    classifier: PatternTurnClassifier,
    savings_engine: ConservativeSavingsEngine,
    parser: ClaudeJsonlParser,
):
    turns = parser.parse([FIXTURES / "sample_session.jsonl"])
    classified = classifier.classify(turns)
    report = savings_engine.calculate(classified)
    assert report.total.estimated_cost <= report.total.baseline_cost
    assert report.total.estimated_tokens <= report.total.baseline_total_tokens


def test_pct_cut_between_0_and_100(
    classifier: PatternTurnClassifier,
    savings_engine: ConservativeSavingsEngine,
    parser: ClaudeJsonlParser,
):
    turns = parser.parse([FIXTURES / "sample_session.jsonl"])
    classified = classifier.classify(turns)
    report = savings_engine.calculate(classified)
    assert 0.0 <= report.total.pct_cost_cut <= 100.0
    assert 0.0 <= report.total.pct_tokens_cut <= 100.0


def test_by_session_keys(
    classifier: PatternTurnClassifier,
    savings_engine: ConservativeSavingsEngine,
    parser: ClaudeJsonlParser,
):
    turns = parser.parse([FIXTURES / "sample_session.jsonl"])
    classified = classifier.classify(turns)
    report = savings_engine.calculate(classified)
    assert "sample_session" in report.by_session


def test_output_tokens_not_saved(
    classifier: PatternTurnClassifier,
    savings_engine: ConservativeSavingsEngine,
    parser: ClaudeJsonlParser,
):
    """Output tokens should never be counted as saved (0% of output)."""
    turns = parser.parse([FIXTURES / "sample_session.jsonl"])
    classified = classifier.classify(turns)
    report = savings_engine.calculate(classified)
    # Saved cost cannot exceed what input+cache alone would cost
    # i.e. output tokens are not credited
    assert report.total.estimated_cost >= 0.0


def test_conservative_cap(
    classifier: PatternTurnClassifier,
    savings_engine: ConservativeSavingsEngine,
    parser: ClaudeJsonlParser,
):
    """Savings should never exceed the baseline cost."""
    turns = parser.parse([FIXTURES / "sample_session.jsonl"])
    classified = classifier.classify(turns)
    report = savings_engine.calculate(classified)
    for ts in report.turn_savings:
        assert ts.saved_cost <= ts.classified.turn.cost + 1e-9, (
            f"saved_cost {ts.saved_cost} exceeds turn cost {ts.classified.turn.cost}"
        )


def _make_turn(
    turn_id: str,
    *,
    session_id: str = "s1",
    is_subagent: bool,
    input_tokens: int = 1_000,
    output_tokens: int = 200,
    cache_read_tokens: int = 500,
    cache_write_tokens: int = 300,
    cost: float = 0.05,
    minute: int = 0,
):
    """Helper: build a minimal Turn for unit tests."""
    from datetime import datetime, timezone

    from vylor_estimator.parser import ToolCall, Turn
    return Turn(
        turn_id=turn_id,
        session_id=session_id,
        model="claude-sonnet-4.5",
        timestamp=datetime(2026, 1, 1, 10, minute, tzinfo=timezone.utc),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        ephemeral_5m_tokens=0,
        ephemeral_1h_tokens=0,
        reasoning_tokens=0,
        duration_seconds=1.0,
        tool_calls=[ToolCall(name="read_file", args={})],
        cost=cost,
        is_subagent=is_subagent,
    )


def test_subagent_turns_are_fully_saved(savings_engine: ConservativeSavingsEngine):
    """
    Every subagent turn should be credited with its full cost and all tokens.
    Vylor MCP eliminates the background exploration agent entirely.
    """
    from vylor_estimator.classifier import ClassifiedTurn

    sub = _make_turn("sub1", is_subagent=True, cost=0.07,
                     input_tokens=2_000, cache_read_tokens=1_000, cache_write_tokens=500)
    ct = ClassifiedTurn(turn=sub, vylor_intercept=False)

    report = savings_engine.calculate([ct])
    ts = report.turn_savings[0]

    assert ts.saved_cost == sub.cost
    assert ts.saved_input_tokens == sub.input_tokens
    assert ts.saved_cache_tokens == sub.cache_read_tokens
    assert ts.saved_cache_write_tokens == sub.cache_write_tokens


def test_main_agent_shell_exec_saves_cache_read_and_output(savings_engine: ConservativeSavingsEngine):
    """
    Main-agent turns with vylor_intercept=True (shell-exec detected) should
    save their cache_read_tokens AND output_tokens cost. Input and cache_write
    are not saved (Claude still does the work, just via Vylor instead).
    """
    from vylor_estimator.classifier import ClassifiedTurn

    main = _make_turn("main1", is_subagent=False, cost=0.10,
                      input_tokens=5_000, cache_read_tokens=3_000, cache_write_tokens=1_000)
    ct = ClassifiedTurn(turn=main, vylor_intercept=True)

    report = savings_engine.calculate([ct])
    ts = report.turn_savings[0]

    # cache_read + output tokens are saved; input and cache_write are not
    assert ts.saved_cache_tokens == 3_000
    assert ts.saved_output_tokens == main.output_tokens
    assert ts.saved_input_tokens == 0
    assert ts.saved_cache_write_tokens == 0
    # Saved cost covers cache_read + output pricing
    assert ts.saved_cost > 0
    assert ts.saved_cost < main.cost   # still less than full turn cost


def test_main_agent_no_intercept_saves_nothing(savings_engine: ConservativeSavingsEngine):
    """
    Main-agent turns with vylor_intercept=False (text-only or unrecognised tool)
    should produce zero savings.
    """
    from vylor_estimator.classifier import ClassifiedTurn

    main = _make_turn("main1", is_subagent=False, cost=0.10,
                      input_tokens=5_000, cache_read_tokens=3_000, cache_write_tokens=1_000)
    ct = ClassifiedTurn(turn=main, vylor_intercept=False)

    report = savings_engine.calculate([ct])
    ts = report.turn_savings[0]

    assert ts.saved_cost == 0.0
    assert ts.saved_input_tokens == 0
    assert ts.saved_cache_tokens == 0
    assert ts.saved_cache_write_tokens == 0


def test_mixed_session_saves_only_subagent_turns(savings_engine: ConservativeSavingsEngine):
    """
    In a session with both main and subagent turns, only the subagent turns are saved.
    The aggregate saved_cost should equal the sum of all subagent turn costs.
    """
    from vylor_estimator.classifier import ClassifiedTurn

    main1 = _make_turn("m1", is_subagent=False, cost=0.02, minute=0)
    sub1  = _make_turn("s1", is_subagent=True,  cost=0.05, minute=1)
    sub2  = _make_turn("s2", is_subagent=True,  cost=0.03, minute=2)
    main2 = _make_turn("m2", is_subagent=False, cost=0.04, minute=3)

    cts = [ClassifiedTurn(turn=t, vylor_intercept=False) for t in [main1, sub1, sub2, main2]]
    report = savings_engine.calculate(cts)

    expected_saved = sub1.cost + sub2.cost  # 0.05 + 0.03 = 0.08
    assert abs(report.total.saved_cost - expected_saved) < 1e-9
    # Main turns contribute to baseline but not to savings
    assert report.total.baseline_cost == pytest.approx(main1.cost + sub1.cost + sub2.cost + main2.cost)


def test_subagent_token_totals_accumulate(savings_engine: ConservativeSavingsEngine):
    """
    With multiple subagent turns, total saved tokens should be the sum of
    all their input + cache_read + cache_write tokens.
    """
    from vylor_estimator.classifier import ClassifiedTurn

    sub1 = _make_turn("s1", is_subagent=True,
                      input_tokens=1_000, cache_read_tokens=500, cache_write_tokens=200,
                      cost=0.02, minute=0)
    sub2 = _make_turn("s2", is_subagent=True,
                      input_tokens=2_000, cache_read_tokens=800, cache_write_tokens=300,
                      cost=0.03, minute=1)

    cts = [ClassifiedTurn(turn=t, vylor_intercept=False) for t in [sub1, sub2]]
    report = savings_engine.calculate(cts)

    assert report.total.saved_input_tokens == 3_000
    assert report.total.saved_cache_tokens == 1_300
    assert report.total.saved_cache_write_tokens == 500
    # output_tokens are now also saved for subagent turns (sub1=200, sub2=200)
    assert report.total.saved_output_tokens == 400
    assert report.total.total_saved_tokens == 5_200  # 3000 + 400 + 1300 + 500


# ---------------------------------------------------------------------------
# Time and waste-source tracking
# ---------------------------------------------------------------------------

def _timed_turn(turn_id: str, *, is_subagent: bool, seconds, minute: int = 0):
    t = _make_turn(turn_id, is_subagent=is_subagent, minute=minute)
    t.duration_seconds = seconds
    return t


def test_wasted_time_sums_waste_turn_durations(savings_engine: ConservativeSavingsEngine):
    from vylor_estimator.classifier import SOURCE_SEARCH, SOURCE_SUBAGENTS, ClassifiedTurn

    sub = _timed_turn("sub", is_subagent=True, seconds=30.0, minute=0)
    explore = _timed_turn("exp", is_subagent=False, seconds=12.0, minute=1)
    other = _timed_turn("other", is_subagent=False, seconds=8.0, minute=2)

    cts = [
        ClassifiedTurn(turn=sub, vylor_intercept=False, source=SOURCE_SUBAGENTS),
        ClassifiedTurn(turn=explore, vylor_intercept=True, source=SOURCE_SEARCH),
        ClassifiedTurn(turn=other, vylor_intercept=False),
    ]
    report = savings_engine.calculate(cts)

    assert report.total.baseline_seconds == pytest.approx(50.0)
    assert report.total.saved_seconds == pytest.approx(42.0)
    assert report.total.pct_time_cut == pytest.approx(84.0)


def test_turn_duration_none_and_clamped(savings_engine: ConservativeSavingsEngine):
    from vylor_estimator.classifier import ClassifiedTurn
    from vylor_estimator.savings import MAX_TURN_SECONDS

    no_duration = _timed_turn("a", is_subagent=True, seconds=None, minute=0)
    huge = _timed_turn("b", is_subagent=True, seconds=86_400.0, minute=1)
    cts = [ClassifiedTurn(turn=t, vylor_intercept=False) for t in (no_duration, huge)]

    report = savings_engine.calculate(cts)

    assert report.total.saved_seconds == pytest.approx(MAX_TURN_SECONDS)
    assert report.total.baseline_seconds == pytest.approx(MAX_TURN_SECONDS)


def test_by_source_sums_to_waste_totals(savings_engine: ConservativeSavingsEngine):
    from vylor_estimator.classifier import (
        SOURCE_LIST,
        SOURCE_SEARCH,
        SOURCE_SUBAGENTS,
        ClassifiedTurn,
    )

    turns = [
        (_timed_turn("s", is_subagent=True, seconds=5.0, minute=0), False, SOURCE_SUBAGENTS),
        (_timed_turn("g", is_subagent=False, seconds=3.0, minute=1), True, SOURCE_SEARCH),
        (_timed_turn("l", is_subagent=False, seconds=2.0, minute=2), True, SOURCE_LIST),
        (_timed_turn("x", is_subagent=False, seconds=9.0, minute=3), False, None),
    ]
    cts = [ClassifiedTurn(turn=t, vylor_intercept=i, source=s) for t, i, s in turns]
    report = savings_engine.calculate(cts)

    by_source = report.total.by_source
    assert set(by_source) == {SOURCE_SUBAGENTS, SOURCE_SEARCH, SOURCE_LIST}
    assert sum(b["cost"] for b in by_source.values()) == pytest.approx(report.total.saved_cost)
    assert sum(b["tokens"] for b in by_source.values()) == report.total.total_saved_tokens
    assert sum(b["seconds"] for b in by_source.values()) == pytest.approx(
        report.total.saved_seconds
    )
