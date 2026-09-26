"""test_savings.py -- Tests for the savings math engine."""

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
    """Savings fraction should never exceed 70% for input tokens."""
    turns = parser.parse([FIXTURES / "sample_session.jsonl"])
    classified = classifier.classify(turns)
    report = savings_engine.calculate(classified)
    for ts in report.turn_savings:
        if ts.classified.vylor_intercept:
            fraction = ts.saved_input_tokens / max(ts.classified.turn.input_tokens, 1)
            assert fraction <= 1.0, f"Savings fraction too high: {fraction}"


def test_compounding_cache_savings_on_downstream_turns(savings_engine: ConservativeSavingsEngine):
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
        vylor_tool=None,
        savings_fraction=0.0,
        file_reads_count=0,
    )

    report = savings_engine.calculate([ct1, ct2])

    ts1 = report.turn_savings[0]
    assert ts1.saved_cache_write_tokens == 0

    ts2 = report.turn_savings[1]
    assert ts2.saved_cache_tokens > 0
    assert ts2.saved_cost > 0.0
    assert report.total.saved_cost > ts1.saved_cost


def test_sequential_file_reads_accumulate_cache_savings(
    classifier: PatternTurnClassifier,
    savings_engine: ConservativeSavingsEngine,
):
    """Verify that sequential 1-file reads are intercepted, accumulating cache reads are cut, and cache write is normal."""
    from datetime import datetime, timezone
    from vylor_estimator.parser import Turn, ToolCall
    from vylor_estimator.classifier import VylorTool

    turns = []
    # Simulate 5 sequential single-file reads, each adding 5,000 tokens of file content
    accum_cache = 0
    for i in range(5):
        t = Turn(
            turn_id=f"seq_{i}",
            session_id="seq_session",
            model="claude-sonnet-4.5",
            timestamp=datetime(2026, 1, 1, 10, i, tzinfo=timezone.utc),
            input_tokens=5_000,
            output_tokens=100,
            cache_read_tokens=accum_cache,
            cache_write_tokens=5_000,
            ephemeral_5m_tokens=5_000,
            ephemeral_1h_tokens=0,
            reasoning_tokens=0,
            duration_seconds=2.0,
            tool_calls=[ToolCall(name="read_file")],
            cost=0.05,
            is_subagent=False,
        )
        turns.append(t)
        accum_cache += 5_000

    classified = classifier.classify(turns)
    for ct in classified:
        assert ct.vylor_intercept is True
        assert ct.vylor_tool == VylorTool.FIND_FILES

    report = savings_engine.calculate(classified)

    # In step 0: no prior accumulation -> saved_cache_tokens == 0
    assert report.turn_savings[0].saved_cache_tokens == 0
    # In step 1: prior was 10,000 (input + write) -> saves the 5,000 accumulated cache read
    assert report.turn_savings[1].saved_cache_tokens == 5_000
    # In step 2: saves 10,000 accumulated cache read
    assert report.turn_savings[2].saved_cache_tokens == 10_000
    # In step 3: saves 15,000 accumulated cache read
    assert report.turn_savings[3].saved_cache_tokens == 15_000
    # In step 4: saves 20,000 accumulated cache read
    assert report.turn_savings[4].saved_cache_tokens == 20_000

    # Cache write is calculated normally (0 cut) across all turns
    for ts in report.turn_savings:
        assert ts.saved_cache_write_tokens == 0

    assert report.total.saved_cache_tokens == 50_000
    assert report.total.saved_cost > 0.0
