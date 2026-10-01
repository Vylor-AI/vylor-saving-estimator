from __future__ import annotations

from pathlib import Path

import pytest

from vylor_estimator.parser import ClaudeJsonlParser
from vylor_estimator.pricing import IPricingCalculator

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def parser() -> ClaudeJsonlParser:
    return ClaudeJsonlParser()


def test_parse_sample_session(parser: ClaudeJsonlParser):
    files = [FIXTURES / "sample_session.jsonl"]
    turns = parser.parse(files)
    assert len(turns) == 4, f"Expected 4 turns, got {len(turns)}"


def test_turn_fields(parser: ClaudeJsonlParser):
    files = [FIXTURES / "sample_session.jsonl"]
    turns = parser.parse(files)
    for t in turns:
        assert t.turn_id
        assert t.model == "claude-sonnet-4.5"
        assert t.input_tokens >= 0
        assert t.output_tokens >= 0
        assert t.cost >= 0.0


def test_tool_calls_extracted(parser: ClaudeJsonlParser):
    files = [FIXTURES / "sample_session.jsonl"]
    turns = parser.parse(files)
    # First turn has list_dir
    list_dir_turns = [t for t in turns if any(tc.name == "list_dir" for tc in t.tool_calls)]
    assert len(list_dir_turns) >= 1

    # Second turn has grep_search
    grep_turns = [t for t in turns if any(tc.name == "grep_search" for tc in t.tool_calls)]
    assert len(grep_turns) >= 1


def test_no_negative_costs(parser: ClaudeJsonlParser):
    files = [FIXTURES / "sample_session.jsonl"]
    turns = parser.parse(files)
    for t in turns:
        assert t.cost >= 0.0, f"Negative cost: {t.cost} for turn {t.turn_id}"


def test_subagent_stitching(tmp_path, parser: ClaudeJsonlParser):
    """Verify that a subagent file under <parent_sess>/subagents/<agent>.jsonl is stitched."""
    import json

    parent_sess = "parent-session-12345"
    sess_dir = tmp_path / parent_sess
    sub_dir = sess_dir / "subagents"
    sub_dir.mkdir(parents=True)

    parent_file = sess_dir.with_suffix(".jsonl")
    sub_file = sub_dir / "agent-sub123.jsonl"

    # Parent event
    parent_msg = {
        "type": "assistant",
        "uuid": "p_msg_1",
        "timestamp": "2026-01-01T10:00:00Z",
        "message": {
            "id": "p_msg_1",
            "model": "claude-sonnet-4.5",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "content": [{"type": "text", "text": "hello"}]
        }
    }
    parent_file.write_text(json.dumps(parent_msg) + "\n", encoding="utf-8")

    # Subagent event
    sub_msg = {
        "type": "assistant",
        "uuid": "s_msg_1",
        "timestamp": "2026-01-01T10:01:00Z",
        "message": {
            "id": "s_msg_1",
            "model": "claude-sonnet-4.5",
            "usage": {"input_tokens": 500, "output_tokens": 100},
            "content": [{"type": "tool_use", "name": "read_file", "input": {"file": "a.py"}}]
        }
    }
    sub_file.write_text(json.dumps(sub_msg) + "\n", encoding="utf-8")

    turns = parser.parse([parent_file, sub_file])
    assert len(turns) == 2

    # Both turns must share the parent session_id
    assert turns[0].session_id == parent_sess
    assert turns[1].session_id == parent_sess

    # Subagent flag
    p_turn = next(t for t in turns if t.turn_id == "p_msg_1")
    s_turn = next(t for t in turns if t.turn_id == "s_msg_1")

    assert p_turn.is_subagent is False
    assert s_turn.is_subagent is True


class StubPricingCalculator(IPricingCalculator):
    """Stub pricing calculator with fixed rate for testing injection."""

    def __init__(self, flat_rate_per_m: float = 100.0) -> None:
        self.flat_rate_per_m = flat_rate_per_m

    def get_pricing(self, model_name: str) -> tuple[float, float, float, float, float]:
        return (self.flat_rate_per_m, 0.0, 0.0, self.flat_rate_per_m, self.flat_rate_per_m)

    def compute_turn_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        ephemeral_5m_tokens: int = 0,
        ephemeral_1h_tokens: int = 0,
    ) -> float:
        return (input_tokens + output_tokens + cache_read_tokens) * self.flat_rate_per_m / 1_000_000.0


def test_pricing_calculator_injection_in_parser():
    """Verify that ClaudeJsonlParser uses the injected IPricingCalculator."""
    stub_pricing = StubPricingCalculator(flat_rate_per_m=100.0)
    parser = ClaudeJsonlParser(pricing_calculator=stub_pricing)
    turns = parser.parse([FIXTURES / "sample_session.jsonl"])

    assert len(turns) > 0
    for t in turns:
        expected = (t.input_tokens + t.output_tokens + t.cache_read_tokens) * 100.0 / 1_000_000.0
        assert pytest.approx(t.cost, rel=1e-5) == expected


def test_corrupted_session_null_usage(tmp_path):
    """Parser must not crash when 'usage' is null in a session file (corrupted/partial session)."""
    import json

    session_file = tmp_path / "corrupt.jsonl"
    good_msg = {
        "type": "assistant",
        "uuid": "good_1",
        "timestamp": "2026-01-01T10:00:00Z",
        "message": {
            "id": "good_1",
            "model": "claude-sonnet-4.5",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "content": [{"type": "text", "text": "hello"}],
        },
    }
    corrupted_msg = {
        "type": "assistant",
        "uuid": "bad_1",
        "timestamp": "2026-01-01T10:01:00Z",
        "message": {
            "id": "bad_1",
            "model": "claude-sonnet-4.5",
            "usage": None,          # null usage — the bug case
            "content": [],
        },
    }
    session_file.write_text(
        json.dumps(good_msg) + "\n" + json.dumps(corrupted_msg) + "\n",
        encoding="utf-8",
    )

    parser = ClaudeJsonlParser()
    # Must not raise — should parse at least the good turn
    turns = parser.parse([session_file])
    assert len(turns) >= 1
    good_turn = next((t for t in turns if t.turn_id == "good_1"), None)
    assert good_turn is not None
    assert good_turn.input_tokens == 100

