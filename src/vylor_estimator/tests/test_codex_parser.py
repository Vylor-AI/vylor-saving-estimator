from __future__ import annotations

from pathlib import Path

import pytest

from vylor_estimator.classifier import PatternTurnClassifier
from vylor_estimator.codex_parser import CodexRolloutParser
from vylor_estimator.parser import CompositeSessionParser
from vylor_estimator.savings import ConservativeSavingsEngine

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def codex_parser() -> CodexRolloutParser:
    return CodexRolloutParser()


def test_parse_single_codex_session(codex_parser: CodexRolloutParser):
    turns = codex_parser.parse([FIXTURES / "sample_codex_session.jsonl"])
    assert len(turns) == 4

    # Turn 0: Get-ChildItem -Force; rg --files
    t0 = turns[0]
    assert t0.turn_id == "resp_main_001"
    assert t0.session_id == "codex_sess_001"
    assert t0.model == "gpt-5.6-terra"
    assert t0.input_tokens == 2000  # 10000 - 8000 cached
    assert t0.cache_read_tokens == 8000
    assert t0.output_tokens == 150
    assert t0.reasoning_tokens == 30
    assert t0.agent == "codex"
    assert not t0.is_subagent
    assert len(t0.tool_calls) == 1
    assert t0.tool_calls[0].name == "exec_command"

    # Turn 1: wait inheriting search cell
    t1 = turns[1]
    assert t1.turn_id == "resp_main_002"
    assert t1.tool_calls[0].name == "wait"
    assert t1.tool_calls[0].args.get("inherits") == "Directory listing"

    # Turn 2: apply_patch
    t2 = turns[2]
    assert t2.tool_calls[0].name == "apply_patch"

    # Turn 3: text only
    t3 = turns[3]
    assert len(t3.tool_calls) == 0


def test_parse_stitched_subagent_session(codex_parser: CodexRolloutParser):
    turns = codex_parser.parse([
        FIXTURES / "sample_codex_session.jsonl",
        FIXTURES / "sample_codex_subagent.jsonl",
    ])
    assert len(turns) == 6  # 4 main + 2 subagent turns

    sub_turns = [t for t in turns if t.is_subagent]
    assert len(sub_turns) == 2
    for st in sub_turns:
        assert st.session_id == "codex_sess_001"
        assert st.agent == "codex"
        assert st.subagent_type == "Explore"  # classified as explore because read-only

    main_turns = [t for t in turns if not t.is_subagent]
    assert len(main_turns) == 4


def test_codex_savings_calculation(codex_parser: CodexRolloutParser):
    turns = codex_parser.parse([
        FIXTURES / "sample_codex_session.jsonl",
        FIXTURES / "sample_codex_subagent.jsonl",
    ])
    classifier = PatternTurnClassifier()
    classified = classifier.classify(turns)

    # Check classifications
    # t0 (rg --files) -> vylor_intercept True, Directory listing
    # t1 (wait cell 1) -> vylor_intercept True, Directory listing
    # t2 (apply_patch) -> vylor_intercept False
    # t3 (text only) -> vylor_intercept False
    # sub_turns -> is_subagent True, subagent_type Explore
    engine = ConservativeSavingsEngine()
    report = engine.calculate(classified)

    assert report.total.turns_analyzed == 6
    assert report.total.subagent_turns == 2
    assert report.total.saved_cost > 0
    assert report.total.total_saved_tokens > 0
    assert "Directory listing" in report.total.by_source
    assert "Sub-agents" in report.total.by_source


def test_composite_session_parser():
    composite = CompositeSessionParser()
    turns = composite.parse([
        FIXTURES / "sample_session.jsonl",
        FIXTURES / "sample_codex_session.jsonl",
    ])
    assert len(turns) == 4 + 4  # 4 claude turns + 4 codex turns

    claude_turns = [t for t in turns if t.agent == "claude"]
    codex_turns = [t for t in turns if t.agent == "codex"]
    assert len(claude_turns) == 4
    assert len(codex_turns) == 4
