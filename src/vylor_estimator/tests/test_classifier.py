from datetime import datetime, timezone
from pathlib import Path

import pytest

from vylor_estimator.classifier import (
    SOURCE_LIST,
    SOURCE_SEARCH,
    SOURCE_SUBAGENTS,
    PatternTurnClassifier,
    classify_command,
)
from vylor_estimator.parser import ClaudeJsonlParser, ToolCall, Turn

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def classifier() -> PatternTurnClassifier:
    return PatternTurnClassifier()


@pytest.fixture
def parser() -> ClaudeJsonlParser:
    return ClaudeJsonlParser()


def _turn(name: str, tool_names: list, session_id: str = "s1", is_subagent: bool = False) -> Turn:
    """Build a Turn. Each tool entry is a name, or a (name, args) tuple."""
    calls = [
        ToolCall(name=t[0], args=t[1]) if isinstance(t, tuple) else ToolCall(name=t)
        for t in tool_names
    ]
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
        tool_calls=calls,
        cost=0.01,
        is_subagent=is_subagent,
    )


def _bash(command: str, tool: str = "Bash") -> tuple:
    return (tool, {"command": command})


# ---------------------------------------------------------------------------
# Exploration tools / commands → vylor_intercept=True
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", [
    "ls -la src/",
    "find . -name '*.py'",
    "grep -rn TODO src",
    "cd src && grep -r foo . | head -20",
    "rg 'a|b' --glob '*.py'",
    "git grep -n handler",
    "grep foo file.txt | wc -l",
    "ls a; echo ---; ls b",
])
def test_bash_exploration_commands_intercepted(classifier: PatternTurnClassifier, command: str):
    ct = classifier.classify_turn(_turn("t", [_bash(command)]))
    assert ct.vylor_intercept is True, command


@pytest.mark.parametrize("command", [
    "Get-ChildItem -Recurse -Filter *.py",
    "gci src | Select-String TODO",
    "dir src",
    "findstr /s foo *.py",
])
def test_powershell_exploration_commands_intercepted(
    classifier: PatternTurnClassifier, command: str
):
    ct = classifier.classify_turn(_turn("t", [_bash(command, tool="PowerShell")]))
    assert ct.vylor_intercept is True, command


def test_run_command_uses_commandline_arg(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", [("run_command", {"CommandLine": "ls src"})]))
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


def test_read_file_not_intercepted_for_main_agent(classifier: PatternTurnClassifier):
    """File reads by the main agent are legitimate work and not counted as waste."""
    ct = classifier.classify_turn(_turn("t", ["Read", "Read", "Read"]))
    assert ct.vylor_intercept is False
    assert ct.source is None

    ct2 = classifier.classify_turn(_turn("t", ["read_file"]))
    assert ct2.vylor_intercept is False
    assert ct2.source is None


def test_multiple_exploration_tools_intercepted(classifier: PatternTurnClassifier):
    """A turn made only of search/list exploration calls is waste."""
    ct = classifier.classify_turn(_turn("t", [_bash("ls src"), "grep_search", "Glob"]))
    assert ct.vylor_intercept is True


# ---------------------------------------------------------------------------
# Not pure exploration → vylor_intercept=False
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", [
    "cat README.md",
    "Get-Content README.md",
    "git commit -m 'x'",
    "npm test",
    "git log | grep fix",
    "ls src && npm run build",
    "find . -name '*.pyc' -delete",
    "python -c 'print(1)'",
    "",
])
def test_bash_non_exploration_commands_not_intercepted(
    classifier: PatternTurnClassifier, command: str
):
    ct = classifier.classify_turn(_turn("t", [_bash(command)]))
    assert ct.vylor_intercept is False, command


def test_bash_without_command_arg_not_intercepted(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", ["Bash"]))
    assert ct.vylor_intercept is False


def test_mixed_exploration_and_edit_not_intercepted(classifier: PatternTurnClassifier):
    """Exploration mixed with any non-exploration tool call is not waste."""
    ct = classifier.classify_turn(_turn("t", ["Grep", "Edit"]))
    assert ct.vylor_intercept is False
    ct = classifier.classify_turn(_turn("t", ["Read", _bash("npm test")]))
    assert ct.vylor_intercept is False
    ct = classifier.classify_turn(_turn("t", ["Grep", "Read"]))
    assert ct.vylor_intercept is False


# ---------------------------------------------------------------------------
# Waste source attribution
# ---------------------------------------------------------------------------

def test_source_by_tool(classifier: PatternTurnClassifier):
    assert classifier.classify_turn(_turn("t", ["Grep"])).source == SOURCE_SEARCH
    assert classifier.classify_turn(_turn("t", ["Glob"])).source == SOURCE_LIST
    assert classifier.classify_turn(_turn("t", ["Read"])).source is None


def test_source_by_shell_command(classifier: PatternTurnClassifier):
    assert classifier.classify_turn(_turn("t", [_bash("grep -r x .")])).source == SOURCE_SEARCH
    assert classifier.classify_turn(_turn("t", [_bash("ls")])).source == SOURCE_LIST
    assert classifier.classify_turn(_turn("t", [_bash("cat a.py")])).source is None


def test_source_priority_search_over_list(classifier: PatternTurnClassifier):
    assert classifier.classify_turn(_turn("t", ["Glob", "Grep"])).source == SOURCE_SEARCH


def test_subagent_turn_source(classifier: PatternTurnClassifier):
    ct = classifier.classify_turn(_turn("t", ["Read"], is_subagent=True))
    assert ct.source == SOURCE_SUBAGENTS
    ct = classifier.classify_turn(_turn("t", [], is_subagent=True))
    assert ct.source == SOURCE_SUBAGENTS


# ---------------------------------------------------------------------------
# classify_command
# ---------------------------------------------------------------------------

def test_classify_command_whole_word_matching():
    assert classify_command("concat a b") is None
    assert classify_command("category list") is None
    assert classify_command("cat a b") is None
    assert classify_command("grep a b") == SOURCE_SEARCH


def test_classify_command_quotes_protect_separators():
    assert classify_command('grep "a|b && c; d" file.txt') == SOURCE_SEARCH


def test_classify_command_env_prefix_and_windows_path():
    assert classify_command("LC_ALL=C grep x f") == SOURCE_SEARCH
    assert classify_command("& 'C:\\tools\\rg.exe' foo") == SOURCE_SEARCH
    assert classify_command("ls 2>&1") == SOURCE_LIST


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
    t1 = _turn("t1", ["grep_search"], session_id="standard_session")
    t2_a = _turn("t2_a", ["grep_search"], session_id="vylor_session")
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
