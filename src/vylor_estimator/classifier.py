from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vylor_estimator.parser import ToolCall, Turn
from vylor_estimator.shell import (
    SOURCE_LIST,
    SOURCE_SEARCH,
    _best_source,
    classify_command,
    extract_command,
)

SOURCE_SUBAGENTS = "Sub-agents"

__all__ = [
    "ClassifiedTurn",
    "ITurnClassifier",
    "PatternTurnClassifier",
    "SOURCE_LIST",
    "SOURCE_SEARCH",
    "SOURCE_SUBAGENTS",
    "classify_command",
]

# Dedicated exploration tools (matched by tool name).
_SEARCH_TOOLS: frozenset[str] = frozenset(["grep", "grep_search", "ripgrep"])
_LIST_TOOLS: frozenset[str] = frozenset(["glob", "list_dir", "ls", "find"])

# Shell tools: only wasteful when the command itself is exploration.
_SHELL_TOOLS: frozenset[str] = frozenset(["bash", "powershell", "run_command", "cmd"])

_SHELL_EXEC_TOOLS: frozenset[str] = frozenset([
    "bash", "powershell", "run_command", "cmd",
    "list_dir", "ls", "find", "glob",
    "grep", "grep_search", "ripgrep",
])

_VYLOR_TOOLS: frozenset[str] = frozenset([
    "mcp__vylor__request_repo_map",
    "request_repo_map",
    "mcp__vylor__find_files",
    "find_files",
    "mcp__vylor__find_code_definition",
    "find_code_definition",
])


@dataclass
class ClassifiedTurn:
    turn: Turn
    vylor_intercept: bool
    source: str | None = None


class ITurnClassifier(ABC):
    @abstractmethod
    def classify_turn(self, turn: Turn) -> ClassifiedTurn:
        """Classify a single turn."""

    @abstractmethod
    def classify(self, turns: list[Turn]) -> list[ClassifiedTurn]:
        """Classify a sequence of turns."""


class PatternTurnClassifier(ITurnClassifier):
    def __init__(
        self,
        shell_exec_tools: frozenset[str] = _SHELL_EXEC_TOOLS,
        vylor_tools: frozenset[str] = _VYLOR_TOOLS,
    ) -> None:
        self.shell_exec_tools = shell_exec_tools
        self.vylor_tools = vylor_tools
        self.skipped_sessions: set[str] = set()

    def is_vylor_tool(self, name: str) -> bool:
        n = name.lower().strip()
        return (
            n.startswith(("mcp__vylor__", "vylor__"))
            or n in self.vylor_tools
        )

    def _tool_call_source(self, tc: ToolCall) -> str | None:
        """Exploration source of one tool call, or None if it is not pure exploration."""
        name = tc.name.lower().strip()
        if name not in self.shell_exec_tools:
            return None
        if name in _SEARCH_TOOLS:
            return SOURCE_SEARCH
        if name in _LIST_TOOLS:
            return SOURCE_LIST
        if name in _SHELL_TOOLS:
            return classify_command(extract_command(tc.args))
        return None

    def classify_turn(self, turn: Turn) -> ClassifiedTurn:
        sub_source = SOURCE_SUBAGENTS if turn.is_subagent else None

        if any(self.is_vylor_tool(tc.name) for tc in turn.tool_calls):
            return ClassifiedTurn(turn=turn, vylor_intercept=False, source=sub_source)

        if not turn.tool_calls:
            return ClassifiedTurn(turn=turn, vylor_intercept=False, source=sub_source)

        # Pure exploration only: every tool call in the turn must be exploration.
        sources = {self._tool_call_source(tc) for tc in turn.tool_calls}
        if None in sources:
            return ClassifiedTurn(turn=turn, vylor_intercept=False, source=sub_source)

        explore_source = _best_source({s for s in sources if s is not None})
        return ClassifiedTurn(
            turn=turn,
            vylor_intercept=True,
            source=sub_source or explore_source,
        )

    def classify(self, turns: list[Turn]) -> list[ClassifiedTurn]:
        self.skipped_sessions = {
            t.session_id
            for t in turns
            if any(self.is_vylor_tool(tc.name) for tc in t.tool_calls)
        }
        eligible_turns = [t for t in turns if t.session_id not in self.skipped_sessions]
        return [self.classify_turn(t) for t in eligible_turns]
