from abc import ABC, abstractmethod
from dataclasses import dataclass
from vylor_estimator.parser import Turn


_SHELL_EXEC_TOOLS: frozenset[str] = frozenset([
    "bash", "powershell", "run_command", "cmd",
    "list_dir", "ls", "find", "glob",
    "read_file", "read", "cat", "view_file",
    "grep", "grep_search", "ripgrep"
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
            n.startswith("mcp__vylor__")
            or n.startswith("vylor__")
            or n in self.vylor_tools
        )

    def classify_turn(self, turn: Turn) -> ClassifiedTurn:
        tool_names = [tc.name.lower().strip() for tc in turn.tool_calls]

        if any(self.is_vylor_tool(n) for n in tool_names):
            return ClassifiedTurn(turn=turn, vylor_intercept=False)

        if any(n in self.shell_exec_tools for n in tool_names):
            return ClassifiedTurn(turn=turn, vylor_intercept=True)

        return ClassifiedTurn(turn=turn, vylor_intercept=False)

    def classify(self, turns: list[Turn]) -> list[ClassifiedTurn]:
        self.skipped_sessions = {
            t.session_id
            for t in turns
            if any(self.is_vylor_tool(tc.name) for tc in t.tool_calls)
        }
        eligible_turns = [t for t in turns if t.session_id not in self.skipped_sessions]
        return [self.classify_turn(t) for t in eligible_turns]
