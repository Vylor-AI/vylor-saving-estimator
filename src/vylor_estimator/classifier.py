
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from vylor_estimator.parser import Turn


class VylorTool(str, Enum):
    REPO_MAP = "request_repo_map"
    FIND_FILES = "find_files"
    FIND_CODE_DEFINITION = "find_code_definition"



_REPO_MAP_TOOLS: frozenset[str] = frozenset([
    "list_dir", "ls", "find", "tree", "directory_listing",
    "list_directory", "get_directory_structure", "explore_directory",
    "list_files_in_dir", "listdir", "glob",
])

_FILE_READ_TOOLS: frozenset[str] = frozenset([
    "read_file", "cat", "view_file", "open_file", "get_file_contents",
    "read_files", "fetch_file", "file_read", "read",
])

_SYMBOL_SEARCH_TOOLS: frozenset[str] = frozenset([
    "grep_search", "search_files", "ripgrep", "grep", "search_in_file",
    "find_in_file", "code_search", "search_code", "codebase_search",
    "semantic_search", "search_symbol", "find_symbol",
])

_VYLOR_TOOLS: frozenset[str] = frozenset([
    "mcp__vylor__request_repo_map",
    "vylor__request_repo_map",
    "request_repo_map",
    "mcp__vylor__find_files",
    "vylor__find_files",
    "find_files",
    "mcp__vylor__find_code_definition",
    "vylor__find_code_definition",
    "find_code_definition",
])


@dataclass
class ClassifiedTurn:
    turn: Turn
    vylor_intercept: bool
    vylor_tool: VylorTool | None = None
    file_reads_count: int = 0


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
        repo_map_tools: frozenset[str] = _REPO_MAP_TOOLS,
        file_read_tools: frozenset[str] = _FILE_READ_TOOLS,
        symbol_search_tools: frozenset[str] = _SYMBOL_SEARCH_TOOLS,
        vylor_tools: frozenset[str] = _VYLOR_TOOLS,
    ) -> None:
        self.repo_map_tools = repo_map_tools
        self.file_read_tools = file_read_tools
        self.symbol_search_tools = symbol_search_tools
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

        # If turn already uses Vylor, it is not an unoptimized baseline turn to intercept
        if any(self.is_vylor_tool(n) for n in tool_names):
            return ClassifiedTurn(
                turn=turn,
                vylor_intercept=False,
                file_reads_count=0,
            )

        file_reads = sum(1 for n in tool_names if n in self.file_read_tools)
        has_repo_map = any(n in self.repo_map_tools for n in tool_names)
        has_symbol_search = any(n in self.symbol_search_tools for n in tool_names)

        tool: VylorTool | None = None
        if has_repo_map:
            tool = VylorTool.REPO_MAP
        elif has_symbol_search:
            tool = VylorTool.FIND_CODE_DEFINITION
        elif file_reads > 0:
            tool = VylorTool.FIND_FILES

        if tool is not None:
            return ClassifiedTurn(
                turn=turn,
                vylor_intercept=True,
                vylor_tool=tool,
                file_reads_count=file_reads,
            )

        return ClassifiedTurn(
            turn=turn,
            vylor_intercept=False,
            file_reads_count=file_reads,
        )

    def classify(self, turns: list[Turn]) -> list[ClassifiedTurn]:
        self.skipped_sessions = {
            t.session_id
            for t in turns
            if any(self.is_vylor_tool(tc.name) for tc in t.tool_calls)
        }
        eligible_turns = [t for t in turns if t.session_id not in self.skipped_sessions]
        return [self.classify_turn(t) for t in eligible_turns]
