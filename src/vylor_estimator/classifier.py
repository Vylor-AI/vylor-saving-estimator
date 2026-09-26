
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

_VYLOR_REPO_MAP_NAMES: frozenset[str] = frozenset([
    "mcp__vylor__request_repo_map",
    "vylor__request_repo_map",
    "request_repo_map",
])
_VYLOR_FIND_FILES_NAMES: frozenset[str] = frozenset([
    "mcp__vylor__find_files",
    "vylor__find_files",
])
_VYLOR_FIND_CODE_NAMES: frozenset[str] = frozenset([
    "mcp__vylor__find_code_definition",
    "vylor__find_code_definition",
])


@dataclass
class ClassifiedTurn:
    turn: Turn
    vylor_intercept: bool
    vylor_tool: VylorTool | None = None
    savings_fraction: float = 0.0
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
        vylor_repo_map_names: frozenset[str] = _VYLOR_REPO_MAP_NAMES,
        vylor_find_files_names: frozenset[str] = _VYLOR_FIND_FILES_NAMES,
        vylor_find_code_names: frozenset[str] = _VYLOR_FIND_CODE_NAMES,
    ) -> None:
        self.repo_map_tools = repo_map_tools
        self.file_read_tools = file_read_tools
        self.symbol_search_tools = symbol_search_tools
        self.vylor_repo_map_names = vylor_repo_map_names
        self.vylor_find_files_names = vylor_find_files_names
        self.vylor_find_code_names = vylor_find_code_names

    def classify_turn(self, turn: Turn) -> ClassifiedTurn:
        tool_names = [tc.name.lower().strip() for tc in turn.tool_calls]

        file_reads = sum(1 for n in tool_names if n in self.file_read_tools)

        has_repo_map = any(n in self.repo_map_tools for n in tool_names)
        has_symbol_search = any(n in self.symbol_search_tools for n in tool_names)

        has_vylor_repo_map = any(n in self.vylor_repo_map_names for n in tool_names)
        has_vylor_find_files = any(n in self.vylor_find_files_names for n in tool_names)
        has_vylor_find_code = any(n in self.vylor_find_code_names for n in tool_names)

        tool: VylorTool | None = None

        if has_vylor_repo_map:
            tool = VylorTool.REPO_MAP
        elif has_vylor_find_code:
            tool = VylorTool.FIND_CODE_DEFINITION
        elif has_vylor_find_files:
            tool = VylorTool.FIND_FILES
        elif has_repo_map:
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
                savings_fraction=1.0,
                file_reads_count=file_reads,
            )

        return ClassifiedTurn(
            turn=turn,
            vylor_intercept=False,
            file_reads_count=file_reads,
        )

    def classify(self, turns: list[Turn]) -> list[ClassifiedTurn]:
        return [self.classify_turn(t) for t in turns]