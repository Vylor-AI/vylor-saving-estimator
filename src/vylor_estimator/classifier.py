"""classifier.py -- Detect tool-use patterns that Vylor MCP would have replaced.

Each Turn is analyzed by its tool_calls list and a savings_fraction is assigned.
Conservative model: 1.0 base for input tokens on intercepted turns.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from vylor_estimator.parser import Turn


class VylorTool(str, Enum):
    REPO_MAP = "request_repo_map"
    FIND_FILES = "find_files"
    FIND_CODE_DEFINITION = "find_code_definition"
    NONE = "none"


# Conservative savings fractions (input tokens)
SAVINGS_FRACTION: dict[VylorTool, float] = {
    VylorTool.REPO_MAP: 1.0,
    VylorTool.FIND_FILES: 1.0,
    VylorTool.FIND_CODE_DEFINITION: 1.0,
    VylorTool.NONE: 0.0,
}

# Cache-read savings fraction (served from Vylor index, not context window)
CACHE_READ_SAVINGS_FRACTION = 1.0

# I/O latency fraction of duration that Vylor eliminates
TIME_SAVINGS_FRACTION = 0.60

# --- Tool name patterns -------------------------------------------------------

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

_FILE_READ_THRESHOLD = 3


@dataclass
class ClassifiedTurn:
    turn: Turn
    vylor_intercept: bool
    vylor_tool: VylorTool
    savings_fraction: float      # fraction of input tokens saved
    file_reads_count: int        # number of file-read tool calls in this turn


class ITurnClassifier(ABC):
    """Abstract interface for classifying turns that Vylor can optimize."""

    @abstractmethod
    def classify_turn(self, turn: Turn) -> ClassifiedTurn:
        """Classify a single turn."""
        pass

    @abstractmethod
    def classify(self, turns: list[Turn]) -> list[ClassifiedTurn]:
        """Classify a sequence of turns."""
        pass


class PatternTurnClassifier(ITurnClassifier):
    """Detects tool-use patterns using configurable tool sets, thresholds, and savings fractions."""

    def __init__(
        self,
        savings_fractions: dict[VylorTool, float] | None = None,
        file_read_threshold: int = _FILE_READ_THRESHOLD,
        repo_map_tools: frozenset[str] = _REPO_MAP_TOOLS,
        file_read_tools: frozenset[str] = _FILE_READ_TOOLS,
        symbol_search_tools: frozenset[str] = _SYMBOL_SEARCH_TOOLS,
        vylor_repo_map_names: frozenset[str] = _VYLOR_REPO_MAP_NAMES,
        vylor_find_files_names: frozenset[str] = _VYLOR_FIND_FILES_NAMES,
        vylor_find_code_names: frozenset[str] = _VYLOR_FIND_CODE_NAMES,
    ) -> None:
        self.savings_fractions = savings_fractions or dict(SAVINGS_FRACTION)
        self.file_read_threshold = file_read_threshold
        self.repo_map_tools = repo_map_tools
        self.file_read_tools = file_read_tools
        self.symbol_search_tools = symbol_search_tools
        self.vylor_repo_map_names = vylor_repo_map_names
        self.vylor_find_files_names = vylor_find_files_names
        self.vylor_find_code_names = vylor_find_code_names

    def classify_turn(self, turn: Turn) -> ClassifiedTurn:
        """Classify a single turn and return its Vylor interception metadata."""
        tool_names = [tc.name.lower().strip() for tc in turn.tool_calls]

        # Count file reads
        file_reads = sum(1 for n in tool_names if n in self.file_read_tools)

        # Check for patterns
        has_repo_map = any(n in self.repo_map_tools for n in tool_names)
        has_symbol_search = any(n in self.symbol_search_tools for n in tool_names)

        # Check cache-heavy turns
        is_cache_heavy = (
            turn.cache_read_tokens > 0
            and turn.cache_read_tokens > turn.input_tokens * 0.3
        )

        # Check if Vylor was already used
        has_vylor_repo_map = any(n in self.vylor_repo_map_names for n in tool_names)
        has_vylor_find_files = any(n in self.vylor_find_files_names for n in tool_names)
        has_vylor_find_code = any(n in self.vylor_find_code_names for n in tool_names)

        # Priority 1: Vylor already used
        if has_vylor_repo_map:
            return ClassifiedTurn(
                turn=turn,
                vylor_intercept=True,
                vylor_tool=VylorTool.REPO_MAP,
                savings_fraction=self.savings_fractions.get(VylorTool.REPO_MAP, 1.0),
                file_reads_count=file_reads,
            )

        if has_vylor_find_code:
            return ClassifiedTurn(
                turn=turn,
                vylor_intercept=True,
                vylor_tool=VylorTool.FIND_CODE_DEFINITION,
                savings_fraction=self.savings_fractions.get(VylorTool.FIND_CODE_DEFINITION, 1.0),
                file_reads_count=file_reads,
            )

        if has_vylor_find_files:
            return ClassifiedTurn(
                turn=turn,
                vylor_intercept=True,
                vylor_tool=VylorTool.FIND_FILES,
                savings_fraction=self.savings_fractions.get(VylorTool.FIND_FILES, 1.0),
                file_reads_count=file_reads,
            )

        # Priority 2: Standard un-intercepted patterns
        if has_repo_map:
            return ClassifiedTurn(
                turn=turn,
                vylor_intercept=True,
                vylor_tool=VylorTool.REPO_MAP,
                savings_fraction=self.savings_fractions.get(VylorTool.REPO_MAP, 1.0),
                file_reads_count=file_reads,
            )

        if has_symbol_search:
            return ClassifiedTurn(
                turn=turn,
                vylor_intercept=True,
                vylor_tool=VylorTool.FIND_CODE_DEFINITION,
                savings_fraction=self.savings_fractions.get(VylorTool.FIND_CODE_DEFINITION, 1.0),
                file_reads_count=file_reads,
            )

        if file_reads >= self.file_read_threshold:
            return ClassifiedTurn(
                turn=turn,
                vylor_intercept=True,
                vylor_tool=VylorTool.FIND_FILES,
                savings_fraction=self.savings_fractions.get(VylorTool.FIND_FILES, 1.0),
                file_reads_count=file_reads,
            )

        # Cache-heavy turns with file navigation
        if is_cache_heavy and file_reads > 0:
            return ClassifiedTurn(
                turn=turn,
                vylor_intercept=True,
                vylor_tool=VylorTool.FIND_FILES,
                savings_fraction=self.savings_fractions.get(VylorTool.FIND_FILES, 1.0) * 0.5,
                file_reads_count=file_reads,
            )

        return ClassifiedTurn(
            turn=turn,
            vylor_intercept=False,
            vylor_tool=VylorTool.NONE,
            savings_fraction=0.0,
            file_reads_count=file_reads,
        )

    def classify(self, turns: list[Turn]) -> list[ClassifiedTurn]:
        """Classify a list of turns."""
        return [self.classify_turn(t) for t in turns]


# Default classifier instance
_DEFAULT_CLASSIFIER = PatternTurnClassifier()


def classify_turn(turn: Turn) -> ClassifiedTurn:
    """Backward-compatible helper function."""
    return _DEFAULT_CLASSIFIER.classify_turn(turn)


def classify_turns(turns: list[Turn]) -> list[ClassifiedTurn]:
    """Backward-compatible helper function."""
    return _DEFAULT_CLASSIFIER.classify(turns)
