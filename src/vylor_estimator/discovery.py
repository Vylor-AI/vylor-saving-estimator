"""discovery.py -- Auto-discover Claude session JSONL files by OS platform."""
from __future__ import annotations

import os
import platform
import sys
from abc import ABC, abstractmethod
from pathlib import Path


class ISessionDiscoverer(ABC):
    """Abstract interface for session discovery."""

    @abstractmethod
    def discover(self) -> list[Path]:
        """Return a deduplicated, sorted list of .jsonl session files."""
        pass


class ClaudeSessionDiscoverer(ISessionDiscoverer):
    """Discovers Claude session JSONL files from platform defaults or explicit paths."""

    def __init__(
        self,
        explicit_path: str | Path | None = None,
        platform_name: str | None = None,
        home_path: Path | None = None,
    ) -> None:
        self.explicit_path = Path(explicit_path) if explicit_path else None
        self.platform_name = platform_name or platform.system()
        self.home_path = home_path or Path.home()

    def get_candidate_paths(self) -> list[Path]:
        """Return platform-specific candidate directories where Claude stores sessions."""
        home = self.home_path
        candidates: list[Path | None] = []

        if self.platform_name == "Windows":
            appdata = os.environ.get("APPDATA", "")
            candidates = [
                # Claude Code CLI (claude command) stores here: ~/.claude/projects/
                home / ".claude" / "projects",
                # Claude Desktop app stores here: %APPDATA%\Claude\projects\
                Path(appdata) / "Claude" / "projects" if appdata else None,
                Path(appdata) / "claude" / "projects" if appdata else None,
            ]
        elif self.platform_name == "Darwin":
            candidates = [
                home / "Library" / "Application Support" / "Claude" / "projects",
                home / ".claude" / "projects",
            ]
        else:  # Linux and others
            xdg = os.environ.get("XDG_CONFIG_HOME", "")
            candidates = [
                Path(xdg) / "Claude" / "projects" if xdg else None,
                home / ".config" / "Claude" / "projects",
                home / ".claude" / "projects",
            ]

        return [c for c in candidates if c is not None]

    def resolve_default_path(self) -> Path | None:
        """Find the first existing platform candidate path, or None."""
        for candidate in self.get_candidate_paths():
            if candidate.exists():
                return candidate
        return None

    def walk_jsonl(self, directory: Path) -> list[Path]:
        """Recursively collect all .jsonl files in a directory, sorted by name."""
        results: list[Path] = []
        for root, _dirs, fnames in os.walk(directory):
            for fname in fnames:
                if fname.lower().endswith(".jsonl"):
                    results.append(Path(root) / fname)
        return sorted(set(results))

    def discover(self) -> list[Path]:
        """
        Discover session files.

        If explicit_path is provided:
          - If file, returns [explicit_path].
          - If directory, recursively collects .jsonl files.
        Otherwise:
          - Auto-discovers from default platform locations.

        Raises FileNotFoundError if no sessions found.
        Raises ValueError if explicit file is not a .jsonl file.
        """
        if self.explicit_path is not None:
            target = self.explicit_path
            if target.is_file():
                if target.suffix.lower() != ".jsonl":
                    raise ValueError(f"Not a .jsonl file: {target}")
                return [target]
            if target.is_dir():
                return self.walk_jsonl(target)
            raise FileNotFoundError(f"Path does not exist: {target}")

        # Auto-discover
        default = self.resolve_default_path()
        if default is None:
            raise FileNotFoundError(
                "Could not auto-discover Claude session directory.\n"
                "Pass an explicit path: vylor-estimate /path/to/sessions/"
            )

        files = self.walk_jsonl(default)
        if not files:
            raise FileNotFoundError(
                f"No .jsonl session files found in: {default}\n"
                "Pass an explicit path: vylor-estimate /path/to/sessions/"
            )
        return files


def _default_claude_path() -> Path | None:
    """Backward-compatible helper."""
    return ClaudeSessionDiscoverer().resolve_default_path()


def _walk_for_jsonl(directory: Path) -> list[Path]:
    """Backward-compatible helper."""
    return ClaudeSessionDiscoverer().walk_jsonl(directory)


def discover_sessions(path: str | Path | None = None) -> list[Path]:
    """Backward-compatible functional API."""
    return ClaudeSessionDiscoverer(explicit_path=path).discover()
