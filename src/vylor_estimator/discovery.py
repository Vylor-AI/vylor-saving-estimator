from __future__ import annotations

import os
import platform
from abc import ABC, abstractmethod
from pathlib import Path


class ISessionDiscoverer(ABC):
    @abstractmethod
    def discover(self) -> list[Path]:
        """Return a deduplicated, sorted list of .jsonl session files."""


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
        """Recursively collect all .jsonl files in a directory, sorted by name (excluding Codex rollouts)."""
        results: list[Path] = []
        for root, _dirs, fnames in os.walk(directory):
            for fname in fnames:
                if fname.lower().endswith(".jsonl") and not fname.lower().startswith("rollout-"):
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


class CodexSessionDiscoverer(ISessionDiscoverer):
    """Discovers OpenAI Codex session rollout JSONL files from platform defaults or explicit paths."""

    def __init__(
        self,
        explicit_path: str | Path | None = None,
        home_path: Path | None = None,
    ) -> None:
        self.explicit_path = Path(explicit_path) if explicit_path else None
        self.home_path = home_path or Path.home()

    def get_candidate_paths(self) -> list[Path]:
        codex_home = os.environ.get("CODEX_HOME")
        base = Path(codex_home) if codex_home else (self.home_path / ".codex")
        candidates = [
            base / "sessions",
            base / "archived_sessions",
        ]
        return [c for c in candidates if c.exists()]

    def walk_rollouts(self, directory: Path) -> list[Path]:
        results: list[Path] = []
        for root, _dirs, fnames in os.walk(directory):
            for fname in fnames:
                if fname.lower().startswith("rollout-") and fname.lower().endswith(".jsonl"):
                    results.append(Path(root) / fname)
        return sorted(set(results))

    def discover(self) -> list[Path]:
        if self.explicit_path is not None:
            target = self.explicit_path
            if target.is_file():
                if target.suffix.lower() != ".jsonl":
                    raise ValueError(f"Not a .jsonl file: {target}")
                return [target]
            if target.is_dir():
                return self.walk_rollouts(target)
            raise FileNotFoundError(f"Path does not exist: {target}")

        candidates = self.get_candidate_paths()
        if not candidates:
            raise FileNotFoundError(
                "Could not auto-discover Codex session directory.\n"
                "Pass an explicit path: vylor-estimate /path/to/sessions/"
            )

        files: list[Path] = []
        for c in candidates:
            files.extend(self.walk_rollouts(c))

        if not files:
            raise FileNotFoundError(
                "No Codex rollout-.jsonl session files found.\n"
                "Pass an explicit path: vylor-estimate /path/to/sessions/"
            )
        return sorted(set(files))


def _is_codex_file(path: Path) -> bool:
    if path.name.lower().startswith("rollout-"):
        return True
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    return '"type":"session_meta"' in line or '"type": "session_meta"' in line
    except OSError:
        pass
    return False


class CompositeDiscoverer(ISessionDiscoverer):
    """Combines Claude and Codex session discovery."""

    def __init__(
        self,
        explicit_path: str | Path | None = None,
        claude_discoverer: ISessionDiscoverer | None = None,
        codex_discoverer: ISessionDiscoverer | None = None,
        agent: str = "all",
    ) -> None:
        self.explicit_path = Path(explicit_path) if explicit_path else None
        self.claude_discoverer = claude_discoverer or ClaudeSessionDiscoverer(explicit_path=explicit_path)
        self.codex_discoverer = codex_discoverer or CodexSessionDiscoverer(explicit_path=explicit_path)
        self.agent = agent.lower().strip()

    def discover(self) -> list[Path]:
        if self.explicit_path is not None:
            target = self.explicit_path
            if target.is_file():
                if target.suffix.lower() != ".jsonl":
                    raise ValueError(f"Not a .jsonl file: {target}")
                is_codex = _is_codex_file(target)
                if self.agent == "claude" and is_codex:
                    raise FileNotFoundError(
                        f"File {target.name} is a Codex session, but agent filter is set to 'claude'."
                    )
                if self.agent == "codex" and not is_codex:
                    raise FileNotFoundError(
                        f"File {target.name} is a Claude session, but agent filter is set to 'codex'."
                    )
                return [target]

            if target.is_dir():
                all_jsonl: list[Path] = []
                for root, _dirs, fnames in os.walk(target):
                    for fname in fnames:
                        if fname.lower().endswith(".jsonl"):
                            all_jsonl.append(Path(root) / fname)

                claude_files: list[Path] = []
                codex_files: list[Path] = []
                for p in all_jsonl:
                    if _is_codex_file(p):
                        if self.agent in ("all", "codex"):
                            codex_files.append(p)
                    else:
                        if self.agent in ("all", "claude"):
                            claude_files.append(p)

                all_files = sorted(set(claude_files + codex_files))
                if not all_files:
                    raise FileNotFoundError(f"No matching session files found in: {target}")
                return all_files

            raise FileNotFoundError(f"Path does not exist: {target}")

        files: list[Path] = []
        if self.agent in ("all", "claude"):
            try:
                files.extend(self.claude_discoverer.discover())
            except FileNotFoundError:
                pass

        if self.agent in ("all", "codex"):
            try:
                files.extend(self.codex_discoverer.discover())
            except FileNotFoundError:
                pass

        if not files:
            raise FileNotFoundError(
                "Could not auto-discover any Claude or Codex session directories.\n"
                "Pass an explicit path: vylor-estimate /path/to/sessions/"
            )
        return sorted(set(files))
