from __future__ import annotations

import re
from typing import Any

SOURCE_SEARCH = "Searches"
SOURCE_LIST = "Directory listing"

_SOURCE_PRIORITY: tuple[str, ...] = (SOURCE_SEARCH, SOURCE_LIST)

_COMMAND_ARG_KEYS: tuple[str, ...] = ("command", "CommandLine", "cmd", "script")

_SEARCH_COMMANDS: frozenset[str] = frozenset([
    "grep", "egrep", "fgrep", "rg", "ag", "ack", "findstr", "select-string", "sls",
])
_LIST_COMMANDS: frozenset[str] = frozenset([
    "ls", "ll", "find", "tree", "dir", "get-childitem", "gci", "fd", "locate",
])

_FILTER_COMMANDS: frozenset[str] = frozenset([
    "wc", "sort", "uniq", "cut", "tr", "head", "tail",
    "select-object", "sort-object", "measure-object", "where-object",
    "format-table", "out-string",
])

_NEUTRAL_COMMANDS: frozenset[str] = frozenset([
    "cd", "pushd", "popd", "pwd", "echo", "set-location", "sl", "write-host", "write-output",
])
_COMMAND_PREFIXES: frozenset[str] = frozenset(["sudo", "time", "command", "builtin", "&", "nohup"])

_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _split_command(command: str) -> list[tuple[str, bool]]:
    """
    Split a shell command into ``(segment, is_pipe_stage)`` pairs.

    Splits on ``&&``, ``||``, ``;``, newlines (new command) and ``|`` (pipe
    stage), ignoring separators inside single or double quotes so patterns
    like ``grep "a|b" file`` stay intact.
    """
    segments: list[tuple[str, bool]] = []
    buf: list[str] = []
    quote: str | None = None
    piped = False
    i, n = 0, len(command)

    def flush(next_piped: bool) -> None:
        nonlocal piped
        text = "".join(buf).strip()
        if text:
            segments.append((text, piped))
        buf.clear()
        piped = next_piped

    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n and command[i + 1] == '"':
                buf.append('"')
                i += 1
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        two = command[i:i + 2]
        if two in ("&&", "||"):
            flush(False)
            i += 2
            continue
        if ch in (";", "\n", "\r"):
            flush(False)
        elif ch == "|":
            flush(True)
        else:
            buf.append(ch)
        i += 1
    flush(False)
    return segments


def _command_word(segment: str) -> tuple[str, list[str]]:
    """Return the lowercased command word of a segment and its remaining tokens."""
    tokens = segment.split()
    idx = 0
    while idx < len(tokens) and (
        tokens[idx].lower() in _COMMAND_PREFIXES or _ENV_ASSIGNMENT.match(tokens[idx])
    ):
        idx += 1
    if idx >= len(tokens):
        return "", []
    word = tokens[idx].strip("'\"").replace("\\", "/").rsplit("/", 1)[-1].lower()
    word = word.removesuffix(".exe")
    return word, tokens[idx + 1:]


def _segment_source(word: str, args: list[str]) -> str | None:
    """Classify one command word as an exploration source, or None."""
    if word in _SEARCH_COMMANDS:
        return SOURCE_SEARCH
    if word in _LIST_COMMANDS:
        if word == "find" and any(a in ("-delete", "-exec", "-execdir") for a in args):
            return None
        return SOURCE_LIST
    if word == "git" and args:
        if args[0] == "grep":
            return SOURCE_SEARCH
        if args[0] == "ls-files":
            return SOURCE_LIST
    return None


def _best_source(sources: set[str]) -> str | None:
    for source in _SOURCE_PRIORITY:
        if source in sources:
            return source
    return None


def classify_command(command: str) -> str | None:
    """
    Decide whether a shell command is pure file exploration.

    Returns the overhead source (search / list) or None when the command
    does anything else (reading files, editing, testing, building, etc.).
    """
    found: set[str] = set()
    for segment, is_pipe_stage in _split_command(command):
        word, args = _command_word(segment)
        if not word:
            continue
        source = _segment_source(word, args)
        if source is not None:
            found.add(source)
            continue
        if is_pipe_stage and word in _FILTER_COMMANDS:
            continue
        if not is_pipe_stage and word in _NEUTRAL_COMMANDS:
            continue
        return None
    return _best_source(found)


def extract_command(args: dict[str, Any]) -> str:
    """Extract shell command string from tool args across different schemas."""
    for key in _COMMAND_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""
