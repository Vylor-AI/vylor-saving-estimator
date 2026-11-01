from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vylor_estimator.pricing import (
    ClaudePricingCalculator,
    IPricingCalculator,
)
from vylor_estimator.shell import extract_command, is_read_only_command

_MUTATING_WRITE_TOOLS: frozenset[str] = frozenset([
    "write", "edit", "multiedit", "notebookedit", "write_to_file",
    "replace_file_content", "multi_replace_file_content", "apply_patch",
])
_SHELL_TOOLS: frozenset[str] = frozenset([
    "bash", "powershell", "run_command", "cmd",
    "exec_command", "shell", "shell_command", "local_shell",
])


def _is_mutating_tool_call(name: str, args: dict[str, Any]) -> bool:
    """True if a tool call edits files or executes a non-read shell command."""
    n = (name or "").lower().strip()
    if n in _MUTATING_WRITE_TOOLS:
        return True
    if n in _SHELL_TOOLS:
        cmd = extract_command(args if isinstance(args, dict) else {})
        return not is_read_only_command(cmd)
    return False


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    turn_id: str
    session_id: str
    model: str
    timestamp: datetime | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    ephemeral_5m_tokens: int
    ephemeral_1h_tokens: int
    reasoning_tokens: int
    duration_seconds: float | None
    tool_calls: list[ToolCall]
    cost: float
    is_subagent: bool
    subagent_type: str | None = None
    agent: str = "claude"


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


class ISessionParser(ABC):
    @abstractmethod
    def parse(self, paths: list[Path]) -> list[Turn]:
        """Parse session files and return a flat list of Turns."""


class ClaudeJsonlParser(ISessionParser):
    """Parser for Claude session .jsonl files with sub-agent stitching and cost calculation."""

    def __init__(self, pricing_calculator: IPricingCalculator | None = None) -> None:
        self.pricing_calculator = pricing_calculator or ClaudePricingCalculator()

    def parse(self, paths: list[Path]) -> list[Turn]:
        events_by_uuid: dict[str, dict] = {}
        messages_map: dict[str, dict] = {}
        message_order: list[str] = []
        tool_use_subagent_type: dict[str, str] = {}
        agent_id_to_subagent_type: dict[str, str] = {}
        agent_id_to_tool_use_id: dict[str, str] = {}
        tool_use_args_by_id: dict[str, dict[str, Any]] = {}

        # First pass: index all events
        for p in paths:
            subagent_file_id: str | None = None
            if "subagents" in p.parts:
                sub_idx = p.parts.index("subagents")
                session_id = p.parts[sub_idx - 1]
                is_subagent_file = True
                subagent_file_id = p.stem.replace("agent-", "")
            else:
                session_id = p.stem
                is_subagent_file = False
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        uuid = data.get("uuid")
                        if uuid:
                            events_by_uuid[uuid] = data

                        tur = data.get("toolUseResult")
                        if isinstance(tur, dict) and tur.get("agentId"):
                            aid = str(tur["agentId"])
                            if tur.get("subagent_type"):
                                agent_id_to_subagent_type[aid] = tur["subagent_type"]
                            msg_content = (data.get("message") or {}).get("content")
                            if isinstance(msg_content, list):
                                for c_item in msg_content:
                                    if isinstance(c_item, dict) and c_item.get("type") == "tool_result":
                                        tuid = c_item.get("tool_use_id")
                                        if tuid:
                                            agent_id_to_tool_use_id[aid] = tuid
                                            if aid not in agent_id_to_subagent_type and tuid in tool_use_subagent_type:
                                                agent_id_to_subagent_type[aid] = tool_use_subagent_type[tuid]

                        content_str = data.get("content")
                        if isinstance(content_str, str) and "<task-id>" in content_str and "<tool-use-id>" in content_str:
                            tid_m = re.search(r"<task-id>([^<]+)</task-id>", content_str)
                            tuid_m = re.search(r"<tool-use-id>([^<]+)</tool-use-id>", content_str)
                            if tid_m and tuid_m:
                                tid = tid_m.group(1)
                                tuid = tuid_m.group(1)
                                if tuid in tool_use_subagent_type:
                                    agent_id_to_subagent_type[tid] = tool_use_subagent_type[tuid]
                                agent_id_to_tool_use_id[tid] = tuid

                        if data.get("type") == "assistant" and "message" in data:
                            msg = data["message"]
                            msg_id = msg.get("id") or data.get("uuid", "")
                            if not msg_id:
                                continue

                            if msg_id not in messages_map:
                                messages_map[msg_id] = {
                                    "contents": [],
                                    "tool_calls_raw": [],
                                    "usage": msg.get("usage") or {},
                                    "model": msg.get("model", ""),
                                    "parent_uuid": data.get("parentUuid"),
                                    "first_timestamp": data.get("timestamp"),
                                    "last_timestamp": data.get("timestamp"),
                                    "session_id": session_id,
                                    "is_subagent": is_subagent_file,
                                    "subagent_file_id": subagent_file_id,
                                }
                                message_order.append(msg_id)

                            entry = messages_map[msg_id]
                            entry["last_timestamp"] = data.get("timestamp")
                            if msg.get("usage"):
                                entry["usage"] = msg.get("usage") or {}
                            if not entry["model"] and msg.get("model"):
                                entry["model"] = msg.get("model", "")

                            for item in msg.get("content", []):
                                if isinstance(item, dict) and item.get("type") == "tool_use":
                                    name = item.get("name", "")
                                    args = item.get("input", {}) or {}
                                    if name in ("Task", "Agent") and isinstance(args, dict):
                                        st = args.get("subagent_type") or args.get("type")
                                        if st and item.get("id"):
                                            tool_use_subagent_type[item["id"]] = st
                                        if item.get("id"):
                                            tool_use_args_by_id[item["id"]] = args

                                    key = f"{name}:{json.dumps(args, sort_keys=True)}"
                                    if key not in {
                                        f"{tc['name']}:{json.dumps(tc['args'], sort_keys=True)}"
                                        for tc in entry["tool_calls_raw"]
                                    }:
                                        entry["tool_calls_raw"].append({"name": name, "args": args})
            except OSError:
                continue

        for msg_id, entry in messages_map.items():
            parent_uuid = entry.get("parent_uuid")
            if parent_uuid and parent_uuid in events_by_uuid:
                parent_event = events_by_uuid[parent_uuid]
                if parent_event.get("type") in ("tool_result", "tool_use"):
                    entry["is_subagent"] = True

        # Behavioral inspection for sub-agents (e.g., general-purpose):
        # If a sub-agent mutates files or runs non-read shell commands, classify it as Worker.
        mutated_subagents: set[str] = set()
        for entry in messages_map.values():
            if not entry.get("is_subagent"):
                continue
            has_mutation = any(
                _is_mutating_tool_call(tc["name"], tc["args"])
                for tc in entry["tool_calls_raw"]
            )
            if not has_mutation:
                continue
            sub_fid = entry.get("subagent_file_id")
            if sub_fid:
                mutated_subagents.add(sub_fid)
            else:
                entry["subagent_type_override"] = "Worker"

        for sub_fid in mutated_subagents:
            if agent_id_to_subagent_type.get(sub_fid) != "Plan":
                agent_id_to_subagent_type[sub_fid] = "Worker"
                tuid = agent_id_to_tool_use_id.get(sub_fid)
                if tuid and tuid in tool_use_args_by_id:
                    tool_use_args_by_id[tuid]["subagent_type"] = "Worker"

        turns: list[Turn] = []
        skipped = 0
        for msg_id in message_order:
            entry = messages_map[msg_id]
            try:
                tool_calls = [
                    ToolCall(name=tc["name"], args=tc["args"])
                    for tc in entry["tool_calls_raw"]
                ]

                end_dt = _parse_iso(entry["last_timestamp"])
                parent_uuid = entry.get("parent_uuid")
                parent_dt = None
                if parent_uuid and parent_uuid in events_by_uuid:
                    parent_dt = _parse_iso(events_by_uuid[parent_uuid].get("timestamp"))

                duration: float | None = None
                if end_dt and parent_dt and end_dt >= parent_dt:
                    duration = (end_dt - parent_dt).total_seconds()
                elif end_dt and entry["first_timestamp"]:
                    start_dt = _parse_iso(entry["first_timestamp"])
                    if start_dt and end_dt >= start_dt:
                        duration = (end_dt - start_dt).total_seconds()

                usage = entry["usage"] or {}
                model = entry["model"] or "unknown"
                cache_creation = usage.get("cache_creation") or {}

                e_5m = cache_creation.get("ephemeral_5m_input_tokens", 0) or 0
                e_1h = cache_creation.get("ephemeral_1h_input_tokens", 0) or 0
                cc_total = usage.get("cache_creation_input_tokens", e_5m + e_1h) or 0
                input_tokens = usage.get("input_tokens", 0) or 0
                output_tokens = usage.get("output_tokens", 0) or 0
                cache_read = usage.get("cache_read_input_tokens", 0) or 0
                reasoning = (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0) or 0

                cost = self.pricing_calculator.compute_turn_cost(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read,
                    ephemeral_5m_tokens=e_5m,
                    ephemeral_1h_tokens=e_1h,
                )

                subagent_type = entry.get("subagent_type_override")
                if not subagent_type and entry.get("subagent_file_id"):
                    subagent_type = agent_id_to_subagent_type.get(entry["subagent_file_id"])

                turns.append(Turn(
                    turn_id=msg_id,
                    session_id=entry["session_id"],
                    model=model,
                    timestamp=end_dt,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cc_total,
                    ephemeral_5m_tokens=e_5m,
                    ephemeral_1h_tokens=e_1h,
                    reasoning_tokens=reasoning,
                    duration_seconds=duration,
                    tool_calls=tool_calls,
                    cost=cost,
                    is_subagent=entry["is_subagent"],
                    subagent_type=subagent_type,
                ))
            except Exception:
                skipped += 1
                continue

        if skipped:
            import sys
            print(f"[dim]Skipped {skipped} malformed turn(s).[/dim]", file=sys.stderr)

        turns.sort(key=lambda t: t.timestamp or datetime.min.replace(tzinfo=timezone.utc))
        return turns


def _is_codex_file(path: Path) -> bool:
    """Check if the first non-empty line of a jsonl file indicates a Codex session."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    return '"type":"session_meta"' in line or '"type": "session_meta"' in line
    except OSError:
        pass
    return False


class CompositeSessionParser(ISessionParser):
    """Sniffs and dispatches session files to ClaudeJsonlParser or CodexRolloutParser."""

    def __init__(
        self,
        pricing_calculator: IPricingCalculator | None = None,
        claude_parser: ISessionParser | None = None,
        codex_parser: ISessionParser | None = None,
    ) -> None:
        from vylor_estimator.codex_parser import CodexRolloutParser
        from vylor_estimator.pricing import RoutingPricingCalculator

        pricing = pricing_calculator or RoutingPricingCalculator()
        self.claude_parser = claude_parser or ClaudeJsonlParser(pricing_calculator=pricing)
        self.codex_parser = codex_parser or CodexRolloutParser(pricing_calculator=pricing)

    def parse(self, paths: list[Path]) -> list[Turn]:
        claude_paths: list[Path] = []
        codex_paths: list[Path] = []

        for p in paths:
            if _is_codex_file(p):
                codex_paths.append(p)
            else:
                claude_paths.append(p)

        turns: list[Turn] = []
        if claude_paths:
            turns.extend(self.claude_parser.parse(claude_paths))
        if codex_paths:
            turns.extend(self.codex_parser.parse(codex_paths))

        turns.sort(key=lambda t: t.timestamp or datetime.min.replace(tzinfo=timezone.utc))
        return turns
