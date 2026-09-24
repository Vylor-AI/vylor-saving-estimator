"""parser.py -- Parse Claude session JSONL files into structured Turn dataclasses.

Handles:
- Multi-file session directories
- Sub-agent sessions (parentUuid chains stitched into root session)
- Deduplication of messages by message ID
- Duration calculation using parent event timestamps
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vylor_estimator.pricing import (
    ClaudePricingCalculator,
    IPricingCalculator,
    compute_turn_cost,
)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def args_str(self) -> str:
        return json.dumps(self.args, ensure_ascii=False) if self.args else ""


@dataclass
class Turn:
    turn_id: str
    session_id: str
    model: str
    timestamp: datetime | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int   # total cache creation tokens
    ephemeral_5m_tokens: int
    ephemeral_1h_tokens: int
    reasoning_tokens: int
    duration_seconds: float | None
    tool_calls: list[ToolCall]
    cost: float               # baseline cost (pre-savings)
    is_subagent: bool

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _extract_tool_calls(content: list[dict]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    seen: set[str] = set()
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "tool_use":
            name = item.get("name", "")
            args = item.get("input", {}) or {}
            key = f"{name}:{json.dumps(args, sort_keys=True)}"
            if key not in seen:
                seen.add(key)
                calls.append(ToolCall(name=name, args=args))
    return calls


class ISessionParser(ABC):
    """Abstract interface for session parsers."""

    @abstractmethod
    def parse(self, paths: list[Path]) -> list[Turn]:
        """Parse session files and return a flat list of Turns."""
        pass


class ClaudeJsonlParser(ISessionParser):
    """Parser for Claude session .jsonl files with sub-agent stitching and cost calculation."""

    def __init__(self, pricing_calculator: IPricingCalculator | None = None) -> None:
        self.pricing_calculator = pricing_calculator or ClaudePricingCalculator()

    def parse(self, paths: list[Path]) -> list[Turn]:
        """
        Parse a list of .jsonl files and return a flat, deduplicated list of Turns.

        Sub-agent sessions are stitched into their parent session via parentUuid.
        Turns are ordered by timestamp.
        """
        # Index all raw events by UUID across all files
        events_by_uuid: dict[str, dict] = {}

        # Collect raw assistant message data keyed by message id
        messages_map: dict[str, dict] = {}
        message_order: list[str] = []

        # First pass: index all events
        for p in paths:
            if "subagents" in p.parts:
                sub_idx = p.parts.index("subagents")
                session_id = p.parts[sub_idx - 1]
                is_subagent_file = True
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

                        if data.get("type") == "assistant" and "message" in data:
                            msg = data["message"]
                            msg_id = msg.get("id") or data.get("uuid", "")
                            if not msg_id:
                                continue

                            if msg_id not in messages_map:
                                messages_map[msg_id] = {
                                    "contents": [],
                                    "tool_calls_raw": [],
                                    "usage": msg.get("usage", {}),
                                    "model": msg.get("model", ""),
                                    "parent_uuid": data.get("parentUuid"),
                                    "first_timestamp": data.get("timestamp"),
                                    "last_timestamp": data.get("timestamp"),
                                    "session_id": session_id,
                                    "is_subagent": is_subagent_file,
                                }
                                message_order.append(msg_id)

                            entry = messages_map[msg_id]
                            entry["last_timestamp"] = data.get("timestamp")
                            if msg.get("usage"):
                                entry["usage"] = msg.get("usage")
                            if not entry["model"] and msg.get("model"):
                                entry["model"] = msg.get("model", "")

                            # Accumulate tool calls
                            for item in msg.get("content", []):
                                if isinstance(item, dict) and item.get("type") == "tool_use":
                                    name = item.get("name", "")
                                    args = item.get("input", {}) or {}
                                    key = f"{name}:{json.dumps(args, sort_keys=True)}"
                                    if key not in {
                                        f"{tc['name']}:{json.dumps(tc['args'], sort_keys=True)}"
                                        for tc in entry["tool_calls_raw"]
                                    }:
                                        entry["tool_calls_raw"].append({"name": name, "args": args})
            except OSError:
                continue

        # Resolve sub-agent status
        for msg_id, entry in messages_map.items():
            parent_uuid = entry.get("parent_uuid")
            if parent_uuid and parent_uuid in events_by_uuid:
                parent_event = events_by_uuid[parent_uuid]
                if parent_event.get("type") in ("tool_result", "tool_use"):
                    entry["is_subagent"] = True

        # Second pass: build Turn objects
        turns: list[Turn] = []
        for msg_id in message_order:
            entry = messages_map[msg_id]

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

            usage = entry["usage"]
            model = entry["model"] or "unknown"
            cache_creation = usage.get("cache_creation", {})

            e_5m = cache_creation.get("ephemeral_5m_input_tokens", 0) or 0
            e_1h = cache_creation.get("ephemeral_1h_input_tokens", 0) or 0
            cc_total = usage.get("cache_creation_input_tokens", e_5m + e_1h) or 0
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            reasoning = usage.get("output_tokens_details", {}).get("thinking_tokens", 0) or 0

            cost = self.pricing_calculator.compute_turn_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                ephemeral_5m_tokens=e_5m,
                ephemeral_1h_tokens=e_1h,
            )

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
            ))

        # Sort by timestamp
        turns.sort(key=lambda t: t.timestamp or datetime.min.replace(tzinfo=timezone.utc))
        return turns


def parse_files(paths: list[Path], pricing_calculator: IPricingCalculator | None = None) -> list[Turn]:
    """Backward-compatible helper function."""
    return ClaudeJsonlParser(pricing_calculator=pricing_calculator).parse(paths)
