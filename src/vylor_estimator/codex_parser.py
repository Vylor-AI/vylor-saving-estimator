from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vylor_estimator.classifier import SOURCE_SUBAGENTS
from vylor_estimator.parser import ISessionParser, ToolCall, Turn, _parse_iso
from vylor_estimator.pricing import IPricingCalculator, RoutingPricingCalculator
from vylor_estimator.shell import (
    _command_word,
    _split_command,
    classify_command,
    is_read_only_command,
)

NESTED_TOOL_RE = re.compile(r"tools\.([A-Za-z0-9_]+)\s*\(")
CELL_ID_RE = re.compile(r"cell ID (\d+)", re.IGNORECASE)

_is_read_only_command = is_read_only_command


def _extract_json_object(src: str, start: int) -> dict[str, Any] | None:
    """Extract a balanced JSON {...} object starting at src[start]."""
    if start >= len(src) or src[start] != "{":
        return None
    depth = 0
    i = start
    quote: str | None = None
    while i < len(src):
        c = src[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in ('"', "'"):
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    res = json.loads(src[start:i + 1])
                    return res if isinstance(res, dict) else None
                except json.JSONDecodeError:
                    return None
        i += 1
    return None


def _extract_nested_calls(js_code: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse nested tool calls (e.g. tools.exec_command(...)) from an exec JS block."""
    calls: list[tuple[str, dict[str, Any]]] = []
    for m in NESTED_TOOL_RE.finditer(js_code):
        tool_name = m.group(1)
        j = m.end()
        while j < len(js_code) and js_code[j].isspace():
            j += 1
        args = _extract_json_object(js_code, j) or {}
        calls.append((tool_name, args))
    return calls


class CodexRolloutParser(ISessionParser):
    """
    Parser for OpenAI Codex rollout JSONL files.

    Handles:
    - Multi-thread stitching: links subagent rollouts to parent sessions via session_id / parent_thread_id.
    - Forked history trimming: skips duplicate parent events prior to subagent_history_start_ordinal.
    - JavaScript exec tool unpacking: extracts tools.exec_command, tools.apply_patch, etc.
    - Async polling inheritance: correlates wait and wait_agent calls to the commands they poll.
    - Subagent classification: identifies pure read-only Explore subagents vs mutating Worker subagents.
    """

    def __init__(self, pricing_calculator: IPricingCalculator | None = None) -> None:
        self.pricing_calculator = pricing_calculator or RoutingPricingCalculator()

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            pass
        return rows

    def parse(self, paths: list[Path]) -> list[Turn]:
        # Step 1: Pre-read files and inspect session_meta
        thread_files: list[dict[str, Any]] = []
        for p in paths:
            rows = self._read_jsonl(p)
            if not rows:
                continue

            meta: dict[str, Any] = {}
            for r in rows:
                if r.get("type") == "session_meta" and isinstance(r.get("payload"), dict):
                    meta = r["payload"]
                    break

            thread_id = meta.get("id") or p.stem
            session_id = meta.get("session_id") or thread_id
            is_subagent = meta.get("thread_source") == "subagent"
            start_ordinal = meta.get("subagent_history_start_ordinal") or 0
            agent_path = meta.get("agent_path") or "/root"
            spawn = ((meta.get("source") or {}).get("subagent") or {}).get("thread_spawn") or {} if is_subagent else {}
            role = spawn.get("agent_role")

            thread_files.append({
                "path": p,
                "rows": rows,
                "meta": meta,
                "thread_id": thread_id,
                "session_id": session_id,
                "is_subagent": is_subagent,
                "start_ordinal": start_ordinal,
                "agent_path": agent_path,
                "role": role,
            })

        # Step 2: Group by session_id and determine subagent explore classification
        by_session: dict[str, list[dict[str, Any]]] = {}
        for tf in thread_files:
            by_session.setdefault(tf["session_id"], []).append(tf)

        all_turns: list[Turn] = []

        for session_id, threads in by_session.items():
            # Analyze subagents to classify Explore vs Worker
            sub_explore_map: dict[str, bool] = {}
            for tf in threads:
                if not tf["is_subagent"]:
                    continue
                if tf["role"] == "explorer":
                    sub_explore_map[tf["thread_id"]] = True
                elif tf["role"]:
                    sub_explore_map[tf["thread_id"]] = False
                else:
                    # Behavioral inspection: check for edits or non-read commands
                    has_edits = False
                    has_mutation_cmds = False
                    for r in tf["rows"]:
                        if (r.get("ordinal") or 0) < tf["start_ordinal"]:
                            continue
                        p = r.get("payload") or {}
                        if r.get("type") == "response_item" and p.get("type") == "custom_tool_call" and p.get("name") == "exec":
                            for name, args in _extract_nested_calls(p.get("input", "")):
                                if name == "apply_patch":
                                    has_edits = True
                                elif name == "exec_command":
                                    cmd = args.get("cmd", "")
                                    if not _is_read_only_command(cmd):
                                        has_mutation_cmds = True
                    is_explore = not has_edits and not has_mutation_cmds
                    sub_explore_map[tf["thread_id"]] = is_explore

            all_subs_explore = bool(sub_explore_map) and all(sub_explore_map.values())

            # Step 3: Parse turns for each thread in the session
            for tf in threads:
                is_sub = tf["is_subagent"]
                sub_type = "Explore" if sub_explore_map.get(tf["thread_id"], False) else ("Worker" if is_sub else None)
                thread_id = tf["thread_id"]

                current_model = "unknown"
                pending_calls: list[ToolCall] = []
                boundary_dt: datetime | None = None
                cell_sources: dict[str, str] = {}
                last_exec_source: str | None = None
                seen_responses: set[str] = set()

                for r in tf["rows"]:
                    # Skip forked history prior to thread start
                    if is_sub and (r.get("ordinal") or 0) < tf["start_ordinal"]:
                        continue

                    rtype = r.get("type")
                    payload = r.get("payload") or {}
                    ptype = payload.get("type") if isinstance(payload, dict) else None

                    if rtype == "turn_context" and isinstance(payload, dict):
                        if payload.get("model"):
                            current_model = payload["model"]

                    elif rtype == "response_item" and ptype in ("message", "agent_message"):
                        # User prompt or incoming message marks start of model generation
                        if payload.get("role") == "user" or ptype == "agent_message":
                            boundary_dt = _parse_iso(r.get("timestamp"))

                    elif rtype == "response_item" and ptype == "custom_tool_call" and payload.get("name") == "exec":
                        nested = _extract_nested_calls(payload.get("input", ""))
                        for name, args in nested:
                            if name == "exec_command":
                                cmd = args.get("cmd", "")
                                src = classify_command(cmd)
                                last_exec_source = src
                                pending_calls.append(ToolCall(name="exec_command", args=args))
                            else:
                                pending_calls.append(ToolCall(name=name, args=args))

                    elif rtype == "response_item" and ptype == "function_call":
                        fname = payload.get("name", "")
                        fargs: dict[str, Any] = {}
                        try:
                            fargs = json.loads(payload.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            pass

                        if fname == "wait":
                            cell_id = str(fargs.get("cell_id", ""))
                            inh = cell_sources.get(cell_id)
                            args_with_inh = dict(fargs)
                            if inh:
                                args_with_inh["inherits"] = inh
                            pending_calls.append(ToolCall(name="wait", args=args_with_inh))
                        elif fname == "wait_agent":
                            args_with_inh = dict(fargs)
                            if all_subs_explore:
                                args_with_inh["inherits"] = SOURCE_SUBAGENTS
                            pending_calls.append(ToolCall(name="wait_agent", args=args_with_inh))
                        else:
                            pending_calls.append(ToolCall(name=fname, args=fargs))

                    elif rtype == "response_item" and str(ptype).endswith("_output"):
                        out_content = payload.get("output", "")
                        out_str = out_content if isinstance(out_content, str) else json.dumps(out_content)
                        m_cell = CELL_ID_RE.search(out_str)
                        if m_cell and last_exec_source:
                            cell_sources[m_cell.group(1)] = last_exec_source
                        boundary_dt = _parse_iso(r.get("timestamp"))

                    elif rtype == "token_usage_record":
                        # Verify ownership
                        rec_thread = payload.get("thread_id", thread_id)
                        if rec_thread != thread_id:
                            continue

                        resp_id = payload.get("response_id") or f"{thread_id}:{r.get('ordinal')}"
                        if resp_id in seen_responses:
                            continue
                        seen_responses.add(resp_id)

                        usage = payload.get("usage") or {}
                        tot_in = usage.get("input_tokens", 0) or 0
                        cached_in = usage.get("cached_input_tokens", 0) or 0
                        uncached_in = max(0, tot_in - cached_in)
                        out_tok = usage.get("output_tokens", 0) or 0
                        reason_tok = usage.get("reasoning_output_tokens", 0) or 0
                        cw_tok = usage.get("cache_write_input_tokens", 0) or 0

                        end_dt = _parse_iso(r.get("timestamp"))
                        duration: float | None = None
                        if end_dt and boundary_dt and end_dt >= boundary_dt:
                            duration = (end_dt - boundary_dt).total_seconds()

                        cost = self.pricing_calculator.compute_turn_cost(
                            model=current_model,
                            input_tokens=uncached_in,
                            output_tokens=out_tok,
                            cache_read_tokens=cached_in,
                            ephemeral_5m_tokens=0,
                            ephemeral_1h_tokens=0,
                        )

                        all_turns.append(Turn(
                            turn_id=resp_id,
                            session_id=session_id,
                            model=current_model,
                            timestamp=end_dt,
                            input_tokens=uncached_in,
                            output_tokens=out_tok,
                            cache_read_tokens=cached_in,
                            cache_write_tokens=cw_tok,
                            ephemeral_5m_tokens=0,
                            ephemeral_1h_tokens=0,
                            reasoning_tokens=reason_tok,
                            duration_seconds=duration,
                            tool_calls=pending_calls,
                            cost=cost,
                            is_subagent=is_sub,
                            subagent_type=sub_type,
                            agent="codex",
                        ))

                        pending_calls = []
                        boundary_dt = end_dt

        all_turns.sort(key=lambda t: t.timestamp or datetime.min.replace(tzinfo=timezone.utc))
        return all_turns
