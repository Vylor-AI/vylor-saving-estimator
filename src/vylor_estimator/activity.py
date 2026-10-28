from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field

from vylor_estimator.parser import Turn
from vylor_estimator.pricing import (
    ClaudePricingCalculator,
    IPricingCalculator,
)
from vylor_estimator.shell import (
    classify_command,
    extract_command,
)

READ_TOOLS: frozenset[str] = frozenset([
    "read", "view_file", "notebookread", "view_image",
])
SEARCH_TOOLS: frozenset[str] = frozenset([
    "grep", "glob", "ls", "list_dir", "grep_search", "ripgrep", "find_by_name",
])
WRITE_TOOLS: frozenset[str] = frozenset([
    "write", "edit", "multiedit", "notebookedit", "write_to_file",
    "replace_file_content", "multi_replace_file_content", "apply_patch",
])
SHELL_TOOLS: frozenset[str] = frozenset([
    "bash", "powershell", "run_command", "cmd",
    "exec_command", "shell", "shell_command", "local_shell",
])
SUBAGENT_TOOLS: frozenset[str] = frozenset([
    "task", "agent", "spawn_agent", "wait_agent", "send_message",
])
WEB_TOOLS: frozenset[str] = frozenset([
    "webfetch", "websearch", "web_search",
])

_READ_COMMANDS: frozenset[str] = frozenset([
    "cat", "type", "get-content", "gc", "head", "tail", "less", "more", "sed", "nl",
])


def classify_tool_call(name: str, args: dict | None = None) -> str:
    """Classify a single tool call into a high-level category."""
    n = (name or "").lower().strip()
    if n in WRITE_TOOLS:
        return "Write"
    if n in READ_TOOLS:
        return "Read"
    if n in SEARCH_TOOLS:
        return "Search"
    if n in SUBAGENT_TOOLS:
        return "Sub-agent"
    if n in WEB_TOOLS:
        return "Web"
    if n == "wait":
        args_dict = args if isinstance(args, dict) else {}
        inh = args_dict.get("inherits")
        if inh in ("Searches", "Directory listing"):
            return "Search"
        if inh == "Sub-agents":
            return "Sub-agent"
        return "Commands"
    if n in SHELL_TOOLS:
        args_dict = args if isinstance(args, dict) else {}
        cmd = extract_command(args_dict)
        if classify_command(cmd) is not None:
            return "Search"
        first = cmd.strip().split()[0].lower() if cmd.strip() else ""
        first = first.removesuffix(".exe").replace("\\", "/").rsplit("/", 1)[-1]
        if first in _READ_COMMANDS:
            return "Read"
        return "Commands"
    return "Other"


@dataclass
class ActivityReport:
    """Aggregated activity analysis across analyzed turns."""
    total_calls: int = 0
    calls_by_category: dict[str, int] = field(default_factory=dict)
    subagent_types: dict[str, int] = field(default_factory=dict)

    # Cost breakdown across Claude token components
    cache_read_cost_pct: float = 0.0
    cache_creation_cost_pct: float = 0.0
    output_cost_pct: float = 0.0
    input_cost_pct: float = 0.0
    thinking_pct_of_output: float = 0.0

    @property
    def has_activity(self) -> bool:
        return self.total_calls > 0 or (
            self.cache_read_cost_pct > 0 or self.output_cost_pct > 0
        )


class IActivityAnalyzer(ABC):
    @abstractmethod
    def analyze(self, turns: list[Turn]) -> ActivityReport:
        """Analyze tool activity and token cost breakdown across turns."""


class ToolActivityAnalyzer(IActivityAnalyzer):
    """Analyzes tool call distributions and calculates overall Claude token cost breakdown."""

    def __init__(self, pricing_calculator: IPricingCalculator | None = None) -> None:
        self.pricing_calculator = pricing_calculator or ClaudePricingCalculator()

    def analyze(self, turns: list[Turn]) -> ActivityReport:
        if not turns:
            return ActivityReport()

        calls_by_category: Counter[str] = Counter()
        subagent_types: Counter[str] = Counter()
        recorded_agents: set[str] = set()

        tot_input_cost = 0.0
        tot_cw_cost = 0.0
        tot_cr_cost = 0.0
        tot_out_cost = 0.0

        tot_thinking_tokens = 0
        tot_output_tokens = 0

        for t in turns:
            # Tool calls distribution
            for tc in t.tool_calls:
                cat = classify_tool_call(tc.name, tc.args)
                calls_by_category[cat] += 1
                if cat == "Sub-agent" and isinstance(tc.args, dict):
                    st = tc.args.get("subagent_type") or tc.args.get("type")
                    if st and isinstance(st, str):
                        subagent_types[st] += 1

            # Sub-agent turn tracking
            if t.is_subagent and t.subagent_type and t.session_id not in recorded_agents:
                recorded_agents.add(t.session_id)
                # If subagent was launched without a parent tool call in these turns, record type
                if sum(subagent_types.values()) == 0:
                    subagent_types[t.subagent_type] += 1

            # Cost accumulation
            r = self.pricing_calculator.get_pricing(t.model)
            tot_input_cost += (t.input_tokens * r[0]) / 1_000_000.0

            if t.ephemeral_5m_tokens or t.ephemeral_1h_tokens:
                tot_cw_cost += (
                    t.ephemeral_5m_tokens * r[1] + t.ephemeral_1h_tokens * r[2]
                ) / 1_000_000.0
            else:
                tot_cw_cost += (t.cache_write_tokens * r[1]) / 1_000_000.0

            tot_cr_cost += (t.cache_read_tokens * r[3]) / 1_000_000.0
            tot_out_cost += (t.output_tokens * r[4]) / 1_000_000.0

            tot_thinking_tokens += t.reasoning_tokens
            tot_output_tokens += t.output_tokens

        tot_cost = tot_input_cost + tot_cw_cost + tot_cr_cost + tot_out_cost

        cr_pct = round(tot_cr_cost / tot_cost * 100, 1) if tot_cost else 0.0
        cw_pct = round(tot_cw_cost / tot_cost * 100, 1) if tot_cost else 0.0
        out_pct = round(tot_out_cost / tot_cost * 100, 1) if tot_cost else 0.0
        in_pct = round(tot_input_cost / tot_cost * 100, 1) if tot_cost else 0.0
        think_pct = (
            round(tot_thinking_tokens / tot_output_tokens * 100, 1)
            if tot_output_tokens
            else 0.0
        )

        return ActivityReport(
            total_calls=sum(calls_by_category.values()),
            calls_by_category=dict(calls_by_category),
            subagent_types=dict(subagent_types),
            cache_read_cost_pct=cr_pct,
            cache_creation_cost_pct=cw_pct,
            output_cost_pct=out_pct,
            input_cost_pct=in_pct,
            thinking_pct_of_output=think_pct,
        )
