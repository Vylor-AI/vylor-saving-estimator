from __future__ import annotations

from datetime import timezone

from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from vylor_estimator.classifier import NON_EXPLORE_SUBAGENT_TYPES
from vylor_estimator.report.base import IReportRenderer
from vylor_estimator.savings import SavingsReport

ACCENT = "#6C47FF"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_time(seconds: float) -> str:
    """Compact duration: ``45s``, ``5m``, ``1h 12m``."""
    total = round(seconds)
    if total < 60:
        return f"{total}s"
    minutes = round(total / 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def _pct(part: float, whole: float) -> float:
    return round(part / whole * 100, 1) if whole else 0.0


class TerminalReportRenderer(IReportRenderer):
    """
    Renders the Vylor MCP savings estimate.

    Default output is a single compact table (total vs. Vylor MCP savings for
    cost, tokens and time) plus tool activity breakdown. ``debug=True`` appends
    the detailed breakdowns.
    """

    def __init__(self, console: Console | None = None, debug: bool = False) -> None:
        self.console = console or Console(highlight=False)
        self.debug = debug

    def render(
        self,
        report: SavingsReport,
        date_range_label: str = "All-time",
    ) -> None:
        agg = report.total
        c = self.console

        c.print()
        c.print(f"[bold white on {ACCENT}]  VYLOR MCP SAVINGS ESTIMATE  [/]", justify="center")

        if agg.first_turn_at is not None and agg.last_turn_at is not None:
            start_d = agg.first_turn_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
            end_d = agg.last_turn_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
            date_str = f"({date_range_label}) {start_d} to {end_d}"
        else:
            date_str = date_range_label

        if len(agg.by_agent) > 1:
            agents_label = " (" + ", ".join(k.capitalize() for k in sorted(agg.by_agent.keys())) + ")"
        elif len(agg.by_agent) == 1:
            agents_label = f" ({next(iter(agg.by_agent)).capitalize()})"
        else:
            agents_label = ""

        c.print(
            f"[dim]Analyzed: {len(report.by_session)} session(s){agents_label}  |  {date_str}[/dim]",
            justify="center",
        )
        c.print()

        table = Table(
            box=box.ROUNDED, show_header=True,
            header_style=f"bold {ACCENT}", border_style=ACCENT,
        )
        table.add_column("",       style="bold white", min_width=19)
        table.add_column("Cost",   justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Time",   justify="right")

        table.add_row(
            "Total",
            f"${agg.baseline_cost:,.4f}",
            _fmt_tokens(agg.baseline_total_tokens),
            _fmt_time(agg.baseline_seconds),
            style="yellow",
        )
        table.add_row(
            "Vylor MCP savings",
            f"${agg.saved_cost:,.4f} ({agg.pct_cost_cut:.1f}%)",
            f"{_fmt_tokens(agg.total_saved_tokens)} ({agg.pct_tokens_cut:.1f}%)",
            f"{_fmt_time(agg.saved_seconds)} ({agg.pct_time_cut:.1f}%)",
            style="bold green",
        )
        c.print(table, justify="center")

        sources_line = self._sources_line(report)
        if sources_line:
            c.print(sources_line, justify="center")

        if report.activity and report.activity.has_activity:
            act = report.activity
            if act.total_calls > 0:
                core_cats = ["Read", "Search", "Commands", "Write"]
                call_parts = []
                for cat, count in sorted(act.calls_by_category.items(), key=lambda kv: -kv[1]):
                    pct = round(count / act.total_calls * 100)
                    if pct >= 1:
                        call_parts.append(f"{cat} {pct}%")
                for c_cat in core_cats:
                    if c_cat not in [p.split()[0] for p in call_parts]:
                        call_parts.append(f"{c_cat} 0%")
                if call_parts:
                    c.print(f"[dim]Tool calls ({act.total_calls}): " + " | ".join(call_parts) + "[/dim]", justify="center")

            cost_parts = []
            if act.cache_read_cost_pct > 0:
                cost_parts.append(f"cache_read {act.cache_read_cost_pct:.0f}%")
            if act.cache_creation_cost_pct > 0:
                cost_parts.append(f"cache_creation {act.cache_creation_cost_pct:.0f}%")
            if act.output_cost_pct > 0:
                out_str = f"output_tokens {act.output_cost_pct:.0f}%"
                if act.thinking_pct_of_output > 0:
                    out_str += f" (thinking_tokens {act.thinking_pct_of_output:.0f}%)"
                cost_parts.append(out_str)
            if act.input_cost_pct >= 1.0:
                cost_parts.append(f"input_tokens {act.input_cost_pct:.0f}%")

            if cost_parts:
                c.print("[dim]Cost breakdown: " + " | ".join(cost_parts) + "[/dim]", justify="center")

        if self.debug:
            self._render_details(report)
        else:
            c.print("[dim]Run with -d for details.[/dim]", justify="center")
        c.print()

    @staticmethod
    def _sources_line(report: SavingsReport) -> str:
        sources = report.total.by_source
        if not sources:
            return "[dim]No Vylor MCP savings detected.[/dim]"

        total_cost = sum(s["cost"] for s in sources.values())
        key = "cost" if total_cost > 0 else "tokens"
        total = total_cost if total_cost > 0 else sum(s["tokens"] for s in sources.values())
        ranked = sorted(sources.items(), key=lambda kv: -kv[1][key])
        parts = []
        for name, info in ranked:
            if info[key] <= 0:
                continue
            if name == "Sub-agents" and report.activity and report.activity.subagent_types:
                sub_types = {
                    st: cnt
                    for st, cnt in report.activity.subagent_types.items()
                    if st not in NON_EXPLORE_SUBAGENT_TYPES
                }
                if sub_types:
                    tot_sub = sum(sub_types.values())
                    sub_strs = [
                        f"{st} {_pct(cnt, tot_sub):.0f}%"
                        for st, cnt in sorted(sub_types.items(), key=lambda kv: -kv[1])
                    ]
                    parts.append(f"{name.lower()} {_pct(info[key], total):.0f}% ({', '.join(sub_strs)})")
                else:
                    parts.append(f"{name.lower()} {_pct(info[key], total):.0f}%")
            else:
                parts.append(f"{name.lower()} {_pct(info[key], total):.0f}%")

        if not parts:
            return ""
        return "[dim]Top savings sources: " + "  |  ".join(parts) + "[/dim]"

    def _render_details(self, report: SavingsReport) -> None:
        agg = report.total
        c = self.console

        c.print()
        c.print(
            f"[dim]{agg.turns_analyzed} turns analyzed  |  "
            f"{agg.subagent_turns} sub-agent turns[/dim]",
            justify="center",
        )
        c.print()

        if agg.by_source:
            c.print(Rule(f"[bold {ACCENT}]Savings by Source", style=ACCENT))
            t = Table(box=box.SIMPLE_HEAD, header_style="bold white",
                      border_style="dim", expand=True)
            t.add_column("Source",       style="bold white")
            t.add_column("Turns",        justify="right")
            t.add_column("Cost",         justify="right", style="green")
            t.add_column("Tokens",       justify="right")
            t.add_column("Time",         justify="right")
            t.add_column("% of savings",  justify="right", style="cyan")

            total_cost = sum(s["cost"] for s in agg.by_source.values())
            for name, info in sorted(agg.by_source.items(), key=lambda kv: -kv[1]["cost"]):
                t.add_row(
                    name,
                    str(info["turns"]),
                    f"${info['cost']:,.4f}",
                    _fmt_tokens(info["tokens"]),
                    _fmt_time(info["seconds"]),
                    f"{_pct(info['cost'], total_cost):.1f}%",
                )
            c.print(t)
            c.print()

        if len(agg.by_agent) > 1:
            c.print(Rule(f"[bold {ACCENT}]Breakdown by Agent", style=ACCENT))
            t_agent = Table(box=box.SIMPLE_HEAD, header_style="bold white",
                            border_style="dim", expand=True)
            t_agent.add_column("Agent",         style="bold white")
            t_agent.add_column("Turns",         justify="right")
            t_agent.add_column("Baseline Cost", justify="right", style="yellow")
            t_agent.add_column("Savings",       justify="right", style="green")
            t_agent.add_column("% Savings",     justify="right", style="cyan")

            for agent_name, info in sorted(agg.by_agent.items(), key=lambda x: -x[1]["baseline_cost"]):
                pct = _pct(info["saved_cost"], info["baseline_cost"]) if info["baseline_cost"] > 0 else 0.0
                t_agent.add_row(
                    agent_name.capitalize(),
                    str(info["turns"]),
                    f"${info['baseline_cost']:,.4f}",
                    f"${info['saved_cost']:,.4f}",
                    f"{pct:.1f}%",
                )
            c.print(t_agent)
            c.print()

        if agg.by_model:
            c.print(Rule(f"[bold {ACCENT}]Breakdown by Model", style=ACCENT))
            t3 = Table(box=box.SIMPLE_HEAD, header_style="bold white",
                       border_style="dim", expand=True)
            t3.add_column("Model",         style="bold white")
            t3.add_column("Turns",         justify="right")
            t3.add_column("Baseline Cost", justify="right", style="yellow")
            t3.add_column("Savings",       justify="right", style="green")

            for model_name, info in sorted(agg.by_model.items(), key=lambda x: -x[1]["baseline_cost"]):
                short = model_name.replace("claude-", "").strip()
                t3.add_row(
                    short,
                    str(info["turns"]),
                    f"${info['baseline_cost']:,.4f}",
                    f"${info['saved_cost']:,.4f}",
                )
            c.print(t3)
            c.print()

        if report.by_session:
            c.print(Rule(f"[bold {ACCENT}]Top Sessions by Savings", style=ACCENT))
            t4 = Table(box=box.SIMPLE_HEAD, header_style="bold white",
                       border_style="dim")
            t4.add_column("Session",    style="dim", no_wrap=True)
            t4.add_column("Turns",      justify="right", no_wrap=True)
            t4.add_column("Cost",       justify="right", style="yellow", no_wrap=True)
            t4.add_column("Savings",    justify="right", style="green", no_wrap=True)
            t4.add_column("% Savings",  justify="right", style="cyan", no_wrap=True)
            t4.add_column("Time",       justify="right", no_wrap=True)
            t4.add_column("Agents",     justify="right", no_wrap=True)

            top = sorted(report.by_session.items(), key=lambda x: -x[1].saved_cost)[:10]
            for sid, sa in top:
                t4.add_row(
                    (sid if len(sid) <= 15 else f"{sid[:8]}...{sid[-4:]}"),
                    str(sa.turns_analyzed),
                    f"${sa.baseline_cost:,.4f}",
                    f"${sa.saved_cost:,.4f}",
                    f"{sa.pct_cost_cut:.1f}%",
                    _fmt_time(sa.saved_seconds),
                    str(sa.subagent_turns) if sa.subagent_turns > 0 else "-",
                )
            c.print(t4)
            c.print()
