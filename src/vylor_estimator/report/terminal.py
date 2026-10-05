from __future__ import annotations

from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from vylor_estimator.report.base import IReportRenderer
from vylor_estimator.savings import MAX_TURN_SECONDS, SavingsReport

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
    Renders the overhead estimate.

    Default output is a single compact table (total vs. avoidable overhead for
    cost, tokens and time) plus a one-line hint about the biggest overhead
    sources.  ``debug=True`` appends the detailed breakdowns.
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
        c.print(f"[bold white on {ACCENT}]  AGENT OVERHEAD ESTIMATE  [/]", justify="center")
        c.print(
            f"[dim]Analyzed: {len(report.by_session)} session(s)  |  {date_range_label}[/dim]",
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
            "Avoidable overhead",
            f"${agg.saved_cost:,.4f} ({agg.pct_cost_cut:.1f}%)",
            f"{_fmt_tokens(agg.total_saved_tokens)} ({agg.pct_tokens_cut:.1f}%)",
            f"{_fmt_time(agg.saved_seconds)} ({agg.pct_time_cut:.1f}%)",
            style="bold red",
        )
        c.print(table, justify="center")

        sources_line = self._sources_line(report)
        if sources_line:
            c.print(sources_line, justify="center")

        if self.debug:
            self._render_details(report)
        else:
            c.print("[dim]Run with -d for details.[/dim]", justify="center")
        c.print()

    @staticmethod
    def _sources_line(report: SavingsReport) -> str:
        sources = report.total.by_source
        if not sources:
            return "[dim]No avoidable overhead detected.[/dim]"

        total_cost = sum(s["cost"] for s in sources.values())
        key = "cost" if total_cost > 0 else "tokens"
        total = total_cost if total_cost > 0 else sum(s["tokens"] for s in sources.values())
        ranked = sorted(sources.items(), key=lambda kv: -kv[1][key])
        parts = [
            f"{name.lower()} {_pct(info[key], total):.0f}%"
            for name, info in ranked
            if info[key] > 0
        ]
        if not parts:
            return ""
        return "[dim]Top overhead sources: " + "  |  ".join(parts) + "[/dim]"

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
            c.print(Rule(f"[bold {ACCENT}]Overhead by Source", style=ACCENT))
            t = Table(box=box.SIMPLE_HEAD, header_style="bold white",
                      border_style="dim", expand=True)
            t.add_column("Source",       style="bold white")
            t.add_column("Turns",        justify="right")
            t.add_column("Cost",         justify="right", style="red")
            t.add_column("Tokens",       justify="right")
            t.add_column("Time",         justify="right")
            t.add_column("% of overhead", justify="right", style="cyan")

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

        if agg.by_model:
            c.print(Rule(f"[bold {ACCENT}]Breakdown by Model", style=ACCENT))
            t3 = Table(box=box.SIMPLE_HEAD, header_style="bold white",
                       border_style="dim", expand=True)
            t3.add_column("Model",         style="bold white")
            t3.add_column("Turns",         justify="right")
            t3.add_column("Baseline Cost", justify="right", style="yellow")
            t3.add_column("Overhead Cost", justify="right", style="red")

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
            c.print(Rule(f"[bold {ACCENT}]Top Sessions by Overhead", style=ACCENT))
            t4 = Table(box=box.SIMPLE_HEAD, header_style="bold white",
                       border_style="dim")
            t4.add_column("Session",    style="dim", no_wrap=True)
            t4.add_column("Turns",      justify="right", no_wrap=True)
            t4.add_column("Cost",       justify="right", style="yellow", no_wrap=True)
            t4.add_column("Overhead",   justify="right", style="red", no_wrap=True)
            t4.add_column("% Overhead", justify="right", style="cyan", no_wrap=True)
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

        c.print(
            "[dim]Notes: avoidable overhead is an estimate of what Vylor tools could reduce, not a guaranteed "
            "saving. Time is agent-time (sum of turn durations, each capped at "
            f"{_fmt_time(MAX_TURN_SECONDS)}); it excludes tool run time and parallel "
            "sub-agents can overlap.[/dim]"
        )
