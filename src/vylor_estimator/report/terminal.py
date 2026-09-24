"""terminal.py -- Rich-formatted terminal report (Windows-safe, no emojis)."""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box
from rich.text import Text

from vylor_estimator.report.base import IReportRenderer
from vylor_estimator.savings import Aggregate, SavingsReport

# Force UTF-8 on Windows to avoid cp1252 charmap errors
console = Console(highlight=False)


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    h = int(td.total_seconds() // 3600)
    m = int((td.total_seconds() % 3600) // 60)
    s = int(td.total_seconds() % 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


class TerminalReportRenderer(IReportRenderer):
    """Renders the savings report to the terminal using rich tables."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(highlight=False)

    def render(
        self,
        report: SavingsReport,
        date_range_label: str = "All-time",
        output_path: Path | str | None = None,
    ) -> None:
        agg = report.total
        c = self.console

        # Header
        c.print()
        c.print(Align.center(
            Text("  VYLOR SAVINGS ESTIMATE  ", style="bold white on #6C47FF")
        ))
        c.print(Align.center(Text(
            f"Analyzed: {len(report.by_session)} session(s)  |  "
            f"{agg.turns_analyzed} turns  |  {date_range_label}",
            style="dim"
        )))
        if agg.subagent_turns > 0:
            c.print(Align.center(Text(
                f"(includes {agg.subagent_turns} sub-agent turns)",
                style="dim italic"
            )))
        c.print()

        # Main metrics table
        table = Table(
            box=box.ROUNDED, show_header=True,
            header_style="bold #6C47FF", border_style="#6C47FF", expand=True
        )
        table.add_column("Metric",          style="bold white", min_width=18)
        table.add_column("Baseline (paid)", justify="right", style="yellow")
        table.add_column("With Vylor MCP",  justify="right", style="green")
        table.add_column("Saved",           justify="right", style="bold green")
        table.add_column("% Cut",           justify="right", style="bold cyan")

        table.add_row(
            "Total Cost",
            f"${agg.baseline_cost:,.4f}",
            f"${agg.estimated_cost:,.4f}",
            f"-${agg.saved_cost:,.4f}",
            f"-{agg.pct_cost_cut:.1f}%",
        )
        table.add_row(
            "Total Tokens",
            _fmt_tokens(agg.baseline_total_tokens),
            _fmt_tokens(agg.estimated_tokens),
            f"-{_fmt_tokens(agg.total_saved_tokens)}",
            f"-{agg.pct_tokens_cut:.1f}%",
        )
        bl_t = _fmt_time(agg.baseline_time_seconds) if agg.baseline_time_seconds > 0 else "N/A"
        es_t = _fmt_time(agg.estimated_time_seconds) if agg.baseline_time_seconds > 0 else "N/A"
        sv_t = f"-{_fmt_time(agg.saved_time_seconds)}" if agg.saved_time_seconds > 0 else "N/A"
        pc_t = f"-{agg.pct_time_cut:.1f}%" if agg.pct_time_cut > 0 else "N/A"
        table.add_row("Total Time", bl_t, es_t, sv_t, pc_t)

        c.print(table)
        c.print()

        # Breakdown by Vylor tool
        if agg.by_tool:
            c.print(Rule("[bold #6C47FF]Breakdown by Vylor Tool", style="#6C47FF"))
            t2 = Table(box=box.SIMPLE_HEAD, header_style="bold white",
                       border_style="dim", expand=True)
            t2.add_column("Vylor Tool",        style="bold cyan")
            t2.add_column("Intercepted Turns", justify="right")
            t2.add_column("Cost Saved",        justify="right", style="green")
            t2.add_column("Tokens Saved",      justify="right", style="green")

            for tool_name, info in sorted(agg.by_tool.items(), key=lambda x: -x[1]["saved_cost"]):
                t2.add_row(
                    tool_name,
                    str(info["turns"]),
                    f"-${info['saved_cost']:,.4f}",
                    f"-{_fmt_tokens(info['saved_tokens'])}",
                )
            c.print(t2)
            c.print()

        # Breakdown by model
        if agg.by_model:
            c.print(Rule("[bold #6C47FF]Breakdown by Model", style="#6C47FF"))
            t3 = Table(box=box.SIMPLE_HEAD, header_style="bold white",
                       border_style="dim", expand=True)
            t3.add_column("Model",         style="bold white")
            t3.add_column("Turns",         justify="right")
            t3.add_column("Baseline Cost", justify="right", style="yellow")
            t3.add_column("Cost Saved",    justify="right", style="green")

            for model_name, info in sorted(agg.by_model.items(), key=lambda x: -x[1]["baseline_cost"]):
                short = model_name.replace("claude-", "").strip()
                t3.add_row(
                    short,
                    str(info["turns"]),
                    f"${info['baseline_cost']:,.4f}",
                    f"-${info['saved_cost']:,.4f}",
                )
            c.print(t3)
            c.print()

        # Top sessions
        if report.by_session:
            c.print(Rule("[bold #6C47FF]Top Sessions by Savings", style="#6C47FF"))
            t4 = Table(box=box.SIMPLE_HEAD, header_style="bold white",
                       border_style="dim", expand=True)
            t4.add_column("Session ID",    style="dim", no_wrap=True)
            t4.add_column("Turns",         justify="right")
            t4.add_column("Baseline Cost", justify="right", style="yellow")
            t4.add_column("Cost Saved",    justify="right", style="green")
            t4.add_column("% Cut",         justify="right", style="cyan")
            t4.add_column("Sub-agents",    justify="right", min_width=11)

            top = sorted(report.by_session.items(), key=lambda x: -x[1].saved_cost)[:10]
            for sid, sa in top:
                t4.add_row(
                    (sid if len(sid) <= 26 else f"{sid[:16]}...{sid[-4:]}"),
                    str(sa.turns_analyzed),
                    f"${sa.baseline_cost:,.4f}",
                    f"-${sa.saved_cost:,.4f}",
                    f"-{sa.pct_cost_cut:.1f}%",
                    str(sa.subagent_turns) if sa.subagent_turns > 0 else "-",
                )
            c.print(t4)
            c.print()

        # Footer
        c.print(Panel(
            "[dim]Direct cache model: 100% avoided file reads | 100% downstream cache reads | 0% output tokens[/dim]\n"
            "[dim]Powered by [/dim][bold #6C47FF]Vylor MCP[/bold #6C47FF]"
            "[dim] -- https://github.com/vylor-ai/vylor-mcp[/dim]",
            box=box.ROUNDED, border_style="dim #6C47FF", expand=True
        ))


def print_report(report: SavingsReport, date_range_label: str = "All-time") -> None:
    """Backward-compatible helper function."""
    TerminalReportRenderer().render(report, date_range_label)
