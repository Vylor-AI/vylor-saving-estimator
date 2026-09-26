from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from vylor_estimator.report.base import IReportRenderer
from vylor_estimator.savings import SavingsReport

ACCENT = "#6C47FF"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class TerminalReportRenderer(IReportRenderer):

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(highlight=False)

    def render(
        self,
        report: SavingsReport,
        date_range_label: str = "All-time",
    ) -> None:
        agg = report.total
        c = self.console

        c.print()
        c.print(f"[bold white on {ACCENT}]  VYLOR SAVINGS ESTIMATE  [/]", justify="center")
        c.print(
            f"[dim]Analyzed: {len(report.by_session)} session(s)  |  "
            f"{agg.turns_analyzed} turns  |  {date_range_label}[/dim]",
            justify="center",
        )
        if agg.subagent_turns > 0:
            c.print(f"[dim italic](includes {agg.subagent_turns} sub-agent turns)[/]", justify="center")
        c.print()

        table = Table(
            box=box.ROUNDED, show_header=True,
            header_style=f"bold {ACCENT}", border_style=ACCENT, expand=True
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
        c.print(table)
        c.print()

        if agg.by_tool:
            c.print(Rule(f"[bold {ACCENT}]Breakdown by Vylor Tool", style=ACCENT))
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

        if agg.by_model:
            c.print(Rule(f"[bold {ACCENT}]Breakdown by Model", style=ACCENT))
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

        if report.by_session:
            c.print(Rule(f"[bold {ACCENT}]Top Sessions by Savings", style=ACCENT))
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

        c.print(Panel(
            "[dim]Direct cache model: 100% avoided file reads | 100% downstream cache reads | 0% output tokens[/dim]\n"
            f"[dim]Powered by [/dim][bold {ACCENT}]Vylor MCP[/bold {ACCENT}]"
            "[dim] -- https://github.com/vylor-ai/vylor-mcp[/dim]",
            box=box.ROUNDED, border_style=f"dim {ACCENT}", expand=True
        ))
