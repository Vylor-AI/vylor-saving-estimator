"""cli.py -- Click entry-point for vylor-estimate."""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import click
from rich.console import Console

from vylor_estimator import __version__
from vylor_estimator.service import DateFilter, create_estimator

console = Console()


def _parse_date(ctx, param, value) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise click.BadParameter("Expected format: YYYY-MM-DD")


@click.command(name="vylor-estimate")
@click.argument("path", required=False, default=None, metavar="PATH")
@click.option("--week", is_flag=True, default=False, help="Analyze last 7 days only.")
@click.option("--month", is_flag=True, default=False, help="Analyze last 30 days only.")
@click.option("--since", default=None, metavar="DATE", callback=_parse_date, is_eager=False,
              help="Analyze sessions on or after DATE (YYYY-MM-DD).")
@click.option("--format", "output_format",
              type=click.Choice(["terminal", "html", "json"], case_sensitive=False),
              default="terminal", show_default=True,
              help="Output format.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write report to file (required for html/json).")
@click.version_option(__version__, "--version", prog_name="vylor-estimate")
def main(
    path: str | None,
    week: bool,
    month: bool,
    since: date | None,
    output_format: str,
    output: str | None,
) -> None:
    """
    Estimate how much Vylor MCP would have saved on your Claude sessions.

    Automatically discovers Claude session files if no PATH is given.
    Default date range is all-time unless --week / --month / --since is specified.

    Examples:

    \b
      vylor-estimate                            # auto-discover, all-time
      vylor-estimate --week                     # last 7 days
      vylor-estimate --since 2025-09-01
      vylor-estimate /path/to/session.jsonl
      vylor-estimate --format html -o report.html
    """
    try:
        # Build domain filter
        date_filter = DateFilter(week=week, month=month, since=since)

        # Assemble pipeline via Dependency Injection composition root
        estimator = create_estimator(
            path=path,
            output_format=output_format,
            show_progress=(output_format == "terminal"),
        )

        # Determine target file for exported formats
        out_path: Path | None = None
        if output:
            out_path = Path(output)
        elif output_format == "html":
            out_path = Path("report.html")
        elif output_format == "json":
            out_path = Path("report.json")

        # Run pipeline
        estimator.run(date_filter=date_filter, output_path=out_path)

        if output_format in ("html", "json") and out_path:
            console.print(f"[green]{output_format.upper()} report written to:[/green] {out_path.resolve()}")

    except FileNotFoundError as e:
        console.print(f"[yellow]{e}[/yellow]")
        sys.exit(0)
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
