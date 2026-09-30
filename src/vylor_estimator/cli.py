from __future__ import annotations

import sys
from datetime import date, datetime

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
@click.option("--month", is_flag=True, default=False, help="Analyze last 30 days only (default).")
@click.option("--all", "all_time", is_flag=True, default=False, help="Analyze all sessions (all-time).")
@click.option("--since", default=None, metavar="DATE", callback=_parse_date, is_eager=False,
              help="Analyze sessions on or after DATE (YYYY-MM-DD).")
@click.version_option(__version__, "--version", prog_name="vylor-estimate")
def main(
    path: str | None,
    week: bool,
    month: bool,
    all_time: bool,
    since: date | None,
) -> None:
    try:
        date_filter = DateFilter.resolve(
            week=week,
            month=month,
            since=since,
            all_time=all_time,
        )
        estimator = create_estimator(path=path)
        estimator.run(date_filter=date_filter)

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
