from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console

from vylor_estimator.classifier import (
    ITurnClassifier,
    PatternTurnClassifier,
)
from vylor_estimator.discovery import (
    ClaudeSessionDiscoverer,
    ISessionDiscoverer,
)
from vylor_estimator.parser import (
    ClaudeJsonlParser,
    ISessionParser,
)
from vylor_estimator.pricing import (
    ClaudePricingCalculator,
    IPricingCalculator,
)
from vylor_estimator.report.base import IReportRenderer
from vylor_estimator.report.terminal import TerminalReportRenderer
from vylor_estimator.savings import (
    ConservativeSavingsEngine,
    ISavingsEngine,
    SavingsReport,
)


@dataclass
class DateFilter:
    week: bool = False
    month: bool = False
    since: date | None = None

    def get_cutoff_date(self) -> date | None:
        today = date.today()
        if self.week:
            return today - timedelta(days=7)
        if self.month:
            return today - timedelta(days=30)
        return self.since

    @property
    def label(self) -> str:
        if self.week:
            return "Last 7 days"
        if self.month:
            return "Last 30 days"
        if self.since:
            return f"Since {self.since.isoformat()}"
        return "All-time"


class EstimatorService:
    def __init__(
        self,
        discoverer: ISessionDiscoverer,
        parser: ISessionParser,
        classifier: ITurnClassifier,
        savings_engine: ISavingsEngine,
        renderer: IReportRenderer | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.discoverer = discoverer
        self.parser = parser
        self.classifier = classifier
        self.savings_engine = savings_engine
        self.renderer = renderer
        self.progress_callback = progress_callback

    def run(
        self,
        date_filter: DateFilter | None = None,
    ) -> SavingsReport:
        session_files = self.discoverer.discover()
        if not session_files:
            raise FileNotFoundError("No session files found.")

        if self.progress_callback:
            self.progress_callback(f"Found {len(session_files)} session file(s). Parsing...")

        turns = self.parser.parse(session_files)

        if date_filter is not None:
            cutoff = date_filter.get_cutoff_date()
            if cutoff is not None:
                cutoff_dt = datetime(cutoff.year, cutoff.month, cutoff.day, tzinfo=timezone.utc)
                turns = [
                    t for t in turns
                    if t.timestamp is None or t.timestamp >= cutoff_dt
                ]

        if not turns:
            raise ValueError("No turns found in the specified date range.")

        classified = self.classifier.classify(turns)

        if self.progress_callback and getattr(self.classifier, "skipped_sessions", None):
            skipped_count = len(self.classifier.skipped_sessions)
            if skipped_count > 0:
                self.progress_callback(f"Skipped {skipped_count} session(s) already using Vylor MCP.")

        report = self.savings_engine.calculate(classified)

        label = date_filter.label if date_filter else "All-time"
        if self.renderer is not None:
            self.renderer.render(report, date_range_label=label)

        return report


def create_estimator(
    path: str | Path | None = None,
    pricing_calculator: IPricingCalculator | None = None,
    discoverer: ISessionDiscoverer | None = None,
    parser: ISessionParser | None = None,
    classifier: ITurnClassifier | None = None,
    savings_engine: ISavingsEngine | None = None,
    renderer: IReportRenderer | None = None,
    show_progress: bool = True,
) -> EstimatorService:
    pricing = pricing_calculator or ClaudePricingCalculator()
    disc = discoverer or ClaudeSessionDiscoverer(explicit_path=path)
    pars = parser or ClaudeJsonlParser(pricing_calculator=pricing)
    clsf = classifier or PatternTurnClassifier()
    seng = savings_engine or ConservativeSavingsEngine()
    rend = renderer or TerminalReportRenderer()

    console = Console()
    progress_cb = (lambda msg: console.print(f"[dim]{msg}[/dim]")) if show_progress else None

    return EstimatorService(
        discoverer=disc,
        parser=pars,
        classifier=clsf,
        savings_engine=seng,
        renderer=rend,
        progress_callback=progress_cb,
    )
