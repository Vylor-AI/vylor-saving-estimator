"""base.py -- Abstract base class and factory for report renderers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vylor_estimator.savings import SavingsReport


class IReportRenderer(ABC):
    """Abstract interface for savings report formatters."""

    @abstractmethod
    def render(
        self,
        report: SavingsReport,
        date_range_label: str = "All-time",
        output_path: Path | str | None = None,
    ) -> None:
        """Render or export the savings report."""
        pass


class ReportRendererFactory:
    """Factory to create appropriate report renderer strategy by format name."""

    @staticmethod
    def get_renderer(format_name: str) -> IReportRenderer:
        format_lower = format_name.lower().strip()
        if format_lower == "terminal":
            from vylor_estimator.report.terminal import TerminalReportRenderer
            return TerminalReportRenderer()
        elif format_lower == "html":
            from vylor_estimator.report.html_report import HtmlReportRenderer
            return HtmlReportRenderer()
        elif format_lower == "json":
            from vylor_estimator.report.json_report import JsonReportRenderer
            return JsonReportRenderer()
        raise ValueError(f"Unknown report format: {format_name}. Expected 'terminal', 'html', or 'json'.")
