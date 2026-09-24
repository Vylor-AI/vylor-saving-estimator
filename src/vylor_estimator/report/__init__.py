"""report package -- output formatters for the savings report."""
from __future__ import annotations

from vylor_estimator.report.base import IReportRenderer, ReportRendererFactory
from vylor_estimator.report.html_report import HtmlReportRenderer, write_html_report
from vylor_estimator.report.json_report import JsonReportRenderer, build_json_report, write_json_report
from vylor_estimator.report.terminal import TerminalReportRenderer, print_report

__all__ = [
    "IReportRenderer",
    "ReportRendererFactory",
    "TerminalReportRenderer",
    "HtmlReportRenderer",
    "JsonReportRenderer",
    "print_report",
    "write_html_report",
    "write_json_report",
    "build_json_report",
]
