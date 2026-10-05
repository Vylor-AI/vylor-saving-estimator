from __future__ import annotations

from abc import ABC, abstractmethod

from vylor_estimator.savings import SavingsReport


class IReportRenderer(ABC):
    @abstractmethod
    def render(
        self,
        report: SavingsReport,
        date_range_label: str = "All-time",
    ) -> None:
        """Render the overhead report."""

