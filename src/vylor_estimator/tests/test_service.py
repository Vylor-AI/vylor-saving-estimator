from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from vylor_estimator.classifier import PatternTurnClassifier
from vylor_estimator.discovery import (
    ClaudeSessionDiscoverer,
    ISessionDiscoverer,
)
from vylor_estimator.parser import ClaudeJsonlParser
from vylor_estimator.pricing import IPricingCalculator
from vylor_estimator.report.base import IReportRenderer
from vylor_estimator.report.terminal import TerminalReportRenderer
from vylor_estimator.savings import (
    ConservativeSavingsEngine,
    SavingsReport,
)
from vylor_estimator.service import (
    DateFilter,
    EstimatorService,
    create_estimator,
)

FIXTURES = Path(__file__).parent / "fixtures"


class StubPricingCalculator(IPricingCalculator):
    """Stub pricing calculator with fixed rate for testing service."""

    def __init__(self, flat_rate_per_m: float = 50.0) -> None:
        self.flat_rate_per_m = flat_rate_per_m

    def get_pricing(self, model_name: str) -> tuple[float, float, float, float, float]:
        return (self.flat_rate_per_m, 0.0, 0.0, self.flat_rate_per_m, self.flat_rate_per_m)

    def compute_turn_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        ephemeral_5m_tokens: int = 0,
        ephemeral_1h_tokens: int = 0,
    ) -> float:
        return (input_tokens + output_tokens + cache_read_tokens) * self.flat_rate_per_m / 1_000_000.0


class MockDiscoverer(ISessionDiscoverer):
    """Mock discoverer returning predefined fixture files."""

    def __init__(self, files: list[Path]) -> None:
        self.files = files

    def discover(self) -> list[Path]:
        return self.files


class RecordingRenderer(IReportRenderer):
    """Renderer spy recording render calls."""

    def __init__(self) -> None:
        self.rendered_reports: list[SavingsReport] = []
        self.labels: list[str] = []

    def render(
        self,
        report: SavingsReport,
        date_range_label: str = "All-time",
    ) -> None:
        self.rendered_reports.append(report)
        self.labels.append(date_range_label)


def test_estimator_service_orchestration():
    """Verify EstimatorService coordinates all injected components end-to-end."""
    stub_pricing = StubPricingCalculator(flat_rate_per_m=50.0)
    mock_discoverer = MockDiscoverer([FIXTURES / "sample_session.jsonl"])
    parser = ClaudeJsonlParser(pricing_calculator=stub_pricing)
    classifier = PatternTurnClassifier()
    savings_engine = ConservativeSavingsEngine(pricing_calculator=stub_pricing)
    renderer_spy = RecordingRenderer()

    service = EstimatorService(
        discoverer=mock_discoverer,
        parser=parser,
        classifier=classifier,
        savings_engine=savings_engine,
        renderer=renderer_spy,
    )

    date_filter = DateFilter(since=date(2025, 1, 1))
    report = service.run(date_filter=date_filter)

    assert report.total.turns_analyzed == 4
    assert len(renderer_spy.rendered_reports) == 1
    assert renderer_spy.rendered_reports[0] is report
    assert renderer_spy.labels[0] == "Since 2025-01-01"


def test_create_estimator_composition_root():
    """Verify create_estimator wires up default production dependencies."""
    service = create_estimator(path=FIXTURES / "sample_session.jsonl", show_progress=False)

    assert isinstance(service, EstimatorService)
    assert isinstance(service.discoverer, ClaudeSessionDiscoverer)
    assert isinstance(service.parser, ClaudeJsonlParser)
    assert isinstance(service.classifier, PatternTurnClassifier)
    assert isinstance(service.savings_engine, ConservativeSavingsEngine)
    assert isinstance(service.renderer, TerminalReportRenderer)


def test_date_filter_cutoff_calculations():
    """Verify DateFilter logic for week, month, and explicit since dates."""
    today = date.today()

    wf = DateFilter(week=True)
    assert wf.get_cutoff_date() == today - timedelta(days=7)
    assert wf.label == "Last 7 days"

    mf = DateFilter(month=True)
    assert mf.get_cutoff_date() == today - timedelta(days=30)
    assert mf.label == "Last 30 days"

    sf = DateFilter(since=date(2025, 1, 1))
    assert sf.get_cutoff_date() == date(2025, 1, 1)
    assert sf.label == "Since 2025-01-01"

    af = DateFilter()
    assert af.get_cutoff_date() is None
    assert af.label == "All-time"


def test_date_filter_resolve():
    """Verify DateFilter.resolve default fallback and explicit flags."""
    # Default fallback when no parameters provided
    default_filter = DateFilter.resolve()
    assert default_filter.month is True
    assert default_filter.label == "Last 30 days"

    # All-time flag
    all_filter = DateFilter.resolve(all_time=True)
    assert all_filter.get_cutoff_date() is None
    assert all_filter.label == "All-time"

    # Week flag
    week_filter = DateFilter.resolve(week=True)
    assert week_filter.week is True
    assert week_filter.label == "Last 7 days"

    # Since date
    since_filter = DateFilter.resolve(since=date(2025, 3, 1))
    assert since_filter.since == date(2025, 3, 1)

    # Conflicting --all with other flags raises ValueError
    with pytest.raises(ValueError, match="Cannot combine --all"):
        DateFilter.resolve(all_time=True, week=True)

    with pytest.raises(ValueError, match="Cannot combine --all"):
        DateFilter.resolve(all_time=True, month=True)

    with pytest.raises(ValueError, match="Cannot combine --all"):
        DateFilter.resolve(all_time=True, since=date(2025, 3, 1))
