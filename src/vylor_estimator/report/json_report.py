"""json_report.py -- Machine-readable JSON export of savings report."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from vylor_estimator.report.base import IReportRenderer
from vylor_estimator.savings import SavingsReport


def _agg_to_dict(agg, label: str = "") -> dict:
    return {
        "label": label,
        "turns_analyzed": agg.turns_analyzed,
        "turns_intercepted": agg.turns_intercepted,
        "subagent_turns": agg.subagent_turns,
        "baseline": {
            "cost": round(agg.baseline_cost, 6),
            "input_tokens": agg.baseline_input_tokens,
            "output_tokens": agg.baseline_output_tokens,
            "cache_read_tokens": agg.baseline_cache_read_tokens,
            "cache_write_tokens": agg.baseline_cache_write_tokens,
            "total_tokens": agg.baseline_total_tokens,
            "time_seconds": round(agg.baseline_time_seconds, 2),
        },
        "with_vylor": {
            "cost": round(agg.estimated_cost, 6),
            "total_tokens": agg.estimated_tokens,
            "time_seconds": round(agg.estimated_time_seconds, 2),
        },
        "savings": {
            "cost": round(agg.saved_cost, 6),
            "cost_pct": agg.pct_cost_cut,
            "input_tokens": agg.saved_input_tokens,
            "cache_tokens": agg.saved_cache_tokens + agg.saved_cache_write_tokens,
            "cache_read_tokens": agg.saved_cache_tokens,
            "cache_write_tokens": agg.saved_cache_write_tokens,
            "total_tokens": agg.total_saved_tokens,
            "tokens_pct": agg.pct_tokens_cut,
            "time_seconds": round(agg.saved_time_seconds, 2),
            "time_pct": agg.pct_time_cut,
        },
        "breakdown_by_tool": agg.by_tool,
        "breakdown_by_model": agg.by_model,
    }


def build_json_report(report: SavingsReport, date_range_label: str = "all-time") -> dict:
    """Build a dictionary representation of the savings report."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date_range": date_range_label,
        "sessions_analyzed": len(report.by_session),
        "total": _agg_to_dict(report.total, "all-time"),
        "by_session": {
            sess_id: _agg_to_dict(agg, sess_id)
            for sess_id, agg in sorted(
                report.by_session.items(), key=lambda x: -x[1].saved_cost
            )
        },
        "by_day": {
            str(d): _agg_to_dict(agg, str(d))
            for d, agg in report.by_day.items()
        },
    }


class JsonReportRenderer(IReportRenderer):
    """Renders the savings report to a JSON file."""

    def render(
        self,
        report: SavingsReport,
        date_range_label: str = "All-time",
        output_path: Path | str | None = None,
    ) -> None:
        target_path = Path(output_path) if output_path else Path("report.json")
        data = build_json_report(report, date_range_label)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)


def write_json_report(
    report: SavingsReport,
    output_path: Path | str,
    date_range_label: str = "all-time",
) -> None:
    """Backward-compatible helper function."""
    JsonReportRenderer().render(report, date_range_label=date_range_label, output_path=output_path)
