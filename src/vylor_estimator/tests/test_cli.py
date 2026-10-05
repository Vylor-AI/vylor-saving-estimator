from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from vylor_estimator.cli import main


def test_cli_default_uses_month_filter():
    runner = CliRunner()
    with patch("vylor_estimator.cli.create_estimator") as mock_create:
        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        result = runner.invoke(main, [])
        assert result.exit_code == 0
        mock_estimator.run.assert_called_once()
        date_filter = mock_estimator.run.call_args.kwargs["date_filter"]
        assert date_filter.month is True
        assert date_filter.week is False
        assert date_filter.since is None


def test_cli_all_flag_disables_date_filter():
    runner = CliRunner()
    with patch("vylor_estimator.cli.create_estimator") as mock_create:
        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        result = runner.invoke(main, ["--all"])
        assert result.exit_code == 0
        mock_estimator.run.assert_called_once()
        date_filter = mock_estimator.run.call_args.kwargs["date_filter"]
        assert date_filter.month is False
        assert date_filter.week is False
        assert date_filter.since is None
        assert date_filter.get_cutoff_date() is None
        assert date_filter.label == "All-time"


def test_cli_week_flag_overrides_default():
    runner = CliRunner()
    with patch("vylor_estimator.cli.create_estimator") as mock_create:
        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        result = runner.invoke(main, ["--week"])
        assert result.exit_code == 0
        mock_estimator.run.assert_called_once()
        date_filter = mock_estimator.run.call_args.kwargs["date_filter"]
        assert date_filter.week is True
        assert date_filter.month is False
        assert date_filter.label == "Last 7 days"


def test_cli_since_flag_overrides_default():
    runner = CliRunner()
    with patch("vylor_estimator.cli.create_estimator") as mock_create:
        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        result = runner.invoke(main, ["--since", "2025-06-01"])
        assert result.exit_code == 0
        mock_estimator.run.assert_called_once()
        date_filter = mock_estimator.run.call_args.kwargs["date_filter"]
        assert date_filter.since == date(2025, 6, 1)
        assert date_filter.month is False
        assert date_filter.label == "Since 2025-06-01"


def test_cli_explicit_month_flag():
    runner = CliRunner()
    with patch("vylor_estimator.cli.create_estimator") as mock_create:
        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        result = runner.invoke(main, ["--month"])
        assert result.exit_code == 0
        mock_estimator.run.assert_called_once()
        date_filter = mock_estimator.run.call_args.kwargs["date_filter"]
        assert date_filter.month is True
        assert date_filter.week is False
        assert date_filter.label == "Last 30 days"


def test_cli_conflicting_all_and_week_flags():
    runner = CliRunner()
    result = runner.invoke(main, ["--all", "--week"])
    assert result.exit_code == 0
    assert "Cannot combine --all with other date filter options." in result.output



def test_cli_debug_flag_passed_to_estimator():
    runner = CliRunner()
    with patch("vylor_estimator.cli.create_estimator") as mock_create:
        mock_create.return_value = MagicMock()

        result = runner.invoke(main, ["-d"])
        assert result.exit_code == 0
        assert mock_create.call_args.kwargs["debug"] is True

        mock_create.reset_mock()
        result = runner.invoke(main, [])
        assert mock_create.call_args.kwargs["debug"] is False
