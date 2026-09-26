"""test_discovery.py -- Tests for session auto-discovery."""
from __future__ import annotations

import pytest

from vylor_estimator.discovery import ClaudeSessionDiscoverer


def test_walk_finds_jsonl(tmp_path):
    (tmp_path / "a.jsonl").write_text("{}")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.jsonl").write_text("{}")
    (tmp_path / "not_json.csv").write_text("data")

    results = ClaudeSessionDiscoverer().walk_jsonl(tmp_path)
    names = [p.name for p in results]
    assert "a.jsonl" in names
    assert "b.jsonl" in names
    assert "not_json.csv" not in names


def test_discover_explicit_file(tmp_path):
    f = tmp_path / "session.jsonl"
    f.write_text("{}")
    result = ClaudeSessionDiscoverer(explicit_path=f).discover()
    assert result == [f]


def test_discover_explicit_directory(tmp_path):
    (tmp_path / "s1.jsonl").write_text("{}")
    (tmp_path / "s2.jsonl").write_text("{}")
    result = ClaudeSessionDiscoverer(explicit_path=tmp_path).discover()
    assert len(result) == 2


def test_discover_nonexistent_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ClaudeSessionDiscoverer(explicit_path=tmp_path / "nonexistent.jsonl").discover()
