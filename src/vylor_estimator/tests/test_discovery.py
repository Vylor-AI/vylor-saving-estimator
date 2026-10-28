from __future__ import annotations

import pytest

from vylor_estimator.discovery import (
    ClaudeSessionDiscoverer,
    CodexSessionDiscoverer,
    CompositeDiscoverer,
)


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


def test_codex_discover_walks_rollouts_only(tmp_path):
    (tmp_path / "rollout-1.jsonl").write_text("{}")
    (tmp_path / "other.jsonl").write_text("{}")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "rollout-2.jsonl").write_text("{}")

    disc = CodexSessionDiscoverer(explicit_path=tmp_path)
    res = disc.discover()
    names = [p.name for p in res]
    assert "rollout-1.jsonl" in names
    assert "rollout-2.jsonl" in names
    assert "other.jsonl" not in names


def test_composite_discoverer_filters_by_agent(tmp_path):
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    (claude_dir / "c1.jsonl").write_text("{}")

    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / "rollout-x.jsonl").write_text("{}")

    c_disc = ClaudeSessionDiscoverer(explicit_path=claude_dir)
    x_disc = CodexSessionDiscoverer(explicit_path=codex_dir)

    all_res = CompositeDiscoverer(claude_discoverer=c_disc, codex_discoverer=x_disc, agent="all").discover()
    assert len(all_res) == 2

    claude_only = CompositeDiscoverer(claude_discoverer=c_disc, codex_discoverer=x_disc, agent="claude").discover()
    assert len(claude_only) == 1
    assert claude_only[0].name == "c1.jsonl"

    codex_only = CompositeDiscoverer(claude_discoverer=c_disc, codex_discoverer=x_disc, agent="codex").discover()
    assert len(codex_only) == 1
    assert codex_only[0].name == "rollout-x.jsonl"


def test_composite_discoverer_explicit_mixed_directory(tmp_path):
    (tmp_path / "c1.jsonl").write_text('{"type": "user"}')
    (tmp_path / "rollout-x.jsonl").write_text('{"type": "session_meta"}')

    all_disc = CompositeDiscoverer(explicit_path=tmp_path, agent="all")
    assert len(all_disc.discover()) == 2

    claude_disc = CompositeDiscoverer(explicit_path=tmp_path, agent="claude")
    claude_files = claude_disc.discover()
    assert len(claude_files) == 1
    assert claude_files[0].name == "c1.jsonl"

    codex_disc = CompositeDiscoverer(explicit_path=tmp_path, agent="codex")
    codex_files = codex_disc.discover()
    assert len(codex_files) == 1
    assert codex_files[0].name == "rollout-x.jsonl"


def test_composite_discoverer_explicit_file_agent_mismatch(tmp_path):
    codex_f = tmp_path / "rollout-1.jsonl"
    codex_f.write_text('{"type": "session_meta"}')

    claude_f = tmp_path / "claude_session.jsonl"
    claude_f.write_text('{"type": "user"}')

    # Trying to analyze codex file with agent=claude
    with pytest.raises(FileNotFoundError, match="agent filter is set to 'claude'"):
        CompositeDiscoverer(explicit_path=codex_f, agent="claude").discover()

    # Trying to analyze claude file with agent=codex
    with pytest.raises(FileNotFoundError, match="agent filter is set to 'codex'"):
        CompositeDiscoverer(explicit_path=claude_f, agent="codex").discover()

