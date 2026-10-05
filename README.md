# vylor-savings-estimator

> **See how much your Claude AI sessions waste on file exploration — in tokens, cost, and time. Zero configuration.**

Reads your Claude session `.jsonl` files (Claude Code CLI & Claude Desktop), detects the file-exploration and repo-crawling patterns that [Vylor MCP](https://github.com/Vylor-AI/vylor-mcp-binaries) replaces, and reports how much of your spend was **avoidable waste**.

> [!NOTE]
> The numbers are an *estimate of avoidable waste*, not a guaranteed saving. Vylor can't promise to eliminate all of it; this tool shows how much of your usage looks like exploration that a code-search tool could have handled.

---

## Install

### One command (standalone binary, no Python needed)

**macOS / Linux**
```bash
curl -fsSL https://raw.githubusercontent.com/Vylor-AI/vylor-saving-estimator/main/install.sh | sh
```

**Windows (PowerShell)**
```powershell
irm https://raw.githubusercontent.com/Vylor-AI/vylor-saving-estimator/main/install.ps1 | iex
```

Or download the binary for your platform from **[Releases](../../releases/latest)**:

| Platform | Binary |
|---|---|
| Windows (x64) | `vylor-estimate.exe` |
| macOS (x64 / Apple Silicon) | `vylor-estimate-macos` |
| Linux (x64) | `vylor-estimate-linux` |

Manual install on macOS / Linux:
```bash
chmod +x vylor-estimate-macos
mv vylor-estimate-macos /usr/local/bin/vylor-estimate
```
On Windows, move `vylor-estimate.exe` to any folder in your `PATH`.

### From source

```bash
git clone https://github.com/vylor-ai/vylor-saving-estimator.git
cd vylor-saving-estimator
pip install -e .
```

### Via pip / pipx (Python 3.10+)

```bash
pip install vylor-estimate
# or in an isolated environment
pipx install vylor-estimate
```

---

## Usage

```bash
# Auto-discover all Claude sessions (defaults to last 30 days)
vylor-estimate

# Date filtering
vylor-estimate --week          # last 7 days only
vylor-estimate --month         # last 30 days only
vylor-estimate --all           # all-time (disable default 30-day filter)
vylor-estimate --since 2025-09-01

# Explicit path (single file or custom directory)
vylor-estimate /path/to/session.jsonl
vylor-estimate /path/to/sessions/

# Detailed report (waste by source, model, and session)
vylor-estimate -d
```

Session files are read locally. No data is sent anywhere.

---

## Example Output

Default: one table, one hint.

```
                              AGENT OVERHEAD ESTIMATE
                      Analyzed: 5 session(s)  |  Last 30 days

     +--------------------------------------------------------------------+
     |                     |            Cost |        Tokens |       Time |
     |---------------------+-----------------+---------------+------------|
     | Total               |         $2.6576 |         6.14M |        11m |
     | Avoidable overhead  | $2.2445 (84.5%) | 5.39M (87.8%) | 9m (83.2%) |
     +--------------------------------------------------------------------+
                      Top overhead sources: sub-agents 98%
                            Run with -d for details.
```

- **Total** is what the analyzed sessions cost, in cost, tokens and time.
- **Avoidable overhead** is the portion spent on exploration and reconnaissance that Vylor tools make unnecessary.
- **Top overhead sources** tells you what contributes most to the overhead.

`-d` adds overhead by source (cost, tokens, time), a breakdown by model, and the top sessions by overhead.

---

## How Overhead Is Estimated

A turn counts as avoidable overhead when it is one of:

| Source | What counts |
|---|---|
| **Sub-agents** | Every turn of a background sub-agent (e.g. `Explore` agents spawned via the `Agent` tool). Whole turn: cost, tokens, time. |
| **Searches** | `Grep`, `grep_search`, `ripgrep`, or a shell command using `grep` / `rg` / `findstr` / `Select-String`. |
| **Directory listing** | `Glob`, `list_dir`, `LS`, or a shell command using `find` / `ls` / `tree` / `dir` / `Get-ChildItem`. |

Rules for main-agent turns:

- **Pure exploration only.** Every tool call in the turn must be exploration. A turn that also edits files, builds, or runs tests is not counted.
- **Shell commands are inspected.** `Bash` / `PowerShell` count only when the command is exploration (`cd src && grep -rn foo . | head`). `git commit`, `npm test` or `git log | grep fix` do not.
- For an overhead main-agent turn, the output tokens and the prompt-cache reads it caused are counted. For a sub-agent turn, everything is counted.

### Avoidable time

Time is **agent-time**: the sum of the durations of overhead turns, using the turn timestamps in the session files. It is model generation time only (not tool run time), each turn is capped at 10 minutes so idle gaps don't skew the totals, and parallel sub-agents can overlap, so it can exceed the wall-clock time.

### The compounding cache

Reading files writes their contents into the prompt cache. On every later turn of the session those files are re-read from the cache, so avoided exploration also shrinks the cache reads of later turns. That is why exploration is more expensive than its own token count suggests.

> [!NOTE]
> Sessions that already use Vylor MCP tools (`mcp__vylor__*`, `request_repo_map`, etc.) are detected and skipped automatically, so already-optimized runs aren't counted.

---

## Session File Locations

Automatically discovered from platform defaults:

| OS | Search Locations |
|---|---|
| **Windows** | `~/.claude/projects/` (Claude Code CLI)<br>`%APPDATA%\Claude\projects\` (Claude Desktop) |
| **macOS** | `~/.claude/projects/`<br>`~/Library/Application Support/Claude/projects/` |
| **Linux** | `~/.claude/projects/`<br>`~/.config/Claude/projects/` |

---

## Development

```bash
git clone https://github.com/vylor-ai/vylor-saving-estimator.git
cd vylor-saving-estimator
pip install -e ".[dev]"
pytest
```

Binaries are built by [`build-binary.yml`](.github/workflows/build-binary.yml) on every `v*` tag and published to this repo's [Releases](../../releases).
