# vylor-savings-estimator

> **See how much your Claude and OpenAI Codex sessions spend on file exploration and subagent overhead — in tokens, cost, and time. Zero configuration.**

Reads your **Claude Code**, **Claude Desktop**, and **OpenAI Codex CLI** session files, detects the file-exploration patterns, directory crawls, and background exploration subagents that [Vylor MCP](https://github.com/Vylor-AI/vylor-mcp-binaries) replaces, and reports how much spend and time **Vylor MCP can save**.

> [!NOTE]
> The numbers are an *estimate of avoidable exploration*, not a guaranteed saving. Vylor can't promise to eliminate all of it; this tool shows how much of your usage looks like exploration that a code-search tool could have handled.

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
# Auto-discover all sessions across Claude and Codex (defaults to last 30 days)
vylor-estimate

# Filter by AI agent
vylor-estimate --agent codex    # OpenAI Codex CLI sessions only
vylor-estimate --agent claude   # Claude Code / Desktop sessions only
vylor-estimate --agent all      # Analyze both (default)

# Date filtering
vylor-estimate --week          # last 7 days only
vylor-estimate --month         # last 30 days only
vylor-estimate --all           # all-time (disable default 30-day filter)
vylor-estimate --since 2026-01-01

# Explicit path (single file or custom directory)
vylor-estimate /path/to/rollout-xxx.jsonl
vylor-estimate /path/to/sessions/

# Detailed report (waste by source, agent, model, and session)
vylor-estimate -d
```

Session files are read locally. No data is sent anywhere.

---

## Example Output

Default: compact summary with activity & token cost breakdowns.

```text
                           VYLOR MCP SAVINGS ESTIMATE                          
Analyzed: 14 session(s) (Claude, Codex)  |  (All-time) 2026-08-24 to 2026-10-22 

    +--------------------------------------------------------------------+     
    |                     |            Cost |        Tokens |       Time |     
    |---------------------+-----------------+---------------+------------|     
    | Total               |         $6.1108 |        23.52M |        40m |     
    | Vylor MCP savings   | $1.2956 (21.2%) | 6.64M (28.2%) | 11m (27.5%)|     
    +--------------------------------------------------------------------+     
Top savings sources: sub-agents 52% (Explore 100%)  |  directory listing 26%  |  searches 22%
 Tool calls (663): Read 55% | Search 24% | Write 10% | Commands 8% | Sub-agent 3%
    Cost breakdown: cache_read 49% | cache_creation 24% | output_tokens 27% (thinking_tokens 24%)
                           Run with -d for details.
```

- **Total** is what the analyzed sessions cost, in cost, tokens, and time.
- **Vylor MCP savings** is the portion spent on exploration and reconnaissance that Vylor tools make unnecessary.
- **Top savings sources** tells you what contributes most to savings (e.g. Explore sub-agents, searches, directory listing).
- **Tool calls** summarizes tool usage across core categories (`Read`, `Search`, `Commands`, `Write`, and background sub-agents).
- **Cost breakdown** breaks down token costs across prompt caching (`cache_read`, `cache_creation`) and model outputs (`output_tokens` including `thinking_tokens`).

`-d` adds savings by source (cost, tokens, time), a breakdown by agent (Claude vs. Codex), a breakdown by model, and the top sessions by savings.

---

## How Savings Are Estimated

A turn counts as avoidable overhead when it is one of:

| Source | What counts |
|---|---|
| **Sub-agents** | Turns from **Explore** background sub-agents (spawned via Claude's `Agent` / `Task` or Codex's `spawn_agent` + `wait_agent` polling). Sub-agents that only read/inspect code without editing files are classified as Explore waste. `Plan` sub-agents and worker subagents that modify files are retained in the baseline. Whole turn: cost, tokens, time. |
| **Searches** | `Grep`, `grep_search`, `ripgrep`, or a shell command using `grep` / `rg` / `findstr` / `Select-String`. |
| **Directory listing** | `Glob`, `list_dir`, `LS`, or a shell command using `find` / `ls` / `tree` / `dir` / `Get-ChildItem`. |

Rules for main-agent turns:

- **Pure exploration only.** Every tool call in the turn must be exploration. A turn that also edits files, builds, or runs tests is not counted.
- **Shell commands are inspected.** `Bash` / `PowerShell` count only when the command is exploration (`cd src && grep -rn foo . | head`, `rg --files`). `git commit`, `npm test` or `git log | grep fix` do not.
- For an overhead main-agent turn, the output tokens and the prompt-cache reads it caused are counted. For an Explore sub-agent turn, everything is counted.

### Avoidable time

Time is **agent-time**: the sum of the durations of overhead turns, using the turn timestamps in the session files. It is model generation time only (not tool run time), each turn is capped at 10 minutes so idle gaps don't skew the totals, and parallel sub-agents can overlap, so it can exceed the wall-clock time.

### The compounding cache

Reading files writes their contents into the prompt cache. On every later turn of the session those files are re-read from the cache, so avoided exploration also shrinks the cache reads of later turns. That is why exploration is more expensive than its own token count suggests.

> [!NOTE]
> Sessions that already use Vylor MCP tools (`mcp__vylor__*`, `request_repo_map`, etc.) are detected and skipped automatically, so already-optimized runs aren't counted.

---

## Session File Locations

Automatically discovered from platform defaults:

| Agent | Platform | Search Locations |
|---|---|---|
| **Claude** | Windows | `~/.claude/projects/` (Claude Code CLI)<br>`%APPDATA%\Claude\projects\` (Claude Desktop) |
| | macOS | `~/.claude/projects/`<br>`~/Library/Application Support/Claude/projects/` |
| | Linux | `~/.claude/projects/`<br>`~/.config/Claude/projects/` |
| **OpenAI Codex** | All Platforms | `~/.codex/sessions/`<br>`~/.codex/archived_sessions/`<br>`$CODEX_HOME/sessions/` |

---

## Development

```bash
git clone https://github.com/vylor-ai/vylor-saving-estimator.git
cd vylor-saving-estimator
pip install -e ".[dev]"
pytest
```

Binaries are built by [`build-binary.yml`](.github/workflows/build-binary.yml) on every `v*` tag and published to this repo's [Releases](../../releases).
