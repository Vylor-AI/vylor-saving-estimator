# vylor-savings-estimator

> **Instantly estimate how much [Vylor MCP](https://github.com/vylor-ai/vylor-mcp) cuts costs, tokens, and time on your Claude AI sessions — zero configuration required.**

Reads your Claude session `.jsonl` files (Claude Code CLI & Claude Desktop), detects file-exploration and repo-crawling patterns that Vylor MCP replaces, and produces a rich savings report tracking **cost cut**, **tokens cut**, and **time cut**.

---

## Key Features

- **Automatic Session Discovery**: Automatically discovers sessions from default Claude Code CLI (`~/.claude/projects/`) and Claude Desktop directories across Windows, macOS, and Linux.
- **Automatic Sub-Agent Stitching**: Detects background sub-agents (e.g. `Explore` agents spawned via the `Agent` tool) and stitches them into their root task session, reporting true total task cost and sub-agent turn counts.
- **Direct File-Cache Compounding Model**: Accurately models the prompt cache "snowball effect"—avoiding file dumps on early turns eliminates re-reading those files from the prompt cache on every subsequent turn.
- **Rich Output Formats**: Terminal tables with `rich`, self-contained interactive HTML with Chart.js charts, and machine-readable JSON exports.
- **Extensible OOP Architecture**: Built with SOLID principles, abstract interfaces (`ISessionDiscoverer`, `ISessionParser`, `ITurnClassifier`, `ISavingsEngine`, `IReportRenderer`), and Dependency Injection.

---

## Installation

### From Source / Private Repository

```bash
git clone https://github.com/vylor-ai/vylor-savings-estimator.git
cd vylor-savings-estimator
pip install -e .
```

### Once Published to PyPI

```bash
pip install vylor-savings-estimator
# or via pipx (isolated environment)
pipx install vylor-savings-estimator
```

---

## Usage

```bash
# Auto-discover all Claude sessions, all-time (zero config)
vylor-estimate

# Date filtering
vylor-estimate --week          # last 7 days only
vylor-estimate --month         # last 30 days only
vylor-estimate --since 2025-09-01

# Explicit path (single file or custom directory)
vylor-estimate /path/to/session.jsonl
vylor-estimate /path/to/sessions/

# Export reports
vylor-estimate --format html -o report.html
vylor-estimate --format json -o report.json
```

---

## How Savings Are Calculated

### The Compounding Cache Reality
In Claude Code and Claude Desktop, reading files writes their contents directly into the prompt cache (`cache_creation_input_tokens`). On **every single subsequent turn** for the rest of the session—even when the model is merely editing code or reasoning—those files are repeatedly re-read from the cache (`cache_read_input_tokens`).

### The Direct File-Cache Model
1. **100% Avoided Context**: The actual tokens loaded by file exploration (`Read`, `read_file`, `list_dir`, `grep`) are eliminated from entering the context.
2. **Downstream Cache Reduction**: That avoided volume is subtracted from `cache_read_tokens` on every future turn in the session:
   $$\text{saved\_cache\_read} = \min(\text{turn.cache\_read\_tokens}, \text{accumulated\_avoided\_context})$$
3. **0% Output Tokens**: The model still writes all its code, plans, and answers—output tokens are never credited as saved.
4. **I/O Latency**: Eliminates file-reading disk/network round-trip time on intercepted turns.

---

## What It Detects

| Vylor Tool | Replaces | Supported Tools & Patterns |
|---|---|---|
| `find_files` | Heavy file reads & dumps | `Read`, `read_file`, `cat`, `view_file`, sequential multi-reads |
| `request_repo_map` | Directory exploration & trees | `Glob`, `list_dir`, `ls`, `tree`, `find` |
| `find_code_definition` | Symbol searching & code crawling | `Grep`, `grep_search`, `ripgrep`, regex scans |
| `mcp__vylor__*` | Measured active Vylor runs | Calculates actual savings achieved with Vylor active |

---

## Example Output

```
Found 2 session file(s). Parsing...

                            VYLOR SAVINGS ESTIMATE                             
               Analyzed: 1 session(s)  |  29 turns  |  All-time                
                         (includes 12 sub-agent turns)                         

+-----------------------------------------------------------------------------+
| Metric              |  Baseline (paid) | With Vylor MCP |    Saved |  % Cut |
|---------------------+------------------+----------------+----------+--------|
| Total Cost          |          $0.5565 |        $0.3080 | -$0.2485 | -44.7% |
| Total Tokens        |            1.03M |         377.1K |  -656.0K | -63.5% |
| Total Time          |           2m 48s |         2m 11s |     -36s | -21.9% |
+-----------------------------------------------------------------------------+

--------------------------- Breakdown by Vylor Tool ---------------------------
+-----------------------------------------------------------------------------+
| Vylor Tool              |   Intercepted Turns |  Cost Saved |  Tokens Saved |
|-------------------------+---------------------+-------------+---------------|
| find_files              |                  11 |    -$0.1482 |       -369.7K |
| request_repo_map        |                   4 |    -$0.0563 |       -149.5K |
| find_code_definition    |                   1 |    -$0.0273 |        -53.8K |
+-----------------------------------------------------------------------------+

--------------------------- Top Sessions by Savings ---------------------------
+-----------------------------------------------------------------------------+
| Session ID              | Turns | Baseline Cost | Cost Saved |  % Cut | Sub-agents |
|-------------------------+-------+---------------+------------+--------+------------|
| 4bc32054-f243-42...e29f |    29 |       $0.5565 |   -$0.2485 | -44.7% |         12 |
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
| Direct cache model: 100% avoided file reads | 100% downstream cache reads | |
| 0% output tokens                                                            |
| Powered by Vylor MCP -- https://github.com/vylor-ai/vylor-mcp               |
+-----------------------------------------------------------------------------+
```

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
git clone https://github.com/vylor-ai/vylor-savings-estimator.git
cd vylor-savings-estimator
pip install -e ".[dev]"
pytest
```
