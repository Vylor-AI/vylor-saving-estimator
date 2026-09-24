"""html_report.py -- Self-contained HTML report with Chart.js charts."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from vylor_estimator.report.base import IReportRenderer
from vylor_estimator.savings import SavingsReport
from vylor_estimator.report.json_report import build_json_report


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    h = int(td.total_seconds() // 3600)
    m = int((td.total_seconds() % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m"
    return f"{int(td.total_seconds())}s"


class HtmlReportRenderer(IReportRenderer):
    """Renders the savings report to a self-contained HTML file with Chart.js charts."""

    def render(
        self,
        report: SavingsReport,
        date_range_label: str = "all-time",
        output_path: Path | str | None = None,
    ) -> None:
        target_path = Path(output_path) if output_path else Path("report.html")

        data = build_json_report(report, date_range_label)
        agg = report.total

        # Chart data
        days = list(data["by_day"].keys())
        baseline_costs = [data["by_day"][d]["baseline"]["cost"] for d in days]
        vylor_costs = [data["by_day"][d]["with_vylor"]["cost"] for d in days]

        tool_labels = list(agg.by_tool.keys())
        tool_savings = [agg.by_tool[t]["saved_cost"] for t in tool_labels]

        top_sessions = sorted(data["by_session"].items(), key=lambda x: -x[1]["savings"]["cost"])[:10]

        html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vylor MCP Savings Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
      :root {{
        --purple: #6C47FF;
        --purple-light: #8B6FFF;
        --bg: #0D0D14;
        --bg2: #13131F;
        --bg3: #1A1A2E;
        --text: #E8E8F0;
        --text-dim: #8888AA;
        --green: #4ADE80;
        --yellow: #FACC15;
        --red: #F87171;
        --border: #2A2A40;
      }}
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: var(--bg); color: var(--text); padding: 24px; }}
      h1 {{ font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, var(--purple-light), #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
      .subtitle {{ color: var(--text-dim); font-size: 0.9rem; margin-top: 4px; }}
      .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 24px 0; }}
      .card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
      .card .label {{ font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: .06em; }}
      .card .value {{ font-size: 1.7rem; font-weight: 700; margin: 6px 0 2px; }}
      .card .sub {{ font-size: 0.85rem; }}
      .card.green .value {{ color: var(--green); }}
      .card.yellow .value {{ color: var(--yellow); }}
      .card.purple .value {{ color: var(--purple-light); }}
      .section {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
      .section h2 {{ font-size: 1rem; font-weight: 600; color: var(--purple-light); margin-bottom: 16px; }}
      .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
      @media (max-width: 700px) {{ .charts {{ grid-template-columns: 1fr; }} }}
      table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
      th {{ text-align: left; padding: 8px 12px; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border); }}
      td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
      td.green {{ color: var(--green); }}
      td.yellow {{ color: var(--yellow); }}
      td.dim {{ color: var(--text-dim); font-size: 0.8rem; overflow: hidden; text-overflow: ellipsis; max-width: 200px; white-space: nowrap; }}
      .badge {{ display: inline-block; background: var(--purple); color: #fff; border-radius: 999px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }}
      .footer {{ text-align: center; color: var(--text-dim); font-size: 0.8rem; margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border); }}
      a {{ color: var(--purple-light); text-decoration: none; }}
    </style>
    </head>
    <body>
    <h1>Vylor MCP Savings Report</h1>
    <div class="subtitle">
      {data['sessions_analyzed']} sessions &nbsp;·&nbsp; {agg.turns_analyzed} turns &nbsp;·&nbsp; {date_range_label}
      {f"&nbsp;·&nbsp; {agg.subagent_turns} sub-agent turns" if agg.subagent_turns else ""}
    </div>

    <div class="grid">
      <div class="card green">
        <div class="label">Cost Saved</div>
        <div class="value">-${agg.saved_cost:,.2f}</div>
        <div class="sub" style="color:var(--text-dim)">from ${agg.baseline_cost:,.2f} baseline &nbsp;<span class="badge">-{agg.pct_cost_cut:.1f}%</span></div>
      </div>
      <div class="card green">
        <div class="label">Tokens Saved</div>
        <div class="value">-{_fmt_tokens(agg.total_saved_tokens)}</div>
        <div class="sub" style="color:var(--text-dim)">from {_fmt_tokens(agg.baseline_total_tokens)} baseline &nbsp;<span class="badge">-{agg.pct_tokens_cut:.1f}%</span></div>
      </div>
      <div class="card green">
        <div class="label">Time Saved</div>
        <div class="value">-{_fmt_time(agg.saved_time_seconds)}</div>
        <div class="sub" style="color:var(--text-dim)">from {_fmt_time(agg.baseline_time_seconds)} baseline &nbsp;<span class="badge">-{agg.pct_time_cut:.1f}%</span></div>
      </div>
      <div class="card yellow">
        <div class="label">Turns Intercepted</div>
        <div class="value">{agg.turns_intercepted}</div>
        <div class="sub" style="color:var(--text-dim)">of {agg.turns_analyzed} total turns</div>
      </div>
    </div>

    <div class="charts">
      <div class="section">
        <h2>Daily Cost: Baseline vs With Vylor</h2>
        <canvas id="costChart" height="200"></canvas>
      </div>
      <div class="section">
        <h2>Savings by Vylor Tool</h2>
        <canvas id="toolChart" height="200"></canvas>
      </div>
    </div>

    <div class="section">
      <h2>Top Sessions by Savings</h2>
      <table>
        <thead>
          <tr>
            <th>Session ID</th>
            <th>Turns</th>
            <th>Baseline Cost</th>
            <th>Cost Saved</th>
            <th>% Cut</th>
            <th>Sub-agents</th>
          </tr>
        </thead>
        <tbody>
          {"".join(
              f"<tr>"
              f"<td class='dim' title='{sid}'>{sid[:40]}</td>"
              f"<td>{info['turns_analyzed']}</td>"
              f"<td class='yellow'>${info['baseline']['cost']:,.4f}</td>"
              f"<td class='green'>-${info['savings']['cost']:,.4f}</td>"
              f"<td><span class='badge'>-{info['savings']['cost_pct']:.1f}%</span></td>"
              f"<td>{info['subagent_turns'] if info['subagent_turns'] else '-'}</td>"
              f"</tr>"
              for sid, info in top_sessions
          )}
        </tbody>
      </table>
    </div>

    <div class="section">
      <h2>Breakdown by Model</h2>
      <table>
        <thead>
          <tr><th>Model</th><th>Turns</th><th>Baseline Cost</th><th>Cost Saved</th><th>Tokens Saved</th></tr>
        </thead>
        <tbody>
          {"".join(
              f"<tr>"
              f"<td>{m.replace('claude-','')}</td>"
              f"<td>{info['turns']}</td>"
              f"<td class='yellow'>${info['baseline_cost']:,.4f}</td>"
              f"<td class='green'>-${info['saved_cost']:,.4f}</td>"
              f"<td class='green'>-{_fmt_tokens(info['saved_tokens'])}</td>"
              f"</tr>"
              for m, info in sorted(agg.by_model.items(), key=lambda x: -x[1]['baseline_cost'])
          )}
        </tbody>
      </table>
    </div>

    <div class="footer">
      Generated by <a href="https://github.com/vylor-ai/vylor-savings-estimator">vylor-savings-estimator</a>
      &nbsp;·&nbsp; Conservative model (70% input · 90% cache · 0% output)
      &nbsp;·&nbsp; <a href="https://github.com/vylor-ai/vylor-mcp">Vylor MCP</a>
    </div>

    <script>
    const PURPLE = '#6C47FF';
    const PURPLE_LIGHT = '#8B6FFF';
    const GREEN = '#4ADE80';
    const YELLOW = '#FACC15';
    const GRID_COLOR = 'rgba(255,255,255,0.06)';
    const TEXT_COLOR = '#8888AA';
    const defaults = Chart.defaults;
    defaults.color = TEXT_COLOR;
    defaults.borderColor = GRID_COLOR;

    const days = {json.dumps(days)};
    const baselineCosts = {json.dumps(baseline_costs)};
    const vylorCosts = {json.dumps(vylor_costs)};
    const toolLabels = {json.dumps(tool_labels)};
    const toolSavings = {json.dumps(tool_savings)};

    new Chart(document.getElementById('costChart'), {{
      type: 'line',
      data: {{
        labels: days,
        datasets: [
          {{
            label: 'Baseline Cost ($)',
            data: baselineCosts,
            borderColor: YELLOW,
            backgroundColor: 'rgba(250,204,21,0.08)',
            tension: 0.3,
            fill: true,
            pointRadius: 3,
          }},
          {{
            label: 'With Vylor MCP ($)',
            data: vylorCosts,
            borderColor: GREEN,
            backgroundColor: 'rgba(74,222,128,0.10)',
            tension: 0.3,
            fill: true,
            pointRadius: 3,
          }},
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'bottom' }} }},
        scales: {{
          x: {{ grid: {{ color: GRID_COLOR }} }},
          y: {{ grid: {{ color: GRID_COLOR }}, ticks: {{ callback: v => '$' + v.toFixed(3) }} }}
        }}
      }}
    }});

    new Chart(document.getElementById('toolChart'), {{
      type: 'doughnut',
      data: {{
        labels: toolLabels,
        datasets: [{{
          data: toolSavings,
          backgroundColor: ['#6C47FF', '#4ADE80', '#FACC15', '#F87171', '#A78BFA'],
          borderWidth: 0,
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ position: 'bottom' }},
          tooltip: {{ callbacks: {{ label: function(c) {{ return '-$' + c.parsed.toFixed(4); }} }} }}
        }}
      }}
    }});
    </script>
    </body>
    </html>"""

        target_path.write_text(html, encoding="utf-8")

def write_html_report(
    report: SavingsReport,
    output_path: Path | str,
    date_range_label: str = "all-time",
) -> None:
    """Backward-compatible helper function."""
    HtmlReportRenderer().render(report, date_range_label=date_range_label, output_path=output_path)
