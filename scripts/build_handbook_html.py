"""Build the standalone quantitative ML handbook HTML file.

The source chapters remain simple Markdown files. This script uses only the
Python standard library so rebuilding the handbook does not add a dependency.
"""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDBOOK_DIR = ROOT / "docs" / "handbook"
OUTPUT_PATH = ROOT / "docs" / "quantitative_ml_handbook.html"

CHAPTERS = [
  ("overview", "README.md", "Orientation"),
  ("python-stack", "01_python_and_data_stack.md", "Python & data stack"),
  ("data-preparation", "02_market_data_preparation.md", "Market data"),
  ("statistics-factors", "03_statistics_and_factors.md", "Statistics & factors"),
  ("ml-models", "04_machine_learning_models.md", "ML models"),
  ("validation", "05_time_aware_validation.md", "Time-aware validation"),
  ("portfolios", "06_finance_and_portfolios.md", "Portfolio construction"),
  ("backtesting", "07_backtesting.md", "Backtesting"),
  ("metrics", "08_metrics_and_visualization.md", "Metrics & evidence"),
  ("reproducibility", "09_dashboard_and_reproducibility.md", "Reproducibility"),
  ("mathematics", "10_mathematical_foundations.md", "Mathematical foundations"),
  ("advanced-factors", "11_advanced_factor_engineering.md", "Advanced factors"),
  ("model-recipes", "12_practical_model_recipes.md", "Practical model recipes"),
  ("research-risk", "13_inference_and_research_risk.md", "Inference & research risk"),
  ("portfolio-optimization", "14_portfolio_optimization_and_risk.md", "Optimization & risk"),
  ("case-study", "15_end_to_end_case_study.md", "End-to-end case study"),
  ("glossary", "glossary.md", "Glossary"),
]

FILE_TO_ANCHOR = {filename: anchor for anchor, filename, _ in CHAPTERS}


def slugify(value: str) -> str:
  """Return a stable HTML id fragment."""
  clean = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
  return clean or "section"


def inline_markup(value: str) -> str:
  """Render the small inline Markdown subset used by the handbook."""
  placeholders: list[str] = []

  def preserve(fragment: str) -> str:
    placeholders.append(fragment)
    return f"\x00{len(placeholders) - 1}\x00"

  escaped = html.escape(value, quote=False)
  escaped = re.sub(
    r"`([^`]+)`",
    lambda match: preserve(f"<code>{match.group(1)}</code>"),
    escaped,
  )

  def link(match: re.Match[str]) -> str:
    label, href = match.group(1), html.unescape(match.group(2))
    path, separator, fragment = href.partition("#")
    if path in FILE_TO_ANCHOR:
      href = f"#{FILE_TO_ANCHOR[path]}"
      if separator and fragment:
        href += f"-{slugify(fragment)}"
    attributes = ""
    if href.startswith(("https://", "http://")):
      attributes = ' target="_blank" rel="noreferrer"'
    return preserve(
      f'<a href="{html.escape(href, quote=True)}"{attributes}>{label}</a>'
    )

  escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, escaped)
  escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
  escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
  for index in range(len(placeholders) - 1, -1, -1):
    fragment = placeholders[index]
    escaped = escaped.replace(f"\x00{index}\x00", fragment)
  return escaped


def is_table_separator(line: str) -> bool:
  cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
  return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def render_table(lines: list[str]) -> str:
  rows = [
    [cell.strip() for cell in line.strip().strip("|").split("|")]
    for line in lines
  ]
  header = rows[0]
  body = rows[2:]
  parts = ['<div class="table-wrap"><table><thead><tr>']
  parts.extend(f"<th>{inline_markup(cell)}</th>" for cell in header)
  parts.append("</tr></thead><tbody>")
  for row in body:
    parts.append("<tr>")
    parts.extend(f"<td>{inline_markup(cell)}</td>" for cell in row)
    parts.append("</tr>")
  parts.append("</tbody></table></div>")
  return "".join(parts)


def render_flow_diagram(code: str, diagram_id: str) -> str:
  """Render the simple Mermaid flowcharts in the handbook as inline SVG."""
  node_labels: dict[str, str] = {}
  node_order: list[str] = []
  edges: list[tuple[str, str]] = []

  for line in code.splitlines()[1:]:
    for key, label in re.findall(r"([A-Za-z0-9_]+)\[([^\]]+)\]", line):
      if key not in node_labels:
        node_order.append(key)
      node_labels[key] = label
    edge_text = re.sub(r"\[[^\]]+\]", "", line)
    edge = re.search(r"([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)", edge_text)
    if edge:
      start, end = edge.groups()
      edges.append((start, end))
      for key in (start, end):
        if key not in node_order:
          node_order.append(key)
          node_labels.setdefault(key, key)

  predecessors = {
    node: [start for start, end in edges if end == node]
    for node in node_order
  }
  levels: dict[str, int] = {}

  def resolve_level(node: str, visiting: set[str] | None = None) -> int:
    if node in levels:
      return levels[node]
    visiting = set() if visiting is None else visiting
    if node in visiting:
      return 0
    parents = predecessors.get(node, [])
    level = 0 if not parents else 1 + max(
      resolve_level(parent, visiting | {node}) for parent in parents
    )
    levels[node] = level
    return level

  for node in node_order:
    resolve_level(node)

  max_level = max(levels.values(), default=0)
  width = 760
  level_gap = 112
  top = 52
  box_width = 228
  box_height = 58
  height = top * 2 + max_level * level_gap + box_height
  positions: dict[str, tuple[float, float]] = {}
  for level in range(max_level + 1):
    members = [node for node in node_order if levels[node] == level]
    for position, node in enumerate(members, start=1):
      x = width * position / (len(members) + 1)
      y = top + level * level_gap
      positions[node] = (x, y)

  marker_id = f"{diagram_id}-arrow"
  description = "; ".join(
    f"{node_labels.get(start, start)} to {node_labels.get(end, end)}"
    for start, end in edges
  )
  parts = [
    f'<figure class="native-diagram" id="{diagram_id}">',
    '<figcaption><span>Visual flow</span>Follow the arrows from inputs to evidence</figcaption>',
    f'<svg class="flow-svg" viewBox="0 0 {width} {height}" role="img" '
    f'aria-label="{html.escape(description, quote=True)}">',
    f'<defs><marker id="{marker_id}" viewBox="0 0 10 10" refX="5" refY="5" '
    'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    '<path d="M 0 0 L 10 5 L 0 10 z"></path></marker></defs>',
    '<g class="flow-edges">',
  ]
  for start, end in edges:
    start_x, start_y = positions[start]
    end_x, end_y = positions[end]
    parts.append(
      f'<path d="M {start_x:.1f} {start_y + box_height / 2:.1f} '
      f'C {start_x:.1f} {start_y + level_gap * .58:.1f}, '
      f'{end_x:.1f} {end_y - level_gap * .58:.1f}, '
      f'{end_x:.1f} {end_y - box_height / 2 - 7:.1f}" '
      f'marker-end="url(#{marker_id})"></path>'
    )
  parts.append('</g><g class="flow-nodes">')
  for node in node_order:
    x, y = positions[node]
    label = html.escape(node_labels.get(node, node))
    parts.append(
      f'<g transform="translate({x:.1f} {y:.1f})">'
      f'<rect x="{-box_width / 2:.1f}" y="{-box_height / 2:.1f}" '
      f'width="{box_width}" height="{box_height}" rx="10"></rect>'
      f'<text class="node-key" x="{-box_width / 2 + 18:.1f}" y="4">{html.escape(node)}</text>'
      f'<text class="node-label" x="8" y="5" text-anchor="middle">{label}</text>'
      '</g>'
    )
  parts.append('</g></svg></figure>')
  return "".join(parts)


def render_markdown(source: str, chapter_id: str) -> tuple[str, str]:
  """Render one chapter and return its HTML and H1 title."""
  lines = source.splitlines()
  output: list[str] = []
  index = 0
  title = chapter_id
  used_ids: set[str] = set()

  def heading_id(text: str, level: int) -> str:
    if level == 1:
      return f"{chapter_id}-title"
    base = f"{chapter_id}-{slugify(text)}"
    candidate = base
    suffix = 2
    while candidate in used_ids:
      candidate = f"{base}-{suffix}"
      suffix += 1
    used_ids.add(candidate)
    return candidate

  while index < len(lines):
    raw = lines[index]
    stripped = raw.strip()
    if not stripped:
      index += 1
      continue

    fence = re.match(r"^```([\w+-]*)\s*$", stripped)
    if fence:
      language = fence.group(1) or "text"
      index += 1
      code_lines: list[str] = []
      while index < len(lines) and not lines[index].strip().startswith("```"):
        code_lines.append(lines[index])
        index += 1
      index += 1
      if language == "mermaid":
        output.append(
          render_flow_diagram(
            "\n".join(code_lines),
            f"{chapter_id}-diagram",
          )
        )
        continue
      code = html.escape("\n".join(code_lines))
      output.append(
        '<figure class="code-block">'
        f'<figcaption>{html.escape(language)}</figcaption>'
        f'<pre><code class="language-{html.escape(language)}">{code}</code></pre>'
        "</figure>"
      )
      continue

    if stripped == "$$":
      index += 1
      formula_lines: list[str] = []
      while index < len(lines) and lines[index].strip() != "$$":
        formula_lines.append(lines[index].strip())
        index += 1
      index += 1
      output.append(
        f'<div class="formula" role="math">{html.escape(" ".join(formula_lines))}</div>'
      )
      continue

    heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
    if heading:
      level = len(heading.group(1))
      text = heading.group(2)
      if level == 1:
        title = text
      identifier = heading_id(text, level)
      output.append(
        f'<h{level} id="{identifier}"><a class="heading-anchor" '
        f'href="#{identifier}" aria-label="Link to this section">#</a>'
        f"{inline_markup(text)}</h{level}>"
      )
      index += 1
      continue

    if stripped.startswith("|") and index + 1 < len(lines) and is_table_separator(lines[index + 1]):
      table_lines = [raw, lines[index + 1]]
      index += 2
      while index < len(lines) and lines[index].strip().startswith("|"):
        table_lines.append(lines[index])
        index += 1
      output.append(render_table(table_lines))
      continue

    if stripped.startswith(">"):
      quote_lines: list[str] = []
      while index < len(lines) and lines[index].strip().startswith(">"):
        quote_lines.append(lines[index].strip()[1:].strip())
        index += 1
      output.append(f"<blockquote>{inline_markup(' '.join(quote_lines))}</blockquote>")
      continue

    list_match = re.match(r"^(\s*)([-*]|\d+\.)\s+(.+)$", raw)
    if list_match:
      ordered = list_match.group(2).endswith(".") and list_match.group(2)[0].isdigit()
      tag = "ol" if ordered else "ul"
      items: list[str] = []
      while index < len(lines):
        item_match = re.match(r"^(\s*)([-*]|\d+\.)\s+(.+)$", lines[index])
        if not item_match:
          break
        current_ordered = item_match.group(2).endswith(".") and item_match.group(2)[0].isdigit()
        if current_ordered != ordered:
          break
        item = item_match.group(3).strip()
        index += 1
        continuation: list[str] = []
        while index < len(lines):
          next_line = lines[index]
          if not next_line.strip():
            break
          if re.match(r"^(#{1,4})\s+", next_line.strip()):
            break
          if re.match(r"^(\s*)([-*]|\d+\.)\s+", next_line):
            break
          if next_line.strip().startswith(("```", "$$", "|", ">")):
            break
          continuation.append(next_line.strip())
          index += 1
        if continuation:
          item = " ".join([item, *continuation])
        checkbox = re.match(r"^\[([ xX])\]\s+(.+)$", item)
        if checkbox:
          checked = " checked" if checkbox.group(1).lower() == "x" else ""
          rendered = (
            f'<label class="check-item"><input type="checkbox" disabled{checked}>'
            f"<span>{inline_markup(checkbox.group(2))}</span></label>"
          )
        else:
          rendered = inline_markup(item)
        items.append(f"<li>{rendered}</li>")
        while index < len(lines) and not lines[index].strip():
          index += 1
      output.append(f"<{tag}>{''.join(items)}</{tag}>")
      continue

    paragraph_lines = [stripped]
    index += 1
    while index < len(lines):
      candidate = lines[index].strip()
      if not candidate:
        break
      if re.match(r"^(#{1,4})\s+", candidate):
        break
      if candidate.startswith(("```", "$$", ">")):
        break
      if candidate.startswith("|") and index + 1 < len(lines) and is_table_separator(lines[index + 1]):
        break
      if re.match(r"^(\s*)([-*]|\d+\.)\s+", lines[index]):
        break
      paragraph_lines.append(candidate)
      index += 1
    output.append(f"<p>{inline_markup(' '.join(paragraph_lines))}</p>")

  return "\n".join(output), title


VISUAL_LAB = r"""
<section class="chapter visual-lab" id="visual-lab" data-title="Interactive visual lab">
  <div class="chapter-kicker">Interactive visual lab</div>
  <h1>See the research pipeline move</h1>
  <p class="chapter-lead">These small, deterministic examples connect the equations in the handbook to the objects you will inspect in a real experiment. They are teaching illustrations, not market forecasts.</p>

  <div class="pipeline" aria-label="Research pipeline from prices to evidence">
    <div class="pipe-stage"><span>01</span><strong>Prices</strong><small>OHLCV bars</small></div>
    <div class="pipe-arrow" aria-hidden="true">→</div>
    <div class="pipe-stage"><span>02</span><strong>Features</strong><small>Past-only signals</small></div>
    <div class="pipe-arrow" aria-hidden="true">→</div>
    <div class="pipe-stage"><span>03</span><strong>Ranks</strong><small>Relative forecasts</small></div>
    <div class="pipe-arrow" aria-hidden="true">→</div>
    <div class="pipe-stage"><span>04</span><strong>Weights</strong><small>Long and short</small></div>
    <div class="pipe-arrow" aria-hidden="true">→</div>
    <div class="pipe-stage"><span>05</span><strong>Evidence</strong><small>Net of costs</small></div>
  </div>

  <div class="visual-grid">
    <article class="visual-card">
      <div class="visual-header"><div><span class="eyebrow">Cross-section</span><h2>One date, five stocks</h2></div><span class="status-dot">illustration</span></div>
      <p>Raw momentum becomes a percentile rank. The model cares about order across stocks on the same date.</p>
      <div class="rank-chart" role="img" aria-label="Horizontal bars ranking five example stocks by signal">
        <div class="rank-row"><b>NVDA</b><span><i style="--value:92%"></i></span><em>0.92</em></div>
        <div class="rank-row"><b>MSFT</b><span><i style="--value:74%"></i></span><em>0.74</em></div>
        <div class="rank-row"><b>AAPL</b><span><i style="--value:55%"></i></span><em>0.55</em></div>
        <div class="rank-row negative"><b>KO</b><span><i style="--value:34%"></i></span><em>0.34</em></div>
        <div class="rank-row negative"><b>INTC</b><span><i style="--value:18%"></i></span><em>0.18</em></div>
      </div>
      <div class="legend"><span><i class="swatch positive"></i>long candidates</span><span><i class="swatch muted"></i>short candidates</span></div>
    </article>

    <article class="visual-card">
      <div class="visual-header"><div><span class="eyebrow">Validation</span><h2>Walk forward, never backward</h2></div><span class="status-dot">3 folds</span></div>
      <p>Each test block occurs after its training block. Later history never teaches an earlier prediction.</p>
      <div class="folds" role="img" aria-label="Three expanding walk-forward validation folds">
        <div class="fold"><b>Fold 1</b><span class="train" style="--w:42%">train</span><span class="gap">gap</span><span class="test" style="--w:16%">test</span></div>
        <div class="fold"><b>Fold 2</b><span class="train" style="--w:56%">train</span><span class="gap">gap</span><span class="test" style="--w:16%">test</span></div>
        <div class="fold"><b>Fold 3</b><span class="train" style="--w:70%">train</span><span class="gap">gap</span><span class="test" style="--w:16%">test</span></div>
      </div>
      <div class="legend"><span><i class="swatch train-key"></i>available history</span><span><i class="swatch test-key"></i>unseen future</span></div>
    </article>
  </div>

  <div class="visual-grid secondary-visuals">
    <article class="visual-card">
      <div class="visual-header"><div><span class="eyebrow">Factor engineering</span><h2>From measurement to comparison</h2></div><span class="status-dot">past-only</span></div>
      <p>Every transformation changes the question. The order must be explicit and reproducible.</p>
      <div class="transform-chain" role="img" aria-label="Raw factor transformed through winsorization, neutralization, ranking, and validation">
        <span><b>01</b>Raw value</span><i>→</i><span><b>02</b>Winsorize</span><i>→</i><span><b>03</b>Neutralize</span><i>→</i><span><b>04</b>Rank</span><i>→</i><span><b>05</b>Validate</span>
      </div>
    </article>
    <article class="visual-card">
      <div class="visual-header"><div><span class="eyebrow">Research discipline</span><h2>Five levels of evidence</h2></div><span class="status-dot">promotion gate</span></div>
      <p>A profitable curve is not the first level. Timing and accounting must work before predictive claims matter.</p>
      <div class="evidence-stack" role="img" aria-label="Evidence levels from mechanical correctness to independent replication">
        <span style="--level:100%"><b>05</b>Replicated<small>new period or universe</small></span>
        <span style="--level:86%"><b>04</b>Implementable<small>survives costs and constraints</small></span>
        <span style="--level:72%"><b>03</b>Stable<small>folds and nearby parameters</small></span>
        <span style="--level:58%"><b>02</b>Predictive<small>beats a defined baseline</small></span>
        <span style="--level:44%"><b>01</b>Mechanical<small>timing and formulas verified</small></span>
      </div>
    </article>
  </div>

  <article class="visual-card simulator">
    <div class="visual-header"><div><span class="eyebrow">Backtest laboratory</span><h2>Costs turn a story into a test</h2></div><output id="costOutput" for="costSlider">8 bps</output></div>
    <p>Move the transaction-cost assumption. Gross wealth stays unchanged; net wealth falls as turnover becomes more expensive.</p>
    <label class="slider-label" for="costSlider"><span>Cost per unit of turnover</span><input id="costSlider" type="range" min="0" max="30" value="8" step="1"><span class="slider-scale"><small>0 bps</small><small>30 bps</small></span></label>
    <div class="chart-shell">
      <svg id="equityChart" viewBox="0 0 920 300" role="img" aria-labelledby="equityTitle equityDesc">
        <title id="equityTitle">Illustrative gross and net cumulative wealth</title>
        <desc id="equityDesc">Two line series over sixty periods. The net series becomes lower as transaction cost is increased.</desc>
        <g class="grid-lines"></g>
        <path class="area-path" id="grossArea"></path>
        <path class="line-path gross" id="grossLine"></path>
        <path class="line-path net" id="netLine"></path>
        <line class="baseline" x1="44" x2="900" y1="250" y2="250"></line>
      </svg>
      <div class="chart-legend"><span><i class="line-key gross"></i>gross wealth</span><span><i class="line-key net"></i>net wealth</span></div>
    </div>
    <div class="metric-strip" aria-live="polite">
      <div><span>Gross finish</span><strong id="grossFinish">—</strong></div>
      <div><span>Net finish</span><strong id="netFinish">—</strong></div>
      <div><span>Cost drag</span><strong id="costDrag">—</strong></div>
      <div><span>Worst drawdown</span><strong id="drawdownValue">—</strong></div>
    </div>
  </article>
</section>
"""


CSS = r"""
:root {
  --ink: #e8eee9;
  --ink-soft: #aebdb4;
  --paper: #0c1210;
  --paper-2: #111a16;
  --paper-3: #18231e;
  --line: #2b3a32;
  --acid: #b7f34a;
  --mint: #54d7a3;
  --amber: #f2b84b;
  --coral: #ff8066;
  --blue: #70a8ff;
  --shadow: rgba(0, 0, 0, .28);
  --serif: Georgia, "Times New Roman", serif;
  --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --mono: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
}
[data-theme="light"] {
  --ink: #19221d;
  --ink-soft: #536159;
  --paper: #f5f3eb;
  --paper-2: #fffdf7;
  --paper-3: #e9eee6;
  --line: #c9d1c8;
  --acid: #568000;
  --mint: #087d57;
  --amber: #9a6400;
  --coral: #bd3d2d;
  --blue: #245fad;
  --shadow: rgba(23, 32, 27, .12);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 5rem; }
body { margin: 0; color: var(--ink); background: var(--paper); font: 16px/1.72 var(--sans); }
body::before { content: ""; position: fixed; inset: 0; pointer-events: none; opacity: .035; z-index: 20; background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.7'/%3E%3C/svg%3E"); }
a { color: var(--mint); text-underline-offset: .18em; }
a:hover { color: var(--acid); }
:focus-visible { outline: 3px solid var(--acid); outline-offset: 3px; }
.skip-link { position: fixed; left: 1rem; top: -5rem; z-index: 100; padding: .7rem 1rem; color: #111; background: var(--acid); }
.skip-link:focus { top: 1rem; }
.progress { position: fixed; inset: 0 0 auto; z-index: 60; height: 3px; background: transparent; }
.progress i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--mint), var(--acid)); }
.topbar { position: fixed; z-index: 50; inset: 3px 0 auto; height: 60px; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 1rem; padding: 0 1.2rem; border-bottom: 1px solid var(--line); background: color-mix(in srgb, var(--paper) 88%, transparent); backdrop-filter: blur(14px); }
.brand { display: flex; min-width: 0; align-items: baseline; gap: .7rem; overflow: hidden; color: var(--ink); text-decoration: none; font-weight: 800; letter-spacing: -.02em; white-space: nowrap; }
.brand mark { color: var(--acid); background: none; }
.brand small { color: var(--ink-soft); font: 600 .68rem/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; }
.top-actions { display: flex; flex: none; justify-self: end; gap: .55rem; }
.icon-button { min-width: 40px; min-height: 40px; padding: .55rem .8rem; border: 1px solid var(--line); border-radius: 99px; color: var(--ink); background: var(--paper-2); font: 700 .76rem var(--mono); cursor: pointer; }
.icon-button:hover { border-color: var(--mint); }
.sidebar { position: fixed; z-index: 40; inset: 63px auto 0 0; width: 280px; padding: 1.4rem 1rem 2rem; overflow-y: auto; border-right: 1px solid var(--line); background: var(--paper); }
.search-label { display: block; margin-bottom: 1.2rem; }
.search-label span { display: block; margin: 0 0 .45rem .2rem; color: var(--ink-soft); font: 700 .66rem var(--mono); letter-spacing: .1em; text-transform: uppercase; }
.search-label input { width: 100%; min-height: 42px; padding: .65rem .8rem; border: 1px solid var(--line); border-radius: 8px; color: var(--ink); background: var(--paper-2); font: inherit; }
.toc-label { margin: 1.2rem .3rem .55rem; color: var(--ink-soft); font: 700 .64rem var(--mono); letter-spacing: .12em; text-transform: uppercase; }
.toc { list-style: none; margin: 0; padding: 0; }
.toc li { margin: 2px 0; }
.toc a { display: grid; grid-template-columns: 1.7rem 1fr; gap: .5rem; padding: .55rem .65rem; border-radius: 7px; color: var(--ink-soft); text-decoration: none; font-size: .83rem; line-height: 1.25; }
.toc a span { color: var(--mint); font: 700 .66rem var(--mono); }
.toc a:hover, .toc a.active { color: var(--ink); background: var(--paper-3); }
.toc a.active { box-shadow: inset 2px 0 var(--acid); }
.sidebar-note { margin: 1.5rem .65rem 0; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--ink-soft); font-size: .72rem; line-height: 1.5; }
main { margin-left: 280px; overflow: clip; }
.hero { min-height: 92vh; display: grid; place-items: end start; position: relative; padding: clamp(7rem, 12vw, 10rem) clamp(1.4rem, 6vw, 7rem) clamp(4rem, 8vw, 7rem); border-bottom: 1px solid var(--line); overflow: hidden; }
.hero::before { content: ""; position: absolute; inset: 8% -10% auto auto; width: min(66vw, 800px); aspect-ratio: 1; border: 1px solid var(--line); border-radius: 50%; box-shadow: 0 0 0 7vw color-mix(in srgb, var(--mint) 5%, transparent), 0 0 0 15vw color-mix(in srgb, var(--acid) 3%, transparent); }
.hero-grid { position: absolute; inset: 0; opacity: .24; background-image: linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px); background-size: 56px 56px; mask-image: linear-gradient(to top, #000, transparent 72%); }
.hero-copy { position: relative; max-width: 950px; }
.kicker, .chapter-kicker { color: var(--acid); font: 800 .72rem var(--mono); letter-spacing: .16em; text-transform: uppercase; }
.hero h1 { max-width: 900px; margin: .8rem 0 1.2rem; font: 500 clamp(3.1rem, 8vw, 7.7rem)/.9 var(--serif); letter-spacing: -.065em; }
.hero h1 em { color: var(--acid); font-style: italic; }
.hero p { max-width: 720px; color: var(--ink-soft); font-size: clamp(1rem, 2vw, 1.3rem); }
.hero-meta { display: flex; flex-wrap: wrap; gap: .65rem; margin-top: 2rem; }
.hero-meta span { padding: .55rem .8rem; border: 1px solid var(--line); border-radius: 99px; background: color-mix(in srgb, var(--paper-2) 80%, transparent); font: 700 .7rem var(--mono); }
.chapter { max-width: 1120px; margin: 0 auto; padding: clamp(5rem, 9vw, 9rem) clamp(1.3rem, 6vw, 6rem); border-bottom: 1px solid var(--line); }
.chapter > h1 { margin: .35rem 0 2rem; font: 500 clamp(2.5rem, 5vw, 4.8rem)/1 var(--serif); letter-spacing: -.045em; }
.chapter > h1:first-child { margin-top: 0; }
.chapter-lead { max-width: 760px; color: var(--ink-soft); font-size: 1.15rem; }
.chapter h2 { margin: 4rem 0 1rem; padding-top: 1rem; border-top: 1px solid var(--line); font: 600 clamp(1.55rem, 3vw, 2.35rem)/1.15 var(--serif); letter-spacing: -.025em; }
.chapter h3 { margin: 2.5rem 0 .8rem; color: var(--acid); font-size: 1.07rem; line-height: 1.3; }
.chapter h4 { margin-top: 2rem; }
.heading-anchor { display: inline-block; width: 0; margin-left: -1em; padding-right: 1em; color: var(--line); text-decoration: none; opacity: 0; font: 400 .8em var(--mono); }
h1:hover .heading-anchor, h2:hover .heading-anchor, h3:hover .heading-anchor, .heading-anchor:focus { opacity: 1; }
.chapter p, .chapter > ul, .chapter > ol, .chapter blockquote, .chapter .formula { max-width: 790px; }
.chapter li { margin: .38rem 0; padding-left: .25rem; }
.chapter li::marker { color: var(--mint); font-family: var(--mono); }
.chapter code { padding: .12em .34em; border: 1px solid var(--line); border-radius: 4px; color: var(--acid); background: var(--paper-3); font: .86em var(--mono); }
.code-block { max-width: 900px; margin: 1.6rem 0; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: #09100d; box-shadow: 0 18px 50px var(--shadow); }
.code-block figcaption { padding: .55rem .8rem; border-bottom: 1px solid #29372f; color: #97a89d; font: 700 .67rem var(--mono); letter-spacing: .1em; text-transform: uppercase; }
.code-block pre { margin: 0; padding: 1.2rem; overflow-x: auto; }
.code-block code { padding: 0; border: 0; color: #dcebe1; background: none; font: .84rem/1.65 var(--mono); }
.native-diagram { max-width: 900px; margin: 2rem 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--paper-2); box-shadow: 0 18px 50px var(--shadow); }
.native-diagram figcaption { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: .7rem 1rem; border-bottom: 1px solid var(--line); color: var(--ink-soft); font: 700 .68rem var(--mono); }
.native-diagram figcaption span { color: var(--mint); letter-spacing: .1em; text-transform: uppercase; }
.flow-svg { display: block; width: 100%; height: auto; min-height: 360px; padding: 1.2rem; }
.flow-edges path { fill: none; stroke: var(--mint); stroke-width: 2; stroke-dasharray: 5 6; animation: flow-dash 9s linear infinite; }
.flow-edges marker path, .flow-svg marker path { fill: var(--acid); }
.flow-nodes rect { fill: var(--paper-3); stroke: var(--line); stroke-width: 1.5; }
.flow-nodes g:hover rect { stroke: var(--acid); }
.flow-nodes text { dominant-baseline: middle; font-family: var(--mono); }
.flow-nodes .node-key { fill: var(--mint); font-size: 11px; font-weight: 800; }
.flow-nodes .node-label { fill: var(--ink); font-size: 13px; font-weight: 700; }
.formula { margin: 1.5rem 0; padding: 1.2rem 1.5rem; border-left: 3px solid var(--acid); color: var(--ink); background: var(--paper-2); font: italic 1.05rem/1.6 var(--serif); overflow-x: auto; }
blockquote { margin: 2rem 0; padding: 1.25rem 1.5rem; border: 1px solid var(--line); border-left: 4px solid var(--acid); background: var(--paper-2); font: 500 1.15rem/1.6 var(--serif); }
.table-wrap { max-width: 100%; margin: 1.5rem 0 2rem; overflow-x: auto; border: 1px solid var(--line); border-radius: 9px; }
table { width: 100%; border-collapse: collapse; min-width: 600px; font-size: .86rem; }
th { color: var(--acid); background: var(--paper-3); text-align: left; font: 800 .7rem var(--mono); letter-spacing: .04em; text-transform: uppercase; }
th, td { padding: .8rem .9rem; border-bottom: 1px solid var(--line); vertical-align: top; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: color-mix(in srgb, var(--mint) 5%, transparent); }
.check-item { display: flex; align-items: flex-start; gap: .6rem; }
.check-item input { margin-top: .38rem; accent-color: var(--acid); }
.visual-lab { max-width: 1320px; }
.pipeline { display: grid; grid-template-columns: repeat(9, auto); align-items: center; gap: .65rem; margin: 3rem 0; padding: 1.4rem; border: 1px solid var(--line); border-radius: 12px; background: var(--paper-2); overflow-x: auto; }
.pipe-stage { min-width: 125px; }
.pipe-stage span { display: block; margin-bottom: .55rem; color: var(--mint); font: 700 .68rem var(--mono); }
.pipe-stage strong, .pipe-stage small { display: block; }
.pipe-stage small { color: var(--ink-soft); font-size: .7rem; }
.pipe-arrow { color: var(--acid); font-size: 1.6rem; animation: pulse 2.2s ease-in-out infinite; }
.visual-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
.secondary-visuals { margin-top: 1rem; }
.visual-card { padding: clamp(1.2rem, 3vw, 2rem); border: 1px solid var(--line); border-radius: 12px; background: var(--paper-2); box-shadow: 0 22px 60px var(--shadow); }
.visual-card h2 { margin: .2rem 0 0; padding: 0; border: 0; font-size: 1.7rem; }
.visual-card p { color: var(--ink-soft); font-size: .9rem; }
.visual-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
.eyebrow { color: var(--mint); font: 800 .64rem var(--mono); letter-spacing: .12em; text-transform: uppercase; }
.status-dot, #costOutput { flex: none; padding: .35rem .55rem; border: 1px solid var(--line); border-radius: 99px; color: var(--acid); font: 700 .65rem var(--mono); }
.rank-chart { display: grid; gap: .65rem; margin: 1.6rem 0; }
.rank-row { display: grid; grid-template-columns: 3.2rem 1fr 2.4rem; gap: .7rem; align-items: center; font: .73rem var(--mono); }
.rank-row span { height: 10px; border-radius: 99px; background: var(--paper-3); overflow: hidden; }
.rank-row i { display: block; width: var(--value); height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--mint), var(--acid)); transform-origin: left; animation: grow 1.2s ease-out both; }
.rank-row.negative i { background: linear-gradient(90deg, var(--coral), var(--amber)); }
.rank-row em { color: var(--ink-soft); font-style: normal; text-align: right; }
.legend, .chart-legend { display: flex; flex-wrap: wrap; gap: 1rem; color: var(--ink-soft); font: .67rem var(--mono); }
.legend span, .chart-legend span { display: inline-flex; align-items: center; gap: .4rem; }
.swatch { width: 9px; height: 9px; border-radius: 50%; }
.swatch.positive { background: var(--acid); }.swatch.muted { background: var(--coral); }.train-key { background: var(--blue); }.test-key { background: var(--acid); }
.folds { display: grid; gap: .8rem; margin: 1.7rem 0; }
.fold { display: flex; align-items: stretch; min-width: 0; height: 38px; border-radius: 5px; overflow: hidden; background: var(--paper-3); font: 650 .64rem var(--mono); }
.fold b { display: grid; place-items: center; width: 52px; color: var(--ink-soft); }
.fold span { display: grid; place-items: center; }
.fold .train { width: var(--w); color: #dce8ff; background: color-mix(in srgb, var(--blue) 55%, var(--paper-3)); }
.fold .gap { width: 8%; color: var(--ink-soft); background: repeating-linear-gradient(135deg, var(--paper-3), var(--paper-3) 5px, var(--line) 5px, var(--line) 7px); font-size: .55rem; }
.fold .test { width: var(--w); color: #14200c; background: var(--acid); }
.transform-chain { display: flex; align-items: stretch; gap: .45rem; margin-top: 1.5rem; overflow-x: auto; padding-bottom: .4rem; }
.transform-chain span { display: grid; min-width: 88px; place-items: center; padding: .75rem .5rem; border: 1px solid var(--line); border-radius: 8px; background: var(--paper-3); font: 700 .68rem var(--mono); text-align: center; }
.transform-chain span b { display: block; margin-bottom: .2rem; color: var(--mint); font-size: .58rem; }
.transform-chain i { display: grid; place-items: center; color: var(--acid); font-style: normal; animation: pulse 2.2s ease-in-out infinite; }
.evidence-stack { display: flex; flex-direction: column; align-items: center; gap: .35rem; margin-top: 1.4rem; }
.evidence-stack span { display: grid; grid-template-columns: 2rem 1fr auto; align-items: center; width: var(--level); min-width: 44%; gap: .55rem; padding: .55rem .7rem; border: 1px solid var(--line); border-radius: 6px; background: linear-gradient(90deg, color-mix(in srgb, var(--mint) 15%, var(--paper-3)), var(--paper-3)); font: 700 .7rem var(--mono); }
.evidence-stack b { color: var(--acid); }
.evidence-stack small { color: var(--ink-soft); font-size: .58rem; font-weight: 500; }
.simulator { margin-top: 1rem; }
.slider-label { display: grid; grid-template-columns: minmax(180px, .35fr) minmax(200px, 1fr); gap: 1rem; align-items: center; margin: 1.6rem 0; color: var(--ink-soft); font-size: .8rem; }
.slider-label input { width: 100%; accent-color: var(--acid); }
.slider-scale { grid-column: 2; display: flex; justify-content: space-between; margin-top: -1rem; font: .65rem var(--mono); }
.chart-shell { padding: 1rem; border: 1px solid var(--line); border-radius: 8px; background: var(--paper); }
#equityChart { display: block; width: 100%; height: auto; overflow: visible; }
.grid-lines line { stroke: var(--line); stroke-width: 1; stroke-dasharray: 3 7; }
.line-path { fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
.line-path.gross { stroke: var(--acid); }.line-path.net { stroke: var(--mint); }.area-path { fill: color-mix(in srgb, var(--acid) 10%, transparent); }.baseline { stroke: var(--line); }
.line-key { width: 18px; border-top: 2px solid; }.line-key.gross { border-color: var(--acid); }.line-key.net { border-color: var(--mint); }
.metric-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; margin-top: 1rem; border: 1px solid var(--line); background: var(--line); }
.metric-strip div { padding: .9rem; background: var(--paper-2); }
.metric-strip span, .metric-strip strong { display: block; }
.metric-strip span { color: var(--ink-soft); font: .62rem var(--mono); text-transform: uppercase; }
.metric-strip strong { margin-top: .3rem; color: var(--acid); font: 500 1.25rem var(--serif); }
.no-results { display: none; max-width: 680px; margin: 8rem auto; padding: 2rem; border: 1px solid var(--line); text-align: center; }
.chapter.search-hidden { display: none; }
footer { margin-left: 280px; padding: 3rem clamp(1.3rem, 6vw, 6rem); color: var(--ink-soft); font-size: .78rem; }
@keyframes grow { from { transform: scaleX(0); } }
@keyframes pulse { 50% { transform: translateX(4px); opacity: .55; } }
@keyframes flow-dash { to { stroke-dashoffset: -110; } }
@media (max-width: 920px) {
  .sidebar { transform: translateX(-100%); transition: transform .25s ease; box-shadow: 24px 0 60px var(--shadow); }
  .sidebar.open { transform: none; }
  main, footer { margin-left: 0; }
  .visual-grid { grid-template-columns: 1fr; }
  .brand small { display: none; }
  #menuButton { display: inline-block; }
}
@media (min-width: 921px) { #menuButton { display: none; } }
@media (max-width: 640px) {
  .topbar { padding: 0 .65rem; }
  .brand { font-size: .82rem; }
  .top-actions { gap: .25rem; }
  .icon-button { min-width: 36px; padding: .45rem .55rem; }
  #themeButton, #menuButton { font-size: 0; }
  #themeButton::after { content: "◐"; font-size: 1rem; }
  #menuButton::after { content: "☰"; font-size: 1rem; }
  .hero h1 { font-size: clamp(3rem, 16vw, 5rem); }
  .chapter { padding-inline: 1.15rem; }
  .pipeline { grid-template-columns: 1fr; }
  .pipe-arrow { transform: rotate(90deg); text-align: center; }
  .slider-label { grid-template-columns: 1fr; }
  .slider-scale { grid-column: 1; }
  .metric-strip { grid-template-columns: repeat(2, 1fr); }
  .evidence-stack span { width: 100% !important; grid-template-columns: 2rem 1fr; }
  .evidence-stack small { grid-column: 2; }
  .native-diagram { overflow-x: auto; }
  .flow-svg { width: 700px; max-width: none; min-height: 0; }
  .top-actions .search-button { display: none; }
}
@media print {
  .topbar, .sidebar, .progress, .skip-link { display: none !important; }
  main, footer { margin-left: 0; }
  body { background: #fff; color: #111; font-size: 11pt; }
  .hero { min-height: 0; padding: 3rem 1rem; }
  .chapter { break-before: page; max-width: none; padding: 2rem 1rem; }
  .visual-lab { break-before: auto; }
  a { color: #111; }
  .code-block { box-shadow: none; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
}
"""


JAVASCRIPT = r"""
(() => {
  const root = document.documentElement;
  const themeButton = document.querySelector('#themeButton');
  const storedTheme = localStorage.getItem('quant-handbook-theme');
  if (storedTheme) root.dataset.theme = storedTheme;

  const updateThemeLabel = () => {
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    themeButton.textContent = `${next} mode`;
    themeButton.setAttribute('aria-label', `Switch to ${next} mode`);
  };
  updateThemeLabel();
  themeButton.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
    localStorage.setItem('quant-handbook-theme', root.dataset.theme);
    updateThemeLabel();
  });

  const sidebar = document.querySelector('#sidebar');
  const menuButton = document.querySelector('#menuButton');
  menuButton.addEventListener('click', () => {
    const open = sidebar.classList.toggle('open');
    menuButton.setAttribute('aria-expanded', String(open));
  });
  sidebar.addEventListener('click', event => {
    if (event.target.closest('a') && window.innerWidth <= 920) {
      sidebar.classList.remove('open');
      menuButton.setAttribute('aria-expanded', 'false');
    }
  });

  const progress = document.querySelector('#readingProgress');
  const updateProgress = () => {
    const distance = document.documentElement.scrollHeight - window.innerHeight;
    const value = distance > 0 ? window.scrollY / distance * 100 : 0;
    progress.style.width = `${Math.min(100, Math.max(0, value))}%`;
  };
  document.addEventListener('scroll', updateProgress, { passive: true });
  updateProgress();

  const links = [...document.querySelectorAll('.toc a')];
  const sections = [...document.querySelectorAll('main > section[id]')];
  const activeObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      links.forEach(link => link.classList.toggle('active', link.hash === `#${entry.target.id}`));
    });
  }, { rootMargin: '-18% 0px -70% 0px' });
  sections.forEach(section => activeObserver.observe(section));

  const search = document.querySelector('#chapterSearch');
  const noResults = document.querySelector('#noResults');
  search.addEventListener('input', () => {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    document.querySelectorAll('.chapter').forEach(chapter => {
      const matches = !query || chapter.textContent.toLowerCase().includes(query);
      chapter.classList.toggle('search-hidden', !matches);
      if (matches) visible += 1;
    });
    noResults.style.display = visible ? 'none' : 'block';
  });
  document.querySelector('#searchButton').addEventListener('click', () => search.focus());
  document.addEventListener('keydown', event => {
    if (event.key === '/' && !/input|textarea/i.test(document.activeElement.tagName)) {
      event.preventDefault();
      search.focus();
    }
  });

  const grossReturns = Array.from({ length: 60 }, (_, index) =>
    0.0032 + Math.sin(index * 0.55) * 0.006 + Math.cos(index * 0.19) * 0.004 -
    (index > 31 && index < 39 ? 0.009 : 0)
  );
  const turnover = Array.from({ length: 60 }, (_, index) =>
    0.18 + Math.abs(Math.sin(index * 0.73)) * 0.48
  );
  const compound = returns => {
    let wealth = 1;
    return returns.map(value => (wealth *= 1 + value));
  };
  const grossWealth = compound(grossReturns);
  const slider = document.querySelector('#costSlider');
  const ns = 'http://www.w3.org/2000/svg';
  const grid = document.querySelector('.grid-lines');
  [55, 120, 185, 250].forEach(y => {
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', '44'); line.setAttribute('x2', '900');
    line.setAttribute('y1', String(y)); line.setAttribute('y2', String(y));
    grid.appendChild(line);
  });
  const pointString = values => {
    const all = [...grossWealth, ...values];
    const min = Math.min(...all) * .985;
    const max = Math.max(...all) * 1.015;
    return values.map((value, index) => {
      const x = 44 + index / (values.length - 1) * 856;
      const y = 250 - (value - min) / (max - min) * 205;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
  };
  const pathFromPoints = points => `M ${points.split(' ').join(' L ')}`;
  const drawdown = values => {
    let peak = values[0]; let worst = 0;
    values.forEach(value => { peak = Math.max(peak, value); worst = Math.min(worst, value / peak - 1); });
    return worst;
  };
  const updateChart = () => {
    const bps = Number(slider.value);
    const netReturns = grossReturns.map((value, index) => value - turnover[index] * bps / 10000);
    const netWealth = compound(netReturns);
    const grossPoints = pointString(grossWealth);
    const netPoints = pointString(netWealth);
    document.querySelector('#grossLine').setAttribute('d', pathFromPoints(grossPoints));
    document.querySelector('#netLine').setAttribute('d', pathFromPoints(netPoints));
    document.querySelector('#grossArea').setAttribute('d', `${pathFromPoints(grossPoints)} L 900,250 L 44,250 Z`);
    document.querySelector('#costOutput').textContent = `${bps} bps`;
    document.querySelector('#grossFinish').textContent = `${grossWealth.at(-1).toFixed(3)}×`;
    document.querySelector('#netFinish').textContent = `${netWealth.at(-1).toFixed(3)}×`;
    document.querySelector('#costDrag').textContent = `${((grossWealth.at(-1) - netWealth.at(-1)) * 100).toFixed(1)} pts`;
    document.querySelector('#drawdownValue').textContent = `${(drawdown(netWealth) * 100).toFixed(1)}%`;
  };
  slider.addEventListener('input', updateChart);
  updateChart();
})();
"""


def build_document() -> str:
  rendered_chapters: list[str] = []
  toc_items: list[str] = []
  for number, (chapter_id, filename, short_title) in enumerate(CHAPTERS):
    source = (HANDBOOK_DIR / filename).read_text(encoding="utf-8")
    body, title = render_markdown(source, chapter_id)
    label = "00" if chapter_id == "overview" else f"{number:02d}"
    if chapter_id == "glossary":
      label = "A–Z"
    rendered_chapters.append(
      f'<section class="chapter" id="{chapter_id}" data-title="{html.escape(title, quote=True)}">'
      f'<div class="chapter-kicker">Chapter {label}</div>{body}</section>'
    )
    toc_items.append(
      f'<li><a href="#{chapter_id}"><span>{label}</span>{html.escape(short_title)}</a></li>'
    )
    if chapter_id == "overview":
      toc_items.append('<li><a href="#visual-lab"><span>LAB</span>Interactive visuals</a></li>')
      rendered_chapters.append(VISUAL_LAB)

  return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A practical, visual handbook for cross-sectional quantitative machine learning, portfolio construction, and backtesting.">
  <title>Quantitative ML Foundations — Visual Handbook</title>
  <style>{CSS}</style>
</head>
<body>
  <a class="skip-link" href="#main">Skip to handbook</a>
  <div class="progress" aria-hidden="true"><i id="readingProgress"></i></div>
  <header class="topbar">
    <a class="brand" href="#top"><mark>RE</mark> Quantitative ML <small>field handbook</small></a>
    <div class="top-actions">
      <button class="icon-button search-button" id="searchButton" type="button">search /</button>
      <button class="icon-button" id="themeButton" type="button">light mode</button>
      <button class="icon-button" id="menuButton" type="button" aria-label="Open chapter navigation" aria-controls="sidebar" aria-expanded="false">chapters</button>
    </div>
  </header>
  <aside class="sidebar" id="sidebar" aria-label="Handbook navigation">
    <label class="search-label"><span>Search all chapters</span><input id="chapterSearch" type="search" placeholder="Try ‘leakage’ or ‘Ridge’"></label>
    <div class="toc-label">Contents</div>
    <nav aria-label="Chapters"><ol class="toc">{''.join(toc_items)}</ol></nav>
    <p class="sidebar-note">Single-file edition · Works offline · Press <strong>/</strong> to search · Use your browser to print or save as PDF.</p>
  </aside>
  <main id="main">
    <section class="hero" id="top">
      <div class="hero-grid" aria-hidden="true"></div>
      <div class="hero-copy">
        <div class="kicker">From a price table to defensible evidence</div>
        <h1>Quantitative ML,<br><em>without the shortcuts.</em></h1>
        <p>A complete learning path through data preparation, factor signals, machine-learning models, time-aware validation, portfolio construction, backtesting, metrics, visualization, and reproducible research.</p>
        <div class="hero-meta"><span>15 chapters + glossary</span><span>Practical Python</span><span>Interactive charts</span><span>Worked case study</span></div>
      </div>
    </section>
    <div class="no-results" id="noResults"><h2>No matching chapter</h2><p>Try a broader term such as “return”, “model”, “cost”, or “data”.</p></div>
    {''.join(rendered_chapters)}
  </main>
  <footer><strong>Quantitative ML Foundations.</strong> Educational material only; historical simulations are not investment advice or evidence of future performance.</footer>
  <script>{JAVASCRIPT}</script>
</body>
</html>
"""


def main() -> None:
  OUTPUT_PATH.write_text(build_document(), encoding="utf-8")
  print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
  main()
