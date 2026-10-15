# Relative Edge research app

From the project root:

```powershell
uv run streamlit run dashboard/app.py
```

Keep that terminal running and open the printed local URL (normally localhost:8501).
Do not launch this app with `python dashboard/app.py`.

## Four pages

1. **Stock picks** — choose a model and a daily, weekly, or monthly snapshot.
   A recommendation brief names the first stock to investigate, explains why
   the model favors it, and places its historical evidence beside it. Joint
   leaders are labeled as ties. Past returns are explained in a separate
   **Recent history — not a forecast** section, never as a giant forecast-like KPI.
   Other candidates, complete rankings, and period history remain available.
2. **Model leaderboard** — all retained model families plus the 20-day momentum
   baseline, ranked by mean daily validation Spearman IC. Compare shortlist edge,
   mean forward return, and model choices on the same date.
3. **Model metrics** — per-model correlation summaries, shortlist outcomes,
   baseline comparison, daily/monthly/distribution/rolling charts, fitted settings,
   and downloadable evaluated rows.
4. **Help & methodology** — examples, formulas, metric definitions, timing,
   limitations, and instructions for adding models. Contextual help is also
   expandable beside the results.

## Timing and sources

The app reads the local `data/processed/eda_data.csv`; it never downloads prices
or places trades. The source's last date is displayed in the sidebar.

**Prediction horizon** (1, 5, or 20 trading observations) is chosen before training.
Changing it requires another experiment. **Snapshot frequency** changes only the
view: Day uses every observed date, Week uses the last observed date in each
Monday–Sunday week, Month uses the last observed date in each calendar month.
The latest week/month may be incomplete. These snapshots do not change the
prediction horizon or average later scores into an earlier prediction.

**Post-selection forecasts** use the already fitted models and past-only features
from dates at or after the configured test boundary. No refitting or future target
labels are required, so the last local price date can be scored. Only dates with
finite inputs for the entire configured universe are retained.

**Validation replay** displays the stored, labeled validation scores. Optional
**test replay** and test metrics exist only for the recorded validation winner.
The leaderboard never uses test for model selection. Forecast history has no
realized-return labels attached; historical outcome assessment is kept on the
evaluation pages. The original 2023–2024 test period has already been inspected
and is not fresh holdout evidence.

## What a pick means

The top three score positions form a research signal. Cutoff ties stay together
and may produce more than three stocks. If every score ties, the model abstains.
Rank 1 is most attractive *relative to the other stocks*, not necessarily a stock
with a positive future return. Scores are not probabilities or predicted returns.
Not shortlisted does not mean sell or short.

The page explicitly recommends **further research**, not buying on the model
alone. It reports unfavorable or undefined historical evidence as plainly as
favorable results. A driver is described in words against the correct reference
(same-date peers, or a stock's own normal activity), and score effects are not
presented as expected percentage returns. Company labels for the original ten
stocks are static offline display aliases, not live security metadata.

The muted teal/sage theme uses local fonts and no external font requests.
Research sources, the design rationale, layout plan, and verification scope are
documented in [the recommendation-page design note](../docs/recommendation-page-design.md).

For Ridge, feature contributions exactly explain differences from the same-date
average model score. Other predict-capable models use **single-feature sensitivity**:
replace one input by the same-date universe mean and measure the score difference.
These hypothetical effects need not add up and can reflect unrealistic feature
combinations. Neither method establishes what causes future price changes.

## Metrics are not a backtest

IC compares predicted and realized stock rankings per date. Selection edge is the
top group's mean forward return minus the whole universe's mean forward return.
The top-minus-bottom spread compares the two score groups. Top/bottom groups can
overlap in a small universe. Constant scores yield no meaningful selection outcome.

These are research statistics over overlapping forward windows, not an executable
portfolio return series. Do not compound them into wealth, or calculate a trading
Sharpe ratio on IC. Entry timing, weights, rebalancing, costs, drawdown, and a full
backtest are not implemented. Statistical uncertainty is not quantified.

## Adding models and implementation

Register another search function in `src/factor_model/pipeline.py`, returning the
existing ModelSelection / ModelRun structure with a unique model name, fitted
predict-capable pipeline, and feature schema. Picks, leaderboard, metrics, and
generic sensitivity explanations discover it automatically. Friendly names and
descriptions are optional; generic fallbacks work without a new page.

Shared reporting calculations live in `src/factor_model/reporting.py`; page UI is
in `dashboard/ui.py`, with four small entry files in `dashboard/app_pages/`.
The top-level entry point handles data loading, experiment settings, and session
results. No new dependencies or separate API are required.

Results live in the browser session. Switching pages, dates, models, and charts
does not train again. Restarting the server loses the in-memory results.
Running `main.py` does not populate this app; saved experiment artifacts are not
implemented yet. A source-file change is flagged until another experiment runs.

The concise picks view leads with the company, its rank within the configured
universe, and one observed reason that helped its model score. The evidence panel
shows the average top-group advantage in percentage points (pp), plus the exact
number of validation dates when that group beat the all-stock average. Both use
the displayed forward horizon and refer to the model's historical group—not an
individual stock's expected return or success probability. Missing outcomes show
"Unavailable", not zero. Full reasons, drawbacks, recent observations, and metric
definitions remain in collapsed detail sections.
