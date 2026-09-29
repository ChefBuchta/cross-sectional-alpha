# Dashboards and Reproducibility

## Concrete example: a result no one can reproduce is an anecdote

A notebook shows a Sharpe ratio of 1.4. It does not record the symbols, retrieval
date, adjusted-price setting, feature windows, model hyperparameters, cost
assumption, or train/test split. The number cannot be checked or recreated.

Reproducibility means that the result is connected to its inputs and decisions.

## 1. Separate core calculations from presentation

Good separation:

```text
src/               tested data, features, models, backtest, metrics
notebooks/         exploration and explanation
dashboard/         controls and rendering
data/              versioned references or ignored generated artifacts
artifacts/         experiment outputs
```

The dashboard calls functions from `src/`; it should not contain the only
implementation of a metric.

## 2. Notebooks: useful but stateful

Notebooks are excellent for:

- Exploring distributions.
- Building one small example.
- Comparing charts.
- Writing a research narrative.

Risks:

- Cells can execute out of order.
- Hidden state survives code changes.
- Variables are overwritten silently.
- Important functions remain untested.

Restart the kernel and run all cells before sharing. Move stable transformations
into importable modules.

## 3. Streamlit's execution model

Streamlit reruns the script when widget state changes. Treat the file as a
declarative view:

```python
import streamlit as st

symbols = st.multiselect("Symbols", options=available_symbols)
start = st.date_input("Start date")

if st.button("Run analysis"):
  results = run_analysis(symbols=symbols, start=start)
  st.line_chart(results["wealth"])
```

Do not launch expensive downloads or model fitting unconditionally on every
rerun.

## 4. Caching

Use data caching for deterministic serializable results:

```python
@st.cache_data(ttl="1h")
def load_market_data(symbols: tuple[str, ...], start: str, end: str):
  return loader.load(symbols, start, end)
```

Cache keys depend on function code and arguments. Keep mutable global state out
of cached calculations. Set a time-to-live for changing external data.

Do not cache a result across configurations that affect its meaning.

## 5. Experiment configuration

Represent every research choice explicitly:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
  symbols: tuple[str, ...]
  start_date: str
  end_date: str
  target_horizon: int
  feature_windows: tuple[int, ...]
  model_name: str
  transaction_cost_bps: float
  rebalance_frequency: str
  random_seed: int
```

Serialize the configuration next to predictions and metrics.

## 6. Artifact layout

One experiment directory might contain:

```text
artifacts/2026-09-29T180000Z_ridge_5d/
├── config.json
├── data_manifest.json
├── fold_metrics.csv
├── daily_metrics.csv
├── predictions.parquet
├── positions.parquet
├── returns.parquet
└── summary.html
```

Use a stable experiment identifier rather than overwriting `results.csv`.

## 7. Data manifest

Record:

- Data provider and retrieval timestamp.
- Symbols and requested interval.
- First/last observation and row count.
- Hash or version of raw/processed files.
- Adjustment settings.
- Exclusions and validation warnings.
- Code revision, when available.

This distinguishes a code change from an upstream data revision.

## 8. Tests by layer

### Unit tests

Test one formula or validation rule:

- Return calculation.
- Cross-sectional z-score.
- Target shift.
- Portfolio normalization.
- Drawdown.
- Cost deduction.

### Integration tests

Run several components on a tiny fixed dataset and verify the combined output.

### Invariant/property tests

Check facts such as:

- Future data cannot alter earlier features.
- Increasing costs cannot increase net returns.
- Weights satisfy exposure constraints.
- Shuffled row input produces the same sorted result.

### Network tests

Keep live provider checks separate from deterministic tests. APIs change and
networks fail; such failures should not make formula tests unreliable.

## 9. Git for research

Commit:

- Source code.
- Tests.
- Configuration examples.
- Documentation.
- Small fixtures with clear licenses.
- Dependency lock files.

Usually do not commit:

- Large downloaded datasets.
- Secrets or API keys.
- Virtual environments.
- Generated caches.
- Large experiment artifacts unless intentionally versioned elsewhere.

Use commit messages that record the research change, for example:

```text
Add 20-day volatility factor with past-only rolling window
```

## 10. Dependency reproducibility with uv

```bash
uv sync
uv run python -m unittest discover -s tests -v
```

`pyproject.toml` declares direct requirements. `uv.lock` records the resolved
environment. Update dependencies deliberately and review the resulting lock diff.

## 11. Dashboard design

A research dashboard should expose assumptions near results:

- Universe and sample dates.
- Signal horizon and rebalance rule.
- Cost assumption.
- Model and validation scheme.
- Gross/net exposure.
- Data warnings.

Organize the dashboard by questions rather than by library objects:

1. Is the data usable?
2. Does the factor have a relationship with future returns?
3. Does the model improve on a baseline?
4. Does a portfolio survive costs?
5. Where and when does it fail?

## 12. Security and responsible data use

- Never commit credentials.
- Read provider licenses and terms.
- Validate all user-supplied symbols and paths.
- Avoid unsafe deserialization of untrusted artifacts.
- Do not present educational backtests as investment recommendations.
- Preserve provenance when redistributing data is restricted.

## Common mistakes

- Keeping the only working logic in notebooks.
- Letting dashboard state alter core calculations invisibly.
- Caching external data forever.
- Overwriting artifacts from previous experiments.
- Recording final metrics but not predictions and positions.
- Committing API keys or large raw downloads.
- Changing dependencies without updating the lock file.

## Practice

1. Define a frozen experiment configuration dataclass.
2. Save configuration and fold metrics under a unique experiment ID.
3. Convert one notebook calculation into a tested function.
4. Add a Streamlit cache with an explicit TTL.
5. Write a data manifest for a small downloaded sample.

## Primary documentation

- [Streamlit caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
- [Streamlit session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)
- [uv documentation](https://docs.astral.sh/uv/)
- [Python `unittest`](https://docs.python.org/3/library/unittest.html)
