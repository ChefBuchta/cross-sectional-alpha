# Python and the Data Stack

## Concrete example: one question, four tools

Suppose you want to calculate each stock's 20-day volatility:

```python
returns = prices.sort_values(["symbol", "date"]).copy()
returns["return_1d"] = returns.groupby("symbol")["adjusted_close"].pct_change()
returns["volatility_20d"] = (
  returns.groupby("symbol")["return_1d"]
  .rolling(window=20, min_periods=20)
  .std()
  .reset_index(level=0, drop=True)
)
```

Four layers are working together:

1. **Python** expresses the procedure.
2. **pandas** represents labeled date-symbol data and performs grouping/windows.
3. **NumPy** supplies efficient numerical arrays under much of pandas and ML.
4. **The environment manager** records compatible package versions so the same
   code can run later.

This chapter explains the role of each piece rather than memorizing methods.

## 1. Python: orchestration and readable research

Python is dynamically typed at runtime, but type hints make data boundaries
clear:

```python
from collections.abc import Sequence
from datetime import date

import pandas as pd


def load_prices(
  symbols: Sequence[str],
  start: date,
  end: date,
) -> pd.DataFrame:
  ...
```

Useful language features for research code include:

- Functions for small, testable transformations.
- Dataclasses for immutable experiment configuration.
- Protocols for replaceable data/model interfaces.
- Exceptions for explicit failure instead of silent empty outputs.
- Context managers for files and resources.
- Type hints for public APIs and editor feedback.

Keep imports free of network requests and expensive computation. Importing a
module should define tools; it should not run the experiment.

## 2. Environments and dependency locking

An environment isolates package versions. Without isolation, upgrading pandas
for one project can break another.

With `uv`:

```bash
uv sync
uv run python
uv run python -m unittest discover -s tests -v
```

The important files are:

| File | Purpose |
|---|---|
| `pyproject.toml` | Declares supported Python and direct dependencies |
| `uv.lock` | Records a complete resolved dependency graph |
| `.python-version` | Suggests the intended Python runtime |

Commit the lock file for an application or research repository. It makes the
environment far easier to reproduce.

## 3. NumPy: arrays and vectorized numerical work

A Python loop processes values one at a time through the interpreter. A NumPy
operation applies compiled array routines to entire blocks:

```python
import numpy as np

returns = np.array([0.01, -0.02, 0.005, 0.013])
mean_return = returns.mean()
volatility = returns.std(ddof=1)
```

Important concepts:

- **Shape:** `(rows, columns)` describes array dimensions.
- **dtype:** numerical representation such as `float64` or `int64`.
- **Broadcasting:** operations combine compatible shapes without manual loops.
- **Boolean masks:** select values using logical conditions.
- **NaN:** floating-point missing-value marker; it propagates through many
  operations unless a `nan*` function is used deliberately.

Vectorization improves both speed and clarity when the formula naturally applies
to a whole series. Do not force a vectorized expression if it hides timing.

## 4. pandas: labeled tabular and time-series data

### Long format

Long format keeps one observation per row:

| date | symbol | adjusted_close | volume |
|---|---|---:|---:|
| 2025-01-02 | AAA | 100.0 | 500000 |
| 2025-01-02 | BBB | 42.0 | 180000 |
| 2025-01-03 | AAA | 101.0 | 520000 |

It works naturally with `groupby`, database-style joins, and feature tables.

### Wide format

Wide format uses one column per asset:

| date | AAA | BBB |
|---|---:|---:|
| 2025-01-02 | 100.0 | 42.0 |
| 2025-01-03 | 101.0 | 41.5 |

It can be convenient for covariance matrices and portfolio-return calculations.
Learn to move explicitly between the two with `pivot`, `stack`, and `melt`.

### Split–apply–combine

Most factor calculations follow:

1. Split rows by symbol.
2. Apply a lag or rolling operation inside each symbol.
3. Combine results back in original row order.

Cross-sectional transforms reverse the grouping direction:

```python
frame["momentum_rank"] = frame.groupby("date")["momentum_20d"].rank(pct=True)
```

Here each date is a miniature population of stocks.

### Index discipline

Indexes are labels, not guaranteed row numbers. Always know whether `date` and
`symbol` are columns or index levels. Sort before time-dependent operations:

```python
frame = frame.sort_values(["symbol", "date"], ignore_index=True)
```

After grouped rolling calculations, inspect the resulting index before assigning
the Series back to the frame. Misaligned indexes can create plausible-looking
but incorrect columns.

## 5. yfinance: accessible research data

`yfinance` is a convenient wrapper for downloading Yahoo Finance data:

```python
import yfinance as yf

raw = yf.download(
  ["AAPL", "MSFT"],
  start="2020-01-01",
  end="2025-01-01",
  interval="1d",
  auto_adjust=False,
  group_by="ticker",
  progress=False,
)
```

Important details:

- `start` is inclusive and `end` is exclusive.
- Multiple tickers commonly produce MultiIndex columns.
- `auto_adjust=True` changes the meaning of returned OHLC values.
- Daily and intraday data have different timezone and availability behavior.
- Network responses can be empty, partial, delayed, or revised.
- The library is appropriate for learning and prototypes, not a replacement for
  a contracted production market-data feed.

Always normalize provider-specific output at the ingestion boundary. The rest of
the system should consume one stable internal schema.

## 6. scikit-learn: estimators and safe composition

scikit-learn uses a consistent estimator interface:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Its `Pipeline` combines preprocessing and modeling so transformations are fitted
inside the training fold:

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
  ("scale", StandardScaler()),
  ("ridge", Ridge(alpha=1.0)),
])
```

The interface is simple; choosing valid samples is the difficult part. A normal
pipeline does not automatically make a random cross-validation split safe for
financial time series.

## 7. Seaborn and Streamlit

**Seaborn** is useful during research for distributions, correlations, grouped
comparisons, and residual diagnostics. It is built around tidy DataFrames.

**Streamlit** reruns the script after interactions. Cache deterministic data work
with `st.cache_data`, keep UI state in session state when necessary, and keep
core calculations outside the dashboard module so they remain testable.

## Common mistakes

- Installing packages globally and losing reproducibility.
- Using row order as time without sorting by symbol and date.
- Mixing wide and long shapes without documenting the conversion.
- Calling `groupby.apply` for work supported by faster built-in transforms.
- Treating `NaN` as automatically bad rather than asking why it exists.
- Performing downloads or training at module import time.
- Fitting a scaler before the train/test split.

## Practice

1. Create a six-row long-format DataFrame for two symbols and three dates.
2. Calculate per-symbol daily returns.
3. Pivot adjusted closes into wide format and back to long format.
4. Add a percentile rank of returns inside each date.
5. Write a test showing that a duplicated ticker request does not duplicate rows.

## Primary documentation

- [pandas user guide](https://pandas.pydata.org/docs/user_guide/)
- [pandas window operations](https://pandas.pydata.org/docs/user_guide/window.html)
- [pandas GroupBy reference](https://pandas.pydata.org/docs/reference/groupby.html)
- [yfinance download reference](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [yfinance Multi-Level Columns](https://ranaroussi.github.io/yfinance/advanced/multi_level_columns.html)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Streamlit data caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
