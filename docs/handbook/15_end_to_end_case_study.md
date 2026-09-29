# End-to-End Research Case Study

## The concrete goal

Build a weekly strategy that ranks a small equity universe using only daily
price and volume data. The strategy buys the highest-ranked 20%, shorts the
lowest-ranked 20%, delays execution until the next session, and reports results
after transaction costs.

This chapter connects the individual pieces. The numbers and symbols are
educational; they are not an investment recommendation.

## 1. Write the research contract before coding

```text
Question:
  Do past-only momentum, volatility, and volume features improve the ranking
  of five-session forward returns over a simple momentum baseline?

Universe:
  Fixed educational list, disclosed as survivorship-biased.

Decision time:
  Friday close, or the last available session of the week.

Execution:
  Next session's close in the simple prototype.

Target:
  Five-session forward adjusted-close return, ranked within each date.

Validation:
  Expanding walk-forward folds with a five-session purge gap.

Portfolio:
  Equal-weight top and bottom quintiles; gross exposure 1.0, net exposure 0.

Primary evidence:
  Daily Spearman IC, quintile monotonicity, and net long-short return.
```

This contract makes later changes visible. If execution or target timing changes,
the experiment is a different claim.

## 2. Configuration

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchConfig:
  symbols: tuple[str, ...]
  start_date: str
  end_date: str
  momentum_windows: tuple[int, ...] = (5, 21, 63)
  volatility_window: int = 21
  volume_window: int = 21
  target_horizon: int = 5
  rebalance_frequency: str = "W-FRI"
  transaction_cost_bps: float = 10.0
  ridge_alpha: float = 10.0
  random_seed: int = 42
```

Serialize this object into every experiment directory.

## 3. Load and normalize data

The existing Yahoo Finance loader should return:

```text
date, symbol, open, high, low, close, adjusted_close, volume
```

Immediately check:

```python
required = {
  "date", "symbol", "open", "high", "low",
  "close", "adjusted_close", "volume",
}
missing = required - set(prices.columns)
if missing:
  raise ValueError(f"missing columns: {sorted(missing)}")

if prices[["date", "symbol"]].duplicated().any():
  raise ValueError("duplicate date-symbol observations")

prices = prices.sort_values(["symbol", "date"], ignore_index=True)
```

Save a raw or normalized snapshot with retrieval metadata. Provider history can
change after the experiment.

## 4. Inspect coverage before features

Create a coverage report:

```python
coverage = (
  prices.groupby("symbol")
  .agg(
    first_date=("date", "min"),
    last_date=("date", "max"),
    observations=("date", "size"),
    missing_price=("adjusted_close", lambda x: x.isna().sum()),
    missing_volume=("volume", lambda x: x.isna().sum()),
  )
)
```

Plot observations by date and symbol. Do not discover halfway through model
training that one security begins years later than the others.

## 5. Calculate returns

```python
frame = prices.copy()
grouped = frame.groupby("symbol", sort=False)
frame["return_1d"] = grouped["adjusted_close"].pct_change()
```

The first row for each symbol must be missing. Assert it in a fixture.

## 6. Build past-only features

```python
for window in config.momentum_windows:
  frame[f"momentum_{window}d"] = grouped["adjusted_close"].pct_change(window)

frame["volatility_21d"] = (
  grouped["return_1d"]
  .rolling(config.volatility_window, min_periods=config.volatility_window)
  .std()
  .reset_index(level=0, drop=True)
)

frame["dollar_volume"] = frame["adjusted_close"] * frame["volume"]
frame["average_dollar_volume_21d"] = (
  frame.groupby("symbol")["dollar_volume"]
  .rolling(config.volume_window, min_periods=config.volume_window)
  .mean()
  .reset_index(level=0, drop=True)
)

frame["volume_ratio_21d"] = (
  frame["dollar_volume"] / frame["average_dollar_volume_21d"]
)
```

All rolling windows end at the current row. Whether today's close and volume may
be used depends on the declared decision time.

## 7. Create the future target

```python
horizon = config.target_horizon
future_price = frame.groupby("symbol")["adjusted_close"].shift(-horizon)
frame["forward_return_5d"] = future_price / frame["adjusted_close"] - 1
frame["target_rank"] = (
  frame.groupby("date")["forward_return_5d"].rank(pct=True)
)
```

The last five rows per symbol should have missing targets. The target belongs in
training data only; it must never enter the live feature table.

## 8. Normalize features cross-sectionally

```python
raw_features = [
  "momentum_5d",
  "momentum_21d",
  "momentum_63d",
  "volatility_21d",
  "volume_ratio_21d",
]

for feature in raw_features:
  frame[f"{feature}_rank"] = (
    frame.groupby("date")[feature].rank(pct=True)
  )
```

Ranks put features on a common bounded scale and reduce sensitivity to extreme
magnitudes. Keep the raw columns for diagnostics.

## 9. Define eligibility

```python
model_features = [f"{name}_rank" for name in raw_features]
frame["eligible"] = (
  frame[model_features].notna().all(axis=1)
  & frame["target_rank"].notna()
  & (frame["adjusted_close"] > 5)
  & (frame["average_dollar_volume_21d"] > 1_000_000)
)
```

This simplified universe uses current sample securities and therefore has
survivorship bias. State that next to every result.

## 10. Select rebalance dates

Resampling each symbol independently can produce mismatched dates. Instead,
choose decision dates from a shared calendar, then use rows available on those
dates.

For a prototype:

```python
all_dates = pd.Series(sorted(frame["date"].dropna().unique()))
decision_dates = (
  pd.DataFrame({"date": all_dates})
  .set_index("date")
  .resample(config.rebalance_frequency)
  .last()
  .dropna()
  .index
)
research = frame[frame["date"].isin(decision_dates)].copy()
```

Verify holiday weeks and exchange calendars explicitly.

## 11. Construct expanding folds

Example:

```text
Fold 1: train 2016–2018 | purge | test 2019
Fold 2: train 2016–2019 | purge | test 2020
Fold 3: train 2016–2020 | purge | test 2021
Fold 4: train 2016–2021 | purge | test 2022
Fold 5: train 2016–2022 | purge | test 2023
```

Create splits from unique dates, then map them back to all symbols. Purge
training labels whose five-session outcome interval reaches into the test era.

## 12. Fit baselines and Ridge

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


model = Pipeline([
  ("scale", StandardScaler()),
  ("ridge", Ridge(alpha=config.ridge_alpha)),
])
```

For every fold:

1. Filter eligible training rows.
2. Fit preprocessing and model on training only.
3. Predict every eligible test row.
4. Store date, symbol, fold, prediction, and target.
5. Also store the raw momentum baseline.

Do not concatenate test rows and refit preprocessing afterward.

## 13. Evaluate predictive evidence

Daily Spearman IC:

```python
daily_ic = (
  predictions.groupby("date")
  .apply(lambda group: group["prediction"].corr(
    group["target_rank"], method="spearman"
  ))
)
```

Report:

- Mean, median, and standard deviation of daily IC.
- Positive-IC fraction.
- IC by fold and year.
- Ridge minus baseline IC.
- Prediction coverage and stocks per date.

One average is not enough.

## 14. Inspect quantile monotonicity

```python
predictions["prediction_quintile"] = (
  predictions.groupby("date")["prediction"]
  .transform(lambda values: pd.qcut(
    values.rank(method="first"),
    q=5,
    labels=False,
  ))
)
```

Calculate future return by quintile and fold. Look for an ordered relationship,
not just one lucky long-short spread.

## 15. Convert predictions into weights

On each rebalance date:

```python
def equal_weight_extremes(group: pd.DataFrame) -> pd.Series:
  ranks = group["prediction"].rank(pct=True)
  long_mask = ranks >= 0.8
  short_mask = ranks <= 0.2
  weights = pd.Series(0.0, index=group.index)
  if long_mask.any():
    weights.loc[long_mask] = 0.5 / long_mask.sum()
  if short_mask.any():
    weights.loc[short_mask] = -0.5 / short_mask.sum()
  return weights
```

Assert gross exposure near 1.0 and net exposure near zero when both sides exist.
Handle small universes explicitly.

## 16. Delay execution

If the signal uses the decision-date close, do not earn the return that already
ended at that close. Shift positions to the declared execution interval.

Store separately:

- Signal date.
- Target weight.
- Execution date.
- Executed weight.
- Return start and end timestamps.

A single DataFrame index should not be asked to carry all these meanings
implicitly.

## 17. Calculate gross return and turnover

```python
asset_contributions = executed_weights * realized_returns
gross_return = asset_contributions.groupby("date").sum()

trade = target_weights - drifted_previous_weights
turnover = trade.abs().groupby("date").sum()
```

Target-to-target turnover is acceptable for a labeled prototype, but drift-aware
turnover is the better accounting convention.

## 18. Deduct costs

```python
cost_rate = config.transaction_cost_bps / 10_000
transaction_cost = turnover * cost_rate
net_return = gross_return - transaction_cost
```

Run a sensitivity grid at 0, 5, 10, 20, and 30 bps. Calculate the break-even
cost at which cumulative net performance loses its advantage.

## 19. Build the result report

The report should contain:

1. Data coverage and known universe bias.
2. Feature definitions and availability timing.
3. Fold diagram and exact dates.
4. Baseline versus Ridge IC by fold.
5. Prediction distribution and quintile returns.
6. Gross and net equity curves.
7. Underwater drawdown chart.
8. Turnover and cost drag.
9. Gross, net, sector, and beta exposure.
10. Worst periods and failed folds.
11. Parameter and cost sensitivity.
12. Limitations and next experiment.

## 20. Save reproducible artifacts

```text
artifacts/<experiment_id>/
├── config.json
├── data_manifest.json
├── coverage.csv
├── fold_definitions.csv
├── predictions.parquet
├── target_weights.parquet
├── executed_weights.parquet
├── contributions.parquet
├── daily_returns.csv
├── fold_metrics.csv
└── report.html
```

Hash configuration and data manifests. Never overwrite an earlier experiment.

## 21. Minimum test suite

### Data

- Duplicate date-symbol rows fail.
- Empty provider output fails.
- Symbols are normalized deterministically.

### Features and targets

- Future price mutation cannot alter earlier features.
- The target has the correct sign and horizon.
- Last-horizon rows have missing targets.
- Grouped rolling output aligns with original rows.

### Validation

- Every test date follows every training date.
- No purged label overlaps a test interval.
- All stocks from one date remain in one split.

### Portfolio and backtest

- Gross and net exposure satisfy configuration.
- Return equals summed contributions.
- Higher nonnegative costs cannot raise net wealth.
- No position earns a return before execution.
- Missing live-position prices produce explicit behavior.

## 22. Interpret possible outcomes

### Ridge beats momentum predictively and economically

Inspect whether improvement is stable and identify which features add value.
Then replicate on a new period or point-in-time universe.

### Ridge improves IC but not net return

The model may create excessive turnover or concentrate improvement in names that
are costly to trade. Explore slower rebalancing or turnover-aware construction.

### Ridge and boosted trees are similar

Prefer Ridge unless nonlinear response curves reveal a stable economic pattern.

### Every model fails out of sample

That is useful evidence. Recheck timing and implementation, then reconsider the
hypothesis instead of tuning until the failure disappears.

## 23. What this prototype still cannot prove

- A fixed current universe is not point-in-time membership.
- Free daily bars do not model intraday execution.
- Short borrow availability and fees are absent.
- Corporate-action methodology depends on the provider.
- Market impact is not captured by constant basis-point costs.
- A historical relationship may decay after discovery.
- A research backtest is not a live trading track record.

## 24. Recommended implementation order

1. Keep the tested Yahoo loader.
2. Add deterministic data validators.
3. Implement returns and three transparent features.
4. Implement forward targets and timing tests.
5. Add date-level walk-forward splits.
6. Evaluate the momentum baseline.
7. Add Ridge in a fitted pipeline.
8. Store all out-of-sample predictions.
9. Build equal-weight extreme portfolios.
10. Add delayed execution, turnover, and costs.
11. Produce the evidence report.
12. Only then add advanced models or optimization.

## Final review questions

- Can every number be traced to a timestamp and source row?
- Could information from the future affect this decision?
- Is improvement measured against a meaningful baseline?
- Is the result stable across folds rather than one period?
- Do costs and constraints match the trading claim?
- Are failures visible in the report?
- Can another person reproduce the experiment from artifacts?

If any answer is unclear, the next task is not a more complicated model. It is
to make the evidence chain explicit.
