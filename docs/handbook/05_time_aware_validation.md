# Time-Aware Validation

## Concrete example: a random split lets 2025 teach 2022

Suppose rows from 2018–2025 are shuffled, and 80% become training data. The model
may train on December 2025 and be evaluated on February 2022. That is a valid
generic supervised-learning split only when observations are exchangeable. It is
not a realistic simulation of forecasting through time.

Chronological validation asks what would have happened if the model had been
trained using only the past at each decision point.

## 1. Why financial rows are dependent

Dependence appears in several directions:

- Consecutive rows for one stock share market state and rolling windows.
- Different stocks on one date share macroeconomic events.
- Forward targets overlap when the horizon exceeds one period.
- Model retraining windows overlap.
- Feature and universe definitions can persist through time.

As a result, conventional independent-and-identically-distributed intuition can
overstate the effective sample size.

## 2. One chronological holdout

The simplest honest split is:

```text
2018 ───────── 2022 | 2023 ───── 2024 | 2025
      training      |  validation     | test
```

- Train model parameters on training data.
- Choose features/hyperparameters using validation data.
- Evaluate the chosen design once on test data.

This is easy to explain but produces only one historical test regime.

## 3. Walk-forward evaluation

### Expanding window

```text
Fold 1: [train 2018–2020] [test 2021]
Fold 2: [train 2018–2021] [test 2022]
Fold 3: [train 2018–2022] [test 2023]
```

Training information grows. This matches a process that retains all history.

### Rolling window

```text
Fold 1: [train 2018–2020] [test 2021]
Fold 2:      [train 2019–2021] [test 2022]
Fold 3:           [train 2020–2022] [test 2023]
```

Training length stays fixed. This adapts to change but discards older evidence.

Neither is universally correct. Compare them based on the hypothesis about
market drift and the intended retraining process.

## 4. Date-level splitting for panel data

Split unique dates, not arbitrary rows. Every stock on one date should belong to
the same fold, otherwise the model can train on same-day peers while pretending
to test that date.

```python
dates = frame["date"].drop_duplicates().sort_values()
train_dates = dates[dates < "2023-01-01"]
test_dates = dates[(dates >= "2023-01-01") & (dates < "2024-01-01")]

train = frame[frame["date"].isin(train_dates)]
test = frame[frame["date"].isin(test_dates)]
```

## 5. Overlapping forward labels

A five-day target at Monday uses prices through the following Monday. A target at
Tuesday uses many of the same returns. If the training set ends on Friday and the
test set starts Monday, late training labels may include prices inside the test
period.

### Purging

Remove training observations whose label interval overlaps a test interval.

### Embargo

Leave an additional time gap around a test fold to reduce boundary dependence.

The required gap depends on target horizon and data construction. It is not a
magic constant.

## 6. `TimeSeriesSplit`

scikit-learn's `TimeSeriesSplit` produces ordered splits and supports a `gap`.
It operates on ordered samples, so you must ensure the input rows represent the
intended time units. With multiple stocks per date, splitting raw rows may divide
one date incorrectly. A custom date-level splitter is often clearer.

```python
from sklearn.model_selection import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=5, gap=5)
for train_index, test_index in splitter.split(date_level_samples):
  ...
```

## 7. Validation, test, and research memory

The terms describe how data is used:

- **Training:** fits coefficients or trees.
- **Validation:** chooses model design and hyperparameters.
- **Test:** estimates performance after choices are frozen.

Every time you use test feedback to revise the system, information from the test
enters the design. Maintain an experiment log with all trials so the number of
research decisions is visible.

## 8. Hyperparameter search

Do not tune `alpha`, feature windows, target horizons, portfolio cutoffs, and cost
assumptions all on the same short period. Each additional search dimension makes
it easier to select noise.

A disciplined sequence:

1. Fix a research question and primary metric.
2. Select a small justified hyperparameter grid.
3. Evaluate across multiple chronological validation folds.
4. Prefer stable regions over one sharp optimum.
5. Freeze the design.
6. Run the final test once.

## 9. Regime coverage

Averages can hide dependence on one environment. Break out results by:

- Calendar period.
- Market direction.
- Market volatility.
- Interest-rate or liquidity environment when available.
- Sector.
- Long and short side.

Regime labels created after inspecting results are exploratory. Treat them as new
hypotheses requiring later confirmation.

## 10. Multiple testing and backtest overfitting

If one hundred useless strategies are tried, the best historical result can look
impressive by chance. Standard holdout logic becomes less convincing when the
researcher repeatedly adapts to the same history.

Mitigations:

- Log all trials, not only winners.
- Limit degrees of freedom.
- Use economic motivation before testing.
- Require stability across folds and nearby parameters.
- Preserve a final holdout.
- Replicate on another reasonable universe or period.
- Consider multiple-testing-aware statistics for mature research.

## 11. What to report per fold

Record:

- Exact train, validation, and test dates.
- Number of dates, symbols, and rows.
- Feature coverage.
- Selected hyperparameters.
- Predictive metrics.
- Portfolio metrics and turnover.
- Model coefficients/importance.
- Failures and excluded data.

Aggregate only after inspecting individual folds.

## Common mistakes

- Random train/test split.
- Splitting rows instead of dates.
- Selecting preprocessing using all data.
- Forgetting label overlap.
- Tuning on the test set repeatedly.
- Choosing the best fold rather than summarizing all folds.
- Reporting many strategies but only one winner.

## Practice

1. Draw the information interval for a five-day feature and five-day target.
2. Create three expanding and three rolling folds on monthly data.
3. Explain which training labels overlap a chosen test start date.
4. Implement a date-level splitter for a two-stock panel.
5. Compare the same Ridge grid using random and chronological splits.

## Primary reading

- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Bailey et al., The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb)
