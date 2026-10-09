# Reusable Ridge experiment

The research in `notebooks/05_modeling.ipynb` is implemented in
`src/factor_model/training.py`. The notebook remains a record of the original
experiment; its cells and saved results are unchanged.

## Entry points

- `build_ridge_pipeline(alpha)` creates a fresh, unfitted StandardScaler + Ridge.
- `select_ridge_model(train, valid)` fits candidates on training rows and selects
  the highest finite mean daily validation Spearman correlation. Exact ties
  prefer larger alpha. It never receives test rows and never refits the winner.
- `evaluate_ridge_model(selection, rows)` predicts on labeled rows and compares
  Ridge with the 20-day momentum baseline. It never calls `fit`.

`RidgeSelection` and `ModelEvaluation` in `models.py` are Pydantic containers
for fitted objects and result tables. They are not serialized model files.

## Usage after preparing chronological splits

```python
from factor_model.training import evaluate_ridge_model, select_ridge_model

selection = select_ridge_model(splits.train, splits.valid)
validation = evaluate_ridge_model(selection, splits.valid)

print(selection.best_alpha)
print(selection.comparison)
print(validation.summary)

# Evaluate the already selected model; do not choose settings using this result.
test = evaluate_ridge_model(selection, splits.test)
print(test.summary)
```

The default alpha candidates match the notebook: 0.1, 0.5, 1, 2, 5, and 10.
The feature order and target name are stored with the winner. Evaluation
preserves the supplied row order and returns a copy with `ridge_score` added.
`daily_ic` contains one baseline and Ridge correlation per date; `summary`
contains their aggregate metrics. The fitted coefficients remain accessible as
`selection.pipeline.named_steps["ridge"].coef_`.

For a different target horizon, pass the corresponding target column explicitly
to selection, such as `target_column="target_percentile_20d"`. Construct the
targets and chronological splits for that same horizon first.

## Recorded experiment

The original alpha=10 winner and test summaries are documented in section 10
of the notebook. The extracted functions reproduce those results. This is a
reproducibility check, not a fresh independent test or new model selection.

## Next stages

1. Connect data, features, targets, splits, selection, and evaluation in
   `pipeline.py`, with file access at the outer boundary.
2. Save experiment settings, data identity, candidate comparison, predictions,
   coefficients, and scores so a run can be inspected later.
3. Convert scores into portfolio weights in `portfolio.py`, with explicit
   group size, gross exposure, tie handling, and rebalance timing.
4. Simulate executable trades in `backtest.py`, accounting for next-session
   execution, turnover, costs, and short-side assumptions.
5. Add return, volatility, drawdown, and risk-adjusted portfolio metrics.
6. Present saved results in the dashboard rather than running training during
   every UI interaction.

Ranking correlations are not trading returns. Five-day forward returns overlap
across dates and must not be compounded as independent daily portfolio returns.
The already inspected test period cannot become fresh evidence when new models
or portfolio settings are designed after seeing its results.
