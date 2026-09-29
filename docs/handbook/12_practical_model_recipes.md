# Practical Model Recipes

## Concrete example: beat the mean before tuning a forest

Assume a Ridge model achieves mean daily rank IC of `0.021`. That sounds small
but potentially useful. The correct first comparison is not a deep model; it is
a baseline built from the same dates and universe.

| Model | Mean IC | IC stability | Turnover | Interpretation |
|---|---:|---:|---:|---|
| Predict cross-sectional mean | 0.000 | stable | 0% | no ranking information |
| Raw momentum rank | 0.014 | mixed | 18% | simple economic baseline |
| Ridge | 0.021 | stable across 4/5 folds | 20% | incremental linear value |
| Boosted trees | 0.024 | unstable | 31% | small gain, higher complexity |

The boosted model is not automatically best. Its incremental predictive value
must survive fold variation, costs, and implementation complexity.

## 1. Freeze the dataset interface

Model training should receive four explicit objects:

```python
X_train: pd.DataFrame
y_train: pd.Series
X_test: pd.DataFrame
metadata_test: pd.DataFrame  # date and symbol, not model inputs
```

Keep `date` and `symbol` available for grouping and diagnostics, but do not pass
them to the estimator as arbitrary numbers or strings.

Check before fitting:

- Feature order is fixed.
- All values are finite after documented preprocessing.
- Target horizon and units are recorded.
- Each test date occurs after its training period.
- Sample weights, if any, align by index.

## 2. Baseline recipes

### Constant prediction

Predict the training mean. For date-ranked targets, this produces no useful
cross-sectional order and confirms metric behavior.

### One-factor score

Use the economically strongest single feature directly. This shows whether the
ML model improves on a transparent signal.

### Previous-period relationship

Estimate a coefficient or factor direction in the previous training window and
carry it forward. This exposes whether frequent refitting actually helps.

### Random ranking

Use only as a diagnostic with a fixed seed and many repetitions. One random
ranking can look surprisingly good by chance.

## 3. A complete Ridge pipeline

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge(alpha: float) -> Pipeline:
  return Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=alpha)),
  ])
```

This pipeline learns imputation values and scaling parameters during `fit`.
Because it is fitted separately inside each chronological fold, later data does
not determine earlier preprocessing.

Imputation is a modeling assumption. Adding missingness indicators lets the
model distinguish an imputed median from a genuinely median observation.

## 4. Tune inside the training era

For one outer test fold:

```text
outer training dates
├── inner fold 1: early train → later validation
├── inner fold 2: expanded train → later validation
└── inner fold 3: expanded train → later validation

choose alpha using inner folds
refit on all outer training dates
predict untouched outer test dates
```

Never select `alpha` using the outer test performance. That converts the test
set into another validation set.

A compact grid is often enough:

```python
alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
```

Prefer a broad stable region over one isolated optimum.

## 5. Coefficient interpretation

With standardized features, coefficient sign and relative magnitude are easier
to compare. Still, correlated features share explanatory work, so a small
coefficient does not prove irrelevance.

Track coefficients by fold:

| Feature | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Sign stable? |
|---|---:|---:|---:|---:|---|
| Momentum | 0.06 | 0.04 | 0.05 | 0.02 | yes |
| Volatility | −0.01 | 0.01 | −0.02 | −0.01 | no |
| Volume shock | 0.03 | 0.02 | 0.00 | 0.01 | mostly |

Coefficient instability is evidence about the model and data, not a cosmetic
problem to hide in an average.

## 6. Lasso and Elastic Net recipe

Use Lasso when sparse selection is part of the hypothesis. Use Elastic Net when
you want sparsity while reducing arbitrary competition among correlated
features.

```python
from sklearn.linear_model import ElasticNet

model = Pipeline([
  ("impute", SimpleImputer(strategy="median", add_indicator=True)),
  ("scale", StandardScaler()),
  ("model", ElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=20_000)),
])
```

Check convergence warnings. Increasing iterations can help a genuinely slow
fit, but it does not repair badly scaled inputs or an unreasonable penalty.

## 7. Random forest recipe

Forests capture thresholds and interactions without feature scaling:

```python
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(
  n_estimators=400,
  max_depth=5,
  min_samples_leaf=100,
  max_features=0.7,
  n_jobs=-1,
  random_state=42,
)
```

Large leaves and shallow depth are deliberate regularization for noisy returns.
Tune them chronologically. A deep forest can memorize date- and stock-specific
accidents.

Native impurity importance is biased toward features with more possible split
points and can split importance among correlated features. Compare it with
out-of-sample permutation importance.

## 8. Histogram gradient boosting recipe

Boosting adds shallow trees sequentially to reduce loss. It can model nonlinear
relationships efficiently:

```python
from sklearn.ensemble import HistGradientBoostingRegressor

boosted = HistGradientBoostingRegressor(
  learning_rate=0.04,
  max_iter=300,
  max_leaf_nodes=15,
  min_samples_leaf=100,
  l2_regularization=1.0,
  random_state=42,
)
```

Learning rate and number of iterations trade off against each other. Leaf count
controls interaction complexity. Minimum leaf size prevents the model from
building rules on tiny groups.

Use an explicit chronological validation set for early-stopping decisions when
the library's default split would not respect time.

## 9. Regression, classification, or ranking

### Regression

Predict future return or future cross-sectional rank. It preserves magnitude or
order information and is a natural first formulation.

### Classification

Predict an event such as top-quintile membership. This simplifies the target but
turns nearby observations on opposite sides of a threshold into different
classes.

### Learning to rank

Ranking losses directly compare items within a query group—in this case, within
a date. They align closely with selection but require careful group-aware
training and evaluation. Start with regression on ranks before adding a ranking
framework.

## 10. Sample weighting

Possible weights include:

- Equal weight per row.
- Equal total weight per date so dates with larger universes do not dominate.
- Recency weights to adapt to changing relationships.
- Liquidity weights to emphasize implementable securities.
- Volatility-based weights to limit extreme target influence.

Every weighting rule changes the estimand—the relationship the model is trying
to learn. Report it as part of the model definition.

## 11. Permutation importance

Permutation importance measures how much an out-of-sample metric worsens after
one feature is shuffled. For panel data, naive row shuffling can create
unrealistic values. Consider permuting within dates if the question is
cross-sectional ranking importance.

Correlated features can substitute for one another, making each appear less
important. Importance is predictive evidence under one fitted model, not causal
evidence.

## 12. Partial dependence and response curves

A response curve asks how model predictions change as a feature varies while
other data is treated according to a chosen procedure.

Check whether the curve:

- Is monotonic as expected.
- Depends on sparse extreme regions.
- Changes strongly between folds.
- Extrapolates beyond training support.
- Reflects correlated-feature combinations that rarely occur.

Plot the feature distribution underneath the curve.

## 13. Save every out-of-sample prediction

Store one row per prediction:

| date | symbol | fold | model | prediction | target | eligible |
|---|---|---:|---|---:|---:|---|

Predictions are more valuable than only storing summary metrics. They allow you
to rebuild IC, quantiles, portfolios, turnover, errors, and regime analysis
without retraining.

## 14. A model comparison protocol

1. Freeze universe, features, target, folds, and primary metric.
2. Evaluate a constant and one-factor baseline.
3. Fit Ridge across the same folds.
4. Add Elastic Net only if sparsity is useful.
5. Add one tree ensemble to test nonlinearity.
6. Compare per-fold predictions and portfolio outcomes.
7. Charge identical cost and execution assumptions.
8. Prefer the simplest model whose incremental evidence is stable.

## 15. Model card

Record:

- Intended use and prohibited use.
- Training and evaluation dates.
- Feature list and availability assumptions.
- Target definition.
- Hyperparameters and selection procedure.
- Primary and secondary metrics per fold.
- Known failure regimes.
- Data and execution limitations.
- Code and dataset versions.

## Common mistakes

- Comparing models trained on different samples.
- Fitting preprocessing before splitting by time.
- Tuning dozens of parameters on a small validation period.
- Treating feature importance as causality.
- Keeping only final metrics and discarding predictions.
- Choosing a complex model for a tiny, unstable improvement.

## Practice

1. Implement constant, single-factor, and Ridge baselines.
2. Tune five Ridge penalties with nested chronological folds.
3. Plot coefficient paths across penalties and folds.
4. Compare Ridge with one regularized tree ensemble.
5. Write a model card for the selected design.

## Primary documentation

- [scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [scikit-learn Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [scikit-learn random forest regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [scikit-learn histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
