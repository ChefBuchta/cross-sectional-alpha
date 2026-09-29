# Machine-Learning Models

## Concrete example: the strongest model is not the most complicated one

Suppose a Ridge model achieves a mean out-of-sample rank correlation of 0.025,
while a boosted-tree model achieves 0.028. The tree appears better. But then you
find that:

- Its result comes mostly from one year.
- Its turnover is twice as high.
- Its feature importance changes radically between folds.
- The improvement disappears with slightly higher costs.

The correct conclusion is not “boosting wins.” Model selection includes
stability, cost, interpretability, and uncertainty—not one mean score.

## 1. Frame the supervised-learning problem

For every eligible `(date, symbol)` row:

- `X` contains features available at the decision time.
- `y` contains a future return or future-return rank.
- The model estimates `f(X) -> y`.

Cross-sectional equity data is usually **panel data**: many entities observed
repeatedly through time. Rows are dependent within dates and within symbols.
Ordinary random-sample assumptions are therefore imperfect.

## 2. Baselines before ML

Use at least three baselines:

1. **Random score:** confirms the metric implementation centers near no skill.
2. **Single factor:** shows whether ML improves on the strongest simple signal.
3. **Equal-weight factor score:** combines standardized features without fitting.

If a learned model cannot beat these out of sample, complexity has not earned its
place.

## 3. Ordinary Least Squares

Linear regression predicts:

$$
\hat y = \beta_0 + \sum_{j=1}^{p}\beta_jx_j
$$

OLS chooses coefficients minimizing residual sum of squares:

$$
\min_\beta \|y-X\beta\|_2^2
$$

Benefits:

- Fast and transparent.
- Coefficient signs are inspectable.
- A strong baseline for approximately additive signals.

Weaknesses:

- Correlated features can create unstable coefficients.
- Outliers strongly affect squared error.
- Relationships are linear unless features encode nonlinearity.
- Classical inference assumptions rarely fit financial panels perfectly.

## 4. Ridge regression

Ridge adds an L2 penalty:

$$
\min_\beta \|y-X\beta\|_2^2 + \alpha\sum_{j=1}^{p}\beta_j^2
$$

The penalty shrinks coefficients toward zero. It is especially useful when
momentum horizons and volatility features are correlated.

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ridge = Pipeline([
  ("scale", StandardScaler()),
  ("model", Ridge(alpha=10.0)),
])
```

Scaling matters because the penalty acts on coefficient magnitudes. A feature
measured in millions and a feature measured in decimals should not receive
different effective penalties merely because of units.

As `alpha` increases:

- Variance usually decreases.
- Bias usually increases.
- Coefficients shrink more strongly.

Choose `alpha` using chronological validation, never the final test period.

## 5. Lasso and Elastic Net

Lasso uses an L1 penalty:

$$
\min_\beta \|y-X\beta\|_2^2 + \alpha\sum_j|\beta_j|
$$

It can set coefficients exactly to zero. With strongly correlated features, it
may select one and discard another somewhat arbitrarily.

Elastic Net combines L1 and L2 penalties. It can produce sparse models while
handling correlated groups more smoothly than pure Lasso.

Use zero coefficients as a modeling result, not proof that a factor has no
economic relationship in every period.

## 6. Decision trees and random forests

A decision tree partitions feature space using rules such as:

```text
momentum_rank > 0.8?
├── yes: volatility_rank < 0.6?
│   ├── yes: high score
│   └── no: medium score
└── no: lower score
```

Trees capture nonlinearities and interactions without feature scaling. A single
deep tree has high variance. Random forests reduce variance by averaging trees
trained on resampled observations and feature subsets.

Watch for:

- Excessive depth fitting noise.
- Feature importance biased toward variables with many possible splits.
- Predictions that jump sharply near split boundaries.
- Random bootstrapping that ignores panel/time dependence.

Permutation importance on a chronological holdout is usually more informative
than built-in impurity importance, though correlated features still complicate
interpretation.

## 7. Gradient boosting

Boosting builds weak trees sequentially, with later trees correcting previous
errors. It is powerful for tabular nonlinear data, but it exposes many tuning
choices: depth, learning rate, number of trees, subsampling, regularization, and
early stopping.

That flexibility increases overfitting risk. Do not adopt boosting until the
validation process and experiment ledger are trustworthy.

## 8. Loss functions versus research objectives

Training loss and final objective need not match:

- Squared error cares about numerical target distance.
- Absolute error is less sensitive to large errors.
- Rank-based evaluation cares about ordering.
- Portfolio return cares about scores after thresholding, weights, and costs.

A model can improve mean squared error without improving rank IC. It can improve
IC without improving net portfolio return. Evaluate all relevant layers.

## 9. Bias–variance trade-off

Expected prediction error can be understood as:

```text
irreducible noise + squared bias + variance
```

- A model with too much bias misses real structure.
- A model with too much variance learns accidental sample details.
- Regularization trades a little fit for more stable future behavior.

In noisy finance problems, reducing variance is often more valuable than
extracting the final fraction of training fit.

## 10. Pipelines prevent preprocessing leakage

Incorrect:

```python
X_scaled = scaler.fit_transform(X_all)
# Split happens afterward: future distribution entered the scaler.
```

Correct pattern:

```python
model = Pipeline([
  ("scale", StandardScaler()),
  ("ridge", Ridge()),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The pipeline fits the scaler using the training subset. This solves
preprocessing leakage, but the caller must still provide a valid chronological
split.

## 11. Hyperparameters and nested decisions

Parameters are learned from training data. Hyperparameters, such as Ridge
`alpha` or tree depth, are selected using validation data.

If you inspect the test result, change the model, and test again, the test set
has become another validation set. Keep a final untouched period or report the
iterative nature of the research honestly.

## 12. Model interpretation

For linear models inspect:

- Coefficient sign and magnitude after consistent scaling.
- Coefficient stability across time folds.
- Correlation among features.
- Prediction distribution and concentration.

For nonlinear models inspect:

- Permutation importance on held-out periods.
- Partial dependence only where supported by data.
- Performance after removing one feature family.
- Stability across seeds and market regimes.

Interpretation is diagnostic, not causal proof.

## 13. Recommended model ladder

```text
random score
  ↓
single factor
  ↓
equal-weight factor blend
  ↓
OLS
  ↓
Ridge / Lasso / Elastic Net
  ↓
random forest
  ↓
gradient boosting
```

Advance only when the current level has a measured limitation the next level can
address.

## Common mistakes

- Comparing models on training performance.
- Scaling before splitting.
- Randomly shuffling dependent observations.
- Optimizing many hyperparameters on one short validation window.
- Reading coefficient sign causally.
- Declaring victory from one favorable period.
- Ignoring that a slightly better model may require much more turnover.

## Practice

1. Fit OLS and Ridge to two correlated features and compare coefficients.
2. Plot validation performance versus Ridge `alpha`.
3. Compare a single-factor baseline with equal-weight and learned combinations.
4. Track coefficient signs across chronological folds.
5. Explain why lower MSE may not produce a better long-short portfolio.

## Primary documentation

- [scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)
