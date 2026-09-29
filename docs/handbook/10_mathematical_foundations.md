# Mathematical Foundations for Quantitative ML

## Concrete example: one score is a dot product

Suppose a stock has three standardized features:

| Feature | Value | Model coefficient | Contribution |
|---|---:|---:|---:|
| Momentum | 1.20 | 0.08 | 0.096 |
| Volatility | −0.50 | −0.04 | 0.020 |
| Volume shock | 0.30 | 0.02 | 0.006 |

The linear model score is the sum of the contributions:

$$
\hat y = 1.20(0.08) + (-0.50)(-0.04) + 0.30(0.02) = 0.122
$$

In vector notation, the three feature values are `x`, the three coefficients are
`beta`, and the same calculation is a dot product:

$$
\hat y = x^T\beta
$$

Most of the mathematics in a tabular ML pipeline is an extension of this small
calculation across many stocks and dates.

## 1. Scalars, vectors, matrices, and tensors

A **scalar** is one number, such as a daily return. A **vector** is an ordered
list, such as all factor values for one stock. A **matrix** is a rectangular
table, such as all stocks by all features.

```text
X.shape = (observations, features)
y.shape = (observations,)
beta.shape = (features,)
```

For 100,000 date-stock observations and 12 features, `X` has shape
`(100000, 12)`. The model maps each row to one prediction.

The word **tensor** generalizes arrays to more dimensions. A panel indexed by
date, stock, and feature can be viewed as a three-dimensional tensor, although
this project normally stores it as a long two-dimensional DataFrame.

## 2. Dot products and weighted combinations

For vectors `a` and `b` of length `p`:

$$
a^Tb = \sum_{j=1}^{p}a_jb_j
$$

Dot products appear in:

- Linear predictions.
- Portfolio returns, `w^T r`.
- Covariance and correlation calculations.
- Projection and neutralization.
- Similarity calculations.

The order of elements matters. A coefficient for momentum multiplied by a
volatility column is not a noisy approximation; it is a wrong calculation.
Named columns and explicit feature lists protect against this error.

```python
feature_names = ["momentum_20d", "volatility_20d", "volume_z"]
X = frame.loc[:, feature_names]
predictions = model.predict(X)
```

## 3. Matrix multiplication

If `X` contains `n` observations and `p` features, and `beta` contains `p`
coefficients, then:

$$
\hat y = X\beta
$$

The output has length `n`. Matrix multiplication performs every row-level dot
product at once. NumPy expresses it with `@`:

```python
predictions = X_array @ coefficients
```

Matrix shapes are a debugging tool. Before multiplying, write the shapes on
paper and verify that the inner dimensions agree.

## 4. Centering and scaling as geometry

Centering subtracts a mean. Standardization divides by a standard deviation:

$$
z_{i} = \frac{x_i - \bar{x}}{s_x}
$$

After standardization, one unit means roughly one sample standard deviation.
This lets a linear penalty compare coefficients whose original features had
different units.

For cross-sectional research, the mean and standard deviation are usually
calculated separately inside each date. For an ML preprocessing pipeline, the
scaler is fitted only on training observations and then applied to later data.
These operations answer different questions and may both appear in one system.

## 5. Distance and norms

The L1 norm sums absolute values:

$$
\|\beta\|_1 = \sum_j |\beta_j|
$$

The L2 norm is the square root of the sum of squares:

$$
\|\beta\|_2 = \sqrt{\sum_j \beta_j^2}
$$

Lasso penalizes the L1 norm and can drive some coefficients exactly to zero.
Ridge penalizes the squared L2 norm and usually shrinks correlated coefficients
together. Portfolio gross exposure is also an L1 norm: `sum(abs(weights))`.

## 6. Derivatives and gradients

A derivative describes how an output changes after a small input change. A
gradient collects one derivative per parameter.

For mean-squared error:

$$
L(\beta) = \frac{1}{n}\|y-X\beta\|_2^2
$$

the gradient points toward the direction of fastest increase in loss. Gradient
descent moves in the opposite direction:

$$
\beta_{k+1} = \beta_k - \eta \nabla L(\beta_k)
$$

`eta` is the learning rate. Too small is slow; too large can overshoot or
diverge. Tree models are not fitted by ordinary gradient descent, but gradient
boosting repeatedly fits learners to improve a differentiable loss.

## 7. Probability as a language for uncertainty

A random variable represents an outcome whose value is not known in advance.
The distribution describes possible values and their probabilities.

Important quantities include:

- **Expected value:** probability-weighted average outcome.
- **Variance:** expected squared distance from the mean.
- **Quantile:** value below which a stated proportion falls.
- **Conditional expectation:** expected outcome given known information.
- **Tail probability:** probability of an extreme event.

A model prediction for squared-error regression estimates a conditional mean in
an idealized setting:

$$
\hat y(x) \approx E[Y \mid X=x]
$$

It is not a guarantee and does not describe the full distribution of future
returns.

## 8. Sampling and estimation

The historical sample is one realization from a changing process. An
**estimator** is a rule that turns the sample into a quantity such as a mean,
coefficient, or covariance matrix.

Three useful questions for every estimator are:

1. Is it biased under the assumptions?
2. How variable is it across possible samples?
3. Are the assumptions plausible for this data?

More data does not automatically solve a bad sampling design. Ten thousand
overlapping date-stock rows may contain much less independent information than
their row count suggests.

## 9. Covariance matrices

For `p` assets, a covariance matrix has shape `(p, p)`:

$$
\Sigma_{ij} = \operatorname{Cov}(r_i, r_j)
$$

Its diagonal contains asset variances. Off-diagonal values describe joint
movement. Portfolio variance is:

$$
\sigma_p^2 = w^T\Sigma w
$$

This formula explains diversification. Two volatile assets can form a less
volatile portfolio if their returns do not move together perfectly.

Sample covariance becomes unstable when the number of assets is large relative
to the amount of history. Shrinkage estimators deliberately move noisy sample
estimates toward a structured target.

## 10. Eigenvectors and principal components

An eigenvector of a matrix is a direction that the matrix stretches without
rotating. For a covariance matrix, leading eigenvectors describe directions of
large common variation.

Principal component analysis (PCA) rotates correlated features into orthogonal
components ordered by explained sample variance. It can help diagnose common
structure, but the components may be unstable and difficult to interpret.

PCA must be fitted inside each training fold. Fitting it on all dates leaks the
future covariance structure into earlier predictions.

## 11. Autocorrelation and dependence

Autocorrelation measures association between a series and its lagged values:

$$
\rho_k = \operatorname{Corr}(r_t, r_{t-k})
$$

Even when raw returns have weak autocorrelation, squared or absolute returns can
remain correlated because volatility clusters. Factor values, portfolio
positions, and overlapping targets are often strongly dependent.

Dependence affects standard errors, validation design, bootstrap methods, and
the effective amount of information in a sample.

## 12. Stationarity and regime change

A process is **stationary** under a chosen definition when properties such as
its distribution or moments do not change through time. Financial markets often
violate this approximation:

- Volatility changes.
- Correlations rise during stress.
- Trading costs and market structure evolve.
- Factor popularity changes behavior.
- The eligible universe changes.

Models do not require perfect stationarity, but research claims should state
which relationships are assumed to remain useful and for how long.

## 13. Numerical stability

Mathematically equivalent expressions can behave differently in floating-point
arithmetic. Common problems include:

- Dividing by a near-zero cross-sectional standard deviation.
- Inverting a poorly conditioned covariance matrix.
- Compounding very long return series through repeated multiplication.
- Treating `NaN`, positive infinity, and negative infinity as ordinary values.
- Comparing floats for exact equality.

Prefer stable library solvers over explicitly computing matrix inverses. Add
clear tolerances and fail when inputs are not finite.

```python
if not np.isfinite(features.to_numpy()).all():
  raise ValueError("features contain non-finite values")
```

## Common mistakes

- Memorizing formulas without checking array shapes.
- Interpreting a conditional mean prediction as certainty.
- Assuming more rows means proportionally more independent evidence.
- Using full-sample means or covariance matrices in historical folds.
- Inverting noisy covariance matrices without regularization.
- Treating PCA components as stable economic factors automatically.

## Practice

1. Calculate a three-feature dot product by hand and with NumPy.
2. Show how standardization changes the Ridge penalty for features in dollars
   and percentages.
3. Calculate the variance of a two-asset portfolio for three correlations.
4. Simulate an autocorrelated sequence and compare its apparent row count with
   its effective information.
5. Fit PCA on one training period and inspect whether its loadings remain stable
   in the following period.

## Further reading

- [NumPy linear algebra reference](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [scikit-learn preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [scikit-learn covariance estimation](https://scikit-learn.org/stable/modules/covariance.html)
- [scikit-learn decomposition and PCA](https://scikit-learn.org/stable/modules/decomposition.html)
