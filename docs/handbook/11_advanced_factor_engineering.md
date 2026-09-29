# Advanced Factor Engineering

## Concrete example: momentum may actually be a sector bet

Imagine that every semiconductor stock rises while every utility stock falls.
A raw momentum ranking buys semiconductors and shorts utilities. The strategy
looks like stock selection, but most of its return may come from one sector
spread.

Neutralizing momentum by sector asks a narrower question: which semiconductor
stocks are stronger than other semiconductor stocks, and which utilities are
stronger than other utilities?

Advanced factor engineering is the work of deciding exactly what comparison a
signal should represent.

## 1. Write a factor contract

Every factor should document:

| Property | Example |
|---|---|
| Economic idea | Medium-term continuation |
| Raw input | Adjusted close |
| Formula | 252-day return excluding latest 21 days |
| Availability | After decision-date close |
| Minimum history | 253 valid sessions |
| Cross-sectional transform | Winsorize, sector-neutralize, rank |
| Expected direction | Higher predicts higher future rank |
| Missing rule | Ineligible; never replace with zero |
| Unit test | Future prices cannot change earlier values |

A name such as `momentum` is not a definition. Encode the horizon, skip period,
adjustment, and transformation in configuration or metadata.

## 2. Build an availability matrix

Before filling missing features, inspect which values exist:

```python
coverage = (
  frame.groupby("date")[feature_names]
  .agg(lambda column: column.notna().mean())
)
```

Plot coverage through time and by symbol. A missing value can mean insufficient
lookback, a suspended security, a provider gap, or a fundamental report that
was not published yet. Each cause can imply a different eligibility decision.

Missingness itself may contain information, but using it as a feature requires
a point-in-time reason—not accidental provider behavior.

## 3. A defensible transformation order

A common date-level sequence is:

```text
raw feature
→ eligibility filter
→ suspicious-value handling
→ winsorization
→ sector/size neutralization
→ cross-sectional standardization or ranking
→ optional clipping
```

Changing the order changes the signal. For example, winsorizing before
neutralization limits the influence of extremes on the regression. Winsorizing
afterward limits extreme residuals. Neither is universally correct; document
and test the chosen meaning.

## 4. Winsorization and robust scaling

Winsorization clips values to chosen quantiles:

```python
def winsorize_cross_section(series: pd.Series) -> pd.Series:
  lower = series.quantile(0.01)
  upper = series.quantile(0.99)
  return series.clip(lower=lower, upper=upper)
```

The limits must be calculated inside the current date or training sample,
depending on the intended transformation. Never calculate them from future
dates.

A median and median absolute deviation provide a more robust scale:

$$
\operatorname{MAD}(x)=\operatorname{median}(|x_i-\operatorname{median}(x)|)
$$

Robust does not mean assumption-free. If most of a small cross-section shares
the same value, even robust scaling can become undefined.

## 5. Sector demeaning

The simplest sector adjustment subtracts the group mean:

```python
frame["momentum_sector_relative"] = (
  frame["momentum_12_1"]
  - frame.groupby(["date", "sector"])["momentum_12_1"].transform("mean")
)
```

Small sectors produce unstable comparisons. Require a minimum group size or use
a broader classification. Sector labels must also be point-in-time if companies
can change classifications.

## 6. Regression neutralization

Suppose a raw factor `f` is related to log market capitalization and sector
indicators collected in matrix `Z`:

$$
f = Z\gamma + \epsilon
$$

The residual `epsilon` is the part not linearly explained by those exposures.
Calculate the regression separately on each date using only eligible stocks.

Neutralization changes the hypothesis. It does not make a signal objectively
better; it removes specified sources of variation so the remaining claim is
more focused.

Checks should include:

- Residual correlation with neutralized exposures.
- Number of stocks relative to regression columns.
- Stability under alternative sector granularity.
- Behavior when a category has one member.
- Whether extreme observations dominate the fit.

## 7. Orthogonalizing related factors

Momentum horizons often overlap. To isolate the incremental component of factor
`b` beyond factor `a`, regress `b` on `a` and use residuals.

The result depends on order: residualizing value against quality is not the same
as residualizing quality against value. Symmetric alternatives such as PCA lose
some economic interpretation.

Before orthogonalizing, ask whether the model's own regularization can handle
correlation. Orthogonalization is most useful when the research question
requires a specific incremental interpretation.

## 8. Exponential decay

An exponentially weighted signal gives recent observations more weight:

$$
s_t = \alpha x_t + (1-\alpha)s_{t-1}
$$

The half-life is easier to interpret than `alpha`: it describes how many
periods it takes for an observation's weight to halve.

```python
smoothed = (
  frame.groupby("symbol")["return_1d"]
  .ewm(halflife=20, adjust=False)
  .mean()
  .reset_index(level=0, drop=True)
)
```

Verify grouped alignment carefully. The first values depend on initialization
and minimum-history conventions.

## 9. Multi-horizon features

Instead of one momentum window, calculate several:

- 5-day short-term movement.
- 21-day monthly movement.
- 63-day quarterly movement.
- 252-day annual movement with a recent-month skip.

Closely related horizons increase collinearity. A regularized model can combine
them, but research should examine coefficient stability and incremental value.
Do not select the best horizon on the final test period.

## 10. Composite factors

A transparent composite can average standardized components:

$$
c_{i,t}=\frac{1}{K}\sum_{k=1}^{K}z_{i,t}^{(k)}
$$

Alternatives include:

- Equal-weighted ranks.
- Weights fixed by economic reasoning.
- Weights estimated only from training folds.
- Inverse-volatility weighting of factor-return series.
- A supervised model that learns nonlinear interactions.

Equal weights are a strong baseline because they have low estimation error and
are easy to explain.

## 11. Signal decay and holding horizon

Measure how predictive association changes with future horizon:

| Horizon | Question |
|---|---|
| 1 day | Is the signal immediately reflected? |
| 5 days | Does it persist through a weekly holding period? |
| 21 days | Is the effect slower but more economical to trade? |
| 63 days | Has the signal decayed or reversed? |

Overlapping horizons create dependent measurements. Treat the curve as a
diagnostic, not dozens of independent discoveries.

The trading decision should align feature decay, target horizon, rebalance
frequency, and cost assumptions.

## 12. Monotonicity and portfolio sorts

Sort stocks into quantiles by one factor and calculate later returns for each
group. A clean ranking signal often shows an ordered pattern from low to high,
not only an extreme top-minus-bottom spread.

Inspect:

- Mean and median return by quantile.
- Confidence intervals by fold.
- Number of stocks per bin.
- Turnover within each bin.
- Long and short sides separately.
- Results before and after neutralization.

A single profitable extreme bin with disorder elsewhere may indicate an outlier
or nonlinear threshold rather than a broad ranking relationship.

## 13. Factor interaction hypotheses

An interaction means the effect of one feature depends on another. Examples:

- Momentum may behave differently under high volatility.
- Reversal may be stronger after unusually high volume.
- Value may work differently within profitable and unprofitable firms.

A linear interaction term is a product:

$$
x_{interaction}=x_{momentum}x_{volatility}
$$

Standardize components before multiplying when scale matters. Introduce
interactions sparingly; the number of possible products grows quickly and makes
data snooping easy.

## 14. Factor library design

A useful factor function has an explicit contract:

```python
def momentum(
  prices: pd.DataFrame,
  lookback: int,
  skip: int = 0,
) -> pd.Series:
  """Return past-only momentum aligned to the input row index."""
  ...
```

Store metadata next to the output:

- Name and version.
- Required input columns.
- Lookback and availability lag.
- Expected direction.
- Transformation sequence.
- First valid date.
- Test coverage.

Version a factor when its meaning changes. A column with the same name and a new
formula makes old experiments impossible to interpret.

## 15. Required factor tests

- Shuffling input rows does not change the sorted result.
- Changing a future price does not change an earlier factor.
- First `lookback` observations are missing when required.
- Each date-level rank lies in the documented range.
- Neutralized residuals have near-zero intended exposure within tolerance.
- A constant cross-section returns a documented missing or neutral result.
- A split-adjusted fixture does not create a fake momentum crash.
- Input data is not mutated unexpectedly.

## Common mistakes

- Calling a factor neutral because it was ranked.
- Filling unavailable features with a cross-sectional zero without a flag.
- Using future sector membership or company size.
- Trying many horizons and reporting only the winner.
- Combining highly similar factors and interpreting each coefficient causally.
- Measuring decay with overlapping samples as if observations were independent.

## Practice

1. Write a formal contract for 12-minus-1 momentum.
2. Compare raw, winsorized, ranked, and sector-neutralized signals.
3. Plot IC by future horizon from 1 to 63 sessions.
4. Construct an equal-weight composite and compare it with Ridge predictions.
5. Write a future-mutation test for every factor function.

## Further reading

- [pandas window operations](https://pandas.pydata.org/docs/user_guide/window.html)
- [pandas exponentially weighted windows](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
- [scikit-learn robust scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
