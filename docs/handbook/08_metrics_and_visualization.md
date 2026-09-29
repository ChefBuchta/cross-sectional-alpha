# Metrics and Visual Evidence

## Concrete example: 18% return tells you almost nothing

Two strategies both return 18%:

```text
Strategy A: rises gradually, maximum drawdown −6%, moderate turnover.
Strategy B: loses 35%, recovers after one lucky month, very high turnover.
```

Total return treats them as equal. A research report needs several views because
each metric compresses away different information.

## 1. Predictive metrics

### Rank Information Coefficient

For each date:

$$
IC_t = \operatorname{corr}_{Spearman}(score_{i,t}, future\ return_{i,t})
$$

Report:

- Mean and median IC.
- Standard deviation of IC.
- Percentage of positive IC dates.
- Rolling mean IC.
- IC by year, sector, and regime.

Do not use one universal “good IC” threshold. Horizon, universe, costs, breadth,
and dependence all matter.

### Quintile/decile returns

Split predictions into ordered groups and calculate realized average return.

Idealized monotonic pattern:

```text
Q1 < Q2 < Q3 < Q4 < Q5
```

A profitable Q5–Q1 spread with disorder in the middle may rely on tail behavior
rather than broad ranking skill.

### Prediction error

MSE and MAE are useful for numerical targets, but a low error does not guarantee
good rankings or portfolio results. Always connect model metric to decision use.

## 2. Return aggregation

Cumulative wealth from simple returns:

$$
W_T = W_0\prod_{t=1}^{T}(1+r_t)
$$

```python
wealth = (1 + daily_returns).cumprod()
```

Arithmetic mean multiplied by periods is an estimate of annualized expected
return, not the same as compounded realized growth.

## 3. Annualized volatility

For daily returns under a conventional 252-session assumption:

$$
\sigma_{ann} = \sigma_{daily}\sqrt{252}
$$

This square-root rule assumes a stable variance process and limited serial
dependence. State the convention; do not treat 252 as a physical constant.

## 4. Sharpe ratio

For excess return `R - R_f`:

$$
SR = \frac{E[R-R_f]}{\sigma[R-R_f]}
$$

A common sample annualization:

```python
sharpe = daily_excess.mean() / daily_excess.std(ddof=1) * (252 ** 0.5)
```

Limitations:

- Sensitive to sample period.
- Sampling uncertainty can be large.
- Serial correlation distorts naive annualization.
- Non-normal returns and option-like payoffs are poorly summarized.
- A selected best-of-many Sharpe is upward biased.

Report the return series and drawdown, not only Sharpe.

## 5. Drawdown

Running peak:

$$
Peak_t = \max_{s\le t} W_s
$$

Drawdown:

$$
DD_t = \frac{W_t}{Peak_t} - 1
$$

```python
peak = wealth.cummax()
drawdown = wealth / peak - 1
maximum_drawdown = drawdown.min()
```

Also report drawdown duration. Two equal maximum drawdowns can be psychologically
and operationally different if one recovers in weeks and another takes years.

## 6. Turnover and cost attribution

Plot:

- Turnover through time.
- Gross versus net return.
- Cumulative transaction costs.
- Performance across several cost assumptions.

If all apparent edge disappears at modest costs, that is a primary conclusion.

## 7. Long and short attribution

Separate:

```text
long contribution
short contribution
transaction costs
financing/borrow assumptions
net result
```

A strategy profitable only because the short book benefited from one crash is
different from persistent two-sided stock selection.

## 8. Exposure diagnostics

Over time visualize:

- Gross and net exposure.
- Market beta.
- Sector weights.
- Maximum single-name weight.
- Effective number of positions.
- Feature/factor exposure.

These plots reveal whether a “factor model” is actually a concentrated sector or
market-direction bet.

## 9. Core visualizations

### Data coverage heatmap

**Question:** Which symbols and periods are missing?

### Normalized price chart

**Question:** How did assets move relative to a common starting value?

### Feature distribution

**Question:** Is the factor skewed, heavy-tailed, clipped, or dominated by
outliers?

### Feature correlation heatmap

**Question:** Which inputs carry redundant information?

### Factor quintile bars

**Question:** Does future return change monotonically with factor value?

### Prediction-group equity curves

**Question:** Do high-ranked groups consistently separate from low-ranked groups?

### Daily and rolling IC

**Question:** Is ranking skill stable or concentrated in a short interval?

### Equity and drawdown

**Question:** What path produced the final return?

### Cost-sensitivity curve

**Question:** How fragile is performance to execution assumptions?

### Coefficient-stability chart

**Question:** Does a linear model learn the same relationship across folds?

## 10. Visualization rules

- Label the exact sample, horizon, cost, and universe.
- Show zero lines where sign matters.
- Use consistent colors for long, short, gross, and net series.
- Avoid dual axes unless the relationship is explicit.
- Show distributions or intervals, not only averages.
- Keep failures and bad periods visible.
- Do not truncate axes to exaggerate small differences.
- Make every chart answer a written question.

## 11. Statistical uncertainty

Useful approaches include:

- Confidence intervals that respect time dependence.
- Block bootstrap rather than independently resampling daily rows.
- Per-fold distributions.
- Parameter sensitivity.
- Stability across alternative reasonable universes.

An interval is only as valid as its dependence assumptions.

## 12. A compact report order

1. Data coverage and universe.
2. Feature definitions and distributions.
3. Target and validation design.
4. Baseline versus model predictive metrics.
5. Quintile/decile behavior.
6. Gross and net portfolio performance.
7. Drawdown, turnover, and exposure.
8. Period/regime breakdown.
9. Sensitivity and failure analysis.
10. Limitations and next hypothesis.

## Common mistakes

- Reporting accuracy for a ranking regression without decision relevance.
- Comparing Sharpe ratios from different cost assumptions.
- Hiding a negative fold inside one aggregate number.
- Plotting only cumulative return.
- Using annualized metrics on a very short period without warning.
- Reading feature importance as causal importance.
- Showing too many charts without a question for each.

## Practice

1. Calculate wealth and drawdown for five returns by hand.
2. Compare two return paths with equal total return but different drawdown.
3. Create quintile returns and test monotonic ordering.
4. Plot gross and net equity under three cost assumptions.
5. Write one research question above every plot in a notebook.

## Primary reading

- William F. Sharpe, [The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)
- William F. Sharpe, [Mutual Fund Performance](https://web.stanford.edu/~wfsharpe/art/mfp.pdf)
