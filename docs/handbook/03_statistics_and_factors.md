# Statistics and Factor Signals

## Concrete example: a 10% return can be ordinary or exceptional

Stock A rose 10% over twenty days. Stock B rose 6%. Is A the stronger signal?

Not enough information. If A normally moves 8% in twenty days while B normally
moves 1%, B's smaller raw return may be more exceptional. If their entire sector
rose 12%, both may have underperformed their peers.

Factor research transforms raw observations into comparable statements.

## 1. Random variables, samples, and estimates

A return is treated as an observation from an uncertain process. Historical data
is a sample, not the complete future distribution.

Sample mean:

$$
\bar r = \frac{1}{n}\sum_{t=1}^{n} r_t
$$

Sample variance:

$$
s^2 = \frac{1}{n-1}\sum_{t=1}^{n}(r_t-\bar r)^2
$$

Sample standard deviation is `s`. It has the same unit as returns and is commonly
used as volatility.

Financial returns often have skewness, heavy tails, volatility clustering, and
changing distributions. Mean and standard deviation remain useful summaries,
but they do not fully describe risk.

## 2. Covariance and correlation

Covariance measures joint variation:

$$
\operatorname{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])]
$$

Correlation rescales covariance:

$$
\rho_{X,Y} = \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
$$

Correlation near one means two variables tend to move linearly in the same
direction. It does not imply causality, stability, or identical magnitude.

For factors, high correlation can make linear coefficients unstable. For assets,
correlation determines how risks combine in a portfolio.

## 3. Rolling estimates

A 20-day volatility at date `t` should use returns ending at or before `t`:

```python
frame["volatility_20d"] = (
  frame.groupby("symbol")["return_1d"]
  .rolling(20, min_periods=20)
  .std()
  .reset_index(level=0, drop=True)
)
```

Window length is a modeling choice:

- Short windows react quickly but are noisy.
- Long windows are stable but adapt slowly.
- Overlapping windows produce strongly related consecutive observations.

Do not interpret thousands of overlapping daily measurements as thousands of
independent experiments.

## 4. Cross-sectional normalization

On each date, calculate the mean and standard deviation across eligible stocks:

$$
z_{i,t} = \frac{x_{i,t} - \mu_t}{\sigma_t}
$$

```python
grouped = frame.groupby("date")["momentum_20d"]
frame["momentum_z"] = (
  frame["momentum_20d"] - grouped.transform("mean")
) / grouped.transform("std")
```

A z-score of `+1.5` means the stock's value is 1.5 cross-sectional standard
deviations above that day's mean. It does not mean the value is rare through
time for that stock.

### Percentile ranks

```python
frame["momentum_rank"] = (
  frame.groupby("date")["momentum_20d"].rank(pct=True)
)
```

Ranks are robust to extreme magnitudes and align naturally with a ranking target.
They discard distance information: the gap between ranks 0.90 and 0.80 may be
economically different from the gap between 0.60 and 0.50 even though both rank
gaps are 0.10.

## 5. Momentum

Simple `k`-period momentum:

$$
M_{i,t}^{(k)} = \frac{P_{i,t}}{P_{i,t-k}} - 1
$$

```python
frame["momentum_20d"] = (
  frame.groupby("symbol")["adjusted_close"].pct_change(20)
)
```

Economic interpretations include gradual information diffusion, institutional
trading, underreaction, and behavioral persistence. Momentum is not a law; it can
reverse sharply and its performance depends on horizon and market regime.

Long-term momentum often excludes the most recent period to avoid mixing it with
short-term reversal:

$$
M_{12-1} = \frac{P_{t-21}}{P_{t-252}} - 1
$$

## 6. Short-term reversal

Reversal uses recent returns with the hypothesis that extreme short moves partly
correct:

```python
frame["reversal_5d"] = -frame.groupby("symbol")["adjusted_close"].pct_change(5)
```

The negative sign makes a recent loser receive a positive reversal score. Name
the feature so its direction is unambiguous.

Possible mechanisms include temporary liquidity pressure, bid-ask effects, and
overreaction. At short horizons, trading costs can consume the signal.

## 7. Volatility and range features

Examples:

- Rolling standard deviation of daily returns.
- Downside deviation using only negative returns.
- High–low range divided by close.
- Average True Range divided by price.
- Distance from a rolling maximum.

Volatility may be used as a predictor, a risk control, or both. Mixing these roles
without distinction makes interpretation difficult.

## 8. Volume and liquidity features

Relative volume:

$$
RVOL_{i,t} = \frac{V_{i,t}}{\operatorname{mean}(V_{i,t-19:t})}
$$

Dollar volume approximates trading capacity:

$$
DV_{i,t} = P_{i,t}V_{i,t}
$$

Raw volume is not comparable across stocks with different share prices and share
counts. Ratios, z-scores, and dollar volume often have clearer interpretations.

## 9. Forward-return targets

An `h`-period forward simple return is:

$$
y_{i,t}^{(h)} = \frac{P_{i,t+h}}{P_{i,t}} - 1
$$

```python
grouped_price = frame.groupby("symbol")["adjusted_close"]
frame["forward_return_5d"] = grouped_price.shift(-5) / frame["adjusted_close"] - 1
frame["target_rank"] = (
  frame.groupby("date")["forward_return_5d"].rank(pct=True)
)
```

The negative shift is allowed only in target construction. It is a warning sign
in feature code.

## 10. Information Coefficient

For each date, the rank Information Coefficient is the Spearman correlation
between predictions and realized future returns:

$$
IC_t = \operatorname{corr}_{Spearman}(\hat y_{i,t}, y_{i,t})
$$

Interpretation:

- Positive: higher predictions tended to realize higher returns.
- Zero: no monotonic ranking relationship on that date.
- Negative: ranking tended to be reversed.

Report the distribution and time series, not only the mean. Consecutive ICs can
be dependent because targets overlap.

## 11. Statistical significance versus economic value

A tiny effect can be statistically detectable but too small to trade after
costs. A large estimated effect from a small sample can be economically exciting
but statistically uncertain.

Ask both:

1. Is the effect distinguishable from noise under reasonable assumptions?
2. Is the effect large and stable enough to matter after implementation costs?

Repeatedly trying factors and retaining the best inflates false discoveries.
Record every experiment, including failures, and reserve untouched data.

## 12. Neutralization

Suppose the momentum score is highest for technology stocks. A long-high/short-low
portfolio may secretly be a sector bet. Neutralization removes a modeled
component:

$$
x_{i,t} = \alpha_t + \beta_t^T c_{i,t} + \varepsilon_{i,t}
$$

Use the residual `ε` as the neutralized feature, where `c` contains sector or
other exposure controls. Neutralization changes the research question; it is not
automatically superior.

## Common mistakes

- Normalizing across the entire history instead of inside each date.
- Calculating a target with the wrong sign or horizon.
- Treating correlation as causality.
- Treating overlapping observations as independent.
- Selecting factor direction after observing test performance.
- Using an extreme-value rule fitted on the complete dataset.
- Reporting only average IC and hiding instability.

## Practice

1. Calculate a five-day return and its cross-sectional percentile by hand.
2. Compare raw momentum, z-score momentum, and ranked momentum for five stocks.
3. Show why the final five rows per symbol lack a five-day target.
4. Calculate Spearman IC for one small predicted and realized ranking.
5. Plot a factor's quintile returns and check whether they are monotonic.

## Further reading

- [pandas window operations](https://pandas.pydata.org/docs/user_guide/window.html)
- [Fama and French, Common risk factors in the returns on stocks and bonds](https://doi.org/10.1016/0304-405X%2893%2990023-5)
- Jegadeesh and Titman, *Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency* (1993)
