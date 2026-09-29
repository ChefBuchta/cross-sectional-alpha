# Statistical Inference and Research Risk

## Concrete example: the same mean, different evidence

Two strategies each earn an average of 4 basis points per day.

- Strategy A earns similar small returns across five independent years.
- Strategy B earns nearly everything during one three-month event.

Their full-sample means are equal, but the evidence is not. Strategy A shows
repeatability across time. Strategy B may be a regime-specific accident.

Inference asks how uncertain an estimate is. Research-risk control asks how much
our own search process has made the final estimate too optimistic.

## 1. Estimate, standard error, and interval

An estimate is a sample quantity such as mean IC. Its **standard error**
describes how much that estimator would vary across repeated comparable
samples.

A simplified confidence interval is:

$$
\hat\theta \pm c \cdot \operatorname{SE}(\hat\theta)
$$

The critical value `c` and the validity of the interval depend on assumptions.
An interval built from independent daily rows is unreliable when observations
are autocorrelated or share overlapping forward returns.

## 2. Why ordinary standard errors fail

Financial panels contain dependence in two directions:

- **Through time:** rolling features, positions, regimes, and overlapping labels.
- **Across assets:** stocks share market, sector, and macroeconomic shocks.

Treating every date-stock row as independent can make uncertainty appear far
smaller than it is. Often the decision date—not the row—is the natural unit for
aggregating predictive evidence.

For example, calculate one IC per date and analyze the time series of daily IC
rather than treating all pairs of predictions and outcomes as independent.

## 3. Heteroskedasticity and autocorrelation consistent errors

HAC estimators, often associated with Newey and West, adjust covariance
estimates for heteroskedasticity and a chosen amount of serial dependence.

Conceptually:

1. Estimate residuals or the statistic's time series.
2. Estimate lagged covariance up to a selected lag.
3. Downweight more distant lags.
4. Combine them into a corrected variance estimate.

The lag choice is a research assumption. HAC errors do not fix biased data,
look-ahead leakage, nonstationarity, or multiple testing.

## 4. Block bootstrap

An ordinary bootstrap samples individual observations with replacement and
destroys time dependence. A block bootstrap samples contiguous stretches.

```python
import numpy as np


def moving_block_bootstrap(
  values: np.ndarray,
  block_size: int,
  rng: np.random.Generator,
) -> np.ndarray:
  sample: list[float] = []
  max_start = len(values) - block_size
  while len(sample) < len(values):
    start = int(rng.integers(0, max_start + 1))
    sample.extend(values[start:start + block_size])
  return np.asarray(sample[:len(values)])
```

Repeat this process, calculate the statistic for every resample, and inspect the
resulting distribution. Block length controls a trade-off: short blocks lose
dependence; long blocks produce fewer distinct resamples.

## 5. Cross-sectional uncertainty

On one date, all stocks experience the same market news. A standard error based
only on the number of stocks can overstate independent information.

Useful approaches include:

- Aggregate to one date-level metric.
- Cluster by date when the estimator supports clustering.
- Report dispersion across chronological folds.
- Repeat across reasonable universes.
- Analyze sectors and long/short sides separately.

The right method follows the sampling process and research claim.

## 6. Economic magnitude before significance

A tiny effect can become statistically distinguishable in a very large sample
while remaining untradeable. For every predictive result, translate it into:

- Expected spread between selected groups.
- Turnover required to capture it.
- Break-even transaction cost.
- Capacity and liquidity requirements.
- Drawdown and concentration.

Statistical and economic questions are complementary. Neither replaces the
other.

## 7. Multiple testing

If 100 independent useless ideas are tested at a 5% threshold, some will appear
significant by chance. Real research is worse because trials are correlated and
often undocumented.

The **researcher degrees of freedom** include:

- Universe definitions.
- Start and end dates.
- Feature formulas and horizons.
- Outlier handling.
- Target horizons.
- Model families and hyperparameters.
- Portfolio cutoffs and cost assumptions.

Record every serious trial, not only the final winner.

## 8. Family-wise error and false discovery rate

Family-wise procedures aim to control the probability of at least one false
positive in a family of tests. False-discovery-rate procedures control the
expected proportion of false discoveries among selected results under stated
conditions.

These corrections require a defined family of tests. They cannot repair a
hidden, open-ended research process after the fact.

Use them as one layer alongside economic hypotheses, holdouts, replication, and
trial logs.

## 9. The winner's curse

The best backtest among many noisy candidates tends to overestimate its true
performance. Even if the selected idea has genuine signal, its observed best
parameter or period is likely too favorable.

Practical responses:

- Prefer broad performance plateaus to sharp peaks.
- Shrink expectations toward simple baselines.
- Preserve a final untouched period.
- Replicate on another universe or later data.
- Compare the selected trial with the distribution of all trials.

## 10. Deflated performance thinking

A raw Sharpe ratio should be interpreted in the context of:

- Sample length.
- Non-normality and serial dependence.
- Number and dependence of strategies tried.
- Selection procedure.
- Whether parameters were tuned on the same history.

The **deflated Sharpe ratio** is one formal approach to adjusting a performance
claim for selection and non-normal returns. The important practical lesson is
broader: a Sharpe ratio without the research history behind it is incomplete.

## 11. Probability of backtest overfitting

The probability of backtest overfitting framework asks how often the strategy
selected as best in one part of the data ranks poorly out of sample across many
combinatorial splits.

It is most useful for mature research with many documented alternatives. It is
not a substitute for correct event timing or point-in-time data.

## 12. Parameter sensitivity maps

Plot outcomes across nearby reasonable choices:

| Choice | Values to inspect |
|---|---|
| Momentum lookback | 63, 126, 189, 252 days |
| Rebalance | daily, weekly, monthly |
| Ridge alpha | 0.1, 1, 10, 100 |
| Cost | 0, 5, 10, 20 bps |
| Long/short cutoff | 10%, 20%, 30% |

A smooth region suggests the result does not depend on one exact setting. A
single bright cell surrounded by failures is a warning.

Do not choose the prettiest cell and then call the map validation. The map is
itself part of the research search.

## 13. Placebo and falsification tests

A falsification test asks whether the pipeline also finds effects where none
should exist.

Examples:

- Randomly permute targets within dates.
- Shift signals so they occur after the returns they supposedly predict.
- Replace a factor with random noise of similar distribution.
- Reverse the expected direction.
- Test impossible pre-signal returns.

A system that reports strong results under impossible timing likely contains
leakage or a metric bug.

## 14. Negative controls

A negative control resembles the real analysis but lacks the proposed causal or
predictive link. It helps detect shared confounding or implementation errors.

For example, if a feature constructed only from information at `t` predicts
returns that ended before `t`, the relationship should be near zero. A strong
result indicates alignment problems or shared look-ahead information.

## 15. Research ledger

Record one row per experiment:

| Field | Purpose |
|---|---|
| Experiment ID | immutable reference |
| Timestamp | order of research decisions |
| Hypothesis | reason before seeing results |
| Parent experiment | what changed |
| Data version | provider snapshot |
| Configuration hash | exact choices |
| Primary metric | decision criterion |
| Outcome | including failures |
| Decision | stop, revise, or promote |

The ledger makes the invisible search visible.

## 16. Evidence grades

A simple internal scale can prevent one attractive chart from being treated as
finished research:

1. **Mechanical:** formulas and timing pass hand tests.
2. **Predictive:** out-of-sample relationship beats a baseline.
3. **Stable:** result survives chronological folds and nearby parameters.
4. **Implementable:** survives conservative costs and constraints.
5. **Replicated:** works on new data or another defensible universe.

Do not skip lower grades because a higher-level metric looks impressive.

## Reporting template

For each major claim, report:

- Point estimate.
- Dependence-aware uncertainty or fold distribution.
- Sample dates and number of independent date groups.
- Full trial count and selection process.
- Economic magnitude and break-even cost.
- Sensitivity to reasonable alternatives.
- Known failure periods.
- Whether the result has been replicated.

## Common mistakes

- Reporting a p-value without effect size.
- Using independent-row errors for overlapping panel observations.
- Correcting ten displayed tests while ignoring hundreds of hidden trials.
- Treating one final holdout as untouched after repeatedly inspecting it.
- Using sophisticated inference on a backtest with incorrect timing.
- Calling parameter instability robustness.

## Practice

1. Compare an ordinary bootstrap with a moving-block bootstrap on autocorrelated
   returns.
2. Build a parameter sensitivity heatmap without selecting a new winner.
3. Run a target-permutation placebo through the complete pipeline.
4. Create a research ledger and record every attempted factor horizon.
5. Assign an evidence grade to the current project and justify it.

## Primary reading

- [statsmodels HAC covariance documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html)
- Bailey et al., [The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb)
- Bailey and López de Prado, [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
