# Glossary

## A

**Adjusted close** — Historical close transformed by a data provider to account
for specified corporate actions. Verify whether dividends are included.

**Alpha** — Return or predictive component not explained by a chosen benchmark
or risk model. Its meaning depends on that model.

**Annualization** — Conversion of a periodic statistic to an annual convention,
such as multiplying mean daily return by 252 or volatility by `sqrt(252)` under
specific assumptions.

## B

**Backtest** — Simulation of a strategy on historical data using explicit
information, execution, accounting, and cost rules.

**Basis point (bp)** — One hundredth of one percent: `1 bp = 0.01% = 0.0001`.

**Block bootstrap** — Resampling method that draws contiguous groups of
observations so part of the original time dependence is preserved.

**Beta** — Estimated sensitivity of an asset or portfolio return to a benchmark
return in a specified model and sample.

**Bias–variance trade-off** — Tension between systematic model error and
sensitivity to sample variation.

## C

**Corporate action** — Event such as a split, dividend, merger, or spinoff that
can change price series, share counts, or cash flows.

**Correlation** — Standardized linear association between two variables. It does
not establish causality.

**Covariance matrix** — Square matrix containing each variable's variance on the
diagonal and pairwise covariances off the diagonal.

**Cross-sectional** — Comparison across many entities at one time, such as
ranking stocks within one trading date.

## D

**Data snooping** — Adapting research repeatedly to the same historical sample,
which makes apparent performance increasingly optimistic.

**Delisting bias** — Distortion caused by missing or mishandling securities that
leave the available dataset.

**Drawdown** — Decline of wealth from its previous running maximum.

## E

**Embargo** — Intentional gap near a validation/test period intended to reduce
information overlap across split boundaries.

**Eigenvector** — Direction that a linear transformation stretches without
rotating; leading covariance eigenvectors represent major directions of common
variation.

**Exposure** — Sensitivity or allocation to an asset, market, sector, or factor.

## F

**Factor** — Measurable asset characteristic or return driver used to explain or
predict differences among assets.

**Factor risk model** — Model that represents asset covariance using a smaller
set of common factor exposures plus asset-specific risk.

**Feature** — Model input available at the prediction moment.

**Forward return** — Return measured after a decision date over a specified
future horizon; commonly used as a target.

## G

**Gross exposure** — Sum of absolute portfolio weights.

**Gradient** — Vector of partial derivatives describing how a function changes
with respect to each parameter.

## H

**HAC covariance** — Heteroskedasticity and autocorrelation consistent covariance
estimate designed to account for specified forms of unequal variance and serial
dependence.

**Holdout** — Data excluded from model fitting and used for later evaluation.

**Hyperparameter** — Configuration selected outside ordinary model fitting, such
as Ridge penalty strength or tree depth.

## I

**Information Coefficient (IC)** — Correlation, often Spearman rank correlation,
between predicted scores and realized future returns across assets.

## L

**Leakage** — Information enters training or prediction even though it would not
have been available at the simulated decision time.

**Liquidity** — Ability to trade an asset in size without excessive delay or
price impact.

**Log return** — Difference of log prices; additive across adjacent time periods.

**Long position** — Position that benefits when the asset price rises.

## M

**Market impact** — Price movement caused by the act of trading.

**Momentum** — Tendency or signal based on persistence of past relative or
absolute performance over a defined horizon.

## N

**Net exposure** — Algebraic sum of signed portfolio weights.

**Negative control** — Analysis designed to lack the proposed relationship and
therefore reveal leakage, confounding, or implementation errors when it produces
a strong result.

**Neutralization** — Removal or constraint of exposure to specified variables,
such as sectors or market beta.

## O

**OHLCV** — Open, high, low, close, and volume observations for a bar or period.

**Out of sample** — Data not used for the model-fitting decision being evaluated.

**Overfitting** — Learning sample-specific noise or accidental structure that
does not generalize.

## P

**Panel data** — Repeated observations for many entities through time.

**Permutation importance** — Out-of-sample score deterioration after a feature
is shuffled according to a stated scheme.

**Point-in-time data** — Dataset that represents values and membership as they
were known at each historical timestamp.

**Principal component analysis (PCA)** — Rotation of correlated variables into
orthogonal sample-variance directions ordered from largest to smallest.

**Purging** — Removing training samples whose label intervals overlap a
validation or test interval.

## R

**Rank** — Ordered position or percentile rather than raw numerical magnitude.

**Rebalance** — Trade from current/drifted holdings to new target weights.

**Regularization** — Penalty or constraint used to control model complexity.

**Residual** — Observed value minus model prediction; in neutralization, the
component not explained by selected exposures.

**Risk parity** — Portfolio approach that seeks balanced contributions to risk
rather than equal capital weights.

**Reversal** — Signal based on the hypothesis that a recent move will partly
reverse.

## S

**Sharpe ratio** — Mean excess return divided by its volatility, under a stated
sampling and annualization convention.

**Shrinkage** — Deliberately moving a noisy estimate toward a structured target
to reduce estimation variance.

**Short position** — Borrowed-and-sold position that benefits from a later price
decline before fees and obligations.

**Signal** — Numerical output used to rank or size positions.

**Slippage** — Difference between an assumed/reference price and actual execution
price.

**Survivorship bias** — Bias from studying only assets that remain observable at
a later date.

## T

**Target** — Outcome a supervised model is trained to predict.

**Transaction cost** — Commission, spread, slippage, impact, financing, borrow,
or other cost associated with implementing positions.

**Turnover** — Amount of portfolio weight traded during rebalancing under a
specified convention.

## U

**Universe** — Set of assets eligible for ranking and portfolio selection at a
given date.

## V

**Volatility** — Dispersion of returns, commonly estimated by standard deviation
over a specified window and frequency.

## W

**Walk-forward validation** — Repeated evaluation in which models train on an
earlier period and test on a later period, moving the cutoff through time.

**Winsorization** — Clipping extreme values to selected lower and upper bounds
rather than removing the corresponding rows.

**Weight drift** — Change in portfolio weights caused by different asset returns
between rebalances.
