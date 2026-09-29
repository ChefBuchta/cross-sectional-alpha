# Portfolio Optimization and Risk Models

## Concrete example: the highest scores need not receive the largest weights

Suppose two stocks receive equally strong predictions. One is a stable large-cap
stock; the other is highly volatile and strongly correlated with several
existing positions. Equal score-proportional weighting gives them similar
capital. A risk-aware portfolio may allocate less to the volatile, redundant
position.

The model answers, “which assets look attractive?” Portfolio optimization
answers, “how can those views be held under risk, exposure, turnover, and
trading constraints?”

## 1. Inputs to an optimizer

A portfolio optimizer commonly receives:

- Expected-return or ranking scores `mu`.
- Covariance estimate `Sigma`.
- Previous weights `w_previous`.
- Sector, beta, country, or style exposure matrix `B`.
- Position, gross, net, turnover, and liquidity constraints.
- Risk-aversion and cost parameters.

Every input is estimated. An exact optimizer can amplify estimation error, so
simple constrained weighting remains an important baseline.

## 2. Portfolio variance

For weights `w` and covariance matrix `Sigma`:

$$
\sigma_p^2 = w^T\Sigma w
$$

Marginal risk depends on both individual volatility and covariance with the
rest of the portfolio:

$$
\frac{\partial \sigma_p}{\partial w_i}=\frac{(\Sigma w)_i}{\sigma_p}
$$

An asset with modest standalone volatility can still add substantial risk if it
duplicates existing exposures.

## 3. Estimating covariance

The ordinary sample covariance is easy to calculate but noisy when the asset
count is large relative to the time window.

Choices include:

- Longer windows for stability.
- Exponentially weighted observations for adaptation.
- Shrinkage toward a diagonal or structured target.
- Factor covariance models.
- Robust estimators for extreme observations.

Estimate covariance using data available before the rebalance. Recompute it on
the same schedule as the portfolio rather than using one full-sample matrix.

## 4. Covariance shrinkage

Shrinkage combines the sample estimate `S` with a structured target `F`:

$$
\hat\Sigma = \delta F + (1-\delta)S
$$

`delta` controls the compromise between lower variance and target bias. A
Ledoit-Wolf estimator chooses a shrinkage intensity analytically under its
assumptions.

```python
from sklearn.covariance import LedoitWolf

covariance = LedoitWolf().fit(return_matrix).covariance_
```

The return matrix should contain aligned historical observations, and missing
data handling must be explicit.

## 5. Minimum-variance portfolio

The minimum-variance problem ignores expected returns:

$$
\min_w w^T\Sigma w
$$

subject to constraints such as full investment. It provides a useful risk-only
reference but can concentrate in assets whose volatility is estimated as low.
Bounds and covariance regularization are usually necessary.

## 6. Mean-variance objective

A common objective trades expected return against variance:

$$
\max_w \quad \mu^Tw - \lambda w^T\Sigma w
$$

Higher `lambda` means greater risk aversion. In alpha research, `mu` may be a
model score rather than a calibrated expected return. Its scale then interacts
directly with `lambda`, so standardize or calibrate deliberately.

## 7. Turnover and cost penalties

Penalize changes from drifted current weights:

$$
\max_w \quad \mu^Tw - \lambda w^T\Sigma w
- \kappa\|w-w_{previous}\|_1
$$

An L1 turnover term resembles proportional trading costs and can create a
no-trade region. A quadratic term is smoother and easier to optimize but may be
less economically faithful.

The cost penalty should be consistent with the backtest's explicit cost model.
Do not optimize under one assumption and report under another without showing
the difference.

## 8. Common constraints

| Constraint | Example | Purpose |
|---|---|---|
| Net exposure | `sum(w) = 0` | dollar-neutral target |
| Gross exposure | `sum(abs(w)) <= 1` | leverage control |
| Position bound | `-0.02 <= w_i <= 0.02` | concentration control |
| Sector exposure | `abs(B_sector.T @ w) <= 0.03` | sector neutrality |
| Beta exposure | `abs(beta @ w) <= 0.05` | market sensitivity control |
| Turnover | `sum(abs(w-w_prev)) <= 0.30` | trading control |
| Liquidity | `trade_i <= fraction * ADV_i` | capacity control |

Constraints can conflict. A production-quality optimizer must detect
infeasibility and report which assumptions cannot be satisfied together.

## 9. A constrained optimization skeleton

SciPy offers general constrained solvers. The following simplified long-only
example minimizes negative score plus variance:

```python
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def objective(
  weights: np.ndarray,
  scores: np.ndarray,
  covariance: np.ndarray,
  risk_aversion: float,
) -> float:
  expected_score = scores @ weights
  variance = weights @ covariance @ weights
  return -expected_score + risk_aversion * variance


n_assets = len(scores)
bounds = Bounds(np.zeros(n_assets), np.full(n_assets, 0.05))
fully_invested = LinearConstraint(np.ones((1, n_assets)), 1.0, 1.0)
result = minimize(
  objective,
  x0=np.full(n_assets, 1 / n_assets),
  args=(scores, covariance, 5.0),
  method="SLSQP",
  bounds=bounds,
  constraints=[fully_invested],
)
if not result.success:
  raise RuntimeError(result.message)
```

This is educational, not a complete long-short optimizer. Validate all returned
constraints numerically; solver success alone is not enough.

## 10. Risk parity

Risk parity seeks more balanced contributions to portfolio risk rather than
equal capital weights. For asset `i`, risk contribution is related to:

$$
RC_i = w_i(\Sigma w)_i
$$

Equal-risk-contribution portfolios can still contain concentration through
correlation structure. They also require a covariance estimate and constraints.

## 11. Volatility targeting

Scale a portfolio toward a target volatility:

$$
scale_t = \frac{\sigma_{target}}{\hat\sigma_t}
$$

Clip the scale to leverage bounds. Estimated volatility is backward-looking and
can rise after losses, forcing deleveraging during stress. Use past-only
estimates and simulate the timing explicitly.

## 12. Factor risk models

A factor model decomposes asset returns:

$$
r = Bf + \epsilon
$$

where `B` contains asset exposures, `f` contains factor returns, and `epsilon`
contains idiosyncratic returns. Covariance becomes:

$$
\Sigma = B\Sigma_fB^T + D
$$

This reduces the number of covariance quantities and makes exposure sources
visible. The factors, exposures, and residual variances still require
point-in-time estimation.

## 13. Beta estimation

A simple market beta comes from:

$$
r_i = \alpha_i + \beta_i r_m + \epsilon_i
$$

Beta depends on window, frequency, benchmark, and regime. Shrinking noisy betas
toward one can improve stability. A portfolio with net weights of zero can have
nonzero beta when long and short assets have different sensitivities.

## 14. Ex-ante and ex-post risk

**Ex-ante risk** is forecast before trading from estimated covariance and
exposures. **Ex-post risk** is measured from realized returns afterward.

Compare them:

- Is realized volatility systematically above forecast?
- Which sectors or factors produced unexpected risk?
- Did correlations rise during drawdowns?
- Did position bounds or missing data distort exposures?

Large persistent differences indicate a risk-model or timing problem.

## 15. Optimizer failure modes

- Extreme weights from an ill-conditioned covariance matrix.
- Score scale changes causing radically different leverage.
- Constraints silently violated within loose tolerances.
- High turnover from small score changes.
- Unstable solutions near a constraint boundary.
- Infeasibility after combining sector, beta, position, and turnover limits.
- False precision from uncertain expected returns.

Always maintain a deterministic fallback such as capped rank weights.

## 16. Portfolio validation checklist

- [ ] Weights map to the intended symbols in the intended order.
- [ ] Gross and net exposure match configuration within tolerance.
- [ ] Position and group constraints are checked after solving.
- [ ] Covariance uses only historical observations.
- [ ] Optimizer failure is explicit, not silently replaced.
- [ ] Previous weights reflect drift before turnover is calculated.
- [ ] Score and risk scales are logged.
- [ ] Small input perturbations do not cause unexplained weight jumps.
- [ ] A simple weighting baseline is reported.
- [ ] Ex-ante risk is compared with realized risk.

## Practice

1. Calculate two-asset portfolio variance at correlations −0.5, 0, and 0.8.
2. Compare sample and shrinkage covariance matrices.
3. Add position bounds and dollar neutrality to a toy optimizer.
4. Stress scores and covariance by 10% and inspect weight stability.
5. Compare optimized weights with capped percentile-rank weights after costs.

## Primary documentation

- [scikit-learn covariance estimation](https://scikit-learn.org/stable/modules/covariance.html)
- [scikit-learn Ledoit-Wolf estimator](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)
- [SciPy constrained minimization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [SciPy linear constraints](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html)
