# Finance and Portfolio Construction

## Concrete example: correct ranking, bad portfolio

A model ranks five stocks correctly. You invest 90% in the highest-ranked stock
and spread the other 10% across the rest. The concentrated stock suffers a
company-specific shock. The ranking model may have been useful, but the portfolio
construction turned a modest forecasting edge into a large idiosyncratic bet.

Prediction and portfolio construction are separate optimization problems.

## 1. Long and short positions

- A **long** position gains when the asset price rises.
- A **short** position borrows and sells an asset, then later repurchases it. It
  gains when the repurchase price is lower, before borrow fees and other costs.

Short positions have practical constraints:

- Shares may be unavailable to borrow.
- Borrow fees can change.
- Losses are not capped by the initial position size.
- Dividends and corporate actions create obligations.
- Brokers impose margin requirements.

A daily educational backtest usually simplifies these mechanics. State the
simplification.

## 2. Portfolio weights

Let `w[i,t]` be asset `i`'s weight for a period. A positive weight is long and a
negative weight is short.

One-period portfolio return before costs:

$$
r_{p,t+1} = \sum_i w_{i,t}r_{i,t+1}
$$

The timing subscript matters: weights chosen at `t` earn returns afterward.

## 3. Gross and net exposure

Net exposure:

$$
E_{net,t} = \sum_i w_{i,t}
$$

Gross exposure:

$$
E_{gross,t} = \sum_i |w_{i,t}|
$$

Example:

```text
Long book  = +100%
Short book = −100%
Net        =    0%
Gross      =  200%
```

Zero net dollar exposure does not guarantee zero market beta, sector exposure,
or nonlinear risk.

## 4. Equal-weight top/bottom portfolio

For a simple long-short construction:

1. Rank scores within each date.
2. Select the top `q` fraction and bottom `q` fraction.
3. Assign equal positive weights to the top group.
4. Assign equal negative weights to the bottom group.

If 10 stocks are long and 10 are short with one dollar of gross capital on each
side:

```text
each long weight  = +1 / 10
each short weight = −1 / 10
```

This produces gross exposure 2 and net exposure 0.

## 5. Score-proportional weights

Weights can use centered scores:

$$
w_{i,t} \propto s_{i,t} - \bar s_t
$$

Then normalize to a desired gross exposure. This uses more information than a
top/bottom cutoff, but extreme scores can concentrate positions. Clip scores or
apply position limits.

## 6. Volatility scaling

Inverse-volatility weighting gives smaller weights to more volatile assets:

$$
\tilde w_{i,t} = \frac{s_{i,t}}{\hat\sigma_{i,t}}
$$

Normalize afterward. This can balance predicted risk, but estimated volatility
is noisy and can encourage leverage after calm periods.

## 7. Market beta

A simple market model is:

$$
r_i = \alpha_i + \beta_i r_m + \varepsilon_i
$$

`β` describes sensitivity to market return in the chosen estimation window.
Portfolio beta is approximately:

$$
\beta_p = \sum_i w_i\beta_i
$$

Dollar-neutral weights can still have nonzero beta if long positions are more
market-sensitive than shorts.

## 8. Sector and factor exposure

If all longs are technology and all shorts are utilities, the portfolio may be a
sector trade rather than a stock-selection trade.

Possible controls:

- Rank within sectors.
- Demean scores within sectors.
- Constrain long and short sector weights.
- Regress scores on exposures and use residuals.
- Solve an optimization problem with exposure constraints.

Every control changes the strategy. Report both raw and controlled results when
possible.

## 9. Turnover

A common one-way turnover convention is:

$$
TO_t = \frac{1}{2}\sum_i |w_{i,t}-w_{i,t-1}^{drifted}|
$$

Conventions vary. The factor `1/2` prevents counting both sides of a rebalance as
separate total portfolio replacements. Define the convention in code and docs.

High turnover matters because:

- Costs scale with traded value.
- Short-term signals decay quickly.
- Capacity is limited by liquidity.
- Frequent rebalancing increases operational complexity.

## 10. Transaction costs

A simple linear model:

$$
Cost_t = c \sum_i |\Delta w_{i,t}|
$$

where `c` is cost per unit traded. If `c = 10` bps, use `0.001`, not `10`.

Real costs can include:

- Commissions and exchange fees.
- Bid-ask spread.
- Slippage from delayed or uncertain fills.
- Market impact that grows nonlinearly with order size.
- Borrow fees and locate costs for shorts.
- Taxes and financing.

Run sensitivity analysis instead of presenting one cost assumption as fact.

## 11. Liquidity and capacity

An order small relative to typical dollar volume may have limited impact. A large
order cannot assume execution at the closing price.

Useful controls:

- Maximum fraction of average daily dollar volume.
- Minimum rolling dollar volume.
- Maximum position weight.
- Delayed execution.
- More conservative costs for less liquid securities.

Daily Yahoo data cannot estimate real execution precisely, so describe results as
research approximations.

## 12. Rebalancing

Daily rebalancing reacts quickly but creates more turnover. Weekly or monthly
rebalancing reduces cost and may miss short-lived signals.

The rebalance schedule must be applied after signal availability. “Friday close
signal, Friday close execution” is usually optimistic unless the signal was
known before the closing auction and that execution is explicitly modeled.

## 13. Constraints as risk definitions

Possible constraints:

- Maximum absolute stock weight.
- Target gross and net exposure.
- Sector bounds.
- Beta bound.
- Volatility target.
- Liquidity limit.
- Turnover budget.

Constraints are not cleanup after predictions. They define what risks the
portfolio is allowed to take.

## Common mistakes

- Calling a dollar-neutral portfolio market-neutral.
- Applying predicted scores directly as unbounded weights.
- Forgetting short borrow and dividends.
- Calculating turnover against yesterday's target weights instead of drifted
  holdings.
- Using current liquidity to size historical trades.
- Assuming all trades fill at the signal price.
- Reporting gross return without net return.

## Practice

1. Construct equal long/short weights for ten ranked stocks.
2. Calculate net and gross exposure.
3. Calculate portfolio beta from asset weights and betas.
4. Rebalance a two-stock portfolio and calculate turnover.
5. Deduct 5, 10, and 25 bps and plot net performance sensitivity.

## Further reading

- Fama and French, *Common risk factors in the returns on stocks and bonds*
- Grinold and Kahn, *Active Portfolio Management*
- William F. Sharpe, [The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)
