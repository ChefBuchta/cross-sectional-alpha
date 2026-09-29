# Backtesting Correctly

## Concrete example: the impossible closing-price trade

At 16:00, the official closing price becomes known. A signal uses that exact
close. The backtest then claims the portfolio bought at the same closing price.

The result assumes a trade was submitted using information that did not yet
exist. The numerical shift may be one row; the conceptual error invalidates the
simulation.

Backtesting is primarily an event-timing problem with accounting attached.

## 1. Define the event timeline

For every decision, write the sequence:

```text
observations become available
        ↓
features are calculated
        ↓
model produces scores
        ↓
portfolio decides target weights
        ↓
orders become executable
        ↓
future returns are realized
```

For a simple daily close-to-close backtest:

```text
Close t known → signal t → weight for next period → return t+1
```

Implementation usually requires shifting weights:

```python
executed_weights = target_weights.groupby("symbol").shift(1)
strategy_return = executed_weights * asset_return
```

Whether one shift is sufficient depends on timestamp definitions and execution
assumptions.

## 2. Vectorized versus event-driven simulation

### Vectorized backtest

Represents prices, signals, weights, and returns as aligned arrays/DataFrames.

Advantages:

- Fast and concise.
- Easy to inspect mathematically.
- Suitable for daily cross-sectional research.

Limitations:

- Complex order states and partial fills are awkward.
- Alignment errors can remain silent.
- Execution assumptions are easily hidden in shifts.

### Event-driven backtest

Processes market events, signals, orders, fills, and portfolio updates in time
order.

Advantages:

- Timing and state are explicit.
- Better for intraday execution and order logic.

Limitations:

- More code and more state-related bugs.
- Slower for broad research sweeps.

Start vectorized for daily factor research, but express the event timeline first.

## 3. Return alignment

If `close_return[t] = close[t] / close[t-1] - 1`, that return occurred between
`t-1` and `t`. A weight formed using `close[t]` cannot earn it.

One consistent convention:

| Column at row `t` | Meaning |
|---|---|
| `features_t` | Information available after close `t` |
| `target_weight_t` | Desired position after processing `t` |
| `executed_weight_t+1` | Position applied to next return interval |
| `return_t+1` | Price movement after decision `t` |

Add explicit names rather than generic `return` and `weight` when debugging.

## 4. Portfolio return

For weights known before the period:

$$
r_{p,t} = \sum_i w_{i,t-1}r_{i,t}
$$

Vectorized outline:

```python
contribution = frame["executed_weight"] * frame["return_1d"]
gross_return = contribution.groupby(frame["date"]).sum(min_count=1)
```

Use `min_count` or explicit validity checks so an all-missing day does not become
a reassuring zero.

## 5. Drifted weights

Without rebalancing, asset returns change weights:

$$
w_{i,t}^{drifted} =
\frac{w_{i,t-1}(1+r_{i,t})}{1+r_{p,t}}
$$

Turnover should compare new target weights with these drifted holdings, not
necessarily with yesterday's target weights. Simplified research may ignore
drift, but the convention must be named.

Long-short portfolios require care because leverage and denominator conventions
can differ. Test with two assets by hand.

## 6. Costs

Simple linear costs:

```python
traded_weight = (target_weight - previous_weight).abs()
cost = traded_weight.groupby(date).sum() * cost_rate
net_return = gross_return - cost
```

Confirm whether turnover is one-way or two-way and whether `cost_rate` applies to
each traded side. Keep costs as a separate output column:

```text
gross_return | spread_cost | borrow_cost | net_return
```

This makes impossible improvements visible.

## 7. Cash, leverage, and financing

A portfolio with 100% long and 100% short has 200% gross exposure. The short sale
proceeds, collateral, margin, and financing treatment depend on the real setup.

An educational backtest may report return on a chosen capital base without full
broker accounting. It should state:

- Gross and net exposure.
- Return denominator.
- Financing assumption.
- Whether idle cash earns interest.
- Whether short proceeds are reusable.

## 8. Delistings and missing exits

Dropping a stock when its data disappears can erase a large loss. Possible
responses include:

- Use a trusted delisting-return source.
- Apply a documented conservative fallback.
- Exclude affected experiments and report the limitation.
- Stop the simulation when valuation becomes impossible.

Silently removing the position is rarely defensible.

## 9. Corporate actions

Adjusted returns may represent splits and dividends, while raw execution prices
represent tradable price levels. Keep the roles explicit. Short positions may owe
dividends, and splits change share counts even if economic exposure is unchanged.

## 10. Major backtest biases

### Look-ahead bias

Future information enters a decision, often through alignment, normalization,
revised data, or same-close execution.

### Survivorship bias

Historical universe contains only securities that survived to a later date.

### Selection and multiple-testing bias

Many ideas are tried, but only the best result is reported.

### Data-snooping bias

Research decisions adapt repeatedly to one historical sample.

### Transaction-cost bias

Execution is assumed free or at an unrealistically favorable price.

### Capacity bias

Trade size is assumed not to affect price or fill probability.

### Shorting bias

Every stock is assumed borrowable at negligible cost.

## 11. Backtest invariants

Automated checks should include:

- No position exists before its signal is available.
- Net return equals gross return minus named costs.
- Increasing nonnegative costs never raises net wealth.
- Portfolio return equals the sum of asset contributions.
- Target gross/net exposure matches configuration on rebalance dates.
- No missing price silently removes a live position.
- First valid strategy return occurs after the first valid signal.
- Changing future prices cannot alter an earlier feature or position.

The last property can be tested by modifying future rows and comparing earlier
outputs.

## 12. Tiny hand-worked test

Two assets, one period:

```text
Weights before return: AAA +0.50, BBB −0.50
Returns:               AAA +2%,   BBB −1%
Contributions:         +1.00%,    +0.50%
Gross portfolio return: +1.50%
Turnover: 40%
Cost: 10 bps × 40% = 0.04%
Net return: 1.46%
```

If code cannot reproduce this example, do not trust a ten-year simulation.

## 13. Backtest outputs

Save at least:

- Per-asset scores and target weights.
- Executed weights.
- Per-asset returns and contributions.
- Turnover and each cost component.
- Gross and net portfolio return.
- Gross, net, sector, and beta exposure.
- Experiment configuration and data version.

Aggregated metrics are not sufficient for debugging.

## Common mistakes

- Using return `t` with a signal computed at the end of `t`.
- Comparing weights and returns with different indexes.
- Filling an unavailable return with zero.
- Calculating costs after aggregating away asset trades.
- Ignoring weight drift while claiming exact turnover.
- Annualizing short samples mechanically.
- Selecting the best backtest from many trials without correction.

## Practice

1. Draw the event timeline for close-to-next-close execution.
2. Reproduce the two-asset hand calculation in code.
3. Write an invariant that future price changes cannot alter past weights.
4. Compare target-to-target turnover with drift-aware turnover.
5. Inject a missing exit price and define explicit failure behavior.

## Primary reading

- [Bailey et al., The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb)
- Bailey, Borwein, and López de Prado,
  [Stock Portfolio Design and Backtest Overfitting](https://escholarship.org/uc/item/2np9b5g9)
