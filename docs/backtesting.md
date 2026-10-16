# Console backtest

Run from the repository root after `uv sync`:

```powershell
uv run python main.py
```

The default experiment uses the ten stocks and chronological dates in `main.py`.
It reads the local `data/processed/eda_data.csv`; it does not download data or
start Streamlit. Use `--prices PATH` for a different local price file with the
same universe and enough history to cover the configured training, validation,
and later evaluation periods.

```powershell
uv run python main.py --top-n 3 --rebalance-every 5 --commission-bps 5 --slippage-bps 5
uv run python main.py --commission-bps 0 --slippage-bps 0
uv run python main.py --risk-free-rate 0.03
uv run python main.py --models elastic_net extra_trees
uv run python main.py --models ridge elastic_net extra_trees
```

The console defaults to five families: Ridge, Random Forest, histogram gradient
boosting, Elastic Net, and Extra Trees. `--models` selects a subset; momentum and
the universe benchmark remain included. Existing dashboard calls keep their
original three families. No additional dependencies are needed.

Elastic Net combines L1 and L2 penalties and scales features using training data
only. Its fixed grid tests alpha 0.0001/0.01 and L1 ratio 0.2/0.8. Extra Trees
uses randomized tree splits, with 200 trees, depth 3/5, minimum leaf size 100/20,
and random seed 42. Both select their best of four configurations using mean
daily validation rank IC. Exact within-family ties keep the first candidate.
Neither uses random train/validation splitting or test data to choose settings.
Changing the model list can change the validation-selected overall family.

Model references: [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
and [Extra Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).

The first table measures validation predictions. Each family's parameters and
the overall winning family are selected by validation mean daily Spearman rank
correlation (rank IC). The second table measures frozen models' later-period
predictions. Neither later-period predictions nor simulated returns change
selection. All models use identical stocks and later prediction dates.

- **Selection win %:** fraction of dates where equal-weight top picks beat the
  equal-weight universe over the target horizon. Exact ties are not wins. Dates
  with all equal scores abstain and are excluded from this denominator.
- **Average edge:** average top-pick forward return minus universe forward
  return, in percentage points. This measures relative performance, even when
  both lose money.
- **Rank IC:** daily Spearman correlation across stocks, averaged over defined
  dates. Positive IC % measures how often the entire ordering is useful.
- **Top-bottom spread:** top-pick minus bottom-pick mean forward return, in
  percentage points. Cutoff ties are included on both sides. Groups can overlap
  in small universes, so this is a descriptive spread, not a short-sale backtest.
- **Counts:** defined and undefined IC dates and actual selection dates. Forward
  windows overlap; these are not independent observations or executable returns.

The final table is a daily portfolio simulation for each frozen model, a 20-day
momentum ranking baseline, and an equal-weight universe benchmark. It uses all
available later-period feature rows, including the final rows without known
future labels. It does not compound overlapping target returns.

## Execution and accounting

For example, Monday's close supplies the features and scores. Those scores can
trade at Tuesday's close. Tuesday's entry does not earn Monday-to-Tuesday price
movement. With a five-session rebalance schedule, the next execution is the
following Tuesday's close, using the preceding session's scores.

The portfolio starts with one unit of cash and no leverage or short positions.
Each rebalance equal-weights the highest-ranked N stocks, including all ties at
the cutoff. If every stock has the same score, the portfolio holds cash. Weights
drift with prices between scheduled rebalances; the simulator does not rebalance
silently each day. Missing scheduled signals or incomplete/nonpositive price
tables cause an error rather than forward-filling or changing the stock universe.
Sessions are the observed price dates, not an independently checked exchange
calendar; a date missing for every stock cannot be discovered by this simulator.

The benchmark always invests equally in all stocks, with the same execution
dates, rebalancing frequency, commission and slippage. Costs apply to both buys
and sells. Default commission is 5 basis points and default slippage is 5 basis
points per dollar traded: together 0.1% each way. These are illustrative cost
assumptions, not broker quotes. Post-cost holdings are solved so purchases and
fees fit within available equity. Closing positions at the final observed close
also incurs costs; the last holding interval may be shorter than the others.

Adjusted closes supply a total-return approximation. The simulation uses
synthetic adjusted-price units, not real share counts, market-on-close order
fills, or a full corporate-action ledger. There is no liquidity constraint,
market impact model, tax, fractional-share restriction, or interest on cash.

## Reading the trading metrics

- **Net return:** final wealth divided by initial wealth minus one, after costs.
- **Annual return:** geometric return annualized using 252 price sessions/year.
- **Annual volatility:** sample standard deviation of net daily returns times
  the square root of 252.
- **Sharpe:** average net daily return minus the daily equivalent of the
  configurable annual risk-free rate, divided by daily volatility, annualized
  by the square root of 252. Undefined at zero volatility. This risk-free rate
  is a comparison assumption; it is not credited to portfolio cash.
- **Maximum drawdown:** worst decline from the previous equity high, including
  the initial cash equity before entry fees. Shown as a negative percentage.
- **Up days %:** share of simulated sessions with positive net daily return.
- **Beat universe days %:** share of sessions with net return strictly greater
  than the net universe benchmark. N/A for the benchmark itself.
- **Costs % initial:** sum of paid commissions/slippage divided by initial capital.
  This is not the difference between gross and net compounded performance.
- **Turnover x:** sum of daily traded notional divided by pre-trade equity.
  Purchases and sales both count; a full switch is approximately 2x.
- **Days:** number of actual daily return observations. The first signal date
  anchors initial equity and is excluded. Entry and exit sessions are included;
  flat days count as non-wins in both daily percentages.

The daily net returns compound exactly to reported net wealth. Returned
`BacktestResult` objects also contain daily equity, post-trade weights and a
trade ledger with signal dates, execution dates, notionals and allocated costs.
They remain in memory; the command does not write generated result files.

## Research limits

The default historical dates have already been examined during development.
They provide a reproducibility exercise, not a new independent holdout. A fixed
list of surviving stocks introduces survivorship bias. Later model comparisons
are descriptive; repeatedly choosing winners from them consumes the holdout.
Annualization and Sharpe can be unstable on short samples, and daily returns
can still be serially dependent. This first simulation does not establish
statistical significance or future profitability.

Reference conventions: [next-bar close execution](https://www.backtrader.com/docu/order-creation-execution/order-creation-execution/),
[rank IC](https://quantopian.github.io/alphalens/alphalens.html#alphalens.performance.factor_information_coefficient),
and [portfolio performance statistics](https://github.com/quantopian/pyfolio/blob/master/pyfolio/timeseries.py).
