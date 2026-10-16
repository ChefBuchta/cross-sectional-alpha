"""Daily long-only simulation using delayed signals and adjusted close prices."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from factor_model.data_checks import check_required_columns
from factor_model.portfolio import top_stock_weights


class BacktestConfig(BaseModel):
    """Costs are basis points per dollar bought or sold (10 bps = 0.1%)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    top_n: int = Field(default=3, gt=0, strict=True)
    rebalance_every: int = Field(default=5, gt=0, strict=True)
    commission_bps: float = Field(default=5.0, ge=0, le=100)
    slippage_bps: float = Field(default=5.0, ge=0, le=100)
    annual_risk_free_rate: float = Field(default=0.0, gt=-1)


@dataclass
class BacktestResult:
    """Equity starts at one; trades and costs use those initial-capital units."""

    daily: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    config: BacktestConfig


def _wide_table(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    check_required_columns(rows, ["date", "symbol", column])
    data = rows[["date", "symbol", column]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data.empty or data[["date", "symbol"]].isna().any().any():
        raise ValueError("dates and symbols must be present in a nonempty table")
    if data.duplicated(["date", "symbol"]).any():
        raise ValueError("date and symbol pairs must be unique")
    wide = data.pivot(index="date", columns="symbol", values=column).sort_index()
    if not np.isfinite(wide.to_numpy(dtype=float)).all():
        raise ValueError(f"{column} must be finite with the complete universe on each date")
    return wide.astype(float)


def _execute(
    holdings: np.ndarray, equity: float, target: np.ndarray, cost_rate: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve post-cost equity so buys, sells and fees are self-financing."""
    low, high = 0.0, equity
    for _ in range(60):
        remaining = (low + high) / 2
        required = remaining + cost_rate * np.abs(target * remaining - holdings).sum()
        if required > equity:
            high = remaining
        else:
            low = remaining
    new_holdings = target * ((low + high) / 2)
    traded = new_holdings - holdings
    cost = float(np.abs(traded).sum() * cost_rate)
    return new_holdings, traded, cost


def run_backtest(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    score_column: str,
    config: BacktestConfig | None = None,
    *,
    universe_benchmark: bool = False,
) -> BacktestResult:
    """Trade at next observed close, then rebalance every N price sessions.

    Pass prices bounded to the intended evaluation end. The first prediction
    date is an equity anchor, not a return observation. Terminal holdings are
    liquidated at the last close, even if the last holding interval is shorter.
    Missing prices or scheduled signals fail rather than silently skipping a day.
    Forward-return labels in predictions are never accessed.
    """
    config = config or BacktestConfig()
    scores = _wide_table(predictions, score_column)
    closes = _wide_table(prices, "adjusted_close")
    if set(closes.columns) != set(scores.columns):
        raise ValueError("prices and predictions must contain the same stock universe")
    if (closes <= 0).any().any():
        raise ValueError("adjusted close prices must be positive")
    if not scores.index.isin(closes.index).all():
        raise ValueError("prediction dates must be present in prices")
    closes = closes.loc[scores.index.min():, scores.columns]
    if len(closes) < 3:
        raise ValueError("at least three price dates are required to enter and hold")

    symbols = closes.columns
    holdings = np.zeros(len(symbols))
    cash = equity = 1.0
    rate = (config.commission_bps + config.slippage_bps) / 10_000
    daily = [{"date": closes.index[0], "equity": 1.0, "net_return": np.nan,
              "gross_return": np.nan, "cost": 0.0, "turnover": 0.0,
              "rebalance": False, "liquidation": False}]
    weights = [holdings.copy()]
    trades = []
    values = closes.to_numpy()

    for index in range(1, len(closes)):
        previous_equity = equity
        holdings *= values[index] / values[index - 1]
        before_cost = float(cash + holdings.sum())
        terminal = index == len(closes) - 1
        rebalance = (index - 1) % config.rebalance_every == 0 and not terminal
        cost = turnover = 0.0
        if rebalance or terminal:
            signal_date = closes.index[index - 1]
            if terminal:
                target = np.zeros(len(symbols))
            else:
                if signal_date not in scores.index:
                    raise ValueError(f"missing scheduled signal on {signal_date.date()}")
                if universe_benchmark:
                    target = np.full(len(symbols), 1.0 / len(symbols))
                else:
                    target = top_stock_weights(scores.loc[signal_date], config.top_n).to_numpy()
            holdings, traded, cost = _execute(holdings, before_cost, target, rate)
            turnover = float(np.abs(traded).sum() / before_cost)
            cash = float(before_cost - cost - holdings.sum())
            for symbol, amount in zip(symbols, traded):
                if abs(amount) > 1e-14:
                    trades.append({
                        "date": closes.index[index],
                        "signal_date": pd.NaT if terminal else signal_date,
                        "symbol": symbol, "notional": float(amount),
                        "cost": float(abs(amount) * rate), "liquidation": terminal,
                    })
        equity = float(cash + holdings.sum())
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError("portfolio equity must remain positive and finite")
        daily.append({
            "date": closes.index[index], "equity": equity,
            "net_return": equity / previous_equity - 1,
            "gross_return": before_cost / previous_equity - 1,
            "cost": cost, "turnover": turnover,
            "rebalance": rebalance, "liquidation": terminal,
        })
        weights.append(holdings / equity)

    daily_table = pd.DataFrame(daily).set_index("date")
    return BacktestResult(
        daily=daily_table,
        weights=pd.DataFrame(weights, index=daily_table.index, columns=symbols),
        trades=pd.DataFrame(trades, columns=[
            "date", "signal_date", "symbol", "notional", "cost", "liquidation",
        ]),
        config=config,
    )


def backtest_statistics(result: BacktestResult) -> dict[str, float | int]:
    """Annualize actual daily returns at 252 sessions; include entry/exit costs."""
    daily = result.daily
    returns = daily["net_return"].dropna()
    count = len(returns)
    volatility = float(returns.std(ddof=1))
    risk_free = np.expm1(np.log1p(result.config.annual_risk_free_rate) / 252)
    sharpe = (
        float((returns.mean() - risk_free) / volatility * np.sqrt(252))
        if volatility > 1e-12 else float("nan")
    )
    return {
        "net_return": float(daily["equity"].iloc[-1] - 1),
        "annual_return": float(daily["equity"].iloc[-1] ** (252 / count) - 1),
        "annual_volatility": volatility * np.sqrt(252),
        "sharpe": sharpe,
        "max_drawdown": float((daily["equity"] / daily["equity"].cummax() - 1).min()),
        "profitable_day_fraction": float(returns.gt(0).mean()),
        "cost_initial_capital_fraction": float(daily["cost"].sum()),
        "total_turnover": float(daily["turnover"].sum()),
        "rebalance_count": int(daily["rebalance"].sum()),
        "return_day_count": count,
    }
