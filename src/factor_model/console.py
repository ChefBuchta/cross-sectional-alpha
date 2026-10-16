"""Console research report with a shared execution schedule for every model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_model.backtest import (
    BacktestConfig, BacktestResult, backtest_statistics, run_backtest,
)
from factor_model.metrics import daily_spearman_ic, summarize_daily_ic
from factor_model.models import ExperimentResult
from factor_model.portfolio import top_stock_weights
from factor_model.reporting import forecast_from_prices, retained_models
from factor_model.targets import build_targets


@dataclass
class ConsoleReport:
    validation: pd.DataFrame
    test_predictions: pd.DataFrame
    backtests: dict[str, BacktestResult]
    performance: pd.DataFrame
    selected_model: str


def prediction_statistics(
    rows: pd.DataFrame, score_column: str, horizon: int, top_n: int,
) -> dict[str, float | int]:
    """Daily overlapping prediction windows; these are not trading returns."""
    ic = summarize_daily_ic(daily_spearman_ic(
        rows, score_column=score_column, target_column=f"target_percentile_{horizon}d",
    ))
    edges, spreads = [], []
    for _, day in rows.groupby("date", sort=True):
        scores = day.set_index("symbol")[score_column]
        weights = top_stock_weights(scores, top_n)
        if weights.sum() > 0:
            outcomes = day.set_index("symbol")[f"forward_return_{horizon}d"]
            edges.append(float(weights.dot(outcomes) - outcomes.mean()))
            bottom = top_stock_weights(-scores, top_n)
            spreads.append(float(weights.dot(outcomes) - bottom.dot(outcomes)))
    return {
        "Rank IC": ic["mean"],
        "IC positive %": ic["fraction_positive"] * 100,
        "Selection win %": float(np.mean(np.array(edges) > 0) * 100) if edges else np.nan,
        "Avg edge (pp)": float(np.mean(edges) * 100) if edges else np.nan,
        "Top-bottom (pp)": float(np.mean(spreads) * 100) if spreads else np.nan,
        "Selection dates": len(edges),
        "IC dates": ic["defined_day_count"],
        "Undefined IC": ic["undefined_day_count"],
    }


def build_console_report(
    prices: pd.DataFrame, experiment: ExperimentResult, config: BacktestConfig,
) -> ConsoleReport:
    """Use frozen validation winners; never fit or select using backtest results."""
    forecasts = forecast_from_prices(prices, experiment)
    runs = retained_models(experiment)
    if set(forecasts) != set(runs):
        raise ValueError("every retained model must have post-selection predictions")
    reference = next(iter(forecasts.values()))
    reference_keys = set(map(tuple, reference[["date", "symbol"]].to_numpy()))
    for rows in forecasts.values():
        if set(map(tuple, rows[["date", "symbol"]].to_numpy())) != reference_keys:
            raise ValueError("model comparisons require identical prediction dates and stocks")
    settings = experiment.config
    price_dates = pd.to_datetime(prices["date"])
    bounded = prices.loc[
        prices["symbol"].isin(settings.symbols)
        & (price_dates >= pd.Timestamp(settings.start_date))
        & (price_dates < pd.Timestamp(settings.end_date))
    ].copy()
    bounded["date"] = pd.to_datetime(bounded["date"])
    if set(reference["symbol"]) != set(settings.symbols):
        raise ValueError("predictions must cover the experiment universe")
    # Targets are attached only for reporting; the simulator never receives them.
    targets = build_targets(
        bounded, reference[["date", "symbol"]], settings.target_horizon,
    )
    forecasts["momentum"] = reference.copy()
    score_columns = {name: f"{name}_score" for name in runs}
    score_columns["momentum"] = "baseline_score"
    validation, test_statistics, backtests = {}, {}, {}
    for name, rows in forecasts.items():
        column = score_columns[name]
        validation_rows = (
            next(iter(runs.values())).validation.scored_rows
            if name == "momentum" else runs[name].validation.scored_rows
        )
        validation[name] = prediction_statistics(
            validation_rows, column, settings.target_horizon, config.top_n,
        )
        labeled = rows.merge(targets, on=["date", "symbol"], validate="one_to_one")
        test_statistics[name] = prediction_statistics(
            labeled, column, settings.target_horizon, config.top_n,
        )
        backtests[name] = run_backtest(bounded, rows, column, config)
    backtests["universe"] = run_backtest(
        bounded, reference, "baseline_score", config, universe_benchmark=True,
    )
    benchmark = backtests["universe"].daily["net_return"].dropna()
    performance = {}
    for name, backtest in backtests.items():
        statistics = backtest_statistics(backtest)
        returns = backtest.daily["net_return"].dropna()
        statistics["beat_universe_day_fraction"] = (
            float((returns > benchmark).mean()) if name != "universe" else np.nan
        )
        performance[name] = statistics
    return ConsoleReport(
        validation=pd.DataFrame.from_dict(validation, orient="index"),
        test_predictions=pd.DataFrame.from_dict(test_statistics, orient="index"),
        backtests=backtests,
        performance=pd.DataFrame.from_dict(performance, orient="index"),
        selected_model=experiment.selection.model_name,
    )


def format_console_report(report: ConsoleReport, horizon: int) -> str:
    sample = next(iter(report.backtests.values()))
    config = sample.config
    table = report.performance
    displayed = pd.DataFrame({
        "Net %": table["net_return"] * 100,
        "Annual %": table["annual_return"] * 100,
        "Vol %": table["annual_volatility"] * 100,
        "Sharpe": table["sharpe"],
        "Max DD %": table["max_drawdown"] * 100,
        "Up days %": table["profitable_day_fraction"] * 100,
        "Beat universe days %": table["beat_universe_day_fraction"] * 100,
        "Costs % initial": table["cost_initial_capital_fraction"] * 100,
        "Turnover x": table["total_turnover"],
        "Rebalances": table["rebalance_count"],
        "Days": table["return_day_count"],
    })
    render = {"float_format": lambda value: f"{value:.2f}", "na_rep": "N/A"}
    return "\n\n".join([
        f"Validation-selected model: {report.selected_model} (mean rank IC).",
        f"VALIDATION PREDICTIONS | {horizon}-session horizon, top {config.top_n}\n"
        + report.validation.to_string(**render),
        f"LATER-PERIOD PREDICTIONS | {horizon}-session horizon, top {config.top_n}\n"
        + report.test_predictions.to_string(**render),
        "Selection win = picks beat the equal-weight universe; average edge is in percentage points.\n"
        "Prediction windows overlap; these counts are not independent bets.",
        f"BACKTEST | {sample.daily.index[0]:%Y-%m-%d} to {sample.daily.index[-1]:%Y-%m-%d}\n"
        f"Long only, top {config.top_n}, cutoff ties included, all-score ties hold cash.\n"
        f"Next-session close fills; rebalance every {config.rebalance_every} price sessions.\n"
        f"Commission {config.commission_bps:g} + slippage {config.slippage_bps:g} bps per dollar bought/sold.\n"
        f"Sharpe assumes {config.annual_risk_free_rate:.1%} annual risk-free return; cash earns zero.\n"
        + displayed.to_string(**render),
        "Up days = positive net daily return; beat-universe days compare net daily returns.\n"
        "Both include all simulated sessions, entry/exit fees, and zero-return ties as non-wins.\n"
        "Turnover counts purchases plus sales; costs are a fraction of initial capital.\n"
        "Final positions are liquidated; the final holding interval may be shorter.\n"
        "Annual metrics use 252 sessions/year; short samples can be misleading.\n"
        "Adjusted closes approximate total returns; no liquidity or market-impact model.\n"
        "Historical fixed-universe research; previously inspected periods are not fresh holdouts.\n"
        "Later-period results do not change model selection.",
    ])
