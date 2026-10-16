"""Report integration on synthetic data, without Streamlit or network access."""

from datetime import date
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from factor_model.backtest import BacktestConfig
from factor_model.console import (
    build_console_report, format_console_report, prediction_statistics,
)
from factor_model.features import build_features
from factor_model.models import ExperimentConfig, ExperimentResult, ModelRun
from factor_model.reporting import forecast_from_prices
from factor_model.targets import build_targets
from factor_model.training import evaluate_model, select_ridge_model
from factor_model.validation import split_labeled_data


class ConsoleTest(unittest.TestCase):
    def experiment(self):
        dates = pd.bdate_range("2024-01-02", periods=85)
        prices = pd.DataFrame([
            {"date": day, "symbol": symbol,
             "adjusted_close": 100 * np.exp(.001 * index + .025 * np.sin(index / (4 + stock))),
             "close": 100 * np.exp(.001 * index + .025 * np.sin(index / (4 + stock))),
             "volume": 1000 + index + stock}
            for index, day in enumerate(dates)
            for stock, symbol in enumerate("ABCD")
        ])
        config = ExperimentConfig(
            symbols=tuple("ABCD"), start_date=date(2024, 1, 1),
            validation_start=dates[40].date(), test_start=dates[60].date(),
            end_date=(dates[-1] + pd.Timedelta(days=1)).date(), target_horizon=2,
        )
        labeled = build_targets(prices, build_features(prices).dropna(), horizon=2)
        splits = split_labeled_data(labeled, config)
        selection = select_ridge_model(
            splits.train, splits.valid, alphas=(1.0,), target_column="target_percentile_2d",
        )
        evaluation = evaluate_model(selection, splits.valid)
        result = ExperimentResult(
            config=config, selection=selection, validation=evaluation,
            model_runs={"ridge": ModelRun(selection=selection, validation=evaluation)},
        )
        return prices, result

    def test_report_uses_frozen_models_and_all_unlabeled_tail_prices(self):
        prices, experiment = self.experiment()
        predictions = forecast_from_prices(prices, experiment)
        original = predictions["ridge"].copy()
        config = BacktestConfig(top_n=1, rebalance_every=2)
        with patch.object(experiment.selection.pipeline, "fit", side_effect=AssertionError("refit")):
            report = build_console_report(prices, experiment, config)
        self.assertEqual(set(report.backtests), {"ridge", "momentum", "universe"})
        self.assertEqual(report.selected_model, "ridge")
        self.assertIsNone(experiment.test)
        for result in report.backtests.values():
            self.assertEqual(result.daily.index[-1], prices.date.max())
            self.assertEqual(result.daily.index[0], pd.Timestamp(experiment.config.test_start))
            self.assertEqual(result.daily.rebalance.sum(), 12)
        self.assertEqual(report.test_predictions.loc["ridge", "IC dates"], 23)
        self.assertTrue(np.isnan(report.performance.loc["universe", "beat_universe_day_fraction"]))
        assert_frame_equal(original, forecast_from_prices(prices, experiment)["ridge"])
        output = format_console_report(report, 2)
        for text in ("VALIDATION", "LATER-PERIOD", "BACKTEST", "Selection win %",
                     "Sharpe", "Next-session close", "not fresh holdouts", "Commission 5 + slippage 5 bps"):
            self.assertIn(text, output)

    def test_report_rejects_different_comparison_dates(self):
        prices, experiment = self.experiment()
        forecasts = forecast_from_prices(prices, experiment)
        experiment.model_runs["other"] = experiment.model_runs["ridge"]
        forecasts["other"] = forecasts["ridge"].iloc[4:].copy()
        with patch("factor_model.console.forecast_from_prices", return_value=forecasts):
            with self.assertRaisesRegex(ValueError, "identical prediction dates"):
                build_console_report(prices, experiment, BacktestConfig())

    def test_selection_win_is_relative_even_when_every_stock_loses_money(self):
        rows = pd.DataFrame({
            "date": pd.Timestamp("2024-01-02"), "symbol": list("ABCD"),
            "score": [4, 3, 2, 1], "forward_return_5d": [-.02, -.04, -.06, -.08],
            "target_percentile_5d": [1, .75, .5, .25],
        })
        stats = prediction_statistics(rows, "score", horizon=5, top_n=1)
        self.assertAlmostEqual(stats["Selection win %"], 100)
        self.assertAlmostEqual(stats["Avg edge (pp)"], 3)
        self.assertAlmostEqual(stats["Top-bottom (pp)"], 6)
        self.assertAlmostEqual(stats["Rank IC"], 1)
        tied = prediction_statistics(rows.assign(score=1), "score", horizon=5, top_n=1)
        self.assertTrue(np.isnan(tied["Selection win %"]))
        self.assertEqual(tied["Selection dates"], 0)
        self.assertEqual(tied["Undefined IC"], 1)


if __name__ == "__main__":
    unittest.main()
