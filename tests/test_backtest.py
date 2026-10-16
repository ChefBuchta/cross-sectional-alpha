"""Hand-calculated execution, accounting, and chronology checks."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from factor_model.backtest import BacktestConfig, backtest_statistics, run_backtest
from factor_model.portfolio import top_stock_weights


class BacktestTest(unittest.TestCase):
    def frames(self, stock_a=(100, 200, 220, 242), stock_b=None):
        stock_b = stock_b or [100] * len(stock_a)
        dates = pd.bdate_range("2024-01-02", periods=len(stock_a))
        prices = pd.DataFrame([
            {"date": day, "symbol": symbol, "adjusted_close": price}
            for index, day in enumerate(dates)
            for symbol, price in (("A", stock_a[index]), ("B", stock_b[index]))
        ])
        predictions = prices[["date", "symbol"]].copy()
        predictions["score"] = predictions["symbol"].map({"A": 2.0, "B": 1.0})
        return prices, predictions

    def simulate(self, prices, predictions, **kwargs):
        config = BacktestConfig(
            top_n=1, commission_bps=0, slippage_bps=0, **kwargs,
        )
        return run_backtest(prices, predictions, "score", config)

    def test_trades_next_close_and_does_not_earn_the_signal_to_fill_jump(self):
        prices, predictions = self.frames()
        original_prices, original_predictions = prices.copy(), predictions.copy()
        result = self.simulate(prices, predictions)
        # The 100 -> 200 jump happens before entry. Only 200 -> 242 is earned.
        self.assertAlmostEqual(result.daily["equity"].iloc[-1], 1.21)
        self.assertAlmostEqual(result.daily["net_return"].iloc[1], 0)
        self.assertEqual(result.trades.iloc[0]["date"], pd.Timestamp("2024-01-03"))
        self.assertEqual(result.trades.iloc[0]["signal_date"], pd.Timestamp("2024-01-02"))
        self.assertTrue((result.trades.loc[~result.trades.liquidation, "signal_date"]
                         < result.trades.loc[~result.trades.liquidation, "date"]).all())
        self.assertAlmostEqual(result.weights.iloc[-1].sum(), 0)
        assert_frame_equal(prices, original_prices)
        assert_frame_equal(predictions, original_predictions)

    def test_entry_and_exit_costs_are_self_financing_on_flat_prices(self):
        prices, predictions = self.frames(stock_a=(100, 100, 100, 100))
        config = BacktestConfig(top_n=1, commission_bps=50, slippage_bps=50)
        result = run_backtest(prices, predictions, "score", config)
        # At entry: x + .01*x = 1. At exit: x - .01*x.
        expected = .99 / 1.01
        self.assertAlmostEqual(result.daily["equity"].iloc[-1], expected)
        self.assertAlmostEqual(result.trades["cost"].sum(), 1 - expected)
        self.assertAlmostEqual(result.daily["cost"].sum(), result.trades["cost"].sum())
        self.assertEqual(len(result.trades), 2)
        stats = backtest_statistics(result)
        self.assertAlmostEqual(stats["max_drawdown"], expected - 1)
        self.assertAlmostEqual(stats["profitable_day_fraction"], 0)

    def test_weights_drift_and_returns_compound_without_daily_rebalancing(self):
        prices, predictions = self.frames(stock_a=(100, 100, 200, 400))
        result = run_backtest(
            prices, predictions, "score",
            BacktestConfig(top_n=2, commission_bps=0, slippage_bps=0),
        )
        # Half the initial capital grows 4x; half remains unchanged.
        self.assertAlmostEqual(result.daily["equity"].iloc[-1], 2.5)
        self.assertAlmostEqual(result.weights.iloc[2]["A"], 2 / 3)
        self.assertEqual(result.daily["rebalance"].sum(), 1)

    def test_rebalances_use_previous_date_and_charge_both_sides_of_switch(self):
        prices, predictions = self.frames(stock_a=(100,) * 6)
        dates = sorted(predictions["date"].unique())
        predictions.loc[predictions.date >= dates[2], "score"] *= -1
        config = BacktestConfig(
            top_n=1, rebalance_every=2, commission_bps=100, slippage_bps=0,
        )
        result = run_backtest(prices, predictions, "score", config)
        switch = result.trades.loc[result.trades.date == dates[3]]
        self.assertEqual(set(switch.symbol), {"A", "B"})
        self.assertLess(switch.set_index("symbol").loc["A", "notional"], 0)
        self.assertGreater(switch.set_index("symbol").loc["B", "notional"], 0)
        self.assertGreater(result.daily.loc[dates[3], "turnover"], 1.9)
        self.assertAlmostEqual(result.daily.equity.iloc[-1], (.99 / 1.01) ** 2)

    def test_all_ties_hold_cash_and_a_later_abstention_sells_existing_positions(self):
        prices, predictions = self.frames(stock_a=(100,) * 6)
        tied = self.simulate(prices, predictions.assign(score=1))
        self.assertTrue(tied.trades.empty)
        self.assertAlmostEqual(tied.daily.equity.iloc[-1], 1)
        self.assertTrue(np.isnan(backtest_statistics(tied)["sharpe"]))
        predictions.loc[predictions.date >= prices.date.unique()[2], "score"] = 1
        result = self.simulate(prices, predictions, rebalance_every=2)
        self.assertAlmostEqual(result.weights.iloc[3].sum(), 0)
        self.assertEqual(len(result.trades), 2)

    def test_future_scores_and_targets_cannot_change_earlier_positions(self):
        prices, predictions = self.frames(stock_a=(100,) * 8)
        original = self.simulate(prices, predictions, rebalance_every=2)
        modified = predictions.assign(forward_return_5d=999, target_percentile_5d=-999)
        modified.loc[modified.date >= prices.date.unique()[4], "score"] *= -1
        changed = self.simulate(prices, modified, rebalance_every=2)
        assert_frame_equal(original.weights.iloc[:5], changed.weights.iloc[:5])
        assert_frame_equal(original.daily.iloc[:5], changed.daily.iloc[:5])

    def test_missing_prices_signals_duplicates_and_invalid_values_fail(self):
        prices, predictions = self.frames()
        cases = [
            (prices.iloc[:-1], predictions, "finite"),
            (pd.concat([prices, prices.iloc[:1]]), predictions, "unique"),
            (prices.assign(adjusted_close=0), predictions, "positive"),
            (prices, predictions.assign(score=np.inf), "finite"),
            (prices, predictions.iloc[:-1], "finite"),
        ]
        for price_rows, score_rows, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.simulate(price_rows, score_rows)
        # Entire scheduled signal dates must not silently disappear.
        date = predictions.date.unique()[1]
        with self.assertRaisesRegex(ValueError, "scheduled signal"):
            self.simulate(prices, predictions.loc[predictions.date != date], rebalance_every=1)

    def test_universe_benchmark_invests_despite_tied_scores_and_uses_same_costs(self):
        prices, predictions = self.frames(stock_a=(100, 100, 200, 400))
        result = run_backtest(
            prices, predictions.assign(score=1), "score",
            BacktestConfig(commission_bps=50, slippage_bps=50), universe_benchmark=True,
        )
        self.assertAlmostEqual(result.daily.equity.iloc[-1], 2.5 * .99 / 1.01)

    def test_drawdown_includes_initial_capital_and_first_entry_fee(self):
        prices, predictions = self.frames(stock_a=(100, 100, 50, 75))
        result = self.simulate(prices, predictions)
        stats = backtest_statistics(result)
        self.assertAlmostEqual(stats["max_drawdown"], -.5)
        self.assertAlmostEqual(stats["net_return"], -.25)
        self.assertEqual(stats["return_day_count"], 3)
        self.assertAlmostEqual(stats["profitable_day_fraction"], 1 / 3)
        self.assertAlmostEqual(stats["annual_return"], .75 ** (252 / 3) - 1)
        returns = pd.Series([0, -.5, .5])
        self.assertAlmostEqual(stats["annual_volatility"], returns.std() * np.sqrt(252))

    def test_cutoff_ties_are_equal_weighted_and_invalid_settings_fail(self):
        scores = pd.Series([5, 4, 3, 3, 1], index=list("ABCDE"))
        self.assertEqual(top_stock_weights(scores).tolist(), [.25, .25, .25, .25, 0])
        for settings in (
            {"top_n": 0}, {"top_n": True}, {"rebalance_every": 0},
            {"commission_bps": -1}, {"slippage_bps": np.nan},
            {"annual_risk_free_rate": -1}, {"annual_risk_free_rate": np.inf},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValidationError):
                BacktestConfig(**settings)


if __name__ == "__main__":
    unittest.main()
