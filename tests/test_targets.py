import unittest

import pandas as pd

from factor_model.targets import build_targets


class BuildTargetsTest(unittest.TestCase):
    def test_uses_full_price_history_before_joining_eligible_rows(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=8)
        prices = pd.DataFrame(
            [
                {
                    "date": date,
                    "symbol": symbol,
                    "adjusted_close": start + day * daily_gain,
                }
                for day, date in enumerate(dates)
                for symbol, start, daily_gain in (
                    ("MSFT", 200, 3),
                    ("AAPL", 100, 1),
                )
            ]
        )
        eligible = prices.loc[
            prices["date"].isin(dates[[2, 4, 6, 7]])
        ].copy()
        original_prices = prices.copy(deep=True)
        original_eligible = eligible.copy(deep=True)

        labeled = build_targets(prices, eligible, horizon=2)
        first_date = labeled.loc[labeled["date"] == dates[2]].set_index("symbol")

        pd.testing.assert_frame_equal(prices, original_prices)
        pd.testing.assert_frame_equal(eligible, original_eligible)
        self.assertEqual(len(labeled), 4)
        self.assertEqual(set(labeled["date"]), {dates[2], dates[4]})
        self.assertAlmostEqual(first_date.loc["AAPL", "forward_return_2d"], 104 / 102 - 1)
        self.assertEqual(first_date.loc["AAPL", "label_end_date"], dates[4])
        self.assertAlmostEqual(first_date.loc["AAPL", "target_percentile_2d"], 0.5)
        self.assertAlmostEqual(first_date.loc["MSFT", "target_percentile_2d"], 1.0)

    def test_rejects_nonpositive_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizon must be positive"):
            build_targets(pd.DataFrame(), pd.DataFrame(), horizon=0)


if __name__ == "__main__":
    unittest.main()
