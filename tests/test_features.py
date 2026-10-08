import unittest

import pandas as pd

from factor_model.features import build_features


class BuildFeaturesTest(unittest.TestCase):
    def test_builds_past_only_features_from_interleaved_prices(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=22)
        rows = []
        for day, date in enumerate(dates):
            for symbol, start, daily_gain in (
                ("MSFT", 200, 3),
                ("AAPL", 100, 1),
            ):
                price = start + day * daily_gain
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "adjusted_close": price,
                        "volume": 200 if symbol == "AAPL" and day == 20 else 100,
                    }
                )
        prices = pd.DataFrame(rows)
        original = prices.copy(deep=True)

        features = build_features(prices)
        aapl = features.loc[features["symbol"] == "AAPL"].reset_index(drop=True)
        msft = features.loc[features["symbol"] == "MSFT"].reset_index(drop=True)

        pd.testing.assert_frame_equal(prices, original)
        self.assertTrue(aapl.loc[:19, "trailing_return_20d"].isna().all())
        self.assertAlmostEqual(aapl.loc[20, "trailing_return_5d"], 120 / 115 - 1)
        self.assertAlmostEqual(aapl.loc[20, "relative_volume_20d"], 2.0)
        self.assertAlmostEqual(aapl.loc[20, "average_dollar_volume_20d"], 11_650)
        self.assertAlmostEqual(aapl.loc[20, "momentum_percentile_5d"], 0.5)
        self.assertAlmostEqual(msft.loc[20, "momentum_percentile_5d"], 1.0)


if __name__ == "__main__":
    unittest.main()
