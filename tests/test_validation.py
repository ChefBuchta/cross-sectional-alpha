import unittest
from datetime import date

import pandas as pd

from factor_model.config import ExperimentConfig
from factor_model.validation import split_labeled_data


class SplitLabeledDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            symbols=("AAPL", "MSFT"),
            start_date=date(2024, 12, 1),
            validation_start=date(2025, 1, 6),
            test_start=date(2025, 1, 10),
            end_date=date(2025, 1, 20),
            target_horizon=2,
        )
        dates = pd.to_datetime(
            [
                "2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03",
                "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09",
                "2025-01-10", "2025-01-13", "2025-01-14",
            ]
        )
        self.labeled = pd.DataFrame(
            [
                {
                    "date": prediction_date,
                    "symbol": symbol,
                    "label_end_date": dates[day + 2],
                    "trailing_return_5d": 0.01,
                    "trailing_return_20d": 0.02,
                    "volatility_20d": 0.03,
                    "relative_volume_20d": 1.0,
                    "target_percentile_2d": 0.5 if symbol == "AAPL" else 1.0,
                }
                for day, prediction_date in enumerate(dates[:9])
                for symbol in self.config.symbols
            ]
        )

    def test_purges_boundary_labels_and_preserves_source(self) -> None:
        original = self.labeled.copy(deep=True)

        splits = split_labeled_data(self.labeled, self.config)

        pd.testing.assert_frame_equal(self.labeled, original)
        self.assertEqual([len(splits[name]) for name in splits], [4, 4, 2])
        self.assertEqual(
            set(splits["train"]["date"]),
            set(pd.to_datetime(["2024-12-30", "2024-12-31"])),
        )
        self.assertEqual(
            set(splits["validation"]["date"]),
            set(pd.to_datetime(["2025-01-06", "2025-01-07"])),
        )
        self.assertTrue(
            (splits["train"]["label_end_date"] < pd.Timestamp("2025-01-06")).all()
        )
        self.assertTrue(
            (splits["validation"]["label_end_date"] < pd.Timestamp("2025-01-10")).all()
        )

    def test_rejects_incomplete_daily_universe(self) -> None:
        incomplete = self.labeled.loc[
            ~((self.labeled["date"] == pd.Timestamp("2025-01-07"))
              & (self.labeled["symbol"] == "MSFT"))
        ]

        with self.assertRaisesRegex(ValueError, "incomplete daily universe"):
            split_labeled_data(incomplete, self.config)


if __name__ == "__main__":
    unittest.main()
