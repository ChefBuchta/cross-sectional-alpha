import unittest

import numpy as np
import pandas as pd

from factor_model.data_checks import (
    check_daily_universe,
    check_finite_columns,
    check_no_missing_values,
    check_not_empty,
    check_required_columns,
    check_unique_keys,
)


class DataChecksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-02"] * 2 + ["2025-01-03"] * 2),
            "symbol": ["AAPL", "MSFT"] * 2,
            "score": [0.1, 0.2, 0.3, 0.4],
        })

    def test_valid_checks_do_not_modify_source(self) -> None:
        original = self.df.copy(deep=True)
        check_required_columns(self.df, ["date", "symbol", "score"])
        check_not_empty(self.df, "empty")
        check_no_missing_values(self.df, ["date", "symbol"], "missing")
        check_unique_keys(self.df, ["date", "symbol"], "duplicates")
        check_finite_columns(self.df, ["score"], "nonfinite")
        check_daily_universe(self.df, ["AAPL", "MSFT"], "incomplete")
        pd.testing.assert_frame_equal(self.df, original)

    def test_missing_columns_are_reported_in_sorted_order(self) -> None:
        with self.assertRaisesRegex(ValueError, r"missing columns: \['a', 'z'\]"):
            check_required_columns(self.df, ["z", "a"])

    def test_empty_table_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            check_not_empty(self.df.iloc[:0], "empty")

    def test_missing_key_is_rejected(self) -> None:
        self.df.loc[0, "date"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "missing"):
            check_no_missing_values(self.df, ["date", "symbol"], "missing")

    def test_duplicate_pair_is_rejected_but_repeated_dates_are_valid(self) -> None:
        check_unique_keys(self.df, ["date", "symbol"], "duplicates")
        duplicated = pd.concat([self.df, self.df.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            check_unique_keys(duplicated, ["date", "symbol"], "duplicates")

    def test_nan_and_both_infinities_are_rejected(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.df.loc[0, "score"] = value
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    check_finite_columns(self.df, ["score"], "nonfinite")

    def test_missing_or_unexpected_daily_symbol_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            check_daily_universe(self.df.iloc[1:], ["AAPL", "MSFT"], "incomplete")
        self.df.loc[0, "symbol"] = "JPM"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            check_daily_universe(self.df, ["AAPL", "MSFT"], "incomplete")


if __name__ == "__main__":
    unittest.main()
