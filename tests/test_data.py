import unittest
from unittest.mock import patch

import pandas as pd

from factor_model.data import YahooFinanceDataError, YahooFinanceDataLoader


class YahooFinanceDataLoaderTest(unittest.TestCase):
  def setUp(self) -> None:
    self.loader = YahooFinanceDataLoader()

  @patch("factor_model.data.yf.download")
  def test_load_returns_long_format_for_multiple_symbols(self, download) -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = pd.MultiIndex.from_product(
      [["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    download.return_value = pd.DataFrame(
      [[100, 105, 99, 104, 104, 1_000] * 2] * 2,
      index=dates,
      columns=columns,
    )

    result = self.loader.load(
      ["aapl", "MSFT", "AAPL"],
      "2025-01-01",
      "2025-01-04",
    )

    self.assertEqual(len(result), 4)
    self.assertEqual(set(result["symbol"]), {"AAPL", "MSFT"})
    self.assertEqual(
      list(result.columns),
      [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
      ],
    )
    download.assert_called_once()

  @patch("factor_model.data.yf.download", return_value=pd.DataFrame())
  def test_load_rejects_empty_download(self, download) -> None:
    with self.assertRaisesRegex(YahooFinanceDataError, "returned no data"):
      self.loader.load(["AAPL"], "2025-01-01", "2025-01-04")

  def test_load_rejects_empty_symbols(self) -> None:
    with self.assertRaisesRegex(ValueError, "symbols"):
      self.loader.load([], "2025-01-01", "2025-01-04")

  def test_load_rejects_reversed_dates(self) -> None:
    with self.assertRaisesRegex(ValueError, "start_date"):
      self.loader.load(["AAPL"], "2025-01-04", "2025-01-01")


if __name__ == "__main__":
  unittest.main()
