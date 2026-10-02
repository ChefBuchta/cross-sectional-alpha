"""Download historical daily market data from Yahoo Finance."""

from collections.abc import Sequence
from datetime import date

import pandas as pd
import yfinance as yf


class YahooFinanceDataError(RuntimeError):
    """Raised when Yahoo Finance does not return usable market data."""


class YahooFinanceDataLoader:
    """Download OHLCV data and return one row per date and symbol."""

    _COLUMN_NAMES = {  # noqa: RUF012
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }

    def __init__(self, session=None, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.session = session

    def load(
        self,
        symbols: Sequence[str],
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Download daily, unadjusted OHLCV data for the requested symbols.

        Yahoo treats ``end_date`` as exclusive. For example, an end date of
        ``2025-01-10`` returns observations strictly before that date.
        """
        normalized_symbols = self._normalize_symbols(symbols)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        if start >= end:
            raise ValueError("start_date must be earlier than end_date")

        market_data = yf.download(
            tickers=list(normalized_symbols),
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            progress=False,
            threads=True,
            timeout=self.timeout_seconds,
            session=self.session,
            multi_level_index=True,
        )

        if market_data is None or market_data.empty:
            raise YahooFinanceDataError("Yahoo Finance returned no data for the requested symbols and dates.")

        return self._to_long_format(market_data, normalized_symbols)

    @staticmethod
    def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip()))
        if not normalized:
            raise ValueError("symbols must contain at least one non-empty ticker")
        return normalized

    def _to_long_format(
        self,
        market_data: pd.DataFrame,
        symbols: tuple[str, ...],
    ) -> pd.DataFrame:
        if not isinstance(market_data.columns, pd.MultiIndex):
            if len(symbols) != 1:
                raise YahooFinanceDataError("Yahoo Finance returned unexpected columns for multiple symbols.")
            market_data = pd.concat({symbols[0]: market_data}, axis="columns")

        available_symbols = set(market_data.columns.get_level_values(0))
        frames: list[pd.DataFrame] = []

        for symbol in symbols:
            if symbol not in available_symbols:
                continue
            symbol_data = market_data[symbol].rename(columns=self._COLUMN_NAMES).copy()
            symbol_data.index.name = "date"
            symbol_data["symbol"] = symbol
            frames.append(symbol_data.reset_index())

        if not frames:
            raise YahooFinanceDataError("Yahoo Finance did not return data for any requested symbol.")

        result = pd.concat(frames, ignore_index=True)
        expected_columns = [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ]
        missing_columns = set(expected_columns) - set(result.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise YahooFinanceDataError(f"Yahoo Finance data is missing columns: {missing}")

        return result.loc[:, expected_columns].sort_values(
            ["date", "symbol"],
            ignore_index=True,
        )
