"""Reusable, read-only checks for research DataFrames."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def check_required_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    message: str = "missing columns",
) -> None:
    """Require every named column before accessing it."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{message}: {sorted(missing)}")


def check_not_empty(df: pd.DataFrame, message: str) -> None:
    """Reject a table with no rows or no columns."""
    if df.empty:
        raise ValueError(message)


def check_no_missing_values(
    df: pd.DataFrame, columns: Iterable[str], message: str
) -> None:
    """Reject missing values in the selected columns."""
    if df[list(columns)].isna().any().any():
        raise ValueError(message)


def check_unique_keys(
    df: pd.DataFrame, columns: Iterable[str], message: str
) -> None:
    """Require each combination of key values to occur only once."""
    if df.duplicated(list(columns)).any():
        raise ValueError(message)


def check_finite_columns(
    df: pd.DataFrame, columns: Iterable[str], message: str
) -> None:
    """Reject NaN and infinity in selected numeric columns."""
    if not np.isfinite(df[list(columns)].to_numpy()).all():
        raise ValueError(message)


def check_daily_universe(
    df: pd.DataFrame, expected_symbols: Iterable[str], message: str
) -> None:
    """Require the exact expected set of stocks on every observed date."""
    expected = set(expected_symbols)
    daily_symbols = df.groupby("date")["symbol"].agg(set)
    if not daily_symbols.apply(lambda symbols: symbols == expected).all():
        raise ValueError(message)
