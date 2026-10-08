"""Build past-only stock features from daily market prices."""

import numpy as np
import pandas as pd


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return price history with the features developed in the EDA notebooks.

    Rows without enough history are retained with missing rolling features.
    """
    features = prices.sort_values(["symbol", "date"]).copy()
    by_symbol = features.groupby("symbol")

    features["normalized_price"] = (
        features["adjusted_close"]
        / by_symbol["adjusted_close"].transform("first")
        * 100
    )
    features["dollar_volume"] = features["close"] * features["volume"]

    for column, periods in (
        ("trailing_return_1d", 1),
        ("trailing_return_5d", 5),
        ("trailing_return_20d", 20),
    ):
        features[column] = by_symbol["adjusted_close"].pct_change(periods=periods)

    for return_column, rank_column in (
        ("trailing_return_1d", "trailing_return_rank_1d"),
        ("trailing_return_5d", "trailing_return_rank_5d"),
        ("trailing_return_20d", "trailing_return_rank_20d"),
    ):
        features[rank_column] = features.groupby("date")[return_column].rank(
            ascending=False
        )

    features["volatility_20d"] = by_symbol["trailing_return_1d"].transform(
        lambda returns: returns.rolling(20, min_periods=20).std()
    )

    average_volume_20 = by_symbol["volume"].transform(
        lambda volume: volume.shift(1).rolling(20, min_periods=20).mean()
    )
    features["relative_volume_20d"] = (
        features["volume"] / average_volume_20.where(average_volume_20 > 0)
    )

    features["average_dollar_volume_20d"] = by_symbol["dollar_volume"].transform(
        lambda dollar_volume: dollar_volume.rolling(20, min_periods=20).mean()
    )

    required_features = [
        "trailing_return_5d",
        "trailing_return_20d",
        "volatility_20d",
        "relative_volume_20d",
    ]
    eligible = np.isfinite(features[required_features]).all(axis=1)
    features.loc[eligible, "momentum_percentile_5d"] = (
        features.loc[eligible]
        .groupby("date")["trailing_return_5d"]
        .rank(method="average", ascending=True, pct=True)
    )

    return features.reset_index(drop=True)
