"""Build future-return labels for eligible stock feature rows."""

import pandas as pd

from factor_model.data_checks import check_daily_universe, check_finite_columns


def build_targets(
    df: pd.DataFrame,
    eligible_features: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Join forward returns from full price history onto eligible features.

    A horizon counts stock-specific trading observations. Missing outcomes at
    the end of the price history are excluded from the labeled dataset.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    price_history = df[["symbol", "date", "adjusted_close"]].sort_values(
        ["symbol", "date"]
    ).copy()

    by_symbol = price_history.groupby("symbol")
    future_price = by_symbol["adjusted_close"].shift(-horizon)
    price_history["label_end_date"] = by_symbol["date"].shift(-horizon)

    return_column = f"forward_return_{horizon}d"
    percentile_column = f"target_percentile_{horizon}d"
    price_history[return_column] = future_price / price_history["adjusted_close"] - 1

    labeled = eligible_features.merge(
        price_history.loc[:, ["date", "symbol", return_column, "label_end_date"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    labeled = labeled.dropna(subset=[return_column, "label_end_date"]).copy()

    check_finite_columns(labeled, [return_column], "forward returns must be finite")
    if not (labeled["label_end_date"] > labeled["date"]).all():
        raise ValueError("label end dates must follow prediction dates")

    expected_symbols = set(eligible_features["symbol"])
    check_daily_universe(
        labeled, expected_symbols, "some dates have an incomplete target universe"
    )

    labeled[percentile_column] = (
        labeled.groupby("date")[return_column]
        .rank(method="average", ascending=True, pct=True)
    )
    return labeled
