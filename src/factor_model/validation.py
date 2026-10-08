"""Create chronological model evaluation splits without overlapping labels."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from factor_model.config import ExperimentConfig


MODEL_FEATURE_COLUMNS = (
    "trailing_return_5d",
    "trailing_return_20d",
    "volatility_20d",
    "relative_volume_20d",
)


def split_labeled_data(
    labeled: pd.DataFrame,
    config: ExperimentConfig,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> dict[str, pd.DataFrame]:
    """Split by prediction date and purge labels crossing the next boundary.

    Each result retains feature, target, and metadata columns. Select model
    inputs and targets from these tables when fitting or evaluating a model.
    """
    target_column = f"target_percentile_{config.target_horizon}d"
    required_columns = {
        "date", "symbol", "label_end_date", target_column, *feature_columns
    }
    missing_columns = required_columns - set(labeled.columns)
    if missing_columns:
        raise ValueError(f"labeled data is missing columns: {sorted(missing_columns)}")

    frame = labeled.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["label_end_date"] = pd.to_datetime(frame["label_end_date"])
    if frame.empty:
        raise ValueError("labeled data is empty")
    if frame[["date", "symbol", "label_end_date"]].isna().any().any():
        raise ValueError("labeled data has missing keys or label end dates")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("labeled data has duplicate date and symbol keys")
    if not (frame["label_end_date"] > frame["date"]).all():
        raise ValueError("label end dates must follow prediction dates")
    if not np.isfinite(frame[[*feature_columns, target_column]].to_numpy()).all():
        raise ValueError("model inputs and targets must be finite")

    expected_symbols = set(config.symbols)
    if set(frame["symbol"]) != expected_symbols:
        raise ValueError("labeled data symbols do not match the experiment universe")

    validation_start = pd.Timestamp(config.validation_start)
    test_start = pd.Timestamp(config.test_start)
    splits = {
        "train": frame.loc[
            (frame["date"] < validation_start)
            & (frame["label_end_date"] < validation_start)
        ].copy(),
        "validation": frame.loc[
            (frame["date"] >= validation_start)
            & (frame["date"] < test_start)
            & (frame["label_end_date"] < test_start)
        ].copy(),
        "test": frame.loc[frame["date"] >= test_start].copy(),
    }

    for name, split in splits.items():
        if split.empty:
            raise ValueError(f"{name} split is empty")
        daily_symbols = split.groupby("date")["symbol"].agg(set)
        if not daily_symbols.apply(lambda symbols: symbols == expected_symbols).all():
            raise ValueError(f"{name} split has an incomplete daily universe")

    return splits
