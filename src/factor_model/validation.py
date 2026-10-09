"""Create chronological model evaluation splits without overlapping labels."""

from collections.abc import Sequence

import pandas as pd

from factor_model.data_checks import (
    check_daily_universe,
    check_finite_columns,
    check_no_missing_values,
    check_not_empty,
    check_required_columns,
    check_unique_keys,
)
from factor_model.models import ExperimentConfig, ValidationSplits

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
) -> ValidationSplits:
    """Split by prediction date and purge labels crossing the next boundary.

    Each result retains feature, target, and metadata columns. Select model
    inputs and targets from these tables when fitting or evaluating a model.
    """
    target_column = f"target_percentile_{config.target_horizon}d"
    required_columns = {
        "date", "symbol", "label_end_date", target_column, *feature_columns
    }
    check_required_columns(labeled, required_columns, "labeled data is missing columns")

    df = labeled.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["label_end_date"] = pd.to_datetime(df["label_end_date"])

    check_not_empty(df, "labeled data is empty")
    check_no_missing_values(
        df, ["date", "symbol", "label_end_date"],
        "labeled data has missing keys or label end dates",
    )
    check_unique_keys(
        df, ["date", "symbol"], "labeled data has duplicate date and symbol keys"
    )
    if not (df["label_end_date"] > df["date"]).all():
        raise ValueError("label end dates must follow prediction dates")
    check_finite_columns(
        df, [*feature_columns, target_column], "model inputs and targets must be finite"
    )

    expected_symbols = set(config.symbols)
    if set(df["symbol"]) != expected_symbols:
        raise ValueError("labeled data symbols do not match the experiment universe")

    validation_start = pd.Timestamp(config.validation_start)
    test_start = pd.Timestamp(config.test_start)
    splits = ValidationSplits(
        train=df.loc[
            (df["date"] < validation_start)
            & (df["label_end_date"] < validation_start)
        ].copy(),
        valid=df.loc[
            (df["date"] >= validation_start)
            & (df["date"] < test_start)
            & (df["label_end_date"] < test_start)
        ].copy(),
        test=df.loc[df["date"] >= test_start].copy(),
    )

    for name, split in (
        ("train", splits.train),
        ("valid", splits.valid),
        ("test", splits.test),
    ):
        check_not_empty(split, f"{name} split is empty")
        check_daily_universe(
            split, expected_symbols, f"{name} split has an incomplete daily universe"
        )

    return splits
