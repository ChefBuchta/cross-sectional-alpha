"""Measure how well stock scores rank future outcomes."""

import pandas as pd

from factor_model.data_checks import (
    check_finite_columns,
    check_no_missing_values,
    check_not_empty,
    check_required_columns,
    check_unique_keys,
)


def daily_spearman_ic(
    df: pd.DataFrame,
    score_column: str = "baseline_score",
    target_column: str = "target_percentile_5d",
) -> pd.Series:
    """Return one Spearman rank correlation across stocks for each date."""
    required_columns = {"date", "symbol", score_column, target_column}
    check_required_columns(df, required_columns)
    check_not_empty(df, "cannot evaluate an empty table")
    check_no_missing_values(df, ["date", "symbol"], "dates and symbols must be present")
    check_unique_keys(df, ["date", "symbol"], "date and symbol pairs must be unique")
    check_finite_columns(
        df, [score_column, target_column], "scores and targets must be finite"
    )

    correlations = {}
    for date, stocks in df.groupby("date", sort=True):
        scores = stocks[score_column]
        targets = stocks[target_column]
        if len(stocks) < 2 or scores.nunique() < 2 or targets.nunique() < 2:
            correlations[date] = float("nan")
        else:
            correlations[date] = scores.corr(targets, method="spearman")

    return pd.Series(correlations, name="spearman_ic", dtype=float)


def summarize_daily_ic(daily_ic: pd.Series) -> dict[str, float | int]:
    """Summarize defined daily correlations and count undefined dates."""
    defined = daily_ic.dropna()
    return {
        "mean": float(defined.mean()),
        "median": float(defined.median()),
        "fraction_positive": float(defined.gt(0).mean()),
        "defined_day_count": int(defined.size),
        "undefined_day_count": int(daily_ic.isna().sum()),
    }
