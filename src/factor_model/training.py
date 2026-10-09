"""Train Ridge candidates on the past and select using validation dates only."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from factor_model.data_checks import (
    check_finite_columns,
    check_no_missing_values,
    check_not_empty,
    check_required_columns,
    check_unique_keys,
)
from factor_model.metrics import daily_spearman_ic, summarize_daily_ic
from factor_model.models import ModelEvaluation, RidgeSelection
from factor_model.validation import MODEL_FEATURE_COLUMNS


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Create an unfitted scaler and Ridge model with independent state."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def select_ridge_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    alphas: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    target_column: str = "target_percentile_5d",
) -> RidgeSelection:
    """Fit each candidate on train; maximize mean daily validation Spearman.

    Inputs should come from split_labeled_data. Test rows are intentionally
    not accepted. Exact ties prefer larger alpha; the winner is not refitted.
    """
    columns = list(feature_columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("feature columns must be non-empty and unique")
    if target_column in columns:
        raise ValueError("the target cannot also be a model feature")
    alpha_values = tuple(float(alpha) for alpha in alphas)
    if not alpha_values:
        raise ValueError("at least one alpha is required")
    if any(not np.isfinite(alpha) or alpha < 0 for alpha in alpha_values):
        raise ValueError("alphas must be finite and non-negative")
    if len(alpha_values) != len(set(alpha_values)):
        raise ValueError("alphas must be unique")

    for name, rows in (("train", train), ("valid", valid)):
        check_required_columns(
            rows, ["date", "symbol", "label_end_date", target_column, *columns]
        )
        check_not_empty(rows, f"{name} data is empty")
        check_no_missing_values(
            rows, ["date", "symbol", "label_end_date"],
            f"{name} has missing keys or label end dates",
        )
        check_unique_keys(rows, ["date", "symbol"], f"{name} has duplicate keys")
        check_finite_columns(
            rows, [*columns, target_column], f"{name} features and targets must be finite"
        )

    validation_start = pd.to_datetime(valid["date"]).min()
    if not (pd.to_datetime(train["date"]) < validation_start).all():
        raise ValueError("training dates must precede validation dates")
    if not (pd.to_datetime(train["label_end_date"]) < validation_start).all():
        raise ValueError("training labels must end before validation starts")

    results = []
    candidate_models = {}
    for alpha in alpha_values:
        candidate = build_ridge_pipeline(alpha)
        candidate.fit(train[columns], train[target_column])

        validation_results = valid.copy()
        validation_results["ridge_score"] = candidate.predict(valid[columns])
        daily_ic = daily_spearman_ic(
            validation_results, score_column="ridge_score", target_column=target_column
        )
        results.append({"alpha": alpha, **summarize_daily_ic(daily_ic)})
        candidate_models[alpha] = candidate

    comparison = pd.DataFrame(results)
    ranked = comparison.loc[np.isfinite(comparison["mean"])].sort_values(
        ["mean", "alpha"], ascending=[False, False]
    )
    if ranked.empty:
        raise ValueError("no candidate has a defined validation mean correlation")

    best_alpha = float(ranked.iloc[0]["alpha"])
    return RidgeSelection(
        best_alpha=best_alpha,
        pipeline=candidate_models[best_alpha],
        feature_columns=tuple(columns),
        target_column=target_column,
        comparison=comparison,
    )


def evaluate_ridge_model(
    selection: RidgeSelection, rows: pd.DataFrame
) -> ModelEvaluation:
    """Score labeled rows and compare Ridge with the baseline without fitting.

    Call separately for validation or the final test period. This function
    never changes alpha, scaling statistics, or model weights.
    """
    columns = list(selection.feature_columns)
    check_required_columns(
        rows, ["date", "symbol", "baseline_score", selection.target_column, *columns]
    )
    check_not_empty(rows, "evaluation data is empty")
    check_finite_columns(rows, columns, "evaluation features must be finite")
    scored_rows = rows.copy()
    scored_rows["ridge_score"] = selection.pipeline.predict(rows[columns])

    daily_correlations = {}
    summaries = {}
    for name, score_column in (
        ("baseline", "baseline_score"), ("ridge", "ridge_score")
    ):
        daily_ic = daily_spearman_ic(
            scored_rows, score_column=score_column, target_column=selection.target_column
        )
        daily_correlations[name] = daily_ic
        summaries[name] = summarize_daily_ic(daily_ic)

    return ModelEvaluation(
        scored_rows=scored_rows,
        daily_ic=pd.DataFrame(daily_correlations),
        summary=pd.DataFrame(summaries).T,
    )
