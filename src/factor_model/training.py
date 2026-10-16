"""Train supervised models and select using chronological validation only."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import ParameterGrid
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
from factor_model.models import ModelEvaluation, ModelSelection, RidgeSelection
from factor_model.validation import MODEL_FEATURE_COLUMNS


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Create an unfitted scaler and Ridge model with independent state."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def _check_training_rows(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> None:
    """Check the shared input and chronology requirements for every family."""
    columns = list(feature_columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("feature columns must be non-empty and unique")
    if target_column in columns:
        raise ValueError("the target cannot also be a model feature")
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


def select_ridge_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    alphas: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    target_column: str = "target_percentile_5d",
) -> RidgeSelection:
    """Fit on train and select by mean daily validation Spearman.

    Exact ties prefer larger alpha. The winner is not refitted.
    """
    _check_training_rows(train, valid, feature_columns, target_column)
    columns = list(feature_columns)
    alpha_values = tuple(float(alpha) for alpha in alphas)
    if not alpha_values:
        raise ValueError("at least one alpha is required")
    if any(not np.isfinite(alpha) or alpha < 0 for alpha in alpha_values):
        raise ValueError("alphas must be finite and non-negative")
    if len(alpha_values) != len(set(alpha_values)):
        raise ValueError("alphas must be unique")

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
        best_parameters={"alpha": best_alpha},
        pipeline=candidate_models[best_alpha],
        feature_columns=tuple(columns),
        target_column=target_column,
        comparison=comparison,
    )


def evaluate_model(
    selection: ModelSelection, rows: pd.DataFrame
) -> ModelEvaluation:
    """Score labeled rows and compare a model with the baseline without fitting.

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
    score_column = f"{selection.model_name}_score"
    scored_rows[score_column] = selection.pipeline.predict(rows[columns])

    daily_correlations = {}
    summaries = {}
    for name, score_column in (
        ("baseline", "baseline_score"), (selection.model_name, score_column)
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


def evaluate_ridge_model(
    selection: RidgeSelection, rows: pd.DataFrame
) -> ModelEvaluation:
    """Backward-compatible notebook entry point for Ridge evaluation."""
    return evaluate_model(selection, rows)


def _select_candidates(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    model_name: str,
    candidates: Sequence[
        tuple[dict[str, float | int | str | bool | None], Pipeline]
    ],
    feature_columns: Sequence[str],
    target_column: str,
) -> ModelSelection:
    """Compare a small fixed grid; exact ties keep the first listed candidate."""
    _check_training_rows(train, valid, feature_columns, target_column)
    columns = list(feature_columns)
    results = []
    best_score = -np.inf
    best_pipeline = None
    best_parameters = None

    for parameters, candidate in candidates:
        candidate.fit(train[columns], train[target_column])
        scored_rows = valid.copy()
        score_column = f"{model_name}_score"
        scored_rows[score_column] = candidate.predict(valid[columns])
        daily_ic = daily_spearman_ic(
            scored_rows, score_column=score_column, target_column=target_column
        )
        summary = summarize_daily_ic(daily_ic)
        results.append({**parameters, **summary})
        score = summary["mean"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_pipeline = candidate
            best_parameters = parameters

    if best_pipeline is None or best_parameters is None:
        raise ValueError("no candidate has a defined validation mean correlation")
    return ModelSelection(
        model_name=model_name,
        best_parameters=best_parameters,
        pipeline=best_pipeline,
        feature_columns=tuple(columns),
        target_column=target_column,
        comparison=pd.DataFrame(results),
    )


def select_random_forest_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    target_column: str = "target_percentile_5d",
    *,
    max_depths: Sequence[int] = (3, 5),
    min_samples_leaves: Sequence[int] = (100, 20),
    n_estimators: int = 200,
    random_state: int = 42,
) -> ModelSelection:
    """Search four shallow forest settings by default, without feature scaling."""
    grid = ParameterGrid({
        "max_depth": list(max_depths),
        "min_samples_leaf": list(min_samples_leaves),
    })
    candidates = []
    for parameters in grid:
        settings = {
            **parameters, "n_estimators": n_estimators, "random_state": random_state,
        }
        pipeline = Pipeline([("random_forest", RandomForestRegressor(**settings))])
        candidates.append((settings, pipeline))
    return _select_candidates(
        train, valid, "random_forest", candidates, feature_columns, target_column
    )


def select_gradient_boosting_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    target_column: str = "target_percentile_5d",
    *,
    max_leaf_nodes: Sequence[int] = (7, 15),
    l2_regularizations: Sequence[float] = (1.0, 0.0),
    max_iter: int = 100,
    min_samples_leaf: int = 20,
    random_state: int = 42,
) -> ModelSelection:
    """Search four boosting settings with no internal random validation split."""
    grid = ParameterGrid({
        "max_leaf_nodes": list(max_leaf_nodes),
        "l2_regularization": list(l2_regularizations),
    })
    candidates = []
    for parameters in grid:
        settings = {
            **parameters,
            "max_iter": max_iter,
            "min_samples_leaf": min_samples_leaf,
            "learning_rate": 0.05,
            "random_state": random_state,
            "early_stopping": False,
        }
        pipeline = Pipeline([
            ("gradient_boosting", HistGradientBoostingRegressor(**settings)),
        ])
        candidates.append((settings, pipeline))
    return _select_candidates(
        train, valid, "gradient_boosting", candidates, feature_columns, target_column
    )


def select_elastic_net_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    target_column: str = "target_percentile_5d",
    *,
    alphas: Sequence[float] = (0.0001, 0.01),
    l1_ratios: Sequence[float] = (0.2, 0.8),
) -> ModelSelection:
    """Compare four mixed L1/L2 penalties with train-only feature scaling."""
    candidates = []
    for parameters in ParameterGrid({"alpha": list(alphas), "l1_ratio": list(l1_ratios)}):
        settings = {**parameters, "max_iter": 10_000, "tol": 1e-6}
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("elastic_net", ElasticNet(**settings)),
        ])
        candidates.append((settings, pipeline))
    return _select_candidates(
        train, valid, "elastic_net", candidates, feature_columns, target_column,
    )


def select_extra_trees_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    target_column: str = "target_percentile_5d",
    *,
    max_depths: Sequence[int] = (3, 5),
    min_samples_leaves: Sequence[int] = (100, 20),
    n_estimators: int = 200,
    random_state: int = 42,
) -> ModelSelection:
    """Compare four shallow randomized-tree ensembles with a fixed seed."""
    candidates = []
    grid = ParameterGrid({
        "max_depth": list(max_depths), "min_samples_leaf": list(min_samples_leaves),
    })
    for parameters in grid:
        settings = {
            **parameters, "n_estimators": n_estimators, "random_state": random_state,
        }
        pipeline = Pipeline([("extra_trees", ExtraTreesRegressor(**settings))])
        candidates.append((settings, pipeline))
    return _select_candidates(
        train, valid, "extra_trees", candidates, feature_columns, target_column,
    )
