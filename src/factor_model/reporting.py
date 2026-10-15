"""Prediction snapshots and honest, non-portfolio research summaries for the UI."""

import numpy as np
import pandas as pd

from factor_model.features import build_features
from factor_model.models import ExperimentResult, ModelEvaluation, ModelRun, ModelSelection


def retained_models(result: ExperimentResult) -> dict[str, ModelRun]:
    """Also support an older session that retained only its winning family."""
    return result.model_runs or {
        result.selection.model_name: ModelRun(selection=result.selection, validation=result.validation)
    }


def rank_stocks(rows: pd.DataFrame, score_column: str, limit: int = 3) -> pd.DataFrame:
    """Rank independently on each date; cutoff ties stay together, all ties abstain."""
    if limit < 1:
        raise ValueError("shortlist size must be positive")
    if rows.duplicated(["date", "symbol"]).any():
        raise ValueError("prediction date and symbol pairs must be unique")
    if not np.isfinite(rows[score_column].to_numpy()).all():
        raise ValueError("prediction scores must be finite")
    ranked = rows.copy()
    ranked["date"] = pd.to_datetime(ranked["date"])
    groups = ranked.groupby("date")[score_column]
    ranked["predicted_rank"] = groups.rank(method="average", ascending=False)
    ranked["shortlisted"] = (
        (groups.rank(method="min", ascending=False) <= limit)
        & (groups.transform("nunique") > 1)
    )
    return ranked.sort_values(["date", score_column, "symbol"], ascending=[True, False, True])


def snapshot_dates(rows: pd.DataFrame, cadence: str) -> pd.DataFrame:
    """Last available observation per calendar period, never an average of future scores."""
    dates = pd.Series(pd.to_datetime(rows["date"]).unique()).sort_values()
    if cadence == "Day":
        periods = dates.dt.strftime("%Y-%m-%d")
    elif cadence == "Week":
        periods = "Week of " + dates.dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    elif cadence == "Month":
        periods = dates.dt.strftime("%Y-%m")
    else:
        raise ValueError("cadence must be Day, Week, or Month")
    schedule = pd.DataFrame({"Period": periods, "As of": dates})
    return schedule.groupby("Period", sort=False, as_index=False)["As of"].max().sort_values("As of")


def shortlist_history(rows: pd.DataFrame, score_column: str, cadence: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["Period", "As of", "Most attractive", "Top research group", "Stocks in group"])
    schedule = snapshot_dates(rows, cadence)
    ranked = rank_stocks(rows, score_column)
    records = []
    for period, as_of in schedule.itertuples(index=False, name=None):
        day = ranked.loc[ranked["date"] == as_of]
        chosen = day.loc[day["shortlisted"]]
        best = day.loc[day["predicted_rank"] == day["predicted_rank"].min(), "symbol"]
        records.append({
            "Period": period, "As of": as_of,
            "Most attractive": ", ".join(best) if not chosen.empty else "No preference (all tied)",
            "Top research group": ", ".join(chosen["symbol"]),
            "Stocks in group": len(chosen),
        })
    return pd.DataFrame(records).sort_values("As of", ascending=False)


def forecast_from_prices(prices: pd.DataFrame, result: ExperimentResult) -> dict[str, pd.DataFrame]:
    """Score post-selection features, including rows without known future targets.

    The fitted models are never refitted here. Future returns are neither required
    nor used, and training/selection dates are excluded from this forecast history.
    """
    config = result.config
    dates = pd.to_datetime(prices["date"])
    history = prices.loc[
        prices["symbol"].isin(config.symbols)
        & (dates >= pd.Timestamp(config.start_date))
        & (dates < pd.Timestamp(config.end_date))
    ].copy()
    history["date"] = pd.to_datetime(history["date"])
    features = build_features(history)
    forecasts = {}
    for name, run in retained_models(result).items():
        columns = list(run.selection.feature_columns)
        eligible = features.loc[
            (features["date"] >= pd.Timestamp(config.test_start))
            & np.isfinite(features[columns]).all(axis=1)
        ].copy()
        # Keep the declared universe intact instead of silently changing the competition.
        counts = eligible.groupby("date")["symbol"].transform("nunique")
        eligible = eligible.loc[counts == len(config.symbols)].copy()
        if not eligible.empty:
            eligible[f"{name}_score"] = run.selection.pipeline.predict(eligible[columns])
            forecasts[name] = rank_stocks(eligible, f"{name}_score")
    return forecasts


def score_explanations(selection: ModelSelection, rows: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Model-score explanations only: exact linear differences or local sensitivity."""
    columns = list(selection.feature_columns)
    inputs = rows[columns]
    estimator = selection.pipeline.steps[-1][1]
    if selection.model_name == "ridge" and hasattr(estimator, "coef_"):
        transformed = selection.pipeline[:-1].transform(inputs)
        values = transformed * np.ravel(estimator.coef_)
        values -= values.mean(axis=0)
        kind = "Exact score contributions"
    else:
        actual = selection.pipeline.predict(inputs)
        values = np.empty((len(rows), len(columns)))
        for index, feature in enumerate(columns):
            hypothetical = inputs.copy()
            hypothetical[feature] = inputs[feature].mean()
            values[:, index] = actual - selection.pipeline.predict(hypothetical)
        kind = "Single-feature sensitivity"
    return pd.DataFrame(values, index=rows["symbol"], columns=columns), kind


def selection_performance(rows: pd.DataFrame, score_column: str, horizon: int) -> pd.DataFrame:
    """Forward-window outcomes, NOT an executable backtest or compounded wealth."""
    return_column = f"forward_return_{horizon}d"
    if return_column not in rows:
        return pd.DataFrame(columns=["shortlist_return", "universe_return", "selection_edge", "top_bottom_spread"])
    ranked = rank_stocks(rows, score_column)
    bottom = ranked.groupby("date")[score_column].rank(method="min", ascending=True) <= 3
    universe = ranked.groupby("date")[return_column].mean()
    top = ranked.loc[ranked["shortlisted"]].groupby("date")[return_column].mean()
    low = ranked.loc[bottom].groupby("date")[return_column].mean()
    return pd.DataFrame({
        "shortlist_return": top.reindex(universe.index),
        "universe_return": universe,
        "selection_edge": top - universe,
        "top_bottom_spread": top - low,
    }).sort_index()


def model_statistics(evaluation: ModelEvaluation, name: str, horizon: int) -> dict[str, float | int]:
    summary = evaluation.summary.loc[name].to_dict()
    performance = selection_performance(evaluation.scored_rows, f"{name}_score", horizon)
    edges = performance["selection_edge"].dropna()
    summary.update({
        "mean_shortlist_return": float(performance["shortlist_return"].mean()),
        "mean_selection_edge": float(edges.mean()),
        "fraction_positive_edge": float(edges.gt(0).mean()) if len(edges) else float("nan"),
        "mean_top_bottom_spread": float(performance["top_bottom_spread"].mean()),
        "selection_date_count": int(len(edges)),
    })
    return summary


def model_leaderboard(result: ExperimentResult) -> pd.DataFrame:
    """Rank all retained families by the recorded validation selection criterion."""
    runs = retained_models(result)
    reference_keys = set(map(tuple, result.validation.scored_rows[["date", "symbol"]].to_numpy()))
    records = []
    for name, run in runs.items():
        keys = set(map(tuple, run.validation.scored_rows[["date", "symbol"]].to_numpy()))
        if keys != reference_keys:
            raise ValueError("Model comparison requires the same validation dates and stocks for every model")
        records.append({"model": name, **model_statistics(run.validation, name, result.config.target_horizon)})
    records.append({"model": "baseline", **model_statistics(result.validation, "baseline", result.config.target_horizon)})
    table = pd.DataFrame(records)
    table["performance_rank"] = table["mean"].rank(method="min", ascending=False)
    return table.sort_values(["mean", "model"], ascending=[False, True]).reset_index(drop=True)
