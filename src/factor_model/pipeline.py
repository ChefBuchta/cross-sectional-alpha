"""Coordinate feature preparation, target construction, and model evaluation."""

import numpy as np
import pandas as pd

from factor_model.features import build_features
from factor_model.models import ExperimentConfig, ExperimentResult
from factor_model.targets import build_targets
from factor_model.training import (
    evaluate_model,
    select_gradient_boosting_model,
    select_random_forest_model,
    select_ridge_model,
)
from factor_model.validation import MODEL_FEATURE_COLUMNS, split_labeled_data


def run_experiment(
    prices: pd.DataFrame,
    config: ExperimentConfig,
    *,
    evaluate_test: bool = False,
) -> ExperimentResult:
    """Run the experiment stages and return their results.

    Prices are loaded by the caller. Keep the supplied DataFrame unchanged.
    Test evaluation is optional and must not influence model selection.
    """
    valid_prices = prices[(prices.date >= config.start_date) & (prices.date < config.end_date)].copy()
    features = build_features(valid_prices).copy()
    valid_features = features[MODEL_FEATURE_COLUMNS].dropna
    df = build_targets(valid_prices, valid_features, config.target_horizon)
    splits = split_labeled_data(df, config)
    model_searches = [
        ("ridge", select_ridge_model, evaluate_model),
        ("random_forest", select_random_forest_model, evaluate_model),
        ("gradient_boosting", select_gradient_boosting_model, evaluate_model),
    ]
    target_column = f"target_percentile_{config.target_horizon}d"
    best_score = -np.inf
    selection = None
    validation = None
    best_evaluator = None

    for model_name, search_model, evaluate_model in model_searches:
        candidate = search_model(
            splits.train,
            splits.valid,
            feature_columns=MODEL_FEATURE_COLUMNS,
            target_column=target_column,
        )

        candidate_validation = evaluate_model(candidate, splits.valid)
        score = candidate_validation.summary.loc[model_name, "mean"]
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            selection = candidate
            validation = candidate_validation
            best_evaluator = evaluate_model

    if selection is None or validation is None or best_evaluator is None:
        raise ValueError("No model has a finite mean validation correlation")

    test = None
    if evaluate_test:
        test = best_evaluator(selection, splits.test)

    # 8. Return ExperimentResult with config, selection, validation, and test.
    raise NotImplementedError("Fill in the experiment stages above")
