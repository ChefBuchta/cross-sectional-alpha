"""Model-family coordination tests without fitting on historical holdout data."""

from contextlib import ExitStack
from datetime import date
import unittest
from unittest.mock import patch

import pandas as pd

from factor_model.models import (
    ExperimentConfig, ModelEvaluation, ModelSelection, ValidationSplits,
)
from factor_model.pipeline import run_experiment
from factor_model.training import build_ridge_pipeline
from factor_model.validation import MODEL_FEATURE_COLUMNS


class PipelineModelComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            symbols=("AAPL", "MSFT"), start_date=date(2018, 1, 1),
            validation_start=date(2022, 1, 1), test_start=date(2023, 1, 1),
            end_date=date(2025, 1, 1), target_horizon=5,
        )
        self.prices = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "symbol": ["AAPL", "MSFT"],
            **{column: [0.1, 0.2] for column in MODEL_FEATURE_COLUMNS},
        })
        self.splits = ValidationSplits(
            train=pd.DataFrame({"period": ["train"]}),
            valid=pd.DataFrame({"period": ["valid"]}),
            test=pd.DataFrame({"period": ["test"]}),
        )
        self.scores = {"ridge": 0.1, "random_forest": 0.3, "gradient_boosting": 0.2}

    def selection(self, name: str) -> ModelSelection:
        return ModelSelection(
            model_name=name, best_parameters={}, pipeline=build_ridge_pipeline(),
            feature_columns=MODEL_FEATURE_COLUMNS, target_column="target_percentile_5d",
            comparison=pd.DataFrame({"mean": [self.scores[name]]}),
        )

    def evaluation(self, selection, rows):
        name = selection.model_name
        return ModelEvaluation(
            scored_rows=rows.copy(),
            daily_ic=pd.DataFrame({name: [self.scores[name]]}),
            summary=pd.DataFrame({"mean": [0.0, self.scores[name]]}, index=["baseline", name]),
        )

    def run_mocked(self, evaluate_test=False, model_names=None):
        with ExitStack() as stack:
            stack.enter_context(patch("factor_model.pipeline.build_features", return_value=self.prices.copy()))
            stack.enter_context(patch("factor_model.pipeline.build_targets", return_value=pd.DataFrame()))
            stack.enter_context(patch("factor_model.pipeline.split_labeled_data", return_value=self.splits))
            searches = []
            for name in self.scores:
                searches.append(stack.enter_context(patch(
                    f"factor_model.pipeline.select_{name}_model", return_value=self.selection(name)
                )))
            evaluator = stack.enter_context(patch(
                "factor_model.pipeline.evaluate_model", side_effect=self.evaluation
            ))
            options = {} if model_names is None else {"model_names": model_names}
            result = run_experiment(
                self.prices, self.config, evaluate_test=evaluate_test, **options,
            )
        return result, searches, evaluator

    def test_retains_each_family_and_selects_by_validation_mean(self) -> None:
        result, searches, evaluator = self.run_mocked()
        self.assertEqual(result.selection.model_name, "random_forest")
        self.assertEqual(set(result.model_runs), set(self.scores))
        self.assertIsNone(result.test)
        self.assertEqual(evaluator.call_count, 3)
        for search in searches:
            self.assertIs(search.call_args.args[0], self.splits.train)
            self.assertIs(search.call_args.args[1], self.splits.valid)
        for call in evaluator.call_args_list:
            self.assertIs(call.args[1], self.splits.valid)

    def test_only_validation_winner_is_evaluated_on_test(self) -> None:
        result, _, evaluator = self.run_mocked(evaluate_test=True)
        self.assertIsNotNone(result.test)
        self.assertEqual(evaluator.call_count, 4)
        self.assertEqual(evaluator.call_args.args[0].model_name, "random_forest")
        self.assertIs(evaluator.call_args.args[1], self.splits.test)

    def test_can_compare_new_families_and_select_them_using_validation(self) -> None:
        self.scores.update(elastic_net=.4, extra_trees=.5)
        result, searches, evaluator = self.run_mocked(
            evaluate_test=True, model_names=tuple(self.scores),
        )
        self.assertEqual(set(result.model_runs), set(self.scores))
        self.assertEqual(result.selection.model_name, "extra_trees")
        self.assertEqual(evaluator.call_count, 6)
        self.assertEqual(evaluator.call_args.args[0].model_name, "extra_trees")
        self.assertIs(evaluator.call_args.args[1], self.splits.test)
        for search in searches:
            self.assertIs(search.call_args.args[0], self.splits.train)
            self.assertIs(search.call_args.args[1], self.splits.valid)

    def test_requested_subset_does_not_train_unselected_models(self) -> None:
        self.scores.update(elastic_net=.4, extra_trees=.5)
        result, searches, evaluator = self.run_mocked(model_names=("extra_trees", "elastic_net"))
        self.assertEqual(list(result.model_runs), ["extra_trees", "elastic_net"])
        self.assertEqual(evaluator.call_count, 2)
        for name, search in zip(self.scores, searches):
            self.assertEqual(search.call_count, int(name in result.model_runs))

    def test_rejects_unknown_empty_and_duplicate_model_lists(self) -> None:
        for names in ((), ("typo",), ("ridge", "ridge")):
            with self.subTest(names=names), self.assertRaises(ValueError):
                run_experiment(self.prices, self.config, model_names=names)


if __name__ == "__main__":
    unittest.main()
