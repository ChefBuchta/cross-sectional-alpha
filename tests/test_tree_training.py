import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from factor_model.models import ModelSelection
from factor_model.training import (
    evaluate_model,
    select_elastic_net_model,
    select_extra_trees_model,
    select_gradient_boosting_model,
    select_random_forest_model,
)


class TreeTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = self.make_rows("2025-01-02", 20)
        self.valid = self.make_rows("2025-03-03", 3)
        self.test = self.make_rows("2025-04-01", 3)
        self.searches = (
            ("extra_trees", select_extra_trees_model, {
                "max_depths": (2, 3), "min_samples_leaves": (1, 2),
                "n_estimators": 8,
            }),
            ("random_forest", select_random_forest_model, {
                "max_depths": (2, 3), "min_samples_leaves": (1, 2),
                "n_estimators": 8,
            }),
            ("gradient_boosting", select_gradient_boosting_model, {
                "max_leaf_nodes": (3, 7), "l2_regularizations": (0.0, 1.0),
                "max_iter": 5, "min_samples_leaf": 1,
            }),
        )

    @staticmethod
    def make_rows(start: str, periods: int) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "date": date,
                "symbol": symbol,
                "label_end_date": date + pd.Timedelta(days=1),
                "factor": factor,
                "baseline_score": factor,
                "target_percentile_1d": target,
            }
            for date in pd.bdate_range(start, periods=periods)
            for symbol, factor, target in (
                ("AAPL", 1.0, 1 / 3),
                ("MSFT", 2.0, 2 / 3),
                ("JPM", 3.0, 1.0),
            )
        ]).sample(frac=1.0, random_state=12)

    def select(self, search, options):
        return search(
            self.train, self.valid,
            feature_columns=("factor",), target_column="target_percentile_1d",
            **options,
        )

    def test_each_family_selects_and_evaluates_with_common_result_types(self) -> None:
        for name, search, options in self.searches:
            with self.subTest(model=name):
                train_before = self.train.copy(deep=True)
                valid_before = self.valid.copy(deep=True)
                selection = self.select(search, options)
                self.assertIsInstance(selection, ModelSelection)
                self.assertEqual(selection.model_name, name)
                self.assertEqual(len(selection.comparison), 4)
                self.assertEqual(selection.feature_columns, ("factor",))
                self.assertEqual(selection.target_column, "target_percentile_1d")
                self.assertNotIn("scaler", selection.pipeline.named_steps)
                pd.testing.assert_frame_equal(self.train, train_before)
                pd.testing.assert_frame_equal(self.valid, valid_before)

                expected = selection.pipeline.predict(self.test[["factor"]])
                original_test = self.test.copy(deep=True)
                with patch.object(selection.pipeline, "fit", side_effect=AssertionError):
                    evaluation = evaluate_model(selection, self.test)
                self.assertTrue(evaluation.scored_rows.index.equals(self.test.index))
                np.testing.assert_allclose(
                    evaluation.scored_rows[f"{name}_score"], expected
                )
                self.assertEqual(list(evaluation.summary.index), ["baseline", name])
                self.assertEqual(evaluation.summary.loc[name, "defined_day_count"], 3)
                pd.testing.assert_frame_equal(self.test, original_test)

    def test_selection_uses_validation_mean_not_last_candidate(self) -> None:
        for name, search, options in self.searches:
            with self.subTest(model=name), patch(
                "factor_model.training.summarize_daily_ic",
                side_effect=[{"mean": 0.1}, {"mean": 0.3},
                             {"mean": 0.2}, {"mean": 0.0}],
            ):
                selection = self.select(search, options)
                best_row = selection.comparison.iloc[1]
                for parameter, value in selection.best_parameters.items():
                    self.assertEqual(best_row[parameter], value)

    def test_reproducibility_and_no_internal_boosting_validation(self) -> None:
        for name, search, options in self.searches:
            with self.subTest(model=name):
                first = self.select(search, options)
                second = self.select(search, options)
                self.assertEqual(first.best_parameters, second.best_parameters)
                np.testing.assert_allclose(
                    first.pipeline.predict(self.valid[["factor"]]),
                    second.pipeline.predict(self.valid[["factor"]]),
                )
                if name == "gradient_boosting":
                    model = first.pipeline.named_steps[name]
                    self.assertFalse(model.early_stopping)
                    self.assertFalse(model.do_early_stopping_)

    def test_each_family_rejects_overlapping_training_labels(self) -> None:
        self.train.loc[self.train.index[0], "label_end_date"] = self.valid["date"].min()
        for name, search, options in self.searches:
            with self.subTest(model=name):
                with self.assertRaisesRegex(ValueError, "labels must end before"):
                    self.select(search, options)

    def test_each_family_rejects_all_undefined_candidates(self) -> None:
        self.valid["target_percentile_1d"] = 0.5
        for name, search, options in self.searches:
            with self.subTest(model=name):
                with self.assertRaisesRegex(ValueError, "no candidate has a defined"):
                    self.select(search, options)

    def test_empty_grids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.select(select_random_forest_model, {"max_depths": ()})
        with self.assertRaises(ValueError):
            self.select(select_gradient_boosting_model, {"max_leaf_nodes": ()})
        with self.assertRaises(ValueError):
            self.select(select_extra_trees_model, {"max_depths": ()})


class ElasticNetTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = TreeTrainingTest.make_rows("2025-01-02", 20)
        self.valid = TreeTrainingTest.make_rows("2025-03-03", 3)

    def select(self, **options):
        return select_elastic_net_model(
            self.train, self.valid, feature_columns=("factor",),
            target_column="target_percentile_1d", **options,
        )

    def test_scales_only_training_data_and_evaluation_never_refits(self):
        self.valid["factor"] += 100
        train_before, valid_before = self.train.copy(), self.valid.copy()
        selection = self.select()
        self.assertEqual(selection.model_name, "elastic_net")
        self.assertEqual(len(selection.comparison), 4)
        self.assertAlmostEqual(
            selection.pipeline.named_steps["scaler"].mean_[0], self.train.factor.mean(),
        )
        expected = selection.pipeline.predict(self.valid[["factor"]])
        with patch.object(selection.pipeline, "fit", side_effect=AssertionError("refit")):
            evaluation = evaluate_model(selection, self.valid)
        np.testing.assert_allclose(evaluation.scored_rows.elastic_net_score, expected)
        pd.testing.assert_frame_equal(self.train, train_before)
        pd.testing.assert_frame_equal(self.valid, valid_before)

    def test_selects_by_validation_ic_and_reproduces_predictions(self):
        with patch("factor_model.training.summarize_daily_ic", side_effect=[
            {"mean": .1}, {"mean": .4}, {"mean": .2}, {"mean": .3},
        ]):
            selection = self.select()
        for name, value in selection.best_parameters.items():
            self.assertEqual(selection.comparison.iloc[1][name], value)
        first, second = self.select(), self.select()
        np.testing.assert_allclose(
            first.pipeline.predict(self.valid[["factor"]]),
            second.pipeline.predict(self.valid[["factor"]]),
        )

    def test_rejects_overlapping_labels_empty_grid_and_undefined_scores(self):
        with self.assertRaises(ValueError):
            self.select(alphas=())
        self.valid["target_percentile_1d"] = .5
        with self.assertRaisesRegex(ValueError, "no candidate"):
            self.select()
        self.train["label_end_date"] = self.valid.date.min()
        with self.assertRaisesRegex(ValueError, "labels must end before"):
            self.select()


if __name__ == "__main__":
    unittest.main()
