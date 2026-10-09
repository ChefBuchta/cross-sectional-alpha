import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from factor_model.models import ModelEvaluation, RidgeSelection
from factor_model.training import (
    build_ridge_pipeline,
    evaluate_ridge_model,
    select_ridge_model,
)


class RidgeTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = self.make_rows("2025-01-02", shift=0)
        self.valid = self.make_rows("2025-01-10", shift=10)
        self.test = self.make_rows("2025-01-20", shift=20)

    @staticmethod
    def make_rows(start: str, shift: int) -> pd.DataFrame:
        rows = pd.DataFrame([
            {
                "date": date,
                "symbol": symbol,
                "label_end_date": date + pd.Timedelta(days=1),
                "factor": value + shift,
                "baseline_score": value,
                "target_percentile_1d": percentile,
            }
            for date in pd.bdate_range(start, periods=2)
            for symbol, value, percentile in (
                ("AAPL", 1.0, 1 / 3),
                ("MSFT", 2.0, 2 / 3),
                ("JPM", 3.0, 1.0),
            )
        ])
        rows.index = [90, 10, 50, 20, 80, 30]
        return rows

    def select(self, **kwargs) -> RidgeSelection:
        return select_ridge_model(
            self.train, self.valid,
            feature_columns=("factor",), target_column="target_percentile_1d",
            **kwargs,
        )

    def test_pipeline_instances_are_independent_and_unfitted(self) -> None:
        first = build_ridge_pipeline(1.0)
        second = build_ridge_pipeline(1.0)
        self.assertIsNot(first.named_steps["scaler"], second.named_steps["scaler"])
        self.assertIsNot(first.named_steps["ridge"], second.named_steps["ridge"])
        self.assertFalse(hasattr(first.named_steps["ridge"], "coef_"))

    def test_selects_larger_alpha_in_a_tie_and_scales_only_training_rows(self) -> None:
        original_train = self.train.copy(deep=True)
        original_valid = self.valid.copy(deep=True)
        selection = self.select(alphas=(0.1, 1.0, 10.0))
        self.assertIsInstance(selection, RidgeSelection)
        self.assertEqual(selection.best_alpha, 10.0)
        self.assertEqual(selection.pipeline.named_steps["ridge"].alpha, 10.0)
        np.testing.assert_allclose(selection.comparison["mean"], 1.0)
        np.testing.assert_allclose(selection.pipeline.named_steps["scaler"].mean_, [2.0])
        pd.testing.assert_frame_equal(self.train, original_train)
        pd.testing.assert_frame_equal(self.valid, original_valid)

    def test_evaluation_preserves_row_alignment_and_does_not_refit(self) -> None:
        selection = self.select(alphas=(1.0,))
        means = selection.pipeline.named_steps["scaler"].mean_.copy()
        weights = selection.pipeline.named_steps["ridge"].coef_.copy()
        original_test = self.test.copy(deep=True)
        expected = selection.pipeline.predict(self.test[["factor"]])
        evaluation = evaluate_ridge_model(selection, self.test)
        self.assertIsInstance(evaluation, ModelEvaluation)
        self.assertTrue(evaluation.scored_rows.index.equals(self.test.index))
        np.testing.assert_allclose(evaluation.scored_rows["ridge_score"], expected)
        np.testing.assert_allclose(selection.pipeline.named_steps["scaler"].mean_, means)
        np.testing.assert_allclose(selection.pipeline.named_steps["ridge"].coef_, weights)
        pd.testing.assert_frame_equal(self.test, original_test)
        self.assertAlmostEqual(evaluation.summary.loc["ridge", "mean"], 1.0)
        self.assertAlmostEqual(evaluation.summary.loc["baseline", "mean"], 1.0)
        self.assertEqual(list(evaluation.daily_ic.columns), ["baseline", "ridge"])

    def test_rejects_invalid_candidate_sets(self) -> None:
        for alphas in ((), (-1.0,), (np.inf,), (np.nan,), (1.0, 1.0)):
            with self.subTest(alphas=alphas), self.assertRaises(ValueError):
                self.select(alphas=alphas)

    def test_rejects_invalid_pipeline_alpha(self) -> None:
        for alpha in (-1.0, np.inf, np.nan):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                build_ridge_pipeline(alpha)

    def test_rejects_undefined_validation_scores(self) -> None:
        self.valid["target_percentile_1d"] = 0.5
        with self.assertRaisesRegex(ValueError, "no candidate has a defined"):
            self.select(alphas=(1.0,))

    def test_rejects_training_labels_overlapping_validation(self) -> None:
        self.train.loc[90, "label_end_date"] = self.valid["date"].min()
        with self.assertRaisesRegex(ValueError, "labels must end before"):
            self.select()

    def test_rejects_nonfinite_training_features(self) -> None:
        self.train.loc[90, "factor"] = np.inf
        with self.assertRaisesRegex(ValueError, "features and targets must be finite"):
            self.select()

    def test_selection_uses_validation_mean_before_alpha(self) -> None:
        with patch(
            "factor_model.training.summarize_daily_ic",
            side_effect=[{"mean": 0.1}, {"mean": 0.3}, {"mean": 0.2}],
        ):
            selection = self.select(alphas=(0.1, 1.0, 10.0))
        self.assertEqual(selection.best_alpha, 1.0)

    def test_rejects_target_as_an_input_feature(self) -> None:
        with self.assertRaisesRegex(ValueError, "target cannot also be"):
            select_ridge_model(
                self.train, self.valid,
                feature_columns=("target_percentile_1d",),
                target_column="target_percentile_1d",
            )


if __name__ == "__main__":
    unittest.main()
