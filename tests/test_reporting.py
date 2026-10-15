"""Hand-checked reporting calculations, independent of Streamlit."""

from datetime import date
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from factor_model.models import ExperimentConfig, ExperimentResult, ModelRun, ModelSelection
from factor_model.reporting import (
    forecast_from_prices, model_leaderboard, rank_stocks, score_explanations,
    selection_performance, shortlist_history, snapshot_dates,
)
from factor_model.training import build_ridge_pipeline, evaluate_model


class ReportingTest(unittest.TestCase):
    @staticmethod
    def selection(name="ridge"):
        inputs = pd.DataFrame({"trailing_return_5d": [-0.1, 0.3, 0.1, 0.4]})
        pipeline = build_ridge_pipeline().fit(inputs, [0.1, 0.7, 0.3, 0.9])
        return ModelSelection(
            model_name=name, pipeline=pipeline, feature_columns=tuple(inputs.columns),
            best_parameters={"alpha": 1.0}, target_column="target_percentile_5d", comparison=pd.DataFrame(),
        )

    def test_cutoff_ties_are_included_and_all_ties_abstain(self):
        rows = pd.DataFrame({"date": pd.Timestamp("2024-01-02"),
                             "symbol": list("ABCDE"), "score": [5, 4, 3, 3, 1]})
        ranked = rank_stocks(rows, "score")
        self.assertEqual(ranked.loc[ranked.shortlisted, "symbol"].tolist(), list("ABCD"))
        self.assertEqual(ranked["predicted_rank"].tolist(), [1, 2, 3.5, 3.5, 5])
        self.assertFalse(rank_stocks(rows.assign(score=1), "score")["shortlisted"].any())
        with self.assertRaisesRegex(ValueError, "finite"):
            rank_stocks(rows.assign(score=np.nan), "score")
        with self.assertRaisesRegex(ValueError, "unique"):
            rank_stocks(pd.concat([rows, rows]), "score")

    def test_calendar_snapshots_use_observed_dates_not_future_period_ends(self):
        rows = pd.DataFrame({"date": pd.to_datetime([
            "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05",
        ])})
        self.assertEqual(len(snapshot_dates(rows, "Day")), 5)
        weekly = snapshot_dates(rows, "Week")
        self.assertEqual(weekly["As of"].tolist(), list(pd.to_datetime(["2024-02-02", "2024-02-05"])))
        monthly = snapshot_dates(rows, "Month")
        self.assertEqual(monthly["As of"].tolist(), list(pd.to_datetime(["2024-01-31", "2024-02-05"])))
        with self.assertRaises(ValueError):
            snapshot_dates(rows, "Year")

    def test_history_rankings_do_not_use_realized_returns(self):
        rows = pd.DataFrame({"date": pd.Timestamp("2024-01-31"), "symbol": list("ABCD"),
                             "score": [4, 3, 2, 1], "forward_return_5d": [-1, -2, -3, 100]})
        for cadence in ("Day", "Week", "Month"):
            history = shortlist_history(rows, "score", cadence)
            self.assertEqual(history.iloc[0]["Most attractive"], "A")
            self.assertEqual(history.iloc[0]["Top research group"], "A, B, C")
            tied = shortlist_history(rows.assign(score=1), "score", cadence)
            self.assertEqual(tied.iloc[0]["Most attractive"], "No preference (all tied)")

    def test_forward_window_selection_metrics_by_hand(self):
        rows = pd.DataFrame({"date": pd.Timestamp("2024-01-02"), "symbol": list("ABCDEF"),
                             "score": [6, 5, 4, 3, 2, 1], "forward_return_5d": [.06, .04, .02, -.01, -.02, -.03]})
        performance = selection_performance(rows, "score", 5).iloc[0]
        self.assertAlmostEqual(performance.shortlist_return, .04)
        self.assertAlmostEqual(performance.universe_return, .01)
        self.assertAlmostEqual(performance.selection_edge, .03)
        self.assertAlmostEqual(performance.top_bottom_spread, .06)
        tied = selection_performance(rows.assign(score=1), "score", 5).iloc[0]
        self.assertTrue(np.isnan(tied.selection_edge))
        self.assertTrue(np.isnan(tied.shortlist_return))

    def test_ridge_score_contributions_are_exact_and_aligned(self):
        selection = self.selection()
        rows = pd.DataFrame({"symbol": list("CAB"), "trailing_return_5d": [.2, -.1, .5]}, index=[7, 3, 12])
        contributions, kind = score_explanations(selection, rows)
        predictions = selection.pipeline.predict(rows[["trailing_return_5d"]])
        self.assertEqual(kind, "Exact score contributions")
        np.testing.assert_allclose(contributions.sum(axis=1), predictions - predictions.mean(), atol=1e-12)
        self.assertEqual(contributions.index.tolist(), list("CAB"))

    def test_tree_sensitivity_matches_single_feature_replacement(self):
        inputs = pd.DataFrame({"momentum": [0., 1., 2., 3.], "volume": [2., 4., 1., 6.]})
        pipeline = Pipeline([("model", RandomForestRegressor(n_estimators=5, random_state=42))]).fit(inputs, [0., 3., 2., 6.])
        selection = self.selection().model_copy(update={
            "model_name": "future_tree", "pipeline": pipeline, "feature_columns": tuple(inputs.columns),
        })
        rows = inputs.assign(symbol=list("ABCD"))
        effects, kind = score_explanations(selection, rows)
        changed = inputs.assign(momentum=inputs.momentum.mean())
        expected = pipeline.predict(inputs) - pipeline.predict(changed)
        np.testing.assert_allclose(effects["momentum"], expected)
        self.assertEqual(kind, "Single-feature sensitivity")

    def test_forecasts_include_last_unlabeled_date_without_refitting(self):
        selection = self.selection()
        validation_rows = pd.DataFrame({
            "date": pd.Timestamp("2022-01-03"), "symbol": ["A", "B"],
            "trailing_return_5d": [.1, .2], "baseline_score": [.1, .2],
            "target_percentile_5d": [.5, 1.], "forward_return_5d": [.01, .02],
        })
        evaluation = evaluate_model(selection, validation_rows)
        config = ExperimentConfig(symbols=("A", "B"), start_date=date(2021, 1, 1),
                                  validation_start=date(2022, 1, 1), test_start=date(2023, 1, 1), end_date=date(2025, 1, 1))
        result = ExperimentResult(config=config, selection=selection, validation=evaluation,
                                  model_runs={"ridge": ModelRun(selection=selection, validation=evaluation)})
        prices = pd.DataFrame([
            {"date": day, "symbol": symbol, "adjusted_close": 100 + i * rate,
             "close": 100 + i * rate, "volume": 100 + i}
            for i, day in enumerate(pd.bdate_range("2022-12-01", periods=70))
            for symbol, rate in (("A", 1), ("B", 2))
        ])
        original = prices.copy(deep=True)
        with patch.object(selection.pipeline, "fit", side_effect=AssertionError("Must not refit")):
            forecast = forecast_from_prices(prices, result)["ridge"]
        self.assertEqual(forecast["date"].max(), prices["date"].max())
        self.assertGreaterEqual(forecast["date"].min(), pd.Timestamp(config.test_start))
        self.assertNotIn("forward_return_5d", forecast)
        self.assertEqual(set(forecast.groupby("date").symbol.nunique()), {2})
        pd.testing.assert_frame_equal(prices, original)
        incomplete = prices.loc[~((prices["date"] == prices["date"].max()) & (prices["symbol"] == "B"))]
        complete_forecast = forecast_from_prices(incomplete, result)["ridge"]
        self.assertLess(complete_forecast["date"].max(), prices["date"].max())
        # Scoring a past date is unchanged when later prices are present or removed.
        earlier_prices = prices.loc[prices["date"] <= forecast["date"].unique()[-10]]
        earlier = forecast_from_prices(earlier_prices, result)["ridge"]
        merged = earlier.merge(forecast, on=["date", "symbol"], suffixes=("_early", "_full"))
        np.testing.assert_allclose(merged.ridge_score_early, merged.ridge_score_full)

    def test_leaderboard_is_dynamic_and_requires_matching_coverage(self):
        selection = self.selection()
        rows = pd.DataFrame({"date": pd.Timestamp("2022-01-03"), "symbol": list("ABCD"),
                             "trailing_return_5d": [.1, .4, .2, .3], "baseline_score": [.4, .1, .3, .2],
                             "target_percentile_5d": [.25, 1., .5, .75], "forward_return_5d": [.01, .04, .02, .03]})
        evaluation = evaluate_model(selection, rows)
        config = ExperimentConfig(symbols=tuple("ABCD"), start_date=date(2021, 1, 1), validation_start=date(2022, 1, 1),
                                  test_start=date(2023, 1, 1), end_date=date(2025, 1, 1))
        runs = {}
        for name in ("ridge", "new_model_1", "new_model_2", "new_model_3", "new_model_4"):
            candidate = selection.model_copy(update={"model_name": name})
            runs[name] = ModelRun(selection=candidate, validation=evaluate_model(candidate, rows))
        result = ExperimentResult(config=config, selection=selection, validation=evaluation, model_runs=runs)
        table = model_leaderboard(result)
        self.assertEqual(len(table), 6)
        self.assertEqual(table.iloc[-1]["model"], "baseline")
        self.assertEqual(set(table.loc[table.model != "baseline", "performance_rank"]), {1})
        runs["new_model_4"].validation.scored_rows = rows.iloc[:2].copy()
        with self.assertRaisesRegex(ValueError, "same validation"):
            model_leaderboard(result)


if __name__ == "__main__":
    unittest.main()
