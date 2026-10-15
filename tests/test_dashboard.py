"""Behavior tests for picks, leaderboard, metrics, and help. No expensive training."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from factor_model.models import ExperimentResult, ModelRun, ModelSelection
from factor_model.training import build_ridge_pipeline, evaluate_model


APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
FEATURES = ("trailing_return_5d", "trailing_return_20d", "volatility_20d", "relative_volume_20d")


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data_path = Path(self.directory.name) / "prices.csv"
        self.symbols = ("AAPL", "MSFT", "JNJ", "CAT")
        prices = pd.DataFrame([
            {"date": day, "symbol": symbol, "adjusted_close": 100 + index * (stock + 1),
             "close": 100 + index * (stock + 1), "volume": 1000 + index * (stock + 1)}
            for index, day in enumerate(pd.bdate_range("2022-12-01", periods=100))
            for stock, symbol in enumerate(self.symbols)
        ])
        prices.to_csv(self.data_path, index=False)
        source_patch = patch("factor_model.config.EDA_DATA_PATH", self.data_path)
        source_patch.start()
        self.addCleanup(source_patch.stop)

    @staticmethod
    def fake_run(prices, config, *, evaluate_test=False):
        symbols = config.symbols
        inputs = pd.DataFrame({
            "trailing_return_5d": [-.1, .1, .3, .2],
            "trailing_return_20d": [-.2, .2, .5, .3],
            "volatility_20d": [.01, .02, .04, .03],
            "relative_volume_20d": [.7, 1., 1.8, 1.3],
        })
        pipeline = build_ridge_pipeline().fit(inputs, [.1, .4, .9, .7])
        rows = pd.DataFrame([
            {"date": day, "symbol": symbol, "trailing_return_5d": (stock + 1) * .01,
             "trailing_return_20d": (stock + 1) * .03, "volatility_20d": .01 + stock * .005,
             "relative_volume_20d": .8 + stock * .2, "baseline_score": -stock * .01,
             f"target_percentile_{config.target_horizon}d": (stock + 1) / len(symbols),
             f"forward_return_{config.target_horizon}d": stock * .01 - .01,
             "label_end_date": day + pd.Timedelta(days=7)}
            for day in pd.bdate_range("2022-01-03", periods=35)
            for stock, symbol in enumerate(symbols)
        ])
        runs = {}
        for name in ("ridge", "random_forest", "gradient_boosting"):
            selection = ModelSelection(
                model_name=name, pipeline=pipeline, feature_columns=FEATURES,
                best_parameters={"alpha": 1.}, target_column=f"target_percentile_{config.target_horizon}d",
                comparison=pd.DataFrame({"alpha": [1.], "mean": [1.]}),
            )
            runs[name] = ModelRun(selection=selection, validation=evaluate_model(selection, rows))
        selection = runs["ridge"].selection
        validation = runs["ridge"].validation
        return ExperimentResult(
            config=config, selection=selection, validation=validation, model_runs=runs,
            test=validation.model_copy(deep=True) if evaluate_test else None,
        )

    def start(self, runner):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        app.button(key="run_experiment").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        runner.assert_called_once()
        return app

    def test_initial_page_and_help_are_available_without_training(self):
        with patch("factor_model.pipeline.run_experiment") as runner:
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.title[0].value, "Relative Edge")
            self.assertFalse(app.checkbox[0].value)
            self.assertTrue(any("Your stock recommendation" in item.value for item in app.header))
            app.switch_page("app_pages/help.py").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("What is the actual output?" == item.label for item in app.expander))
            self.assertGreater(len(app.latex), 0)
            runner.assert_not_called()

    def test_daily_weekly_monthly_picks_and_models_do_not_retrain(self):
        with patch("factor_model.pipeline.run_experiment", side_effect=self.fake_run) as runner:
            app = self.start(runner)
            self.assertFalse(runner.call_args.kwargs["evaluate_test"])
            self.assertEqual(app.selectbox(key="picks_source").value, "Post-selection forecasts")
            self.assertTrue(any("next 5 trading days" in item.value for item in app.subheader))
            for cadence in ("Day", "Week", "Month"):
                app.segmented_control(key="picks_cadence").set_value(cadence).run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                history = app.dataframe[-1].value
                self.assertIn("Most attractive", history)
                self.assertEqual(pd.to_datetime(history["As of"]).max(),
                                 pd.read_csv(self.data_path, parse_dates=["date"])["date"].max())
            app.selectbox(key="picks_model").set_value("random_forest").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Single-feature sensitivity" in item.value for item in app.markdown))
            app.selectbox(key="picks_explain").set_value("AAPL").run(timeout=30)
            runner.assert_called_once()

    def test_model_leaderboard_and_all_metrics_pages_do_not_retrain(self):
        with patch("factor_model.pipeline.run_experiment", side_effect=self.fake_run) as runner:
            app = self.start(runner)
            app.switch_page("app_pages/leaderboard.py").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            table = app.dataframe[0].value
            self.assertEqual(len(table), 4)
            self.assertIn("20-day momentum baseline", table["Model"].tolist())
            self.assertIn("Mean selection edge", table)
            app.switch_page("app_pages/metrics.py").run(timeout=30)
            for name in ("ridge", "random_forest", "gradient_boosting"):
                app.selectbox(key="metrics_model").set_value(name).run(timeout=30)
                self.assertEqual(len(app.exception), 0)
            for chart in ("Daily rank correlation", "Monthly rank correlation", "Correlation distribution",
                          "Shortlist edge over time", "Rolling rank correlation"):
                app.selectbox(key="metrics_chart").set_value(chart).run(timeout=30)
                self.assertEqual(len(app.exception), 0, chart)
            app.selectbox(key="metrics_window").set_value(252).run(timeout=30)
            self.assertTrue(any("Not enough defined history" in item.value for item in app.info))
            app.switch_page("app_pages/help.py").run(timeout=30)
            app.switch_page("app_pages/picks.py").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            runner.assert_called_once()

    def test_test_metrics_and_replay_only_exist_for_the_recorded_winner(self):
        with patch("factor_model.pipeline.run_experiment", side_effect=self.fake_run) as runner:
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.checkbox[0].set_value(True)
            app.button(key="run_experiment").click().run(timeout=30)
            app.selectbox(key="picks_source").set_value("Test replay (winner only)").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            app.selectbox(key="picks_model").set_value("gradient_boosting").run(timeout=30)
            self.assertNotIn("Historical test results (selected model only)", app.selectbox(key="picks_source").options)
            app.switch_page("app_pages/metrics.py").run(timeout=30)
            app.segmented_control(key="metrics_period").set_value("Test").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            app.selectbox(key="metrics_model").set_value("random_forest").run(timeout=30)
            self.assertEqual(app.segmented_control(key="metrics_period").value, "Validation")
            self.assertNotIn("Test", app.segmented_control(key="metrics_period").options)
            runner.assert_called_once()

    def test_new_model_is_discovered_on_every_page(self):
        def with_extra_model(*args, **kwargs):
            result = self.fake_run(*args, **kwargs)
            original = result.model_runs["ridge"]
            candidate = original.selection.model_copy(update={"model_name": "my_new_model"})
            result.model_runs["my_new_model"] = ModelRun(
                selection=candidate, validation=evaluate_model(candidate, original.validation.scored_rows),
            )
            return result
        with patch("factor_model.pipeline.run_experiment", side_effect=with_extra_model) as runner:
            app = self.start(runner)
            app.selectbox(key="picks_model").set_value("my_new_model").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            app.switch_page("app_pages/leaderboard.py").run(timeout=30)
            self.assertIn("My New Model", app.dataframe[0].value["Model"].tolist())
            app.switch_page("app_pages/metrics.py").run(timeout=30)
            app.selectbox(key="metrics_model").set_value("my_new_model").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            runner.assert_called_once()

    def test_future_outcomes_never_determine_the_replayed_shortlist(self):
        def reversed_outcomes(*args, **kwargs):
            result = self.fake_run(*args, **kwargs)
            rows = result.model_runs["ridge"].validation.scored_rows
            rows["forward_return_5d"] = -rows["ridge_score"]
            return result
        with patch("factor_model.pipeline.run_experiment", side_effect=reversed_outcomes) as runner:
            app = self.start(runner)
            app.selectbox(key="picks_source").set_value("Validation replay").run(timeout=30)
            rows = next(item.value for item in app.dataframe if "shortlisted" in item.value)
            scores = rows["ridge_score"].to_numpy()
            self.assertTrue(np.all(scores[:-1] >= scores[1:]))
            self.assertEqual(rows.loc[rows["shortlisted"], "symbol"].tolist(), ["MSFT", "JNJ", "CAT"])
            runner.assert_called_once()

    def test_all_tied_scores_abstain_instead_of_inventing_picks(self):
        def tied_run(*args, **kwargs):
            result = self.fake_run(*args, **kwargs)
            result.model_runs["ridge"].validation.scored_rows["ridge_score"] = .5
            return result
        with patch("factor_model.pipeline.run_experiment", side_effect=tied_run) as runner:
            app = self.start(runner)
            app.selectbox(key="picks_source").set_value("Validation replay").run(timeout=30)
            self.assertTrue(any("All scores are tied" in item.value for item in app.info))
            rows = next(item.value for item in app.dataframe if "shortlisted" in item.value)
            self.assertFalse(rows["shortlisted"].any())
            runner.assert_called_once()

    def test_invalid_settings_do_not_run_and_help_survives_missing_csv(self):
        with patch("factor_model.pipeline.run_experiment") as runner:
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.multiselect(key="symbols").set_value([])
            app.button(key="run_experiment").click().run(timeout=30)
            self.assertTrue(len(app.error))
            self.assertEqual(len(app.exception), 0)
            runner.assert_not_called()
        with patch("factor_model.config.EDA_DATA_PATH", self.data_path.parent / "missing.csv"):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.switch_page("app_pages/help.py").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertGreater(len(app.latex), 0)

    def test_changing_horizon_requires_a_new_run_not_a_view_change(self):
        with patch("factor_model.pipeline.run_experiment", side_effect=self.fake_run) as runner:
            app = self.start(runner)
            app.selectbox(key="target_horizon").set_value(20).run(timeout=30)
            runner.assert_called_once()
            self.assertTrue(any("next 5 trading days" in item.value for item in app.subheader))
            app.button(key="run_experiment").click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(runner.call_count, 2)
            self.assertTrue(any("next 20 trading days" in item.value for item in app.subheader))

    def test_recommendation_explains_why_without_a_giant_historical_percentage(self):
        with patch("factor_model.pipeline.run_experiment", side_effect=self.fake_run) as runner:
            app = self.start(runner)
            metrics = {item.label: item.value for item in app.metric}
            self.assertEqual(metrics, {
                "Model rank": "1 of 4",
                "Average top-group advantage": "+0.50 pp",
                "Dates the top group beat the average": "35 / 35",
            })
            self.assertTrue(any("following 5 trading days" in item.value and "percentage points" in item.value for item in app.caption))
            self.assertTrue(any("Validation: Jan 2022–Feb 2022" in item.value and "not this stock's chance" in item.value for item in app.caption))
            self.assertTrue(any(item.label == "Why this stock? Strengths and drawbacks" and not item.proto.expanded for item in app.expander))
            self.assertTrue(any("Recommendation for further research" in item.value for item in app.caption))
            self.assertTrue(any("No—not on this result alone" in item.value for item in app.warning))
            self.assertTrue(any("Why the model favors it" in item.value for item in app.subheader))
            self.assertTrue(any("previous" in item.value or "day-to-day" in item.value or "Trading activity" in item.value for item in app.markdown))
            self.assertTrue(any(item.label == "Recent history — not a forecast" for item in app.expander))
            self.assertTrue(any("A negative past return means a past loss" in item.value for item in app.markdown))
            effects = next(item.value for item in app.dataframe if "Meaning" in item.value)
            self.assertTrue(set(effects["Meaning"]).issubset({"Raised model score", "Lowered model score", "No measured effect"}))
            self.assertTrue(any("not a percentage return" in item.value for item in app.caption)
                            or len(app.get("arrow_vega_lite_chart")) > 0)
            runner.assert_called_once()

    def test_missing_group_outcomes_are_unavailable_not_zero_success(self):
        def missing_outcomes(*args, **kwargs):
            result = self.fake_run(*args, **kwargs)
            result.model_runs["ridge"].validation.scored_rows["forward_return_5d"] = np.nan
            return result
        with patch("factor_model.pipeline.run_experiment", side_effect=missing_outcomes) as runner:
            app = self.start(runner)
            metrics = {item.label: item.value for item in app.metric}
            self.assertEqual(metrics["Average top-group advantage"], "Unavailable")
            self.assertEqual(metrics["Dates the top group beat the average"], "Unavailable")

    def test_negative_group_advantage_keeps_its_sign_and_counts_losses(self):
        def losing_group(*args, **kwargs):
            result = self.fake_run(*args, **kwargs)
            rows = result.model_runs["ridge"].validation.scored_rows
            rows["forward_return_5d"] *= -1
            return result
        with patch("factor_model.pipeline.run_experiment", side_effect=losing_group) as runner:
            app = self.start(runner)
            metrics = {item.label: item.value for item in app.metric}
            self.assertEqual(metrics["Average top-group advantage"], "-0.50 pp")
            self.assertEqual(metrics["Dates the top group beat the average"], "0 / 35")

    def test_weak_evidence_is_visible_beside_the_recommendation(self):
        def weak_run(*args, **kwargs):
            result = self.fake_run(*args, **kwargs)
            result.model_runs["ridge"].validation.summary.loc["ridge", "mean"] = -.2
            return result
        with patch("factor_model.pipeline.run_experiment", side_effect=weak_run) as runner:
            app = self.start(runner)
            self.assertTrue(any("did not show positive average agreement" in item.value for item in app.markdown))
            self.assertTrue(any("No—not on this result alone" in item.value for item in app.warning))

    def test_joint_leaders_are_not_presented_as_one_best_stock(self):
        def tied_leaders(*args, **kwargs):
            result = self.fake_run(*args, **kwargs)
            rows = result.model_runs["ridge"].validation.scored_rows
            rows["ridge_score"] = rows["symbol"].map({"AAPL": 1., "CAT": 1., "JNJ": .5, "MSFT": .2})
            return result
        with patch("factor_model.pipeline.run_experiment", side_effect=tied_leaders) as runner:
            app = self.start(runner)
            app.selectbox(key="picks_source").set_value("Validation replay").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Several stocks share first place" == item.value for item in app.header))
            self.assertTrue(any("no single best candidate" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
