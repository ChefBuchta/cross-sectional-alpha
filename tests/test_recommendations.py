"""Copy must explain what a number means without promising investment returns."""

import unittest

from factor_model.recommendations import evidence_readout, observation_text, score_reason_text


class RecommendationTextTest(unittest.TestCase):
    def test_concise_reason_keeps_past_period_and_model_effect_separate(self):
        detailed = score_reason_text("trailing_return_20d", -.07, -.03, .02)
        concise = score_reason_text("trailing_return_20d", -.07, -.03, .02, concise=True)
        self.assertLess(len(concise), len(detailed))
        self.assertIn("previous 20 trading days", concise)
        self.assertIn("loss of 7.0%", concise)
        self.assertIn("Helped this model's score", concise)
        self.assertIn("Lowered this model's score", score_reason_text("volatility_20d", .01, .02, -.1, concise=True))
        self.assertIn("No clear score effect", score_reason_text("relative_volume_20d", .68, 1., 0., concise=True))

    def test_negative_return_is_a_past_loss_not_a_better_forecast(self):
        text = score_reason_text("trailing_return_20d", -.07, -.03, .02)
        self.assertIn("previous 20 trading days", text)
        self.assertIn("loss of 7.0%", text)
        self.assertIn("loss of 3.0%", text)
        self.assertIn("past, not the next trading window", text)
        self.assertIn("helped its ranking score", text)
        self.assertNotIn("will rise", text)

    def test_return_gain_loss_and_zero_are_spelled_out(self):
        text = observation_text("trailing_return_5d", .02, 0.)
        self.assertIn("gain of 2.0%", text)
        self.assertIn("roughly no change", text)

    def test_volume_is_explained_against_its_own_normal_level(self):
        text = observation_text("relative_volume_20d", .68, 1.2)
        self.assertIn("32.0% below its usual level", text)
        self.assertIn("20.0% above its usual level", text)
        self.assertIn("Each stock is compared with its own recent average", text)
        self.assertIn("around its usual level", observation_text("relative_volume_20d", 1., 1.))

    def test_variability_is_not_presented_as_a_future_loss_bound(self):
        text = score_reason_text("volatility_20d", .01, .02, -.1)
        self.assertIn("less variable", text)
        self.assertIn("not a limit on future losses", text)
        self.assertIn("counted against", text)
        self.assertIn("more variable", observation_text("volatility_20d", .03, .02))
        self.assertIn("about as variable", observation_text("volatility_20d", .02, .02))

    def test_future_features_and_missing_values_have_honest_fallbacks(self):
        self.assertIn("custom factor", observation_text("custom_factor", 2., 1.))
        self.assertIn("unavailable", observation_text("custom_factor", float("nan"), 1.))
        self.assertIn("no meaningful measured score effect", score_reason_text("custom_factor", 1., 1., 0.))

    @staticmethod
    def stats(mean=.2, edge=.001, count=100):
        return {"mean": mean, "mean_selection_edge": edge, "selection_date_count": count}

    def test_favorable_evidence_is_not_a_buy_approval_or_stock_probability(self):
        readout = evidence_readout(self.stats(), .1, 5)
        self.assertIn("not a proven investment case", readout.heading)
        self.assertIn("100 validation dates", readout.selection_result)
        self.assertIn("0.10 percentage points more", readout.selection_result)
        self.assertIn("following 5 trading days", readout.selection_result)
        self.assertIn("not independent proof", readout.limitation)
        self.assertIn("confidence percentage", readout.limitation)

    def test_weak_and_negative_evidence_is_not_hidden(self):
        readout = evidence_readout(self.stats(mean=-.02, edge=-.003), .1, 5)
        self.assertIn("did not show positive", readout.heading)
        self.assertIn("0.30 percentage points less", readout.selection_result)
        readout = evidence_readout(self.stats(mean=.05), .1, 5)
        self.assertIn("did not beat", readout.heading)
        readout = evidence_readout(self.stats(edge=-.001), .1, 20)
        self.assertIn("did not establish better stock picks", readout.heading)
        self.assertIn("20 trading days", readout.selection_result)

    def test_undefined_and_zero_evidence_are_not_filled_with_fake_confidence(self):
        readout = evidence_readout(self.stats(mean=float("nan"), edge=float("nan"), count=0), .1, 5)
        self.assertIn("incomplete", readout.heading)
        self.assertIn("no defined", readout.selection_result)
        self.assertIn("about the same", evidence_readout(self.stats(edge=0.), .1, 5).selection_result)


if __name__ == "__main__":
    unittest.main()
