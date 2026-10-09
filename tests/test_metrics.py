import unittest

import pandas as pd

from factor_model.metrics import daily_spearman_ic, summarize_daily_ic


class DailySpearmanICTest(unittest.TestCase):
    def test_reports_daily_rank_agreement_and_undefined_dates(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": date, "symbol": symbol, "baseline_score": score,
                 "target_percentile_5d": target}
                for date, scores, targets in (
                    ("2025-01-02", [1, 2, 3], [1, 2, 3]),
                    ("2025-01-03", [1, 2, 3], [3, 2, 1]),
                    ("2025-01-06", [1, 1, 1], [1, 2, 3]),
                )
                for symbol, score, target in zip(
                    ("AAPL", "MSFT", "JPM"), scores, targets, strict=True
                )
            ]
        )

        daily_ic = daily_spearman_ic(frame)
        summary = summarize_daily_ic(daily_ic)

        self.assertEqual(daily_ic.loc["2025-01-02"], 1.0)
        self.assertEqual(daily_ic.loc["2025-01-03"], -1.0)
        self.assertTrue(pd.isna(daily_ic.loc["2025-01-06"]))
        self.assertAlmostEqual(summary["mean"], 0.0)
        self.assertAlmostEqual(summary["fraction_positive"], 0.5)
        self.assertEqual(summary["defined_day_count"], 2)
        self.assertEqual(summary["undefined_day_count"], 1)


if __name__ == "__main__":
    unittest.main()
