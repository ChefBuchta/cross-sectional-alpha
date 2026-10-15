from datetime import date

import pandas as pd

from factor_model.config import EDA_DATA_PATH
from factor_model.models import ExperimentConfig
from factor_model.pipeline import run_experiment


def main():
    raw_df = pd.read_csv(EDA_DATA_PATH, parse_dates=["date"])
    config = ExperimentConfig(
        symbols=(
            "AAPL", "MSFT", "JPM", "JNJ", "XOM",
            "WMT", "HD", "CAT", "NEE", "AMGN",
        ),
        start_date=date(2018, 1, 1),
        end_date=date(2025, 1, 1),
        validation_start=date(2022, 1, 1),
        test_start=date(2023, 1, 1),
        target_horizon=5,
    )
    run_experiment(prices=raw_df, config=config, evaluate_test=True)

if __name__ == "__main__":
    main()
