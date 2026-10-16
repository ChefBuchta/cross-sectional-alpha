import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from factor_model.config import EDA_DATA_PATH
from factor_model.backtest import BacktestConfig
from factor_model.console import build_console_report, format_console_report
from factor_model.models import ExperimentConfig
from factor_model.pipeline import AVAILABLE_MODELS, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare stock rankings and long-only backtests.")
    parser.add_argument("--prices", type=Path, default=EDA_DATA_PATH)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--commission-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument(
        "--models", nargs="+", choices=AVAILABLE_MODELS, default=list(AVAILABLE_MODELS),
        help="Model families to compare (default: all five); benchmarks are always included.",
    )
    args = parser.parse_args()
    backtest_config = BacktestConfig(
        top_n=args.top_n, rebalance_every=args.rebalance_every,
        commission_bps=args.commission_bps, slippage_bps=args.slippage_bps,
        annual_risk_free_rate=args.risk_free_rate,
    )
    raw_df = pd.read_csv(args.prices, parse_dates=["date"])
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
    print("Training and selecting models using validation rankings...", flush=True)
    result = run_experiment(prices=raw_df, config=config, model_names=tuple(args.models))
    print("Simulating frozen models and benchmarks...", flush=True)
    report = build_console_report(raw_df, result, backtest_config)
    print(format_console_report(report, config.target_horizon))

if __name__ == "__main__":
    main()
