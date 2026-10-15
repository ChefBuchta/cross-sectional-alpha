"""Launch with: uv run streamlit run dashboard/app.py"""

from datetime import date

import pandas as pd
from pydantic import ValidationError
import streamlit as st

from factor_model.config import EDA_DATA_PATH
from factor_model.models import ExperimentConfig
from factor_model.pipeline import run_experiment
from factor_model.reporting import forecast_from_prices, retained_models


@st.cache_data(show_spinner=False, max_entries=4)
def load_prices(path: str, modified_ns: int) -> pd.DataFrame:
    """The modification timestamp is part of the source-data cache key."""
    return pd.read_csv(path, parse_dates=["date"])


st.set_page_config(
    page_title="Relative Edge | Stock selection", page_icon=":material/query_stats:",
    layout="wide",
)
page = st.navigation([
    st.Page("app_pages/picks.py", title="Stock picks", icon=":material/format_list_numbered:", default=True),
    st.Page("app_pages/leaderboard.py", title="Model leaderboard", icon=":material/leaderboard:"),
    st.Page("app_pages/metrics.py", title="Model metrics", icon=":material/analytics:"),
    st.Page("app_pages/help.py", title="Help & methodology", icon=":material/help:"),
], position="top")

st.title("Relative Edge")
st.caption("Understand the stock suggestion, then judge the evidence behind it. No trades are placed.")

prices = None
source_stamp = None
try:
    source_stamp = EDA_DATA_PATH.stat().st_mtime_ns
    prices = load_prices(str(EDA_DATA_PATH), source_stamp)
    required = {"date", "symbol", "adjusted_close", "close", "volume"}
    if required - set(prices.columns) or prices.empty:
        raise ValueError("The CSV needs date, symbol, adjusted_close, close, volume, and at least one row")
except (OSError, ValueError) as error:
    st.sidebar.error(f"Cannot load the local price dataset: {error}")
    st.sidebar.caption("Export data/processed/eda_data.csv from EDA. Help remains available.")
    prices = None

result = st.session_state.get("experiment_result")
if prices is not None:
    with st.sidebar:
        st.header("Research run")
        st.caption(f"Local prices through {prices['date'].max():%d %b %Y}. No live market feed.")
        with st.expander("Experiment settings", expanded=result is None):
            with st.form("experiment_setup"):
                symbols = st.multiselect(
                    "Stocks to compare", sorted(prices["symbol"].unique()),
                    default=sorted(prices["symbol"].unique()), key="symbols",
                )
                start = st.date_input("History starts", date(2018, 1, 1))
                end = st.date_input("History ends (exclusive)", date(2025, 1, 1))
                validation_start = st.date_input("Validation starts", date(2022, 1, 1))
                test_start = st.date_input("Test starts", date(2023, 1, 1))
                horizon = st.selectbox("Predict the next … trading days", [1, 5, 20], index=1, key="target_horizon")
                st.caption("Changing this horizon requires training a new experiment. Five observations is roughly one trading week.")
                evaluate_test = st.checkbox(
                    "Evaluate the selected winner on test", value=False,
                    help="Never use test results to choose model settings.",
                )
                submitted = st.form_submit_button("Run experiment", type="primary", key="run_experiment")
        st.caption("Changing pages, models, or snapshot frequency does not retrain.")
    if submitted:
        for key in ("experiment_result", "prediction_rows", "prediction_error"):
            st.session_state.pop(key, None)
        try:
            config = ExperimentConfig(
                symbols=tuple(symbols), start_date=start, end_date=end,
                validation_start=validation_start, test_start=test_start, target_horizon=horizon,
            )
            with st.spinner("Preparing data and comparing models on validation…"):
                result = run_experiment(
                    prices.loc[prices["symbol"].isin(config.symbols)].copy(), config,
                    evaluate_test=evaluate_test,
                )
            st.session_state["experiment_result"] = result
            st.session_state["source_stamp"] = source_stamp
        except (ValidationError, ValueError, KeyError, TypeError) as error:
            st.error(f"Experiment could not run: {error}")
            result = None

result = st.session_state.get("experiment_result")
if result is not None:
    with st.sidebar:
        st.caption(f"{len(retained_models(result))} fitted models · {len(result.config.symbols)} stocks · {result.config.target_horizon}-day horizon")
        st.caption(f"Training ends before {result.config.validation_start}. Validation selects settings; forecasts start at {result.config.test_start}.")
        st.warning("The original 2023–2024 test period has already been inspected; it is not fresh holdout evidence.")
    if source_stamp != st.session_state.get("source_stamp"):
        st.warning("The price CSV changed or is unavailable. These results belong to the previous source; run a new experiment to update them.")
    elif prices is not None and "prediction_rows" not in st.session_state:
        try:
            with st.spinner("Scoring post-selection price history without refitting…"):
                st.session_state["prediction_rows"] = forecast_from_prices(prices, result)
        except (ValueError, KeyError, TypeError) as error:
            st.session_state["prediction_rows"] = {}
            st.session_state["prediction_error"] = str(error)

page.run()
st.caption("Research prototype · Rankings are not guaranteed gains · Portfolio sizing, trade execution, costs, and a trading backtest are not implemented.")
