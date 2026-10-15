"""Shared page rendering. Calculations live in factor_model.reporting."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from factor_model.models import ExperimentResult
from factor_model.recommendations import evidence_readout, observation_text, score_reason_text
from factor_model.reporting import (
    model_leaderboard, model_statistics, rank_stocks, retained_models,
    score_explanations, selection_performance, shortlist_history,
)


MODEL_NAMES = {
    "ridge": "Ridge regression", "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting", "baseline": "20-day momentum baseline",
}
STOCK_NAMES = {
    "AAPL": "Apple", "AMGN": "Amgen", "CAT": "Caterpillar", "HD": "Home Depot",
    "JNJ": "Johnson & Johnson", "JPM": "JPMorgan Chase", "MSFT": "Microsoft",
    "NEE": "NextEra Energy", "WMT": "Walmart", "XOM": "ExxonMobil",
}
MODEL_DESCRIPTIONS = {
    "ridge": "Combines standardized inputs with learned weights; regularization keeps those weights from becoming too large.",
    "random_forest": "Averages many decision trees to capture thresholds and interactions between the inputs.",
    "gradient_boosting": "Builds trees in sequence, each correcting previous errors to capture nonlinear patterns.",
}
FEATURE_NAMES = {
    "trailing_return_1d": "Past 1-day return", "trailing_return_5d": "Past 5-day return",
    "trailing_return_20d": "Past 20-day return", "volatility_20d": "Recent daily variability",
    "relative_volume_20d": "Volume vs usual", "average_dollar_volume_20d": "Average dollar volume",
}
METRIC_NAMES = {
    "mean": "Mean rank correlation (IC)", "median": "Median rank correlation",
    "fraction_positive": "Positive-correlation dates", "defined_day_count": "Defined IC dates",
    "undefined_day_count": "Undefined IC dates", "mean_shortlist_return": "Mean shortlist forward return",
    "mean_selection_edge": "Mean selection edge", "fraction_positive_edge": "Positive-edge dates",
    "mean_top_bottom_spread": "Mean top-minus-bottom spread", "selection_date_count": "Defined selection dates",
}
PERCENT_METRICS = {
    "fraction_positive", "mean_shortlist_return", "mean_selection_edge",
    "fraction_positive_edge", "mean_top_bottom_spread",
}


def model_name(name: str) -> str:
    return MODEL_NAMES.get(name, name.replace("_", " ").title())


def feature_name(name: str) -> str:
    return FEATURE_NAMES.get(name, name.replace("_", " ").capitalize())


def stock_name(symbol: str) -> str:
    """Offline display labels, not a live security master or a new data request."""
    return f"{STOCK_NAMES[symbol]} ({symbol})" if symbol in STOCK_NAMES else symbol


def current_result() -> ExperimentResult | None:
    result = st.session_state.get("experiment_result")
    if result is None:
        st.info("Open Experiment settings in the sidebar and run an experiment. No models train until you press Run experiment.")
    return result


def choose_model(result: ExperimentResult, key: str) -> str:
    names = list(retained_models(result))
    return st.selectbox("Model", names, index=names.index(result.selection.model_name),
                        format_func=model_name, key=key)


def metric_table(table: pd.DataFrame) -> None:
    config = {
        METRIC_NAMES[key]: st.column_config.NumberColumn(format="percent" if key in PERCENT_METRICS else "%.4f")
        for key in METRIC_NAMES if key in table
        and key not in ("defined_day_count", "undefined_day_count", "selection_date_count")
    }
    st.dataframe(table.rename(columns=METRIC_NAMES), column_config=config, width="stretch", hide_index=True)


def formatted(value: float, percent: bool = False) -> str:
    if not np.isfinite(value):
        return "Undefined"
    return f"{value:+.2%}" if percent else f"{value:+.4f}"


def metric_help() -> None:
    with st.expander("What do correlation and selection edge actually tell me?"):
        st.markdown(
            "**Rank correlation (IC)** compares the model's stock order with the later-return order "
            "on each prediction date. +1 is perfect ordering, 0 is no rank correlation, −1 is reversed. "
            "It is not stock-to-stock price correlation, profit, or a probability of success.\n\n"
            "**Selection edge** asks whether the highest-scored group had a better later return "
            "than the average of all stocks that day. If the group gained 2% and the universe "
            "gained 1%, the edge is +1 percentage point. It can be positive even when both lost money.\n\n"
            "These are averages of forward windows, not portfolio returns. Multi-day windows overlap; "
            "do not compound them into an account balance. Undefined dates are not counted as zero."
        )


def render_picks() -> None:
    st.header("Your stock recommendation", icon=":material/format_list_numbered:")
    st.markdown("A starting point for research, with the reasons and limitations beside it.")
    result = current_result()
    if result is None:
        for column, title, detail in zip(st.columns(3), (
            "1 / Choose a day, week, or month", "2 / Inspect the top three", "3 / Check the model's track record",
        ), (
            "Browse the most attractive stocks at each available snapshot.",
            "See the ranking, observed inputs, and score explanations—not invented buy reasons.",
            "Compare alternatives and inspect their metrics on separate pages.",
        )):
            with column.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(detail)
        st.caption("Example only: a stock falling 2% can outrank one falling 4%. Relative preference is not a promise of positive returns.")
        return

    with st.expander("Change date or model; browse by day, week, or month"):
        model_control, source_control = st.columns(2)
        with model_control:
            name = choose_model(result, "picks_model")
        run = retained_models(result)[name]
        forecasts = st.session_state.get("prediction_rows", {})
        sources = []
        if name in forecasts and not forecasts[name].empty:
            sources.append("Post-selection forecasts")
        else:
            st.info("No complete post-selection forecast history is available for this model. You can inspect historical evaluation replay below.")
        sources.append("Validation replay")
        if result.test is not None and name == result.selection.model_name:
            sources.append("Test replay (winner only)")
        if st.session_state.get("picks_source") not in sources:
            st.session_state.pop("picks_source", None)
        with source_control:
            source = st.selectbox("Which results?", sources, key="picks_source", format_func=lambda value: {
                "Post-selection forecasts": "Suggestions from local price history",
                "Validation replay": "Historical model-selection results",
                "Test replay (winner only)": "Historical test results (selected model only)",
            }[value])
        if source == "Post-selection forecasts":
            rows = forecasts[name]
            st.caption("Built from information available at each date, including the last local price date. This is historical data, not a live market feed.")
        else:
            evaluation = result.test if source.startswith("Test") else run.validation
            rows = evaluation.scored_rows
            st.caption("Historical evaluation replay. Validation influenced model selection; it is not independent proof of future performance.")
        if error := st.session_state.get("prediction_error"):
            st.warning(f"Latest-price scoring is unavailable: {error}. Evaluation replay remains available.")
        cadence_control, date_control = st.columns([1, 2])
        with cadence_control:
            cadence = st.segmented_control("Browse by", ["Day", "Week", "Month"], default="Day", required=True, key="picks_cadence")
        history = shortlist_history(rows, f"{name}_score", cadence)
        periods = history["Period"].tolist()
        with date_control:
            period = st.selectbox(f"{cadence} snapshot", periods, key=f"picks_period_{cadence}_{source}")
        as_of = history.loc[history["Period"] == period, "As of"].iloc[0]
        st.caption("Week/month = last available date in the period, which may be incomplete. The forecast horizon stays unchanged.")
    selected = rank_stocks(rows.loc[pd.to_datetime(rows["date"]) == as_of], f"{name}_score")
    group = selected.loc[selected["shortlisted"]]
    horizon = result.config.target_horizon
    st.subheader(f"As of {as_of:%d %b %Y}: next {horizon} trading days")
    st.caption(f"Historical example using {model_name(name)}. This is not a recommendation for today's market.")
    explanations, kind = score_explanations(run.selection, selected)
    stats = model_statistics(run.validation, name, horizon)
    baseline = run.validation.summary.loc["baseline", "mean"]
    evidence = evidence_readout(stats, baseline, horizon)

    if group.empty:
        st.info("All scores are tied. There is no first-choice stock or meaningful recommendation on this date.")
    else:
        leaders = selected.loc[selected[f"{name}_score"] == selected[f"{name}_score"].max()]
        lead_symbol = leaders.iloc[0]["symbol"]
        recommendation, track_record = st.columns([1.8, 1])
        with recommendation.container(border=True):
            st.caption("Recommendation for further research")
            if len(leaders) == 1:
                st.header(stock_name(lead_symbol))
                st.markdown(f"**The model's first choice among {len(selected)} stocks for the next {horizon} trading days.**")
            else:
                st.header("Several stocks share first place")
                st.markdown(", ".join(stock_name(symbol) for symbol in leaders["symbol"]))
                st.markdown("The model gives them the same score. There is no single best candidate.")
            st.markdown(
                "Start here if you want to investigate this model's idea. It expects this "
                "stock to do better **relative to the others**—not necessarily to rise."
            )
            if len(leaders) > 1:
                st.caption(f"Example explanation for one tied candidate: {stock_name(lead_symbol)}.")
            st.subheader("Why the model favors it")
            render_plain_reasons(selected, explanations, lead_symbol, maximum=1)
            st.caption("The final rank combines all model inputs. This explains the model's choice—not why the stock's price must rise.")
        with track_record.container(border=True):
            st.subheader("Is this enough to buy?")
            st.warning("No—not on this result alone.")
            st.markdown(f"**{evidence.heading}.**")
            st.markdown(evidence.selection_result)
            st.markdown(evidence.baseline_result)
            st.caption(evidence.limitation)
            st.caption("This is the model's historical group performance, not a forecast or confidence score for the named stock.")

        others = group.loc[~group["symbol"].isin(leaders["symbol"])]
        if not others.empty:
            st.subheader("Other candidates to compare")
            for _, candidate in others.iterrows():
                symbol = candidate["symbol"]
                st.markdown(f"**Rank {candidate['predicted_rank']:g}: {stock_name(symbol)}**")
                render_plain_reasons(selected, explanations, symbol, maximum=1)
            if kind == "Single-feature sensitivity":
                st.caption("Reasons can differ between stocks: tree models evaluate combinations of inputs, not a universal rule such as ‘lower variability is always better’.")
        if len(group) > 3:
            st.caption("The third position is tied. All stocks at that cutoff remain on the shortlist.")

    st.subheader("Understand the suggestion")
    symbol = st.selectbox("Inspect a stock", selected["symbol"].tolist(), key="picks_explain",
                          format_func=stock_name)
    stock = selected.set_index("symbol").loc[symbol]
    st.markdown(f"**What helped and what counted against {stock_name(symbol)}**")
    render_plain_reasons(selected, explanations, symbol, maximum=2, include_negative=True)

    with st.expander("Recent history — not a forecast"):
        st.markdown(
            "**These observations happened before the prediction date.** A negative past "
            "return means a past loss. A larger past loss is not automatically better, "
            f"and none of these percentages predicts the next {horizon} trading days."
        )
        for feature in run.selection.feature_columns:
            st.markdown(f"**{feature_name(feature)}**")
            st.markdown(observation_text(feature, stock[feature], selected[feature].mean()))
            if feature.startswith("volatility"):
                st.caption(f"Measured daily-return variability: {stock[feature]:.2%} (standard deviation). This is not a future loss estimate.")

    with st.expander("How the model-score explanation was calculated"):
        st.markdown(f"**Method: {kind}.**")
        if kind == "Exact score contributions":
            st.markdown(
                "Each input's bar explains part of the difference between this stock's "
                "Ridge score and the same-date average score. The bars add up to that "
                "score difference. They do not add up to an expected return."
            )
        else:
            st.markdown(
                "We replace one input at a time with its same-date peer average and "
                "check how the score changes. This is a hypothetical sensitivity test, "
                "not an exact additive explanation. Interactions and unusual input "
                "combinations can affect it."
            )
        render_score_chart(explanations.loc[symbol])

    with st.expander("Historical evidence and metric definitions"):
        st.markdown(evidence.selection_result)
        st.markdown(f"Average ordering agreement: {formatted(stats['mean'])}, versus {formatted(baseline)} for the momentum rule.")
        st.caption("The agreement scale runs from −1 (reversed order) to +1 (perfect order); 0 means no rank correlation. It is not a percentage return.")
        st.caption(evidence.limitation)
        metric_help()
    with st.expander("Complete ranking and later outcomes (where already known)"):
        columns = ["symbol", "predicted_rank", "shortlisted", f"{name}_score"]
        return_column = f"forward_return_{horizon}d"
        if return_column in selected:
            columns += [return_column, "label_end_date"]
            st.caption("Later outcomes were unknown when making this prediction. They never determine the recommendation.")
        else:
            st.caption("No future-return labels are used in forecast mode. See Model metrics for evaluated historical outcomes.")
        st.dataframe(selected[columns], hide_index=True, width="stretch")
    with st.expander(f"Recommendation history by {cadence.lower()}"):
        st.dataframe(history, hide_index=True, width="stretch")
        st.download_button("Download snapshot history", history.to_csv(index=False).encode(),
                           file_name=f"{name}_{cadence.lower()}_picks.csv", mime="text/csv")
    st.caption("Not shortlisted does not mean sell or short. This page recommends what to investigate; portfolio sizing, execution, costs, and a trading backtest are still missing.")


def render_plain_reasons(selected: pd.DataFrame, explanations: pd.DataFrame, symbol: str, maximum: int = 2,
                         include_negative: bool = False) -> None:
    """Every reason contains an observed comparison and a limited model claim."""
    stock = selected.set_index("symbol").loc[symbol]
    effects = explanations.loc[symbol]
    positive = effects.loc[effects > 1e-10].sort_values(ascending=False).head(maximum)
    negative = effects.loc[effects < -1e-10].sort_values().head(maximum)
    if positive.empty:
        st.markdown("No individual input showed a meaningful positive effect in this comparison. The combined model score determined the rank; we cannot point to a single favorable reason.")
    for feature, effect in positive.items():
        st.markdown(score_reason_text(feature, stock[feature], selected[feature].mean(), effect))
    if include_negative:
        st.markdown("**What counted against it**")
        if negative.empty:
            st.markdown("No meaningful negative input effect was measured against this reference. That does not mean the stock is safe or that the model is reliable.")
        for feature, effect in negative.items():
            st.markdown(score_reason_text(feature, stock[feature], selected[feature].mean(), effect))


def render_score_chart(effects: pd.Series) -> None:
    table = effects.rename(index=feature_name).rename("Score effect").rename_axis("Input").reset_index()
    table["Meaning"] = np.where(table["Score effect"] > 1e-10, "Raised model score",
                                np.where(table["Score effect"] < -1e-10, "Lowered model score", "No measured effect"))
    st.markdown("**Which inputs raised or lowered the model's score?**")
    st.caption("These bars explain a score calculation, not how much money the stock will make. The horizontal axis is a model-score change, not a percentage return.")
    chart = alt.Chart(table).mark_bar().encode(
        x=alt.X("Score effect:Q", title="Model-score effect — not a percentage return"),
        y=alt.Y("Input:N", sort="-x", axis=alt.Axis(labelLimit=240)),
        color=alt.Color("Meaning:N", scale=alt.Scale(
            domain=["Raised model score", "Lowered model score", "No measured effect"],
            range=["#17645B", "#86561B", "#53645F"],
        ), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["Input", "Meaning", alt.Tooltip("Score effect:Q", format=".4f")],
    ).properties(height=max(170, len(table) * 42))
    st.altair_chart(chart, width="stretch")
    st.dataframe(table, hide_index=True, width="stretch")


def render_leaderboard() -> None:
    st.header("Which models performed best?", icon=":material/leaderboard:")
    st.markdown("All fitted models compete on the same validation stocks and dates, alongside a simple momentum rule.")
    result = current_result()
    if result is None:
        return
    try:
        table = model_leaderboard(result)
    except ValueError as error:
        st.error(str(error))
        return
    st.caption(f"Recorded family winner: {model_name(result.selection.model_name)}. Selection criterion: mean daily rank correlation. A baseline can still beat every trained model.")
    shown = table.copy()
    shown["model"] = shown["model"].map(model_name)
    shown = shown.rename(columns={"model": "Model", "performance_rank": "Performance rank"})
    columns = ["Performance rank", "Model", "mean", "mean_selection_edge", "mean_shortlist_return", "fraction_positive_edge", "defined_day_count"]
    metric_table(shown[columns])
    st.bar_chart(shown.set_index("Model")[["mean"]].rename(columns={"mean": "Average rank correlation"}), horizontal=True, sort=False)
    st.caption("Higher correlation means better stock ordering, not higher profits. Selection edge is the top group's mean forward return minus the whole universe's mean return; not a portfolio return.")
    metric_help()

    st.subheader("Do the models choose the same stocks?")
    runs = retained_models(result)
    dates = sorted(pd.to_datetime(result.validation.scored_rows["date"]).unique(), reverse=True)
    chosen_date = st.selectbox("Validation prediction date", dates, key="leaderboard_date",
                               format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y"))
    ranks = {}
    choices = []
    for name, run in runs.items():
        rows = run.validation.scored_rows
        day = rank_stocks(rows.loc[pd.to_datetime(rows["date"]) == chosen_date], f"{name}_score")
        ranks[model_name(name)] = day.set_index("symbol")["predicted_rank"]
        choices.append({"Model": model_name(name), "Research picks": ", ".join(day.loc[day["shortlisted"], "symbol"]) or "No preference"})
    st.dataframe(pd.DataFrame(choices), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(ranks).rename_axis("Stock"), width="stretch")
    st.caption("Ranks are comparable; raw model scores are not. Agreement between models is not a calibrated probability of success.")
    with st.expander("How do these models differ?"):
        for name in runs:
            st.markdown(f"**{model_name(name)}:** {MODEL_DESCRIPTIONS.get(name, 'Uses the fitted estimator and input schema recorded in this experiment.')} ")
    st.warning("This leaderboard uses validation only. The final test is never used to choose a winning family or change its settings.")


def render_metrics() -> None:
    st.header("Evidence for each model", icon=":material/analytics:")
    result = current_result()
    if result is None:
        return
    name = choose_model(result, "metrics_model")
    run = retained_models(result)[name]
    periods = ["Validation"]
    if result.test is not None and name == result.selection.model_name:
        periods.append("Test")
    if st.session_state.get("metrics_period") not in periods:
        st.session_state.pop("metrics_period", None)
    period = st.segmented_control("Evaluation period", periods, default="Validation", required=True, key="metrics_period")
    evaluation = result.test if period == "Test" else run.validation
    st.caption(f"{model_name(name)} · {period} · {result.config.target_horizon}-day target")
    if period == "Validation":
        st.caption("Validation helped select the model and parameters, so these results are not an independent estimate.")
    else:
        st.warning("Test inspection for the recorded winner only. Do not tune the model using this page; the original historical test has already been seen.")
    stats = model_statistics(evaluation, name, result.config.target_horizon)
    baseline_ic = evaluation.summary.loc["baseline", "mean"]
    with st.container(horizontal=True):
        st.metric("Mean rank correlation", formatted(stats["mean"]),
                  delta=f"{stats['mean'] - baseline_ic:+.4f} vs momentum", border=True)
        st.metric("Mean shortlist edge", formatted(stats["mean_selection_edge"], True), border=True,
                  help="Top group's average future return minus the universe average. Not portfolio profit.")
        fraction = stats["fraction_positive"]
        st.metric("Positive-correlation dates", f"{fraction:.1%}" if np.isfinite(fraction) else "Undefined", border=True,
                  help="Fraction of defined dates with IC above zero. Not stock accuracy or a trading win rate.")
    if np.isfinite(stats["mean"]) and np.isfinite(baseline_ic):
        verdict = "beat" if stats["mean"] > baseline_ic else "did not beat"
        st.info(f"On average ranking agreement, this model {verdict} the simple momentum baseline here. That alone does not prove reliable or profitable picks.")
    else:
        st.info("Ranking agreement is undefined. Inspect the undefined-date count before drawing conclusions.")
    metric_help()
    metric_table(pd.DataFrame([{"Model": model_name(name), **stats}]))

    st.subheader("Was the performance consistent?")
    daily = evaluation.daily_ic[["baseline", name]].copy().sort_index()
    daily.index = pd.to_datetime(daily.index)
    bounds = st.date_input("Display dates", value=(daily.index.min().date(), daily.index.max().date()),
                           min_value=daily.index.min().date(), max_value=daily.index.max().date(), key=f"metrics_dates_{name}_{period}")
    if len(bounds) != 2:
        st.info("Choose both endpoints of the display range.")
        return
    start, end = map(pd.Timestamp, bounds)
    chart = st.selectbox("Chart", ["Rolling rank correlation", "Daily rank correlation", "Monthly rank correlation", "Correlation distribution", "Shortlist edge over time"], key="metrics_chart")
    names = {"baseline": "Momentum baseline", name: model_name(name)}
    if chart == "Rolling rank correlation":
        window = st.selectbox("Smoothing window (prediction dates)", [5, 20, 60, 120, 252], index=1, key="metrics_window")
        visible = daily.rolling(window, min_periods=window).mean().loc[start:end].rename(columns=names)
        if visible.dropna(how="all").empty:
            st.info("Not enough defined history for this smoothing window.")
        else:
            st.line_chart(visible, y_label="Mean rank correlation", height=300)
        st.caption("Smoothing changes only the graph. It does not change the prediction horizon or retrain the model.")
    elif chart == "Daily rank correlation":
        st.line_chart(daily.loc[start:end].rename(columns=names), y_label="Rank correlation", height=300)
    elif chart == "Monthly rank correlation":
        st.bar_chart(daily.loc[start:end].resample("MS").mean().rename(columns=names), stack=False, y_label="Mean rank correlation", height=300)
    elif chart == "Correlation distribution":
        bins = np.linspace(-1, 1, 21)
        visible = daily.loc[start:end].rename(columns=names)
        counts = pd.DataFrame({column: np.histogram(visible[column].dropna(), bins=bins)[0] for column in visible}, index=(bins[1:] + bins[:-1]) / 2)
        st.bar_chart(counts, stack=False, x_label="Correlation bin midpoint", y_label="Number of dates", height=300)
    else:
        performance = selection_performance(evaluation.scored_rows, f"{name}_score", result.config.target_horizon)
        st.line_chart(performance.loc[start:end, ["selection_edge"]].rename(columns={"selection_edge": "Top-group edge over universe"}), height=300)
        st.caption("Raw overlapping forward-window differences, not compounded portfolio returns or an equity curve.")
    st.caption("Display filters do not change the full-period numbers. Neighboring multi-day labels overlap; uncertainty has not been quantified.")
    with st.expander("Inspect scored rows and fitted settings"):
        st.json(run.selection.best_parameters)
        st.code("\n".join(run.selection.feature_columns), language=None)
        st.dataframe(run.selection.comparison, hide_index=True, width="stretch")
        st.dataframe(evaluation.scored_rows.head(50), hide_index=True, width="stretch")
        st.download_button("Download evaluated rows", evaluation.scored_rows.to_csv(index=False).encode(),
                           file_name=f"{name}_{period.lower()}_evaluated.csv", mime="text/csv")


def render_help() -> None:
    st.header("From model score to something you can interpret", icon=":material/help:")
    st.markdown("Example: the model ranks AAPL above MSFT for the next five trading days. AAPL could still fall 2% while MSFT falls 4%. The order was right, but buying AAPL would have lost money.")
    with st.expander("What is the actual output?", expanded=True):
        st.markdown("A dated top-three research signal within your stock universe. It says which stocks the chosen model expects to outperform the others—not which stocks must rise. Cutoff ties are included, so a group can contain more than three stocks. If all scores tie, the model abstains. No position sizes or real orders are produced.")
    with st.expander("Day, week, month versus the prediction horizon"):
        st.markdown("Day shows each available trading date. Week shows the last available observation in each Monday–Sunday calendar week. Month does the same for each calendar month. The most recent period may be incomplete. These are viewing snapshots—not averages or promises of monthly returns. The separate 1/5/20-trading-day prediction horizon is set before training. A monthly snapshot can still show a five-day forecast.")
    with st.expander("How are stock explanations different from model performance?"):
        st.markdown("Inputs describe a stock's recent history. Ridge contributions explain its score relative to that date's average score. Other models use single-feature sensitivity: replace one input by the same-date universe average and measure the score change. Such hypothetical effects need not add up and are not causal explanations. Historical IC and selection edge evaluate a model across many dates; they are not reasons a specific stock will rise or confidence percentages.")
    metric_help()
    with st.expander("How are the return and correlation calculated?"):
        st.latex(r"r_{i,t\rightarrow t+h}=\frac{P^{adj}_{i,t+h}}{P^{adj}_{i,t}}-1")
        st.caption("Adjusted closing price today versus h trading observations later. i is the stock; t is the prediction date.")
        st.latex(r"IC_t=\operatorname{Corr}(\operatorname{rank}(s_{i,t}),\operatorname{rank}(r_{i,t\rightarrow t+h}))")
        st.markdown("Spearman correlation compares the two stock orderings on each date. The reported mean gives every defined date equal weight. Constant scores or targets cannot define a correlation, and those dates are counted separately—not changed into zeros.")
        st.latex(r"edge_t=\operatorname{mean}(r_{i,t\rightarrow t+h}:i\in top_t)-\operatorname{mean}(r_{i,t\rightarrow t+h}:i\in universe_t)")
        st.caption("Top and bottom groups use score ranks with cutoff ties. Overlapping groups can occur in a small universe; their spread then has limited meaning.")
    with st.expander("What do the other metrics mean?"):
        st.markdown("**Median IC:** the middle daily correlation. **Positive-correlation dates:** proportion of defined dates with IC > 0, not prediction accuracy. **Mean shortlist return:** average later return of the selected group, not compounded profit. **Top-minus-bottom spread:** mean future return of top-scored stocks minus bottom-scored stocks. **Positive-edge dates:** fraction of defined selection dates when the group beat the universe. All these can be noisy in a small universe.")
    with st.expander("How are models ranked, and why not use test to choose one?"):
        st.markdown("Each candidate is fitted on training data; its settings and family are chosen using mean daily validation IC. The leaderboard includes all retained families plus a simple 20-day momentum baseline. It never ranks families using test results. Only the recorded winner can be evaluated on test. Validation is not independent evidence after tuning, and the original 2023–2024 test period has already been inspected. An attractive leaderboard result alone does not establish a robust advantage.")
    with st.expander("Where does the latest shortlist come from?"):
        st.markdown("The local EDA CSV is the price source—not a live feed. After fitting, the app builds past-only features and scores dates starting at the test boundary, after model selection. It does not need future labels, so it can include the final local price date. Dates without finite inputs for every configured stock are excluded. Validation/test replay uses already evaluated rows instead. A saved historical shortlist is not advice for today's market.")
    with st.expander("What is still missing before this is a trading strategy?"):
        st.markdown("Explicit entry timing, portfolio weights, rebalancing, transaction costs, and a trading backtest. An after-close signal cannot assume a fill at the same closing price. Sharpe ratio and drawdown need a properly defined portfolio-return series; calculating them on rank correlations would not answer the investment question. We also have not established statistical significance, robustness across new periods, or an unbiased universe of stocks.")
    with st.expander("How do I add another model?"):
        st.markdown("Add its training/search function to the pipeline's model-search registry and return the existing ModelSelection / ModelRun structures with a unique name, feature schema, and predict-capable pipeline. The app discovers every retained family automatically in picks, leaderboard, and metrics. Display names have a generic fallback; no additional model-specific page is required. Additional feature engineering belongs in the shared feature pipeline, not in a page.")
