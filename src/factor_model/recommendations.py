"""Plain-language explanations derived from observations, not generated buy claims."""

import numpy as np
from pydantic import BaseModel


class EvidenceReadout(BaseModel):
    """Separate historical group outcomes from a stock-specific recommendation."""

    heading: str
    selection_result: str
    baseline_result: str
    limitation: str


def _past_move(value: float) -> str:
    if abs(value) < 0.00005:
        return "roughly no change"
    return f"a {'gain' if value > 0 else 'loss'} of {abs(value):.1%}"


def _activity(value: float) -> str:
    if abs(value - 1) < 0.0005:
        return "around its usual level"
    return f"{abs(value - 1):.1%} {'above' if value > 1 else 'below'} its usual level"


def observation_text(feature: str, value: float, peer_value: float) -> str:
    """Give each displayed measurement a period, unit, and appropriate reference."""
    if not np.isfinite(value) or not np.isfinite(peer_value):
        return "This observation or its comparison is unavailable."
    if feature.startswith("trailing_return_"):
        period = feature.removeprefix("trailing_return_").removesuffix("d")
        return (
            f"Its return over the previous {period} trading days was {_past_move(value)}, "
            f"compared with {_past_move(peer_value)} for the average stock shown. "
            "This describes the past, not the next trading window."
        )
    if feature.startswith("relative_volume"):
        return (
            f"Trading activity was {_activity(value)}. For the average stock shown, "
            f"activity was {_activity(peer_value)}. Each stock is compared with its own recent average."
        )
    if feature.startswith("volatility"):
        if np.isclose(value, peer_value, rtol=1e-7, atol=1e-12):
            comparison = "about as variable as"
        else:
            comparison = "less variable than" if value < peer_value else "more variable than"
        return (
            f"Its recent day-to-day returns were {comparison} those of the average stock shown. "
            "This describes recent variability, not a limit on future losses."
        )
    label = feature.replace("_", " ")
    return f"Its {label} was {value:,.4g}, compared with {peer_value:,.4g} for the average stock shown."


def score_reason_text(feature: str, value: float, peer_value: float, effect: float,
                      concise: bool = False) -> str:
    observation = observation_text(feature, value, peer_value)
    if concise:
        observation = observation.partition(". ")[0].rstrip(".") + "."
        if not np.isfinite(effect) or abs(effect) < 1e-10:
            return f"{observation} No clear score effect."
        return f"{observation} {'Helped' if effect > 0 else 'Lowered'} this model's score."
    if not np.isfinite(effect) or abs(effect) < 1e-10:
        interpretation = "This input had no meaningful measured score effect in this comparison."
    elif effect > 0:
        interpretation = "In this fitted model, this pattern helped its ranking score."
    else:
        interpretation = "In this fitted model, this pattern counted against its ranking score."
    return f"{observation} {interpretation}"


def evidence_readout(stats: dict[str, float | int], baseline_ic: float, horizon: int) -> EvidenceReadout:
    """No numeric threshold here can declare an investment safe or calibrated."""
    mean = float(stats["mean"])
    edge = float(stats["mean_selection_edge"])
    count = int(stats["selection_date_count"])
    if not np.isfinite(edge) or count == 0:
        result = "There are no defined top-group outcomes available for this historical comparison."
    elif round(abs(edge) * 100, 2) == 0:
        result = (
            f"Across {count} validation dates, the selected group had about the same average "
            f"return as all stocks shown over the following {horizon} trading days."
        )
    else:
        result = (
            f"Across {count} validation dates, the selected group averaged {abs(edge) * 100:.2f} "
            f"percentage points {'more' if edge > 0 else 'less'} than all stocks shown "
            f"over the following {horizon} trading days."
        )
    if not np.isfinite(mean) or not np.isfinite(baseline_ic):
        heading = "Historical evidence is incomplete"
        baseline = "The comparison of stock orderings is undefined; do not infer reliability from this recommendation."
    elif mean <= 0:
        heading = "The historical ordering did not show positive average agreement"
        baseline = "On average, its predicted stock order did not positively match the later-return order in validation."
    elif mean <= baseline_ic:
        heading = "It did not beat the simple comparison rule"
        baseline = "Its stock ordering did not beat simply favoring stocks with stronger past 20-day returns."
    elif not np.isfinite(edge) or edge <= 0:
        heading = "Better ordering did not establish better stock picks"
        baseline = "It beat the simple rule on ordering, but the top group did not outperform the average stock shown."
    else:
        heading = "A favorable comparison, not a proven investment case"
        baseline = "It beat the simple past-return rule on ordering, and its selected group outperformed the universe on average."
    return EvidenceReadout(
        heading=heading, selection_result=result, baseline_result=baseline,
        limitation=(
            "These results helped choose the model. They are not independent proof, a profit "
            "forecast, or a confidence percentage for this stock. Trading costs, execution, "
            "and portfolio risk have not been tested."
        ),
    )
