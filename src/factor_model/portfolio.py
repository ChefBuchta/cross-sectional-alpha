"""Convert one date's stock scores into long-only portfolio weights."""

import numpy as np
import pandas as pd


def top_stock_weights(scores: pd.Series, top_n: int = 3) -> pd.Series:
    """Equal weight the top N, including cutoff ties; all ties hold cash."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if scores.empty or scores.index.has_duplicates or scores.index.isna().any():
        raise ValueError("scores require nonempty, unique stock symbols")
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("scores must be finite")
    weights = pd.Series(0.0, index=scores.index)
    if scores.nunique() > 1:
        selected = scores.rank(ascending=False, method="min") <= top_n
        weights.loc[selected] = 1.0 / selected.sum()
    return weights
