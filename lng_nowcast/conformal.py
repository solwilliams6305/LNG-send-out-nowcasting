"""Adaptive conformal wrapper over the filter's intervals (memo U4).

Whatever the model's bands claim, this online layer earns long-run coverage
with no assumptions on the data-generating process (Gibbs & Candès 2021).
Reference form: normalize each day's error by the model's own band width,
    s_t = |y_t - q50_t| / w_t,   w_t = (q95_t - q05_t) / 2,
track the (1-alpha) quantile of the scores by Robbins-Monro,
    c_{t+1} = c_t + gamma * (1{s_t > c_t} - alpha),
and report q50 ± c_t * w_t. The update uses only past days — walk-forward
honest. gamma is fixed here; DtACI (Gibbs & Candès 2024) removes that choice
by expert aggregation and is left as a refinement for Solomon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wrap_stream(g: pd.DataFrame, alpha: float, lo: str, hi: str,
                gamma: float = 0.02, warm: float = 1.0) -> pd.DataFrame:
    """Add conformalized [lo_c, hi_c] columns for one sorted (terminal, horizon)
    stream at miscoverage alpha, using the model's [lo, hi] band as the shape."""
    g = g.sort_values("gas_day").copy()
    c = warm
    los, his, covered = [], [], []
    for _, r in g.iterrows():
        w = max((r[hi] - r[lo]) / 2.0, 1e-6)
        lo_c, hi_c = r.q50 - c * w, r.q50 + c * w
        los.append(lo_c)
        his.append(hi_c)
        if not np.isnan(r.truth):
            s = abs(r.truth - r.q50) / w
            covered.append(lo_c <= r.truth <= hi_c)
            c = max(c + gamma * (float(s > c) - alpha), 0.05)
    g[f"{lo}_c"], g[f"{hi}_c"] = los, his
    return g


def wrap_eval(ev: pd.DataFrame) -> pd.DataFrame:
    """Conformalize the 90% and 50% bands per (terminal, horizon) stream."""
    out = []
    for _, g in ev.groupby(["terminal", "horizon"]):
        g = wrap_stream(g, alpha=0.10, lo="q05", hi="q95")
        g = wrap_stream(g, alpha=0.50, lo="q25", hi="q75")
        out.append(g)
    return pd.concat(out, ignore_index=True)
