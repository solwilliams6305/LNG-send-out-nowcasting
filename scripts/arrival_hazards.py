#!/usr/bin/env python3
"""U6 groundwork: cargo-arrival renewal hazards from the historical dataset.

The AIS layer will eventually modulate arrival intensity in real time; the
baseline it modulates is estimable today from the 1,446-event arrivals
dataset: per terminal, the inter-arrival gap distribution and its discrete
hazard λ(a) = P(arrival after gap a | gap ≥ a). A memoryless (geometric)
renewal is what the filter's current Bernoulli(λ) input assumes; berth
scheduling should produce structure instead — a refractory dip at short gaps
(the berth is busy / slots are spaced) and a hump near the scheduling cycle.
The Negative-Binomial fit vs geometric log-likelihood quantifies it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lng_nowcast import config
from lng_nowcast.pgas import _nb_loglik


def main() -> int:
    ev = pd.read_csv(config.RAW_DIR / "arrival_events.csv")
    ev["start_day"] = pd.to_datetime(ev.start_day)
    print(f"{len(ev)} arrival events, {ev.terminal.nunique()} terminals\n")
    print(f"{'terminal':>14} {'n_gaps':>6} {'mean':>6} {'var':>7} "
          f"{'disp':>6} {'ΔlogL NB−geo':>13}   empirical hazard by gap (1..8 days)")
    for t, g in ev.groupby("terminal"):
        gaps = g.sort_values("start_day").start_day.diff().dt.days.dropna()
        gaps = gaps[gaps > 0].astype(int).to_numpy()
        if len(gaps) < 20:
            continue
        mean, var = float(gaps.mean()), float(gaps.var())
        ll_geo = _nb_loglik(list(gaps), mean, max(mean - 1, 0.5))  # geometric limit
        ll_nb = _nb_loglik(list(gaps), mean, var)
        # empirical discrete hazard
        haz = []
        for a in range(1, 9):
            at_risk = (gaps >= a).sum()
            haz.append((gaps == a).sum() / at_risk if at_risk >= 10 else np.nan)
        hz = " ".join("  ." if np.isnan(x) else f"{x:.2f}" for x in haz)
        print(f"{t:>14} {len(gaps):6d} {mean:6.1f} {var:7.1f} "
              f"{var/mean:6.2f} {ll_nb - ll_geo:13.1f}   {hz}")
    print("\ndisp > 1 with rising-then-falling hazard = scheduled/refractory "
          "renewals, not Poisson: the filter's constant-λ arrival input is the "
          "memoryless special case, and an age-dependent hazard (days since "
          "last arrival, carried per particle) is the drop-in upgrade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
