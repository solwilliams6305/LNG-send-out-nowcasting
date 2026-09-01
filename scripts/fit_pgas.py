#!/usr/bin/env python3
"""Demo of U5: PGAS parameter posteriors for the semi-Markov send-out model,
run on one flickering terminal (grain) and one baseload terminal (gate).
Compares posterior mean ± sd with the moment fits the filters currently use.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lng_nowcast import config
from lng_nowcast.hsmm_rbpf import HsmmModel
from lng_nowcast.pgas import Theta, run_pgas

FIT_END = "2024-06-01"


def series(terminal: str, end: str = FIT_END) -> np.ndarray:
    if terminal in ("grain", "south_hook", "dragon"):
        ng = pd.read_csv(config.RAW_DIR / "nationalgas_daily.csv")
        ng = ng[(ng.metric == "send_out") & (ng.maturity == "D+1") & (ng.terminal == terminal)]
        s = (ng.assign(g=pd.to_numeric(ng.value, errors="coerce") / 1e6)
               .groupby("gas_day")["g"].sum())
    else:
        d = pd.read_csv(config.RAW_DIR / "entsog_daily_flows.csv")
        d = d[(d.indicator == "Physical Flow") & (d.terminal == terminal)]
        s = (d.assign(g=pd.to_numeric(d.value_kwh, errors="coerce") / 1e6)
               .groupby("gas_day")["g"].sum())
    return s[s.index < end].dropna().to_numpy(float)


def main() -> int:
    for terminal in ("grain", "gate"):
        y = series(terminal)
        m = HsmmModel.from_history(terminal, y, i_max=4000)
        th0 = Theta(m.idle_mean, m.idle_var, m.active_mean, m.active_var,
                    m.sigma_tight, m.sigma_broad, m.p_broad,
                    m.mu_restart, m.sigma_restart)
        t0 = time.time()
        chain = run_pgas(y, th0, sweeps=300, n_part=60)
        burn = chain[100:]
        print(f"\n=== {terminal} ({len(y)} days, 300 sweeps, {time.time()-t0:.0f}s) ===")
        print(f"{'param':>12} {'moment fit':>11} {'posterior':>20}")
        for k in ("idle_mean", "active_mean", "sig_tight", "sig_broad",
                  "p_broad", "mu_r", "sig_r"):
            vals = np.array([getattr(t, k) for t in burn])
            mom = {"idle_mean": m.idle_mean, "active_mean": m.active_mean,
                   "sig_tight": m.sigma_tight, "sig_broad": m.sigma_broad,
                   "p_broad": m.p_broad, "mu_r": m.mu_restart, "sig_r": m.sigma_restart}[k]
            print(f"{k:>12} {mom:11.2f} {vals.mean():12.2f} ± {vals.std():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
