#!/usr/bin/env python3
"""The bridge-kernel question (memo U1 refinement): what process are the
intraday deviations, really?

For each hourly-covered terminal, build the daily deviation curves
    d(u_h) = cum(u_h)/S − u_h,   u_h = h/24,  h = 1..23
(d(0) = d(1) = 0 by construction) and compare their empirical second-order
structure against two nested models:
  Brownian bridge:  Cov[d(s), d(t)] = σ² (min(s,t) − st)
  OU bridge (rate θ): Cov = σ² sinh(θ min)·sinh(θ(1−max)) / (θ sinh θ)
    — θ → 0 recovers the Brownian bridge; large θ decorrelates faster and
    concentrates variance toward mid-day less.
θ is fitted by least squares on the empirical correlation matrix, so the
scale σ drops out; the variance profile is then checked separately. The
observed "smoother than Brownian" behaviour (evening deviations smaller than
the bridge extrapolates from the morning) should surface as θ > 0.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lng_nowcast import config

MIN_S = 50.0  # GWh/d: only clearly-active days inform the profile


def ou_bridge_corr(u: np.ndarray, theta: float) -> np.ndarray:
    s = np.minimum.outer(u, u)
    t = np.maximum.outer(u, u)
    if theta < 1e-4:  # Brownian-bridge limit
        cov = s - np.outer(u, u)
    else:
        cov = np.sinh(theta * s) * np.sinh(theta * (1 - t)) / (theta * np.sinh(theta))
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def fbm_bridge_corr(u: np.ndarray, H: float) -> np.ndarray:
    """Fractional-Brownian-motion bridge: condition fBm (Hurst H) on B(1)=0.
    C_H(s,t) = ½(s^2H + t^2H − |t−s|^2H);  K = C(s,t) − C(s,1)C(t,1)/C(1,1).
    H = ½ recovers the Brownian bridge; H > ½ gives persistent (smoother)
    deviation paths with higher inter-time correlations."""
    def C(a, b):
        return 0.5 * (a ** (2 * H) + b ** (2 * H) - np.abs(a - b) ** (2 * H))
    s, t = np.meshgrid(u, u, indexing="ij")
    K = C(s, t) - C(s, 1.0) * C(t, 1.0) / C(1.0, 1.0)
    d = np.sqrt(np.diag(K))
    return K / np.outer(d, d)


def main() -> int:
    h = pd.read_csv(config.RAW_DIR / "entsog_hourly_flows.csv")
    h = h[h.indicator == "Physical Flow"].copy()
    h["gwh"] = pd.to_numeric(h.value_kwh, errors="coerce") / 1e6
    local = pd.to_datetime(h.period_from.str[:19])
    shifted = local - pd.Timedelta(hours=6)
    h["gas_day"] = shifted.dt.strftime("%Y-%m-%d")
    h["hour"] = shifted.dt.hour + 1

    hours = np.arange(1, 24)
    u = hours / 24.0
    for term, g in h.groupby("terminal"):
        piv = g.pivot_table(index="gas_day", columns="hour", values="gwh", aggfunc="sum")
        piv = piv.dropna()
        if piv.shape[1] < 23 or len(piv) < 100:
            print(f"{term}: insufficient hourly coverage")
            continue
        cum = piv.sort_index(axis=1).cumsum(axis=1)
        S = cum.iloc[:, -1]
        keep = S > MIN_S
        if keep.sum() < 100:
            print(f"{term}: only {int(keep.sum())} usable active days, skipping")
            continue
        d = (cum.loc[keep, hours].div(S[keep], axis=0)
             .sub(u, axis=1)).to_numpy()
        n = len(d)

        emp_corr = np.corrcoef(d, rowvar=False)
        sse_bb = float(np.nansum((emp_corr - ou_bridge_corr(u, 1e-5)) ** 2))
        Hs = np.linspace(0.5, 0.98, 49)
        sse_h = [float(np.nansum((emp_corr - fbm_bridge_corr(u, H)) ** 2)) for H in Hs]
        k = int(np.argmin(sse_h))
        H_hat, sse_fbm = float(Hs[k]), sse_h[k]

        sd = d.std(axis=0)
        ratio = sd / np.sqrt(u * (1 - u))
        print(f"{term}: {n} active days | fBm-bridge Ĥ = {H_hat:.2f} "
              f"(corr SSE {sse_fbm:.1f} vs Brownian {sse_bb:.1f}, "
              f"−{100*(1-sse_fbm/max(sse_bb,1e-9)):.0f}%) | "
              f"sd/√(u(1−u)) morning→noon→evening: "
              f"{ratio[3]:.4f} → {ratio[11]:.4f} → {ratio[19]:.4f}")
        print(f"    corr(d(4h), d(11h)): empirical {emp_corr[3, 10]:.2f}, "
              f"Brownian 0.49, fBm(Ĥ) {fbm_bridge_corr(u, H_hat)[3, 10]:.2f}")
    print("\nĤ > ½ = deviation paths are persistent (a ramp, once begun, holds) —"
          " the bridge family the filter should use is fractional, and the"
          " conditional law d(1-day) | d(u) tightens accordingly at late hours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
