#!/usr/bin/env python3
"""First-pass event study: do TTF prices respond to LNG send-out surprises?

Surprise definitions (per gas day D, aggregated over the EU core terminals):
  h0_surprise  = Σ (truth − H0 median)   — the day's total flow news relative
                 to the day-ahead expectation; fully public only D+1 ~09:20
                 via ENTSOG, but knowable intraday via the hourly feed.
  intraday_news = Σ (H2 median − H0 median) over terminals with an intraday
                 channel — the part of the surprise already visible by ~19:00
                 CET on D itself.

Response: TTF front-month log-return over calendar day D (close_D vs
close_{D-1}) and over D+1. A negative surprise (less LNG than expected) is a
supply shock → expected positive price response, i.e. negative slope.

This is deliberately a first pass: no demand/weather controls, aggregate NW
surprise only. A null here bounds the effect size; the W7 study refines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lng_nowcast import config

EU_CORE = ["gate", "eems", "zeebrugge", "dunkerque"]


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 30:
        return np.nan, np.nan, np.nan, n
    X = np.column_stack([np.ones(n), x])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.sum(resid**2) / (n - 2) / np.sum((x - x.mean()) ** 2))
    r = np.corrcoef(x, y)[0, 1]
    return float(beta[1]), float(beta[1] / se), float(r), n


def main() -> int:
    ev = pd.read_csv(config.RAW_DIR / "nowcast_eval.csv")
    px = pd.read_csv(config.RAW_DIR / "prices.csv")

    ttf = px[px.series == "ttf_front_eur_mwh"].copy()
    ttf["value"] = pd.to_numeric(ttf.value, errors="coerce")
    ttf = ttf.sort_values("date").set_index("date")["value"]
    ret = np.log(ttf).diff() * 100  # % log return, indexed by calendar date

    core = ev[ev.terminal.isin(EU_CORE)]
    h0 = core[core.horizon == "H0"]
    surp = (h0.assign(s=h0.truth - h0.q50).groupby("gas_day")["s"].sum()
            .rename("h0_surprise"))
    h2 = core[core.horizon == "H2"]
    h0m = h0.set_index(["terminal", "gas_day"]).q50
    h2m = h2.set_index(["terminal", "gas_day"]).q50
    intr = (h2m - h0m).groupby("gas_day").sum().rename("intraday_news")

    df = pd.concat([surp, intr], axis=1)
    df["ret_d"] = df.index.map(ret)          # TTF return on gas day D
    df["ret_d1"] = df.index.map(ret.shift(-1))  # return the following day

    print(f"{len(df)} gas days; surprise sd {df.h0_surprise.std():.0f} GWh, "
          f"intraday-news sd {df.intraday_news.std():.0f} GWh\n")
    rows = []
    for xcol in ("h0_surprise", "intraday_news"):
        for ycol, label in (("ret_d", "same-day"), ("ret_d1", "next-day")):
            b, t, r, n = ols(df[xcol].to_numpy(float), df[ycol].to_numpy(float))
            rows.append({"signal": xcol, "response": label, "n": n,
                         "slope_%_per_100GWh": b * 100, "t": t, "corr": r})
        big = df[df[xcol].abs() > df[xcol].abs().quantile(0.8)]
        b, t, r, n = ols(big[xcol].to_numpy(float), big["ret_d"].to_numpy(float))
        rows.append({"signal": xcol, "response": "same-day, top-20% |surprise|",
                     "n": n, "slope_%_per_100GWh": b * 100, "t": t, "corr": r})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(res.round(3).to_string(index=False))
    out = config.RAW_DIR / "event_study_firstpass.csv"
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("Sign convention: negative slope = less-LNG-than-expected ⇒ higher TTF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
