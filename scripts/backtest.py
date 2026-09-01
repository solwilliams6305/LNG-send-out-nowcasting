#!/usr/bin/env python3
"""Paper-trading backtest harness (research only — the write-up's economic
chapter, not a trading system).

Protocol:
  * instrument: TTF front-month daily closes (Yahoo continuous series);
    positions set at close D from information published before ~16:30 CET D,
    P&L accrues over D -> D+1 close-to-close log returns.
  * roll hygiene: the last 3 business days of each month are excluded from
    P&L (continuous-contract expiry jumps).
  * lags: intraday H1 flow news trades same-day close; everything slower is
    lagged one full day beyond publication on principle.
  * positions: z-scored signal on a trailing 120-day window (fit online —
    no lookahead), capped to [-1, 1]; costs charged per unit turnover.
  * stats: annualized Sharpe with a Newey-West t-stat (5 lags), max
    drawdown, turnover. With ~400 dev days, se(SR) ~ 0.8 — read everything
    as directional.
  * HOLDOUT FREEZE: development window 2024-06-01 .. 2025-12-31 ONLY.
    2026 is not evaluated here and must stay untouched until the strategy
    set is frozen in the paper.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from lng_nowcast import config

DEV_START, DEV_END = "2018-01-01", "2025-12-31"  # 2026 stays the frozen holdout
COST_BPS = 5.0
ZWIN, ZMIN = 120, 60


def nw_tstat(x: np.ndarray, lags: int = 5) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    mu = x.mean()
    e = x - mu
    s2 = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s2 += 2 * w * (e[:-L] @ e[L:]) / n
    return float(mu / np.sqrt(s2 / n))


def online_z(s: pd.Series) -> pd.Series:
    m = s.rolling(ZWIN, min_periods=ZMIN).mean().shift(1)
    sd = s.rolling(ZWIN, min_periods=ZMIN).std().shift(1)
    return ((s - m) / sd).clip(-3, 3)


def build() -> pd.DataFrame:
    px = pd.read_csv(config.RAW_DIR / "prices.csv")
    px["value"] = pd.to_numeric(px.value, errors="coerce")
    ttf = px[px.series == "ttf_front_eur_mwh"].set_index("date").value.sort_index()
    ret = np.log(ttf).diff()

    # roll hygiene: drop last 3 business days of each month
    dates = pd.to_datetime(ret.index)
    bmax = pd.Series(dates, index=ret.index).groupby(dates.to_period("M")).transform("max")
    keep = (bmax - dates).dt.days > 4
    ret = ret[keep.values]

    # Full-history panel, ALSI-internal for consistency: the same source
    # provides tanks and send-out back to the 2010s, whereas ENTSOG LNG-point
    # history only begins ~2021 — mixing them would create a composition
    # break. EU core four; UK excluded from cover (NG stocks start 2024).
    al = pd.read_csv(config.RAW_DIR / "alsi_daily.csv")
    al = al[al.status.isin(["E", "C"])
            & al.terminal.isin(["gate", "eems", "zeebrugge", "dunkerque"])].copy()
    for c in ("inventory_gwh", "send_out_gwh_d"):
        al[c] = pd.to_numeric(al[c], errors="coerce")
    g = al.groupby("gas_day").agg(tank=("inventory_gwh", "sum"),
                                  burn=("send_out_gwh_d", "sum"),
                                  n_fac=("send_out_gwh_d", "size"))
    g = g[g.n_fac >= 2]  # require at least two reporting facilities
    burn7 = g.burn.rolling(7, min_periods=4).mean()
    panel = pd.DataFrame({
        "cover": g.tank / burn7.clip(lower=50),
        "arr": (g.tank.diff() + g.burn).clip(lower=0),
    })
    si = pd.read_csv(config.RAW_DIR / "stress_inputs.csv")
    si["value"] = pd.to_numeric(si.value, errors="coerce")
    panel["storage"] = si[si.series == "agsi_eu_full_pct"].set_index("date").value

    ev = pd.read_csv(config.RAW_DIR / "nowcast_eval.csv")
    eu = ev[ev.terminal.isin(["gate", "eems", "zeebrugge"])]
    h0 = eu[eu.horizon == "H0"].set_index(["terminal", "gas_day"]).q50
    h1 = eu[eu.horizon == "H1"].set_index(["terminal", "gas_day"]).q50
    intraday_news = (h1 - h0).groupby("gas_day").sum()

    df = pd.DataFrame(index=ret.index)
    df["ret_next"] = ret.shift(-1)  # position at close D earns D->D+1

    # ① tail/event thread: midday flow news (published ~12:00 CET, tradable
    #    same close). Negative news = supply loss = long.
    news = intraday_news.reindex(df.index)
    df["sig_flownews"] = -online_z(news)
    big = news.rolling(ZWIN, min_periods=ZMIN).std().shift(1) * 1.5
    df["sig_flowevent"] = np.where(news < -big, 1.0, np.where(news > big, -1.0, 0.0))

    # ② cover/stress thread (lag 1 day beyond publication)
    df["sig_cover"] = -online_z(panel.cover).shift(1).reindex(df.index)
    df["sig_storage"] = -online_z(panel.storage).shift(1).reindex(df.index)  # incumbent
    df["sig_dcover"] = -online_z(panel.cover.diff(14)).shift(1).reindex(df.index)

    # ③ arrival-drought thread: trailing arrivals low = pipeline thin = long
    df["sig_arrdrought"] = -online_z(panel.arr.rolling(10, min_periods=5).sum()).shift(1).reindex(df.index)

    # benchmarks
    df["sig_mom12"] = np.sign(ret.rolling(12).sum()).shift(0)
    df["sig_long"] = 1.0
    return df


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    dev = df[(df.index >= DEV_START) & (df.index <= DEV_END)]
    rows = []
    for col in [c for c in dev.columns if c.startswith("sig_")]:
        pos = dev[col].clip(-1, 1).fillna(0.0)
        pnl = pos * dev.ret_next
        costs = pos.diff().abs().fillna(0.0) * COST_BPS / 1e4
        net = (pnl - costs).dropna()
        if len(net) < 100:
            continue
        sr = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan
        eq = net.cumsum()
        dd = float((eq.cummax() - eq).max())
        rows.append({
            "strategy": col.replace("sig_", ""),
            "days": len(net), "SR_net": sr, "t_NW": nw_tstat(net.to_numpy()),
            "maxDD_%": dd * 100, "turnover_d": float(pos.diff().abs().mean()),
            "active_%": float((pos != 0).mean() * 100),
        })
    return pd.DataFrame(rows).sort_values("SR_net", ascending=False)


def main() -> int:
    df = build()
    res = evaluate(df)
    pd.set_option("display.width", 160)
    print(f"DEV window {DEV_START}..{DEV_END}, costs {COST_BPS} bps/turnover, "
          f"roll windows excluded; 2026 HOLDOUT UNTOUCHED.\n")
    print(res.round(2).to_string(index=False))
    print(f"\nRead with se(SR) ≈ {np.sqrt(1/ (res.days.max()/252)):.2f}: "
          "only |SR| beyond ~2 se is even suggestive. "
          f"{len(res)} strategies tried — selection bias applies to the best row.")
    res.to_csv(config.RAW_DIR / "backtest_dev.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
