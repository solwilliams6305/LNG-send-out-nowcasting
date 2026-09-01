#!/usr/bin/env python3
"""Preliminary EU intraday revision analysis + UK live-trajectory validation.

Part 1 — ALSI E→C trajectories: for each (facility, gas day), how the reported
send-out/inventory evolved across our 3×-daily snapshots, from the intraday
Estimated row to the Confirmed value. Grows automatically with the archive;
with ~2 days it prints raw trajectories rather than statistics.

Part 2 — UK live trajectories (nationalgas_live): integrate the 2-minutely
day-so-far flow curve and compare its implied average rate with the official
D+1 daily figure. Note: the 22:15 UTC snapshot reaches gas-hour ~18¼, so the
integral is partial by construction — which is exactly how it will be used as
a cum_h-style observation (u ≈ 0.76) in the UK filter horizons.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob

import numpy as np
import pandas as pd

from lng_nowcast import config

MCM_TO_GWH = 11.0  # ~39.6 MJ/m3 CV; refine with the per-terminal CV series


def alsi_trajectories() -> None:
    files = sorted(glob.glob(str(config.SNAPSHOT_DIR / "alsi" / "*.csv")))
    if not files:
        print("no alsi snapshots yet")
        return
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df[df.status.isin(["E", "C"])]
    core = df[df.terminal.isin(["gate", "eems", "zeebrugge", "dunkerque"])]
    days = sorted(core.gas_day.unique())[-3:]
    print(f"=== ALSI E→C trajectories ({len(files)} snapshots, showing {days}) ===")
    for (t, d), g in core[core.gas_day.isin(days)].groupby(["terminal", "gas_day"]):
        g = g.sort_values("snapshot_utc")
        if g.send_out_gwh_d.nunique() <= 1 and g.status.nunique() <= 1 and len(g) < 3:
            continue
        path = " → ".join(
            f"{r.status}:{r.send_out_gwh_d:.1f}@{r.snapshot_utc[11:16]}Z"
            for r in g.itertuples()
        )
        print(f"  {t:>10} {d}: {path}")
    # summary scaffold (meaningful once weeks of snapshots exist)
    stats = []
    for (t, d), g in core.groupby(["terminal", "gas_day"]):
        e = g[g.status == "E"].send_out_gwh_d.dropna()
        c = g[g.status == "C"].send_out_gwh_d.dropna()
        if len(e) and len(c):
            stats.append({"terminal": t, "gas_day": d,
                          "e_last": e.iloc[-1], "c_final": c.iloc[-1],
                          "e_range": e.max() - e.min(),
                          "e_to_c": c.iloc[-1] - e.iloc[-1]})
    if stats:
        s = pd.DataFrame(stats)
        print(f"\n  days with both E and C captured: {len(s)}; "
              f"|E_last − C| mean {s.e_to_c.abs().mean():.2f} GWh, "
              f"max {s.e_to_c.abs().max():.2f}; intraday E-range mean {s.e_range.mean():.2f}")


def uk_live_validation() -> None:
    files = sorted(glob.glob(str(config.SNAPSHOT_DIR / "nationalgas_live" / "*.csv")))
    if not files:
        print("no nationalgas_live snapshots yet")
        return
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    traj = df[df.series == "area_trajectory"].dropna(subset=["epoch_ms", "value_mcm_d"])
    traj["ts"] = pd.to_datetime(traj.epoch_ms, unit="ms", utc=True)
    # UK gas day starts 05:00 local; label by shifting 5h London time
    local = traj.ts.dt.tz_convert("Europe/London")
    traj["gas_day"] = (local - pd.Timedelta(hours=5)).dt.strftime("%Y-%m-%d")

    ng = pd.read_csv(config.RAW_DIR / "nationalgas_daily.csv")
    ng["gwh"] = pd.to_numeric(ng.value, errors="coerce") / 1e6
    daily = (ng[(ng.metric == "send_out") & (ng.maturity == "D+1")]
             .groupby(["terminal", "gas_day"])["gwh"].sum())

    print(f"\n=== UK live-trajectory validation ({len(files)} snapshots) ===")
    for (term, day), g in traj.groupby(["terminal", "gas_day"]):
        g = g.drop_duplicates("ts").sort_values("ts")
        hours = (g.ts.max() - g.ts.min()).total_seconds() / 3600 + 2 / 60
        if hours < 2 or len(g) < 30:
            continue
        # trapezoid over the rate curve: mcm/day * day-fraction
        dt_days = g.ts.diff().dt.total_seconds().fillna(120) / 86400
        cum_mcm = float((g.value_mcm_d * dt_days).sum())
        mean_rate_gwh = cum_mcm / (hours / 24) * MCM_TO_GWH
        # compare with the official daily where published (grain area = grain;
        # milford_haven area = south_hook + dragon)
        if term == "milford_haven":
            official = daily.get(("south_hook", day), np.nan) + daily.get(("dragon", day), np.nan)
        else:
            official = daily.get((term, day), np.nan)
        cmp = f"official D+1 {official:.1f} GWh" if not np.isnan(official) else "official D+1 not yet in raw file"
        print(f"  {term:>14} {day}: {len(g)} pts over {hours:.1f} h, "
              f"cum {cum_mcm:.2f} mcm ⇒ implied daily rate {mean_rate_gwh:.1f} GWh/d | {cmp}")


if __name__ == "__main__":
    alsi_trajectories()
    uk_live_validation()
    sys.exit(0)
