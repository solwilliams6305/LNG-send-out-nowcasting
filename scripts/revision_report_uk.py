#!/usr/bin/env python3
"""First quantification of the UK revision process.

Two mechanisms, reported separately:
  1. Within-item republication: does the same publication item ever get
     re-issued for a gas day (multiple generated_at versions)? Needs
     data/raw/nationalgas_versions.csv (backfill --all-versions).
  2. The cross-maturity ladder: D+1 physical -> D+2 commercial -> M+15
     reconciliation differences on matching gas days. Uses the latest-values
     backfill (data/raw/nationalgas_daily.csv).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lng_nowcast import config


def within_item(versions: pd.DataFrame) -> None:
    print("=== 1. Within-item republication (all published versions) ===")
    v = versions.copy()
    v["value"] = pd.to_numeric(v["value"], errors="coerce")
    grp = v.groupby(["metric", "terminal", "pub_id", "gas_day"])
    counts = grp.size()
    multi = counts[counts > 1]
    print(f"(item, gas_day) pairs: {len(counts)}; with >1 version: {len(multi)} "
          f"({100 * len(multi) / max(len(counts), 1):.2f}%)")
    if len(multi) == 0:
        print("-> no within-item revisions: every value is published exactly once;")
        print("   the UK revision process is purely the cross-maturity ladder.")
        return
    rev = grp.agg(first=("value", "first"), last=("value", "last"),
                  n=("value", "size"),
                  t_first=("generated_at", "min"), t_last=("generated_at", "max"))
    rev = rev[rev.n > 1].copy()
    rev["delta_gwh"] = (rev["last"] - rev["first"]) / 1e6
    rev["lag_days"] = (
        pd.to_datetime(rev.t_last) - pd.to_datetime(rev.t_first)
    ).dt.total_seconds() / 86400
    by = rev.groupby(["metric", "terminal"]).agg(
        n_revised=("delta_gwh", "size"),
        med_abs_gwh=("delta_gwh", lambda s: s.abs().median()),
        max_abs_gwh=("delta_gwh", lambda s: s.abs().max()),
        med_lag_d=("lag_days", "median"),
        max_lag_d=("lag_days", "max"),
    )
    print(by.round(3).to_string())
    worst = rev.reindex(rev.delta_gwh.abs().sort_values(ascending=False).index).head(8)
    print("\nlargest within-item revisions:")
    print(worst[["n", "first", "last", "delta_gwh", "lag_days"]].round(2).to_string())


def ladder(daily: pd.DataFrame) -> None:
    print("\n=== 2. Cross-maturity ladder (send-out, GWh/gas day) ===")
    d = daily[daily.metric == "send_out"].copy()
    d["gwh"] = pd.to_numeric(d["value"], errors="coerce") / 1e6
    per = (d.groupby(["terminal", "maturity", "gas_day"])["gwh"].sum()
             .unstack("maturity"))
    for a, b in [("D+1", "D+2"), ("D+2", "M+15")]:
        if a not in per.columns or b not in per.columns:
            continue
        j = per[[a, b]].dropna()
        diff = j[b] - j[a]
        active = j[(j[a].abs() > 1) | (j[b].abs() > 1)]
        adiff = active[b] - active[a]
        print(f"\n{a} -> {b}:")
        print(f"  matched days: {len(j)} (active: {len(active)})")
        print(f"  |diff| median: {diff.abs().median():.3f}  mean: {diff.abs().mean():.3f}  "
              f"p99: {diff.abs().quantile(0.99):.2f}  max: {diff.abs().max():.2f}")
        if len(active):
            print(f"  active-day |diff| as % of active mean flow: "
                  f"{100 * adiff.abs().mean() / active[a].mean():.2f}%")
        by_t = (j[b] - j[a]).abs().groupby("terminal").agg(["median", "mean", "max"])
        print(by_t.round(3).to_string())
        worst = diff.abs().sort_values(ascending=False).head(5)
        print("  largest:")
        for (term, day), gap in worst.items():
            print(f"    {term} {day}: {a}={j.loc[(term, day), a]:.1f} "
                  f"{b}={j.loc[(term, day), b]:.1f} (|diff| {gap:.1f})")


def main() -> int:
    vpath = config.RAW_DIR / "nationalgas_versions.csv"
    dpath = config.RAW_DIR / "nationalgas_daily.csv"
    if vpath.exists():
        within_item(pd.read_csv(vpath))
    else:
        print(f"missing {vpath} — run backfill_nationalgas.py --all-versions")
    if dpath.exists():
        ladder(pd.read_csv(dpath))
    else:
        print(f"missing {dpath} — run backfill_nationalgas.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
