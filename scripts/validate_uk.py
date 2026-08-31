#!/usr/bin/env python3
"""Cross-validate UK send-out: National Gas (D+1 physical) vs ENTSOG.

Mappings: grain = Grain NTS1 + NTS2 <-> ENTSOG LNG-00008;
south_hook + dragon <-> ENTSOG LNG-00049 (Milford Haven).
Gas days align exactly (UK 05:00 local ≡ EU 06:00 CET in UTC), so any
systematic gap is measurement/allocation, not timing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lng_nowcast import config


def main() -> int:
    ng = pd.read_csv(config.RAW_DIR / "nationalgas_daily.csv")
    eg = pd.read_csv(config.RAW_DIR / "entsog_daily_flows.csv")

    ng = ng[(ng.metric == "send_out") & (ng.maturity == "D+1")].copy()
    ng["gwh"] = pd.to_numeric(ng["value"], errors="coerce") / 1e6
    ng_daily = ng.groupby(["terminal", "gas_day"], as_index=False)["gwh"].sum()
    grain_ng = ng_daily[ng_daily.terminal == "grain"].set_index("gas_day")["gwh"]
    milford_ng = (
        ng_daily[ng_daily.terminal.isin(["south_hook", "dragon"])]
        .groupby("gas_day")["gwh"].sum()
    )

    eg = eg[eg.indicator == "Physical Flow"].copy()
    eg["gwh"] = pd.to_numeric(eg["value_kwh"], errors="coerce") / 1e6
    eg_daily = eg.groupby(["terminal", "gas_day"])["gwh"].sum()
    grain_eg = eg_daily.loc["grain"]
    milford_eg = eg_daily.loc["milford_haven"]

    for label, a, b in [("Grain: NG(NTS1+2) vs ENTSOG LNG-00008", grain_ng, grain_eg),
                        ("Milford: NG(SH+Dragon) vs ENTSOG LNG-00049", milford_ng, milford_eg)]:
        j = pd.DataFrame({"ng": a, "entsog": b}).dropna()
        d = j.ng - j.entsog
        active = j[(j.ng > 1) | (j.entsog > 1)]
        print(f"\n{label}")
        print(f"  overlapping days: {len(j)} (active: {len(active)})")
        print(f"  corr: {j.ng.corr(j.entsog):.4f}   mean|diff|: {d.abs().mean():.2f} GWh"
              f"   mean diff (NG-ENTSOG): {d.mean():+.2f} GWh")
        print(f"  mean|diff| on active days: {(active.ng - active.entsog).abs().mean():.2f} GWh "
              f"({100 * (active.ng - active.entsog).abs().mean() / active.entsog.mean():.1f}% of mean flow)")
        worst = d.abs().sort_values(ascending=False).head(3)
        for day, gap in worst.items():
            print(f"  worst: {day}  NG={j.loc[day, 'ng']:.1f}  ENTSOG={j.loc[day, 'entsog']:.1f}  gap={gap:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
