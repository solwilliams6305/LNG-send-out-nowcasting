#!/usr/bin/env python3
"""Backfill the stress-study inputs into data/raw/stress_inputs.csv:
  agsi_eu_*     — EU storage aggregate (TWh in store, % full, net withdrawal)
  uk_linepack_* — NTS opening/closing linepack actuals (PUBOB693/694, mcm)
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lng_nowcast import alsi, config, nationalgas

LP_ITEMS = (nationalgas.Item("PUBOB693", "uk", "linepack_open", "D0", "mcm"),
            nationalgas.Item("PUBOB694", "uk", "linepack_close", "D0", "mcm"))
OUT = config.RAW_DIR / "stress_inputs.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2024, 1, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date.today())
    args = ap.parse_args()
    config.ensure_dirs()

    rows = []
    for r in alsi.agsi_history(args.start, args.end):
        d = r.get("gasDayStart")
        for series, key in (("agsi_eu_twh", "gasInStorage"), ("agsi_eu_full_pct", "full"),
                            ("agsi_eu_withdrawal", "withdrawal"), ("agsi_eu_injection", "injection")):
            rows.append({"date": d, "series": series, "value": r.get(key)})
    for item in LP_ITEMS:
        for r in nationalgas.fetch_item(item, args.start, args.end):
            rows.append({"date": r["gas_day"], "series": f"uk_{item.metric}",
                         "value": r["value"]})

    new = pd.DataFrame(rows).dropna(subset=["date"])
    if OUT.exists():
        old = pd.read_csv(OUT, dtype=str)
        new = pd.concat([old, new.astype(str)], ignore_index=True).drop_duplicates(
            subset=["date", "series"], keep="last")
    new = new.sort_values(["series", "date"])
    new.to_csv(OUT, index=False)
    for s, g in new.groupby("series"):
        print(f"{s}: {len(g)} days ({g.date.min()} → {g.date.max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
