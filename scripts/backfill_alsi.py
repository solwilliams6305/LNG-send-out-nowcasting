#!/usr/bin/env python3
"""Backfill ALSI daily history per registered facility into data/raw/alsi_daily.csv.

Historical rows are mostly Confirmed; the live E->C action is captured by the
snapshot logger — this file exists so the filter can train on 2024->today.
Re-runnable; merges on (facility_eic, gas_day) keeping the newest fetch.
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lng_nowcast import alsi, config, snapshot

OUT = config.RAW_DIR / "alsi_daily.csv"
KEY = ["facility_eic", "gas_day"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2024, 1, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date.today())
    args = ap.parse_args()
    config.ensure_dirs()

    registry = snapshot.load_alsi_registry()
    if not registry:
        print("run scripts/bootstrap_alsi.py first", file=sys.stderr)
        return 1

    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    frames = []
    for terminal, facilities in registry.items():
        for fac in facilities:
            try:
                raws = alsi.facility_history(
                    fac["country"], fac["company_eic"], fac["facility_eic"],
                    args.start, args.end,
                )
            except Exception as e:
                print(f"  ! {terminal} {fac['facility_name']}: {e}", file=sys.stderr)
                continue
            rows = [alsi.normalise(r, terminal, fac) for r in raws]
            for r in rows:
                r["fetched_at"] = fetched_at
            n_real = sum(1 for r in rows if r["status"] not in (None, "N"))
            print(f"{terminal:>14} {fac['facility_name'][:44]:<44} {len(rows)} rows ({n_real} non-N)")
            if rows:
                frames.append(pd.DataFrame(rows))
            time.sleep(0.3)

    if not frames:
        print("nothing fetched", file=sys.stderr)
        return 1
    new = pd.concat(frames, ignore_index=True)
    if OUT.exists():
        old = pd.read_csv(OUT, dtype=str)
        merged = pd.concat([old, new.astype(str)], ignore_index=True).drop_duplicates(
            subset=KEY, keep="last"
        )
    else:
        merged = new
    merged = merged.sort_values(["terminal", "facility_eic", "gas_day"])
    merged.to_csv(OUT, index=False)
    print(f"wrote {len(merged)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
