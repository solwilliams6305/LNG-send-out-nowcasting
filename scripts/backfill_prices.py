#!/usr/bin/env python3
"""Backfill daily price series into data/raw/prices.csv (gitignored)."""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lng_nowcast import config, prices

OUT = config.RAW_DIR / "prices.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2024, 1, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date.today())
    args = ap.parse_args()
    config.ensure_dirs()

    rows = prices.fetch_uk_sap(args.start, args.end) + prices.fetch_ttf(args.start, args.end)
    new = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT, dtype=str)
        new = pd.concat([old, new.astype(str)], ignore_index=True).drop_duplicates(
            subset=["date", "series"], keep="last")
    new = new.sort_values(["series", "date"])
    new.to_csv(OUT, index=False)
    for s, g in new.groupby("series"):
        print(f"{s}: {len(g)} days ({g.date.min()} → {g.date.max()})")
    print(f"wrote {len(new)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
