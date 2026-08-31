#!/usr/bin/env python3
"""Backfill historical daily ENTSOG flows into data/raw/entsog_daily_flows.csv.

Re-runnable: merges with the existing file, keeping the newest fetch of each
(terminal, operator, point, direction, indicator, gas_day). Chunks requests and
sleeps between them — ENTSOG is a shared public service.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lng_nowcast import config, entsog
from lng_nowcast.terminals import tier

KEY = ["terminal", "operator_key", "point_key", "direction", "indicator", "gas_day"]
OUT = config.RAW_DIR / "entsog_daily_flows.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2024, 1, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date.today())
    ap.add_argument("--tier", choices=["core", "secondary", "all"], default="all")
    ap.add_argument("--indicator", default="Physical Flow")
    args = ap.parse_args()

    config.ensure_dirs()
    frames = []
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for t in tier(args.tier):
        for pd_ in t.entsog:
            rows: list[dict] = []
            try:
                for chunk in entsog.fetch_range_chunked(pd_, args.start, args.end, args.indicator):
                    rows.extend(chunk)
            except Exception as e:
                print(f"{t.slug:>14} {pd_.operator_key}/{pd_.point_key}: FAILED ({e})", file=sys.stderr)
                continue
            for r in rows:
                r["terminal"] = t.slug
                r["fetched_at"] = fetched_at
            print(f"{t.slug:>14} {pd_.operator_key}/{pd_.point_key} {pd_.direction}: {len(rows)} days")
            if rows:
                frames.append(pd.DataFrame(rows))

    if not frames:
        print("nothing fetched", file=sys.stderr)
        return 1
    new = pd.concat(frames, ignore_index=True)
    if OUT.exists():
        old = pd.read_csv(OUT, dtype=str)
        new = new.astype({c: str for c in new.columns if c in old.columns})
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=KEY, keep="last")
    else:
        merged = new
    merged = merged.sort_values(["terminal", "gas_day"])
    merged.to_csv(OUT, index=False)
    print(f"wrote {len(merged)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
