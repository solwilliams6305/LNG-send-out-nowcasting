#!/usr/bin/env python3
"""Backfill historical daily ENTSOG flows into data/raw/entsog_daily_flows.csv.

Re-runnable: merges with the existing file, keeping the newest fetch of each
(terminal, operator, point, direction, indicator, gas_day). Chunks requests and
sleeps between them — ENTSOG is a shared public service.
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lng_nowcast import config, entsog
from lng_nowcast.terminals import tier

KEY = ["terminal", "operator_key", "point_key", "direction", "indicator",
       "period_type", "gas_day", "period_from"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2024, 1, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date.today())
    ap.add_argument("--tier", choices=["core", "secondary", "all"], default="all")
    ap.add_argument("--indicator", default="Physical Flow")
    ap.add_argument("--period-type", choices=["day", "hour"], default="day")
    args = ap.parse_args()

    out = (config.RAW_DIR / "entsog_daily_flows.csv" if args.period_type == "day"
           else config.RAW_DIR / "entsog_hourly_flows.csv")
    chunk_days = 120 if args.period_type == "day" else 45

    config.ensure_dirs()
    frames = []
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for t in tier(args.tier):
        for pd_ in t.entsog:
            rows: list[dict] = []
            cursor, failed_chunks = args.start, 0
            while cursor <= args.end:
                chunk_end = min(cursor + dt.timedelta(days=chunk_days - 1), args.end)
                try:
                    rows.extend(entsog.fetch_operational(
                        pd_, cursor, chunk_end, args.indicator, args.period_type))
                except Exception as e:
                    failed_chunks += 1
                    print(f"{t.slug:>14} {cursor}..{chunk_end}: chunk failed ({e})",
                          file=sys.stderr)
                cursor = chunk_end + dt.timedelta(days=1)
                time.sleep(0.4)
            if failed_chunks:
                print(f"{t.slug:>14} {pd_.operator_key}/{pd_.point_key}: "
                      f"{failed_chunks} chunk(s) skipped", file=sys.stderr)
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
    if out.exists():
        old = pd.read_csv(out, dtype=str)
        new = new.astype({c: str for c in new.columns if c in old.columns})
        merged = pd.concat([old, new], ignore_index=True)
        key = [k for k in KEY if k in merged.columns]
        merged = merged.drop_duplicates(subset=key, keep="last")
    else:
        merged = new
    merged = merged.sort_values(["terminal", "gas_day"])
    merged.to_csv(out, index=False)
    print(f"wrote {len(merged)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
