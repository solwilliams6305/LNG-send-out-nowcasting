#!/usr/bin/env python3
"""Backfill National Gas UK LNG data into data/raw/.

Default: latest values of every registry item -> nationalgas_daily.csv.
--all-versions: every published version (latestFlag=N) of the send-out and
stock items -> nationalgas_versions.csv — the retroactive UK revision history.

Re-runnable; merges on (pub_id, gas_day, generated_at) keeping last.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lng_nowcast import config, nationalgas

KEY = ["pub_id", "gas_day", "generated_at"]


def merge_write(frames: list[pd.DataFrame], out: Path) -> int:
    new = pd.concat(frames, ignore_index=True)
    if out.exists():
        old = pd.read_csv(out, dtype=str)
        new = new.astype(str)
        merged = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=KEY, keep="last")
    else:
        merged = new
    merged = merged.sort_values(["terminal", "metric", "maturity", "gas_day"])
    merged.to_csv(out, index=False)
    return len(merged)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2024, 1, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date.today())
    ap.add_argument("--all-versions", action="store_true")
    ap.add_argument("--chunk-days", type=int, default=200)
    args = ap.parse_args()
    config.ensure_dirs()

    items = nationalgas.ITEMS
    latest = True
    out = config.RAW_DIR / "nationalgas_daily.csv"
    if args.all_versions:
        items = tuple(i for i in items if i.metric in ("send_out", "stock"))
        latest = False
        out = config.RAW_DIR / "nationalgas_versions.csv"

    frames = []
    for item in items:
        rows: list[dict] = []
        cursor = args.start
        while cursor <= args.end:
            chunk_end = min(cursor + dt.timedelta(days=args.chunk_days - 1), args.end)
            try:
                rows.extend(nationalgas.fetch_item(item, cursor, chunk_end, latest=latest))
            except Exception as e:
                print(f"  ! {item.pub_id} {cursor}..{chunk_end}: {e}", file=sys.stderr)
            cursor = chunk_end + dt.timedelta(days=1)
        print(f"{item.terminal:>10} {item.metric:>11} {item.maturity:>5} {item.pub_id:>12}: {len(rows)} rows")
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        print("nothing fetched", file=sys.stderr)
        return 1
    n = merge_write(frames, out)
    print(f"wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
