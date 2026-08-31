#!/usr/bin/env python3
"""Continuous (or bounded) AIS logger for the terminal capture boxes.

Writes raw aisstream messages to data/ais/raw/YYYY-MM-DD.jsonl.gz (gitignored;
bulky) and prints a berth-occupancy summary on exit. The 3x-daily snapshot runs
already capture occupancy windows; run this locally for dense coverage while
the laptop is on:

    .venv/bin/python scripts/ais_logger.py               # until Ctrl-C
    .venv/bin/python scripts/ais_logger.py --duration 600
"""

import argparse
import datetime as dt
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lng_nowcast import ais, config


class DailyGzSink:
    def __init__(self, root: Path):
        self.root = root
        self.day = None
        self.fh = None
        self.n = 0

    def __call__(self, raw: dict) -> None:
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        if day != self.day:
            if self.fh:
                self.fh.close()
            self.root.mkdir(parents=True, exist_ok=True)
            self.day = day
            self.fh = gzip.open(self.root / f"{day}.jsonl.gz", "at")
        self.fh.write(json.dumps(raw, separators=(",", ":")) + "\n")
        self.n += 1
        if self.n % 500 == 0:
            print(f"  {self.n} messages logged", flush=True)

    def close(self) -> None:
        if self.fh:
            self.fh.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to run (default: until interrupted)")
    args = ap.parse_args()

    sink = DailyGzSink(config.DATA_DIR / "ais" / "raw")
    try:
        vessels = ais.listen(args.duration, raw_sink=sink)
    except KeyboardInterrupt:
        print("\ninterrupted")
        vessels = {}
    finally:
        sink.close()

    print(f"{sink.n} raw messages logged -> {sink.root}")
    snap = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows = [v.row(snap) for v in vessels.values() if v.lat is not None]
    berthed = [r for r in rows if r["at_berth"]]
    print(f"{len(rows)} vessels seen in capture boxes; {len(berthed)} at berth:")
    for r in sorted(berthed, key=lambda r: (r["terminal"] or "", str(r["name"]))):
        print(f"  {r['terminal']:>14}/{(r['berth_sub'] or ''):<11} {str(r['name']):<22} "
              f"mmsi={r['mmsi']} loa={r['loa']} draught={r['draught_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
