#!/usr/bin/env python3
"""First-token verification of the GFW client: search a carrier our own AIS
archive has seen, print raw responses, and flag any schema drift to fix in
lng_nowcast/gfw.py before building on it."""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lng_nowcast import config, gfw


def main() -> int:
    files = sorted(glob.glob(str(config.SNAPSHOT_DIR / "ais" / "*.csv")))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    carriers = df[df.likely_lng_carrier == True].dropna(subset=["mmsi"])
    if not len(carriers):
        print("no carriers in the archive yet — pass an MMSI manually")
        return 1
    mmsi = int(carriers.iloc[-1].mmsi)
    name = carriers.iloc[-1].get("name")
    print(f"probing GFW for {name} (mmsi {mmsi})...")
    entries = gfw.search_vessel(mmsi)
    print(json.dumps(entries[:1], indent=1)[:1200] or "no entries")
    if entries:
        vid = entries[0].get("id") or entries[0].get("vesselId")
        print(f"\nport visits for vessel id {vid}:")
        ev = gfw.vessel_events(vid, "2026-06-01", "2026-09-01")
        print(json.dumps(ev[:2], indent=1)[:1200] or "no events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
