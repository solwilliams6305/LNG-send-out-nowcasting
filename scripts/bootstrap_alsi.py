#!/usr/bin/env python3
"""Resolve ALSI facility EICs for every registered terminal (run once, needs ALSI_KEY).

GIE warns that EIC codes change over time, so they are resolved from the live
/api/about?show=listing rather than hardcoded. Writes
data/reference/alsi_facilities.json mapping terminal slug -> list of facilities
(Milford Haven legitimately maps to two: South Hook and Dragon).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lng_nowcast import alsi, config
from lng_nowcast.terminals import TERMINALS

OUT = config.REFERENCE_DIR / "alsi_facilities.json"


def main() -> int:
    config.ensure_dirs()
    facilities = alsi.lng_facilities()
    print(f"ALSI listing: {len(facilities)} LSO facilities")

    resolved: dict[str, list[dict]] = {}
    claimed: set[str] = set()
    for t in TERMINALS:
        hits = [
            f for f in facilities
            if any(p in (f["facility_name"] or "").lower() for p in t.alsi_name_patterns)
        ]
        resolved[t.slug] = hits
        for h in hits:
            claimed.add(h["facility_eic"])
        names = ", ".join(f'{h["facility_name"]} ({h["country"]})' for h in hits) or "NO MATCH"
        print(f"{t.slug:>14}: {names}")

    unclaimed = [f for f in facilities if f["facility_eic"] not in claimed]
    if unclaimed:
        print("\nALSI facilities not claimed by any terminal (fine if outside NW Europe):")
        for f in unclaimed:
            print(f"  {f['country']}: {f['facility_name']}")

    OUT.write_text(json.dumps(resolved, indent=2))
    print(f"\nwrote {OUT}")
    missing = [slug for slug, hits in resolved.items() if not hits]
    if missing:
        print(f"WARNING — no ALSI match for: {', '.join(missing)}. "
              f"Fix alsi_name_patterns in lng_nowcast/terminals.py and re-run.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
