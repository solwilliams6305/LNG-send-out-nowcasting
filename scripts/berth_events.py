#!/usr/bin/env python3
"""Preliminary AIS berth-event reconstruction across snapshot windows.

Each 3×-daily snapshot records who was in a berth box during its ~8-minute
listening window. Across snapshots, a vessel's consecutive at-berth
appearances form an occupancy spell; spells by LNG-carrier-sized vessels are
candidate discharge events to be matched against NG inflows (UK) and ALSI
inventory jumps (EU). With ~2 days of archive this prints spells rather than
statistics; the matching table grows on its own.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob

import pandas as pd

from lng_nowcast import config


def main() -> int:
    files = sorted(glob.glob(str(config.SNAPSHOT_DIR / "ais" / "*.csv")))
    if not files:
        print("no ais snapshots yet")
        return 0
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["snap"] = Path(f).stem
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    print(f"{len(files)} AIS snapshots, {df.mmsi.nunique()} distinct vessels seen")

    berthed = df[df.at_berth == True].copy()
    big = berthed[(berthed.loa.fillna(0) >= 180) | berthed.likely_lng_carrier.fillna(False)]

    print("\n=== carrier-sized berth-box occupancy spells ===")
    if not len(big):
        print("  none yet — no LNG carrier observed at berth in the archive so far")
    for (mmsi, term), g in big.groupby(["mmsi", "terminal"]):
        g = g.sort_values("snap")
        name = g.name.dropna().iloc[-1] if g.name.notna().any() else f"mmsi {mmsi}"
        draughts = g.draught_m.dropna().unique()
        print(f"  {str(name):<24} {term:>14}/{str(g.berth_sub.iloc[-1]):<11} "
              f"snapshots {g.snap.iloc[0][:16]} → {g.snap.iloc[-1][:16]} "
              f"({len(g)} sightings), draught(s) {list(draughts) or 'n/a'}")

    # all berth-box occupants, for box-tuning review
    occ = (berthed.groupby(["terminal", "berth_sub"])
           .agg(vessels=("mmsi", "nunique"), sightings=("mmsi", "size")))
    print("\n=== berth-box occupancy by terminal (all vessels, box-tuning view) ===")
    print(occ.to_string())

    # anchorage watch: likely carriers in capture (not berth) boxes
    anchored = df[(df.likely_lng_carrier == True) & (df.at_berth != True)]
    if len(anchored):
        print("\n=== likely carriers waiting/passing in capture boxes ===")
        for (mmsi,), g in anchored.groupby(["mmsi"]):
            g = g.sort_values("snap")
            nm = g.name.dropna().iloc[-1] if g.name.notna().any() else mmsi
            print(f"  {str(nm):<24} near {g.terminal.iloc[-1]:>13}: {len(g)} sightings, "
                  f"sog last {g.sog.iloc[-1]}, dest {g.destination.dropna().iloc[-1] if g.destination.notna().any() else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
