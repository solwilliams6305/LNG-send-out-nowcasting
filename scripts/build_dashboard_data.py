#!/usr/bin/env python3
"""Compile everything the public dashboard needs into docs/data.json.

Runs at the end of every snapshot cron cycle, so the static page on GitHub
Pages auto-updates 3×/day with zero backend. Licensing: TTF (Yahoo-sourced)
is deliberately excluded from the public build; SAP (National Gas open data),
AGSI (GIE, attributed), and our own derived index are redistributable.
"""

import datetime as dt
import glob
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from lng_nowcast import config
from lng_nowcast.terminals import TERMINALS

spec = importlib.util.spec_from_file_location("ss", ROOT / "scripts" / "stress_study.py")
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)

OUT = ROOT / "docs" / "data.json"
STRESS_DAYS = 200
FLOW_DAYS = 90


def _series(s: pd.Series, days: int, nd: int = 1):
    s = s.dropna().tail(days)
    return {"dates": list(s.index), "values": [round(float(v), nd) for v in s]}


def flows_and_tanks():
    eg = pd.read_csv(config.RAW_DIR / "entsog_daily_flows.csv")
    eg = eg[eg.indicator == "Physical Flow"]
    eu = (eg.assign(g=pd.to_numeric(eg.value_kwh, errors="coerce") / 1e6)
            .groupby(["terminal", "gas_day"]).g.sum())
    ng = pd.read_csv(config.RAW_DIR / "nationalgas_daily.csv")
    ng["g"] = pd.to_numeric(ng.value, errors="coerce") / 1e6
    uk = ng[(ng.metric == "send_out") & (ng.maturity == "D+1")].groupby(["terminal", "gas_day"]).g.sum()
    uk_tank = ng[ng.metric == "stock"].groupby(["terminal", "gas_day"]).g.sum()
    al = pd.read_csv(config.RAW_DIR / "alsi_daily.csv")
    for c in ("inventory_gwh",):
        al[c] = pd.to_numeric(al[c], errors="coerce")
    eu_tank = al.groupby(["terminal", "gas_day"]).inventory_gwh.sum()
    return eu, uk, eu_tank, uk_tank


def latest_ais():
    files = sorted(glob.glob(str(config.SNAPSHOT_DIR / "ais" / "*.csv")))
    if not files:
        return [], {}
    latest = pd.read_csv(files[-1])
    keep = latest[(latest.at_berth == True) | (latest.likely_lng_carrier == True)
                  | latest.terminal.astype(str).str.startswith(("transit_", "depart_"))]
    vessels = []
    for r in keep.itertuples():
        vessels.append({
            "name": None if pd.isna(r.name) else str(r.name),
            "lat": None if pd.isna(r.lat) else round(float(r.lat), 4),
            "lon": None if pd.isna(r.lon) else round(float(r.lon), 4),
            "terminal": str(r.terminal),
            "at_berth": bool(r.at_berth),
            "carrier": bool(r.likely_lng_carrier),
            "sog": None if pd.isna(r.sog) else float(r.sog),
            "draught": None if pd.isna(r.draught_m) else float(r.draught_m),
            "loa": None if pd.isna(r.loa) else float(r.loa),
            "dest": None if pd.isna(r.destination) else str(r.destination),
        })
    # transit tallies over the last ~7 days of snapshots (21 files)
    tally: dict = {}
    for f in files[-21:]:
        d = pd.read_csv(f)
        tr = d[d.terminal.astype(str).str.startswith(("transit_", "depart_"))
               & (d.likely_lng_carrier == True)]
        for r in tr.itertuples():
            key = str(r.terminal).replace("transit_", "").replace("depart_", "↑ ")
            tally.setdefault(key, set()).add(int(r.mmsi))
    transits = {k: len(v) for k, v in tally.items()}
    return vessels, transits


def main() -> int:
    df = ss.build_panel()
    eu, uk, eu_tank, uk_tank = flows_and_tanks()
    vessels, transits = latest_ais()

    # manifest.csv grew columns over time (ragged) — parse the last line leniently
    last = (config.SNAPSHOT_DIR / "manifest.csv").read_text().strip().splitlines()[-1].split(",")
    cols = ["snapshot_utc", "entsog_rows", "entsog_hourly_rows", "alsi_rows",
            "nationalgas_rows", "ng_live_rows", "ais_rows",
            "alsi_key_present", "ais_key_present", "errors"]
    manifest = dict(zip(cols, last)) if len(last) >= 9 else {"snapshot_utc": last[0]}

    terminals = []
    for t in TERMINALS:
        send = eu.get(t.slug) if t.slug in eu.index.get_level_values(0) else None
        if t.slug in ("grain", "south_hook", "dragon"):
            send = uk.get(t.slug)
        tank = (uk_tank.get(t.slug) if t.slug in ("grain", "south_hook", "dragon")
                else (eu_tank.get(t.slug) if t.slug in eu_tank.index.get_level_values(0) else None))
        if send is None or not len(send.dropna()):
            continue
        terminals.append({
            "slug": t.slug, "name": t.name, "country": t.country, "tier": t.tier,
            "lat": t.approx_lat, "lon": t.approx_lon,
            "sendout": _series(send, FLOW_DAYS),
            "tank": _series(tank, FLOW_DAYS) if tank is not None and len(tank.dropna()) else None,
            "latest": round(float(send.dropna().iloc[-1]), 1),
        })
    # UK per-sub-terminal entries exist for grain/south_hook/dragon; milford is
    # the ENTSOG aggregate — drop it in favour of its two sub-terminals.
    for extra in ("south_hook", "dragon"):
        if extra not in [x["slug"] for x in terminals] and extra in uk.index.get_level_values(0):
            base = next((t for t in TERMINALS if t.slug == "milford_haven"))
            off = {"south_hook": (0.016, 0.03), "dragon": (-0.02, 0.09)}[extra]
            terminals.append({
                "slug": extra, "name": extra.replace("_", " ").title() + " LNG",
                "country": "UK", "tier": "core",
                "lat": base.approx_lat + off[0], "lon": base.approx_lon + off[1],
                "sendout": _series(uk.get(extra), FLOW_DAYS),
                "tank": _series(uk_tank.get(extra), FLOW_DAYS),
                "latest": round(float(uk.get(extra).dropna().iloc[-1]), 1),
            })
    terminals = [t for t in terminals if t["slug"] != "milford_haven"]

    cover_now = float(df.cover.dropna().iloc[-1])
    data = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "freshness": {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                      for k, v in manifest.items()},
        "stress": {
            "cover": _series(df.cover, STRESS_DAYS),
            "cover_now": round(cover_now, 1),
            "cover_median": round(float(df.cover.median()), 1),
            "storage": _series(df.storage, STRESS_DAYS),
            "storage_now": round(float(df.storage.dropna().iloc[-1]), 1),
            "sap": _series(df.sap, STRESS_DAYS, nd=3),
            "sap_now": round(float(df.sap.dropna().iloc[-1]), 3),
            "burn_now": round(float(df.burn7.dropna().iloc[-1]), 0),
            "tank_now": round(float(df.tank.dropna().iloc[-1]), 0),
        },
        "terminals": terminals,
        "vessels": vessels,
        "transits_7d": transits,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB, "
          f"{len(terminals)} terminals, {len(vessels)} vessels, transits {transits})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
