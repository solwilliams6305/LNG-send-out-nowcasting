"""The revision logger: archive what each source currently claims about recent days.

Why this exists: ALSI rows carry a status flag (E estimated / C confirmed) and
operators retroactively correct values; ENTSOG rows carry lastUpdateDateTime and
are also revised. Neither platform archives its own past states, so the
Estimated->Confirmed revision process — the thing this project wants to model —
is only observable by logging it ourselves, in real time, from day one.

Each run writes timestamped CSVs under data/snapshots/:
  entsog/<UTC>.csv         daily Physical Flow + Nomination, trailing window
  entsog_hourly/<UTC>.csv  hourly Physical Flow, yesterday + today (core terminals)
  alsi/<UTC>.csv           per-facility inventory/send-out/status, trailing window
  manifest.csv             one line per run: row counts + which sources succeeded

Revision analysis later = group rows by (series, gas_day) across snapshot files.
Runs are idempotent-ish and append-only; a missed run just widens one gap.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from . import alsi, ais, config, entsog, nationalgas
from .terminals import TERMINALS, tier

ENTSOG_COLS = [
    "snapshot_utc", "terminal", "gas_day", "period_from", "period_to", "indicator",
    "period_type", "value_kwh", "unit", "operator_key", "point_key", "direction",
    "flow_status", "last_update",
]
ALSI_COLS = [
    "snapshot_utc", "terminal", "gas_day", "facility_eic", "company_eic", "country",
    "facility_name", "inventory_1e3m3", "inventory_gwh", "send_out_gwh_d",
    "dtmi_1e3m3", "dtmi_gwh", "dtrs_gwh_d", "contracted_capacity",
    "available_capacity", "status", "updated_at", "latitude", "longitude", "info",
]
NATIONALGAS_COLS = [
    "snapshot_utc", "terminal", "metric", "maturity", "pub_id", "gas_day",
    "value", "unit", "quality", "generated_at", "detail", "item_name",
]
NG_LIVE_COLS = [
    "snapshot_utc", "series", "terminal", "sub", "epoch_ms", "time_label",
    "value_mcm_d",
]
AIS_COLS = [
    "snapshot_utc", "mmsi", "name", "imo", "ship_type", "loa", "beam",
    "draught_m", "lat", "lon", "sog", "nav_status", "terminal", "berth_sub",
    "at_berth", "likely_lng_carrier", "destination", "n_pos",
    "first_seen", "last_seen",
]


def _stamp(now: dt.datetime) -> str:
    return now.strftime("%Y-%m-%dT%H%MZ")


def _write(rows: list[dict], cols: list[str], path) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df[cols].to_csv(path, index=False)
    return len(df)


def load_alsi_registry() -> dict[str, list[dict]]:
    """slug -> ALSI facilities, as resolved by scripts/bootstrap_alsi.py."""
    path = config.REFERENCE_DIR / "alsi_facilities.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def run(window_days: int = 45, include_hourly: bool = True, ais_window_s: float = 480) -> dict:
    """One snapshot run. Returns a manifest dict (also appended to manifest.csv)."""
    config.ensure_dirs()
    now = dt.datetime.now(dt.timezone.utc)
    snap = now.isoformat(timespec="seconds")
    stamp = _stamp(now)
    today = now.date()
    start = today - dt.timedelta(days=window_days)

    manifest = {"snapshot_utc": snap, "entsog_rows": 0, "entsog_hourly_rows": 0,
                "alsi_rows": 0, "nationalgas_rows": 0, "ng_live_rows": 0,
                "ais_rows": 0, "alsi_key_present": bool(config.ALSI_KEY),
                "ais_key_present": bool(config.AISSTREAM_KEY), "errors": ""}
    errors: list[str] = []

    # --- ENTSOG daily: Physical Flow + Nomination for every registered point ---
    try:
        rows = entsog.fetch_terminals(TERMINALS, start, today, "Physical Flow")
        rows += entsog.fetch_terminals(TERMINALS, start, today + dt.timedelta(days=1), "Nomination")
        for r in rows:
            r["snapshot_utc"] = snap
        d = config.SNAPSHOT_DIR / "entsog"
        d.mkdir(exist_ok=True)
        manifest["entsog_rows"] = _write(rows, ENTSOG_COLS, d / f"{stamp}.csv")
    except Exception as e:
        errors.append(f"entsog_daily: {e}")

    # --- ENTSOG hourly: intraday state of the current gas day, core terminals ---
    if include_hourly:
        try:
            hstart = today - dt.timedelta(days=1)
            hrows = entsog.fetch_terminals(
                tier("core"), hstart, today + dt.timedelta(days=1),
                "Physical Flow", period_type="hour",
            )
            for r in hrows:
                r["snapshot_utc"] = snap
            d = config.SNAPSHOT_DIR / "entsog_hourly"
            d.mkdir(exist_ok=True)
            manifest["entsog_hourly_rows"] = _write(hrows, ENTSOG_COLS, d / f"{stamp}.csv")
        except Exception as e:
            errors.append(f"entsog_hourly: {e}")

    # --- ALSI: inventory / send-out / status per facility ---
    registry = load_alsi_registry()
    if not config.ALSI_KEY:
        print("ALSI: skipped (no ALSI_KEY set — register free at https://alsi.gie.eu/account)")
    elif not registry:
        print("ALSI: skipped (run scripts/bootstrap_alsi.py once to resolve facility EICs)")
    else:
        try:
            arows = alsi.snapshot_terminals(registry, start, today)
            for r in arows:
                r["snapshot_utc"] = snap
            d = config.SNAPSHOT_DIR / "alsi"
            d.mkdir(exist_ok=True)
            manifest["alsi_rows"] = _write(arows, ALSI_COLS, d / f"{stamp}.csv")
        except Exception as e:
            errors.append(f"alsi: {e}")

    # --- National Gas (UK): send-out ladder, stocks, flows, nominations, CV ---
    try:
        grows = nationalgas.fetch_all(start, today + dt.timedelta(days=1))
        for r in grows:
            r["snapshot_utc"] = snap
        d = config.SNAPSHOT_DIR / "nationalgas"
        d.mkdir(exist_ok=True)
        manifest["nationalgas_rows"] = _write(grows, NATIONALGAS_COLS, d / f"{stamp}.csv")
    except Exception as e:
        errors.append(f"nationalgas: {e}")

    # --- National Gas live: UK intraday trajectories (not archived anywhere
    # else — each snapshot preserves the 2-minutely day-so-far curves) ---
    try:
        lrows = nationalgas.snapshot_instantaneous()
        for r in lrows:
            r["snapshot_utc"] = snap
        d = config.SNAPSHOT_DIR / "nationalgas_live"
        d.mkdir(exist_ok=True)
        manifest["ng_live_rows"] = _write(lrows, NG_LIVE_COLS, d / f"{stamp}.csv")
    except Exception as e:
        errors.append(f"nationalgas_live: {e}")

    # --- AIS: bounded listening window -> berth occupancy + static state ---
    if not config.AISSTREAM_KEY:
        print("AIS: skipped (no AISSTREAM_KEY set — register free at https://aisstream.io)")
    elif ais_window_s <= 0:
        pass
    else:
        try:
            arows = ais.snapshot_rows(duration_s=ais_window_s)
            d = config.SNAPSHOT_DIR / "ais"
            d.mkdir(exist_ok=True)
            manifest["ais_rows"] = _write(arows, AIS_COLS, d / f"{stamp}.csv")
        except Exception as e:
            errors.append(f"ais: {e}")

    manifest["errors"] = "; ".join(errors)
    mpath = config.SNAPSHOT_DIR / "manifest.csv"
    pd.DataFrame([manifest]).to_csv(mpath, mode="a", header=not mpath.exists(), index=False)
    return manifest
