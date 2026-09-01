"""Client for National Gas Transmission's data portal (UK LNG terminals).

Fills the UK hole: Grain/South Hook/Dragon stopped reporting to ALSI (>= 2024)
and ENTSOG's UK rows lag ~6 days. The portal is free, keyless, and richer than
either — per terminal it publishes send-out at three maturities (a built-in
revision ladder), tank stock, in/outflows, prevailing nominations, calorific
value, and even a boil-off nomination for Grain.

API (discovered by sniffing the SPA, verified live 2026-09-01):
  POST https://data.nationalgas.com/api/find-gas-data
  body: {"latestFlag": "Y"|"N", "applicableFor": "Y"|"N",
         "dateFrom": "YYYY-MM-DD", "dateTo": "YYYY-MM-DD",
         "dateType": "GASDAY", "ids": "PUBOB...,PUBOB..."}
  rows: value, applicableFor (DD/MM/YYYY gas day), generatedTimeStamp,
        qualityIndicator, itemName, UnitOfMeasure (kWh, displayed as "kw/h")
  latestFlag=N returns every published version of each value — the full UK
  revision history is reconstructable retroactively (unlike ALSI!).

Timing (empirical): D+1 physical send-out generated ~12:01 on D+1; opening
stock generated ~15:56 same gas day. UK gas day 05:00-05:00 local — the same
UTC instant as the EU 06:00 CET gas day year-round, so rows align with ENTSOG.

Intraday (cracked 2026-09-01 by reading the portal's JS bundle): the
gas-system-status endpoints take POST {"request": "<name>"}, keyless.
  /api/gas-system-status-table  {"request": "flowRatesTable"}
      2-minutely instantaneous flow (mcm/day) per SYSTEM ENTRY NAME —
      including GRAIN NTS 1/2 and MILFORD HAVEN - SOUTH HOOK / - DRAGON
      (sub-terminal split!) — but only the ~6 latest readings.
  /api/gas-system-status-graph  {"request": "flowRatesGraphs"}
      the full gas day so far at 2-min cadence per terminal AREA
      (ISLE OF GRAIN, MILFORD HAVEN aggregates), epoch-ms timestamps
      starting at the 05:00 local gas-day start.
Neither is archived anywhere public — the snapshot logger is the archive.
Units are mcm/day instantaneous rate; convert to energy at analysis time
with the per-terminal CV series (metric="cv").
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

import requests

from . import config

BASE = "https://data.nationalgas.com/api"


@dataclass(frozen=True)
class Item:
    pub_id: str
    terminal: str  # grain | south_hook | dragon  (south_hook + dragon = ENTSOG milford_haven)
    metric: str
    maturity: str  # D+1 | D+2 | M+15 | live | n/a
    detail: str = ""


# Commercial M+15 counterparts of the physical D+1 items; physical M+15 also
# exists (PUBOBJ1247/1252/1253/1254) if the allocation-vs-metered gap ever matters.
ITEMS: tuple[Item, ...] = (
    # --- send-out energy (kWh/gas day), the revision ladder ---
    Item("PUBOBJ605", "grain", "send_out", "D+1", "Grain NTS1, physical"),
    Item("PUBOBJ606", "grain", "send_out", "D+1", "Grain NTS2, physical"),
    Item("PUBOB370", "grain", "send_out", "D+2", "Grain NTS1, commercial"),
    Item("PUBOB3474", "grain", "send_out", "D+2", "Grain NTS2, commercial"),
    Item("PUBOB454", "grain", "send_out", "M+15", "Grain NTS1, commercial"),
    Item("PUBOB3477", "grain", "send_out", "M+15", "Grain NTS2, commercial"),
    Item("PUBOBJ607", "south_hook", "send_out", "D+1", "physical"),
    Item("PUBOB3481", "south_hook", "send_out", "D+2", "commercial"),
    Item("PUBOB3484", "south_hook", "send_out", "M+15", "commercial"),
    Item("PUBOBJ600", "dragon", "send_out", "D+1", "physical"),
    Item("PUBOB3565", "dragon", "send_out", "D+2", "commercial"),
    Item("PUBOB3568", "dragon", "send_out", "M+15", "commercial"),
    # --- LNG tank stock (kWh, opening) — the slow state of the filter ---
    Item("PUBOBJ2363", "grain", "stock", "D0", "opening stock"),
    Item("PUBOBJ2371", "south_hook", "stock", "D0", "opening stock"),
    Item("PUBOBJ2372", "dragon", "stock", "D0", "opening stock"),
    # --- storage-actual flows (cargo discharge shows up as inflow) ---
    Item("PUBOBJ2403", "grain", "inflow", "D+1", "LNG importation"),
    Item("PUBOBJ2411", "south_hook", "inflow", "D+1", "LNG importation"),
    Item("PUBOBJ2412", "dragon", "inflow", "D+1", "LNG importation"),
    Item("PUBOBJ2415", "grain", "outflow", "D+1", "LNG importation"),
    Item("PUBOBJ2423", "south_hook", "outflow", "D+1", "LNG importation"),
    Item("PUBOBJ2424", "dragon", "outflow", "D+1", "LNG importation"),
    # --- prevailing nominations (forward-looking) ---
    Item("PUBOBJ1117", "grain", "nomination", "live", "Grain NTS1"),
    Item("PUBOBJ1118", "grain", "nomination", "live", "Grain NTS2"),
    Item("PUBOBJ1119", "south_hook", "nomination", "live", ""),
    Item("PUBOBJ1112", "dragon", "nomination", "live", ""),
    Item("PUBOBJ1141", "grain", "boiloff_nom", "live", "storage boil-off"),
    # --- calorific value (MJ/m3-ish; free GCV calibration) ---
    Item("PUBOB369", "grain", "cv", "D+2", "Grain NTS1"),
    Item("PUBOB3475", "grain", "cv", "D+2", "Grain NTS2"),
    Item("PUBOB3482", "south_hook", "cv", "D+2", ""),
    Item("PUBOB3566", "dragon", "cv", "D+2", ""),
)


def _post(body: dict) -> dict:
    headers = {"User-Agent": config.USER_AGENT, "Content-Type": "application/json"}
    delay = config.RETRY_BACKOFF
    for attempt in range(config.RETRIES):
        try:
            r = requests.post(
                f"{BASE}/find-gas-data", json=body, headers=headers, timeout=config.HTTP_TIMEOUT
            )
            if r.status_code in (429, 500, 502, 503, 504) and attempt < config.RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == config.RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _iso_day(ddmmyyyy: str) -> str | None:
    try:
        return dt.datetime.strptime(ddmmyyyy[:10], "%d/%m/%Y").date().isoformat()
    except (ValueError, TypeError):
        return None


def _iso_ts(ts: str) -> str | None:
    try:
        return dt.datetime.strptime(ts, "%d/%m/%Y %H:%M:%S").isoformat()
    except (ValueError, TypeError):
        return None


def fetch_item(
    item: Item,
    start: dt.date,
    end: dt.date,
    latest: bool = True,
    applicable_for: bool = True,
) -> list[dict]:
    """Rows for one publication item. latest=False returns every published
    version (the retroactive revision history)."""
    payload = _post(
        {
            "latestFlag": "Y" if latest else "N",
            "applicableFor": "Y" if applicable_for else "N",
            "dateFrom": start.isoformat(),
            "dateTo": end.isoformat(),
            "dateType": "GASDAY",
            "ids": item.pub_id,
        }
    )
    rows = []
    for raw in payload.get("data", []):
        rows.append(
            {
                "pub_id": item.pub_id,
                "terminal": item.terminal,
                "metric": item.metric,
                "maturity": item.maturity,
                "detail": item.detail,
                "gas_day": _iso_day(raw.get("applicableFor") or ""),
                "value": raw.get("value"),
                "unit": raw.get("UnitOfMeasure"),
                "quality": raw.get("qualityIndicator"),
                "generated_at": _iso_ts(raw.get("generatedTimeStamp") or ""),
                "item_name": raw.get("itemName"),
            }
        )
    return rows


LIVE_SUBTERMINALS = {
    "GRAIN NTS 1": ("grain", "nts1"),
    "GRAIN NTS 2": ("grain", "nts2"),
    "MILFORD HAVEN - SOUTH HOOK": ("south_hook", ""),
    "MILFORD HAVEN - DRAGON": ("dragon", ""),
}
LIVE_AREAS = {"ISLE OF GRAIN": "grain", "MILFORD HAVEN": "milford_haven"}


def _post_status(endpoint: str, request: str) -> dict:
    headers = {"User-Agent": config.USER_AGENT, "Content-Type": "application/json"}
    r = requests.post(f"{BASE}/{endpoint}", json={"request": request},
                      headers=headers, timeout=config.HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def snapshot_instantaneous() -> list[dict]:
    """Current UK intraday state: the day-so-far 2-min trajectory per LNG area
    (flowRatesGraphs) plus the latest sub-terminal readings (flowRatesTable).
    Values in mcm/day instantaneous rate."""
    rows: list[dict] = []

    payload = _post_status("gas-system-status-graph", "flowRatesGraphs")
    for area, terminal in LIVE_AREAS.items():
        series = (payload.get(area) or {}).get("data") or []
        for pt in series:
            rows.append({
                "series": "area_trajectory", "terminal": terminal, "sub": "",
                "epoch_ms": pt.get("dateTime"), "time_label": None,
                "value_mcm_d": pt.get(area),
            })

    payload = _post_status("gas-system-status-table", "flowRatesTable")
    data = (payload.get("data") or {})
    times = data.get("timeHeaders") or []
    for row in data.get("data") or []:
        name = row.get("SYSTEM ENTRY NAME")
        if name not in LIVE_SUBTERMINALS:
            continue
        terminal, sub = LIVE_SUBTERMINALS[name]
        for tlab in times:
            rows.append({
                "series": "subterminal_latest", "terminal": terminal, "sub": sub,
                "epoch_ms": None, "time_label": tlab,
                "value_mcm_d": row.get(tlab),
            })
    return rows


def fetch_all(
    start: dt.date,
    end: dt.date,
    items: tuple[Item, ...] = ITEMS,
    latest: bool = True,
    pause: float = 0.25,
) -> list[dict]:
    out: list[dict] = []
    for item in items:
        try:
            out.extend(fetch_item(item, start, end, latest=latest))
        except Exception as e:  # one dead item must not sink the run
            print(f"  ! nationalgas {item.pub_id} ({item.terminal}/{item.metric}): {e}")
        time.sleep(pause)
    return out
