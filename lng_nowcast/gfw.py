"""Global Fishing Watch client — the free mid-ocean layer (memo: the one
legal-free route to satellite-AIS-derived positions; ~72 h latency,
non-commercial license, attribution required).

STATUS: KEY-READY, UNVERIFIED. Endpoints below follow the public v3 API
docs (gateway.api.globalfishingwatch.org) but no request has been made yet —
run scripts/probe_gfw.py the moment GFW_TOKEN lands in .env and fix any
schema drift there before trusting this module (the ALSI/NG pattern:
client first, verify on first key).

Intended role: join the two ends of a voyage. Our departure boxes log a
laden carrier leaving the US Gulf; our chokepoints see it again ~10 days
later; GFW's lagged satellite positions fill the ocean in between and give
speed/heading for arrival-hazard updates. Non-commercial use only — GFW data
must not be redistributed through the public dashboard without checking
their attribution/licensing page first.
"""

from __future__ import annotations

import os

import requests

from . import config

BASE = "https://gateway.api.globalfishingwatch.org/v3"
TOKEN = os.environ.get("GFW_TOKEN", "").strip()
IDENTITY_DATASET = "public-global-vessel-identity:latest"


class GfwError(RuntimeError):
    pass


def _get(path: str, params: dict) -> dict:
    if not TOKEN:
        raise GfwError("GFW_TOKEN not set — register free (non-commercial) at "
                       "globalfishingwatch.org, create a token, add to .env")
    r = requests.get(f"{BASE}{path}", params=params,
                     headers={"Authorization": f"Bearer {TOKEN}",
                              "User-Agent": config.USER_AGENT},
                     timeout=config.HTTP_TIMEOUT)
    if r.status_code in (401, 403):
        raise GfwError(f"GFW auth failed ({r.status_code}): token invalid or "
                       f"insufficient scope — check the token's dataset access")
    r.raise_for_status()
    return r.json()


def search_vessel(mmsi: int) -> list[dict]:
    """Vessel-identity records for an MMSI (id, name, class, flag)."""
    payload = _get("/vessels/search", {
        "query": str(mmsi),
        "datasets[0]": IDENTITY_DATASET,
    })
    return payload.get("entries", [])


def vessel_events(vessel_id: str, start: str, end: str,
                  dataset: str = "public-global-port-visits-events:latest") -> list[dict]:
    """Events (port visits by default; also loitering/encounters datasets)."""
    payload = _get("/events", {
        "vessels[0]": vessel_id,
        "datasets[0]": dataset,
        "start-date": start,
        "end-date": end,
        "limit": 50, "offset": 0,
    })
    return payload.get("entries", [])
