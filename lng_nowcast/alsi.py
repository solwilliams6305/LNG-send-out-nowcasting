"""Client for the GIE ALSI API (LNG inventory & send-out).

Requires a free key from https://alsi.gie.eu/account, passed as the `x-key` header
(the key must have ALSI or all-platforms scope — AGSI-only keys are rejected).
The /about?show=listing endpoint is public. Field semantics (API manual V13,
verified live 2026-09-01):
  gasDayStart      gas day the row reports on
  inventory        {"lng": 10^3 m^3, "gwh": GWh} — GIE's own volume->energy
                   conversion, a free per-facility GCV calibration
  sendOut          gas flow out of the facility during the gas day, GWh/d
  dtmi             declared total max inventory, same dual-unit object
  dtrs             declared total reference send-out, GWh/d
  status           E (estimated) / C (confirmed) / N (no data); E rows appear
                   intraday for the *running* gas day, before the 19:30 print
  updatedAt        per-row last-modified timestamp (CET) — revision-study gold
  latitude/longitude  exact facility coordinates (seed AIS berth polygons)

Publication: 19:30 CET for the previous gas day, plus a second pass at 23:00 CET
for late reporters. Operators may retroactively correct at any time — which is
exactly why scripts/snapshot_daily.py archives what the API says twice a day.

Data usage: GIE requires attribution ("GIE ALSI" as source) when repackaging.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Iterable

import requests

from . import config

BASE = "https://alsi.gie.eu/api"


class AlsiError(RuntimeError):
    pass


def _get(params: dict | None = None, path: str = "") -> dict | list:
    if not config.ALSI_KEY:
        raise AlsiError(
            "ALSI_KEY is not set. Register (free) at https://alsi.gie.eu/account, "
            "then put ALSI_KEY=... in the repo .env or the environment."
        )
    headers = {"x-key": config.ALSI_KEY, "User-Agent": config.USER_AGENT}
    delay = config.RETRY_BACKOFF
    for attempt in range(config.RETRIES):
        try:
            r = requests.get(
                f"{BASE}{path}", params=params or {}, headers=headers, timeout=config.HTTP_TIMEOUT
            )
            if r.status_code in (429, 500, 502, 503, 504) and attempt < config.RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise AlsiError(f"ALSI error: {payload.get('message', payload['error'])}")
            return payload
        except (requests.ConnectionError, requests.Timeout):
            if attempt == config.RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def listing() -> list:
    """Company/facility tree with EIC codes: /api/about?show=listing."""
    payload = _get({"show": "listing"}, path="/about")
    if not isinstance(payload, list):
        raise AlsiError(f"unexpected listing payload type: {type(payload)}")
    return payload


def lng_facilities() -> list[dict]:
    """Flatten the listing to LSO facilities: one dict per (company, facility)."""
    out = []
    for company in listing():
        if company.get("type") != "LSO":
            continue
        for fac in company.get("facilities", []):
            out.append(
                {
                    "facility_name": fac.get("name"),
                    "facility_eic": fac.get("eic"),
                    "facility_type": fac.get("type"),
                    "country": fac.get("country"),
                    "company_name": company.get("name"),
                    "company_eic": company.get("eic"),
                    "url": fac.get("url"),
                }
            )
    return out


def facility_history(
    country: str,
    company_eic: str,
    facility_eic: str,
    start: dt.date,
    end: dt.date,
    size: int = 300,
) -> list[dict]:
    """Per-gas-day rows for one facility, [start, end], newest first from the API."""
    rows: list[dict] = []
    page = 1
    while True:
        payload = _get(
            {
                "country": country,
                "company": company_eic,
                "facility": facility_eic,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "size": size,
                "page": page,
            }
        )
        if not isinstance(payload, dict):
            raise AlsiError(f"unexpected facility payload: {payload!r}")
        rows.extend(payload.get("data", []))
        last_page = int(payload.get("last_page") or 1)
        if page >= last_page:
            break
        page += 1
        time.sleep(0.2)
    return rows


def _num(v):
    if v in (None, "", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _volume_energy(v) -> tuple:
    """API >= V13 sends {"lng": <10^3 m^3>, "gwh": <GWh>}; older payloads a bare
    volume. Return (volume_1e3m3, gwh) with None for whatever is absent."""
    if isinstance(v, dict):
        return _num(v.get("lng")), _num(v.get("gwh"))
    return _num(v), None


def normalise(raw: dict, terminal: str, facility: dict) -> dict:
    """One ALSI facility row -> flat snapshot record."""
    inv_vol, inv_gwh = _volume_energy(raw.get("inventory"))
    dtmi_vol, dtmi_gwh = _volume_energy(raw.get("dtmi"))
    info = raw.get("info")
    return {
        "terminal": terminal,
        "facility_eic": facility["facility_eic"],
        "company_eic": facility["company_eic"],
        "country": facility["country"],
        "facility_name": facility["facility_name"],
        "gas_day": raw.get("gasDayStart"),
        "inventory_1e3m3": inv_vol,
        "inventory_gwh": inv_gwh,
        "send_out_gwh_d": _num(raw.get("sendOut")),
        "dtmi_1e3m3": dtmi_vol,
        "dtmi_gwh": dtmi_gwh,
        "dtrs_gwh_d": _num(raw.get("dtrs")),
        "contracted_capacity": _num(raw.get("contractedCapacity")),
        "available_capacity": _num(raw.get("availableCapacity")),
        "status": raw.get("status"),
        "updated_at": raw.get("updatedAt"),
        "latitude": _num(raw.get("latitude")),
        "longitude": _num(raw.get("longitude")),
        "info": "|".join(info) if isinstance(info, list) else info,
    }


def snapshot_terminals(
    facilities_by_terminal: dict[str, list[dict]],
    start: dt.date,
    end: dt.date,
    pause: float = 0.3,
) -> list[dict]:
    """Current ALSI view of [start, end] for every registered facility."""
    out: list[dict] = []
    for terminal, facilities in facilities_by_terminal.items():
        for fac in facilities:
            raws = facility_history(
                fac["country"], fac["company_eic"], fac["facility_eic"], start, end
            )
            out.extend(normalise(r, terminal, fac) for r in raws)
            time.sleep(pause)
    return out
