"""Client for the ENTSOG Transparency Platform REST API (no key required).

Verified live 2026-09-01:
  - operationalData accepts operatorKey/pointKey/directionKey/indicator/periodType
    filters directly; gas day runs 06:00-06:00 CET; values in kWh/d (or kWh/h).
  - Daily Physical Flow for the previous gas day lands ~09:20 CET next morning for
    NW-EU TSOs — 10 hours BEFORE ALSI's 19:30 publication.
  - Hourly Physical Flow trails real time by ~2 hours (intraday nowcasting input).
  - Nomination for gas day D is published ~16:00 CET on D-1 (day-ahead signal).
  - UK-TSO-0001 rows are backfilled in weekly batches, ~6 days late.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Iterable, Iterator

import requests

from . import config
from .terminals import PointDirection, Terminal

BASE = "https://transparency.entsog.eu/api/v1"

# Fields kept from each operationalData row, renamed to snake_case.
_KEEP = {
    "periodFrom": "period_from",
    "periodTo": "period_to",
    "value": "value_kwh",
    "unit": "unit",
    "indicator": "indicator",
    "periodType": "period_type",
    "operatorKey": "operator_key",
    "pointKey": "point_key",
    "directionKey": "direction",
    "pointLabel": "point_label",
    "operatorLabel": "operator_label",
    "lastUpdateDateTime": "last_update",
    "flowStatus": "flow_status",
}


def _get(path: str, params: dict) -> dict:
    """GET with retries; ENTSOG intermittently 500s or stalls under load."""
    url = f"{BASE}/{path}"
    headers = {"User-Agent": config.USER_AGENT, "Accept": "application/json"}
    delay = config.RETRY_BACKOFF
    for attempt in range(config.RETRIES):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=config.HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                # ENTSOG 404s windows containing no rows (e.g. before a point went live).
                return {"operationalData": []}
            if r.status_code in (429, 500, 502, 503, 504) and attempt < config.RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
        except requests.Timeout:
            # A timed-out ENTSOG query is usually a server-side hang that never
            # recovers (verified live: some nomination windows stall with zero
            # bytes indefinitely) — one retry, then let the caller skip the point.
            if attempt >= 1:
                raise
            time.sleep(delay)
        except requests.ConnectionError:
            if attempt == config.RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def fetch_operational(
    pd: PointDirection,
    start: dt.date,
    end: dt.date,
    indicator: str = "Physical Flow",
    period_type: str = "day",
) -> list[dict]:
    """Rows for one point-direction, [start, end] inclusive in gas days."""
    payload = _get(
        "operationalData",
        {
            "operatorKey": pd.operator_key,
            "pointKey": pd.point_key,
            "directionKey": pd.direction,
            "indicator": indicator,
            "periodType": period_type,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "timezone": "CET",
            "limit": -1,
        },
    )
    rows = []
    for raw in payload.get("operationalData", []):
        row = {new: raw.get(old) for old, new in _KEEP.items()}
        # Gas day label = CET date on which the 06:00-06:00 gas day starts.
        pf = row.get("period_from") or ""
        row["gas_day"] = pf[:10]
        rows.append(row)
    return rows


def fetch_range_chunked(
    pd: PointDirection,
    start: dt.date,
    end: dt.date,
    indicator: str = "Physical Flow",
    period_type: str = "day",
    chunk_days: int = 120,
    pause: float = 0.4,
) -> Iterator[list[dict]]:
    """Long ranges in polite chunks (large windows occasionally 500 or truncate)."""
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days - 1), end)
        yield fetch_operational(pd, cursor, chunk_end, indicator, period_type)
        cursor = chunk_end + dt.timedelta(days=1)
        if cursor <= end:
            time.sleep(pause)


def fetch_terminals(
    terminals: Iterable[Terminal],
    start: dt.date,
    end: dt.date,
    indicator: str = "Physical Flow",
    period_type: str = "day",
    pause: float = 0.4,
) -> list[dict]:
    """All point-directions of the given terminals, tagged with the terminal slug."""
    out: list[dict] = []
    for t in terminals:
        for pd in t.entsog:
            try:
                rows = fetch_operational(pd, start, end, indicator, period_type)
            except Exception as e:  # one bad point must not sink the snapshot
                print(f"  ! {t.slug} {pd.operator_key}/{pd.point_key}: {e}")
                continue
            for r in rows:
                r["terminal"] = t.slug
            out.extend(rows)
            time.sleep(pause)
    return out
