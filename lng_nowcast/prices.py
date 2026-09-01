"""Free daily price series for the spread covariate and the event study.

Sources (memo Annex B, verified live 2026-09-01):
  uk_sap  — National Gas "SAP, Actual Day" (PUBOB603), p/kWh: the volume-
            weighted on-the-day commodity market price, NBP-linked; same
            keyless find-gas-data API as everything else. (PUBOB47 is the
            hourly variant when intraday UK prices are wanted.)
  ttf     — Dutch TTF front-month future daily close, EUR/MWh, via Yahoo
            Finance's chart API (symbol TTF=F). Personal-research use:
            derive statistics from it, do not redistribute raw prices in
            the public repo (data/raw/ is gitignored).
Cross-check candidate (not implemented): EEX Neutral Gas Price files.
JKM has no free daily source — the write-up says so honestly.
"""

from __future__ import annotations

import datetime as dt

import requests

from . import config, nationalgas

SAP_ITEM = nationalgas.Item("PUBOB603", "uk", "sap", "D0", "SAP actual day, p/kWh")


def fetch_uk_sap(start: dt.date, end: dt.date) -> list[dict]:
    rows = nationalgas.fetch_item(SAP_ITEM, start, end)
    return [
        {"date": r["gas_day"], "series": "uk_sap_p_kwh", "value": r["value"],
         "source": "nationalgas PUBOB603"}
        for r in rows
    ]


def fetch_ttf(start: dt.date, end: dt.date) -> list[dict]:
    """Daily TTF=F closes from Yahoo's chart API."""
    p1 = int(dt.datetime.combine(start, dt.time(), dt.timezone.utc).timestamp())
    p2 = int(dt.datetime.combine(end + dt.timedelta(days=1), dt.time(), dt.timezone.utc).timestamp())
    r = requests.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/TTF=F",
        params={"period1": p1, "period2": p2, "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0 (research use)"},
        timeout=config.HTTP_TIMEOUT,
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    stamps = res.get("timestamp") or []
    closes = res["indicators"]["quote"][0].get("close") or []
    out = []
    for ts, c in zip(stamps, closes):
        if c is None:
            continue
        d = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()
        out.append({"date": d, "series": "ttf_front_eur_mwh", "value": c,
                    "source": "yahoo TTF=F"})
    return out
