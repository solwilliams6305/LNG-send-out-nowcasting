"""AIS layer: LNG carrier observation around the registered terminals.

Source: aisstream.io websocket (free key, AISSTREAM_KEY in .env / env).
Protocol (verified against docs 2026-09-02): wss://stream.aisstream.io/v0/stream;
subscribe within 3s of connecting with
  {"APIKey": ..., "BoundingBoxes": [[[lat_min, lon_min], [lat_max, lon_max]], ...],
   "FilterMessageTypes": ["PositionReport", "ShipStaticData"]}
Limits: 3 connections per account/IP.

Design: berth calls last many hours and AIS draught persists after departure,
so bounded listening windows inside the 3x-daily snapshot runs already capture
berth occupancy + static state well enough for daily arrival detection; a
continuous local logger (scripts/ais_logger.py) adds density when running.

Geometry is two-tier per terminal: a wide CAPTURE box (subscription + logging,
~±5 km) and a tight BERTH box (the at-berth flag, ~±1.1 km) centred on the
ALSI-published jetty coordinates (Dragon provisional — no ALSI entry). FSRU
terminals' own permanently-moored units (e.g. Energos Igloo at Eemshaven) show
up perpetually at berth: treat vessels berthed across many snapshots as
infrastructure, not arrivals.

Coverage caveat (first live listens, 2026-09-02, ~01:00 UTC): aisstream's
community receiver network delivered nothing around Milford Haven, Dunkerque,
or Mukran while Rotterdam/Elbe/Grain were rich — terrestrial coverage is
area- and time-dependent. Track per-terminal message counts across snapshots
before concluding anything from AIS silence; UK arrivals fall back on National
Gas inflow data regardless.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from dataclasses import dataclass, field

from . import config
from .terminals import TERMINALS

STREAM_URL = "wss://stream.aisstream.io/v0/stream"

# AIS ship-type first digit 8 = tanker; LNG carriers report 80-89.
TANKER_TYPES = range(80, 90)
MIN_CARRIER_LOA = 180.0  # metres; below this it isn't a deep-sea LNG carrier


@dataclass(frozen=True)
class BerthBox:
    terminal: str
    sub: str  # sub-terminal attribution ("" if same as terminal)
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


def _box(terminal: str, sub: str, lat: float, lon: float, dlat: float, dlon: float) -> BerthBox:
    return BerthBox(terminal, sub, lat - dlat, lat + dlat, lon - dlon, lon + dlon)


# Wide capture boxes: one per terminal, from registry coordinates.
CAPTURE_BOXES: tuple[BerthBox, ...] = tuple(
    _box(t.slug, "", t.approx_lat, t.approx_lon, 0.05, 0.08) for t in TERMINALS
)

# Tight berth boxes (~±1.1 km) around the ALSI-published jetty coordinates.
# Milford Haven splits into its two jetties: south_hook uses the ALSI position;
# dragon stays PROVISIONAL (no ALSI entry) until a carrier is observed there.
BERTH_BOXES: tuple[BerthBox, ...] = tuple(
    [
        _box(t.slug, "", t.approx_lat, t.approx_lon, 0.010, 0.016)
        for t in TERMINALS
        if t.slug != "milford_haven"
    ]
    + [
        _box("milford_haven", "south_hook", 51.7210, -5.0809, 0.010, 0.016),
        _box("milford_haven", "dragon", 51.7000, -4.9920, 0.010, 0.016),
    ]
)


def subscription(api_key: str) -> dict:
    return {
        "APIKey": api_key,
        "BoundingBoxes": [
            [[b.lat_min, b.lon_min], [b.lat_max, b.lon_max]] for b in CAPTURE_BOXES
        ],
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    }


def _meta(raw: dict) -> tuple[int | None, str | None, float | None, float | None, str | None]:
    md = raw.get("MetaData") or {}
    lat = md.get("latitude", md.get("Latitude"))
    lon = md.get("longitude", md.get("Longitude"))
    name = (md.get("ShipName") or "").strip() or None
    return md.get("MMSI"), name, lat, lon, md.get("time_utc")


def normalize(raw: dict) -> dict | None:
    """One aisstream message -> flat record (or None if irrelevant)."""
    mtype = raw.get("MessageType")
    mmsi, name, lat, lon, t = _meta(raw)
    if mmsi is None:
        return None
    if mtype == "PositionReport":
        m = (raw.get("Message") or {}).get("PositionReport") or {}
        return {
            "kind": "position",
            "mmsi": mmsi,
            "name": name,
            "lat": m.get("Latitude", lat),
            "lon": m.get("Longitude", lon),
            "sog": m.get("Sog"),
            "nav_status": m.get("NavigationalStatus"),
            "time_utc": t,
        }
    if mtype == "ShipStaticData":
        m = (raw.get("Message") or {}).get("ShipStaticData") or {}
        dim = m.get("Dimension") or {}
        loa = (dim.get("A") or 0) + (dim.get("B") or 0)
        beam = (dim.get("C") or 0) + (dim.get("D") or 0)
        eta = m.get("Eta") or {}
        return {
            "kind": "static",
            "mmsi": mmsi,
            "name": (m.get("Name") or "").strip() or name,
            "imo": m.get("ImoNumber"),
            "ship_type": m.get("Type"),
            "loa": loa or None,
            "beam": beam or None,
            "draught_m": m.get("MaximumStaticDraught"),
            "destination": (m.get("Destination") or "").strip() or None,
            "eta_month": eta.get("Month"),
            "eta_day": eta.get("Day"),
            "eta_hour": eta.get("Hour"),
            "lat": lat,
            "lon": lon,
            "time_utc": t,
        }
    return None


def locate(lat: float | None, lon: float | None) -> tuple[str | None, str | None, bool]:
    """(capture terminal, berth sub-attribution, in_berth_box) for a position."""
    if lat is None or lon is None:
        return None, None, False
    capture = next((b.terminal for b in CAPTURE_BOXES if b.contains(lat, lon)), None)
    berth = next((b for b in BERTH_BOXES if b.contains(lat, lon)), None)
    if berth:
        return capture or berth.terminal, berth.sub or berth.terminal, True
    return capture, None, False


@dataclass
class VesselState:
    mmsi: int
    name: str | None = None
    imo: int | None = None
    ship_type: int | None = None
    loa: float | None = None
    beam: float | None = None
    draught_m: float | None = None
    destination: str | None = None
    lat: float | None = None
    lon: float | None = None
    sog: float | None = None
    nav_status: int | None = None
    n_pos: int = 0
    first_seen: str | None = None
    last_seen: str | None = None
    extra: dict = field(default_factory=dict)

    def update(self, rec: dict) -> None:
        self.first_seen = self.first_seen or rec.get("time_utc")
        self.last_seen = rec.get("time_utc") or self.last_seen
        if rec["kind"] == "position":
            self.n_pos += 1
            for k in ("lat", "lon", "sog", "nav_status"):
                if rec.get(k) is not None:
                    setattr(self, k, rec[k])
            self.name = self.name or rec.get("name")
        else:
            for k in ("name", "imo", "ship_type", "loa", "beam", "draught_m", "destination"):
                if rec.get(k) is not None:
                    setattr(self, k, rec[k])

    def row(self, snapshot_utc: str) -> dict:
        terminal, berth_sub, in_berth = locate(self.lat, self.lon)
        slow = self.sog is not None and self.sog <= 0.5
        moored = self.nav_status == 5  # AIS "moored"
        likely_lng = bool(
            self.loa and self.loa >= MIN_CARRIER_LOA
            and (self.ship_type in TANKER_TYPES or self.ship_type in (None, 0))
            # LNG carriers draw <= ~12.5 m; deeper = crude/products tanker.
            and (self.draught_m is None or self.draught_m <= 12.8)
        )
        return {
            "snapshot_utc": snapshot_utc,
            "mmsi": self.mmsi,
            "name": self.name,
            "imo": self.imo,
            "ship_type": self.ship_type,
            "loa": self.loa,
            "beam": self.beam,
            "draught_m": self.draught_m,
            "lat": self.lat,
            "lon": self.lon,
            "sog": self.sog,
            "nav_status": self.nav_status,
            "terminal": terminal,
            "berth_sub": berth_sub,
            "at_berth": bool(in_berth and (slow or moored)),
            "likely_lng_carrier": likely_lng,
            "destination": self.destination,
            "n_pos": self.n_pos,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


async def _listen_async(
    api_key: str,
    duration_s: float | None,
    raw_sink=None,
) -> dict[int, VesselState]:
    """Collect vessel state until the deadline, reconnecting through the
    periodic server-side drops aisstream is known for ("no close frame
    received or sent"). Partial data is always returned; only a protocol-level
    error payload (e.g. bad key) raises."""
    import websockets  # imported lazily so the package works without it installed

    vessels: dict[int, VesselState] = {}
    loop = asyncio.get_event_loop()
    deadline = None if duration_s is None else loop.time() + duration_s

    while deadline is None or deadline - loop.time() > 1:
        try:
            async with websockets.connect(
                STREAM_URL, ping_interval=20, close_timeout=5, open_timeout=30
            ) as ws:
                await ws.send(json.dumps(subscription(api_key)))
                while True:
                    timeout = None if deadline is None else max(0.1, deadline - loop.time())
                    if deadline is not None and timeout <= 0.11:
                        return vessels
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        return vessels
                    try:
                        raw = json.loads(msg)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(raw, dict) and raw.get("error"):
                        raise RuntimeError(f"aisstream error: {raw['error']}")
                    rec = normalize(raw) if isinstance(raw, dict) else None
                    if rec is None:
                        continue
                    if raw_sink is not None:
                        raw_sink(raw)
                    vessels.setdefault(rec["mmsi"], VesselState(rec["mmsi"])).update(rec)
        except RuntimeError:
            raise
        except Exception:
            # Dropped connection / handshake hiccup: reconnect if time remains.
            if deadline is not None and deadline - loop.time() <= 5:
                return vessels
            await asyncio.sleep(3)
    return vessels


def listen(duration_s: float, raw_sink=None) -> dict[int, VesselState]:
    """Bounded synchronous listening window; returns per-vessel latest state."""
    if not config.AISSTREAM_KEY:
        raise RuntimeError(
            "AISSTREAM_KEY is not set — register free at https://aisstream.io "
            "and add the key to .env / the Actions secrets."
        )
    return asyncio.run(_listen_async(config.AISSTREAM_KEY, duration_s, raw_sink))


def snapshot_rows(duration_s: float = 480) -> list[dict]:
    """One AIS observation window -> snapshot rows for vessels in capture boxes."""
    snap = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    vessels = listen(duration_s)
    rows = [v.row(snap) for v in vessels.values()]
    # Keep everything seen in a capture box; tiny craft are filtered at analysis
    # time, but berthed-or-carrier rows are the signal.
    return [r for r in rows if r["terminal"] is not None]
