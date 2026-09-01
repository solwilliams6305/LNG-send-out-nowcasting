#!/usr/bin/env python3
"""Offline self-test of the AIS layer: geometry, normalization, berth flags,
and the physics inversion — no network, no key. Run after any change to
ais.py/physics.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lng_nowcast import ais, physics


def approx(x, target, rel):
    return abs(x - target) <= rel * target


def main() -> int:
    # --- geometry: Gate berth box contains the Gate jetty; no cross-claims ---
    term, sub, in_berth = ais.locate(51.9737, 4.0728)
    assert (term, sub, in_berth) == ("gate", "gate", True), (term, sub, in_berth)
    term, sub, in_berth = ais.locate(51.7210, -5.0809)
    assert (term, sub, in_berth) == ("milford_haven", "south_hook", True)
    term, sub, in_berth = ais.locate(51.700, -4.992)
    assert (term, sub, in_berth) == ("milford_haven", "dragon", True)
    term, sub, in_berth = ais.locate(51.70, -5.03)  # mid-waterway: capture only
    assert term == "milford_haven" and not in_berth, ais.locate(51.70, -5.03)
    assert ais.locate(48.0, 2.0) == (None, None, False)  # Paris is not a berth
    assert len(ais.subscription("k")["BoundingBoxes"]) == len(ais.CAPTURE_BOXES)

    # --- normalization + state accumulation -> berthed carrier row ---
    static = {
        "MessageType": "ShipStaticData",
        "MetaData": {"MMSI": 229929000, "ShipName": "TEST CARRIER",
                     "latitude": 51.9737, "longitude": 4.0728, "time_utc": "t0"},
        "Message": {"ShipStaticData": {
            "Name": "TEST CARRIER", "ImoNumber": 9700001, "Type": 84,
            "Dimension": {"A": 250, "B": 45, "C": 20, "D": 26},
            "MaximumStaticDraught": 11.6, "Destination": "GATE",
            "Eta": {"Month": 9, "Day": 2, "Hour": 6}}},
    }
    pos = {
        "MessageType": "PositionReport",
        "MetaData": {"MMSI": 229929000, "ShipName": "TEST CARRIER",
                     "latitude": 51.9737, "longitude": 4.0728, "time_utc": "t1"},
        "Message": {"PositionReport": {"Latitude": 51.9739, "Longitude": 4.0731,
                                       "Sog": 0.1, "NavigationalStatus": 5}},
    }
    v = ais.VesselState(229929000)
    for raw in (static, pos):
        rec = ais.normalize(raw)
        assert rec is not None
        v.update(rec)
    row = v.row("2026-09-02T00:00:00+00:00")
    assert row["loa"] == 295 and row["beam"] == 46
    assert row["at_berth"] and row["likely_lng_carrier"]
    assert row["terminal"] == "gate" and row["berth_sub"] == "gate"
    assert row["draught_m"] == 11.6 and row["n_pos"] == 1

    # a fast-moving ferry-sized vessel in the box is neither berthed nor a carrier
    v2 = ais.VesselState(2)
    v2.update({"kind": "static", "mmsi": 2, "loa": 120.0, "beam": 20.0,
               "ship_type": 60, "time_utc": "t0"})
    v2.update({"kind": "position", "mmsi": 2, "lat": 51.9737, "lon": 4.0728,
               "sog": 14.0, "nav_status": 0, "time_utc": "t1"})
    row2 = v2.row("s")
    assert not row2["at_berth"] and not row2["likely_lng_carrier"]

    # --- physics: class lookup, discharge energy, TPC, draught inversion ---
    cap = physics.capacity_from_dimensions(295, 46)
    assert cap is not None and approx(cap.value, 172_000, 0.01)
    assert physics.capacity_from_dimensions(345, 54).value == 265_000  # Q-Max
    assert physics.capacity_from_dimensions(80, 12) is None

    e, half = physics.full_discharge_energy(cap)
    assert approx(e, 1_070, 0.12), e  # ~1.0-1.2 TWh for a standard cargo
    assert 0.05 * e < half < 0.25 * e, (e, half)

    tpc = physics.tonnes_per_cm(295, 46)
    assert approx(tpc.value, 121, 0.05), tpc.value

    # 2.2 m draught decrease with ballast compensation ~ a standard cargo
    ed, ehalf = physics.draught_delta_to_energy(295, 46, 2.2)
    assert approx(ed, 1_015, 0.25), ed
    assert ehalf / ed > 0.3  # honestly wide — draught is the coarse channel

    print("selftest_ais: all assertions passed")
    print(f"  standard carrier (295x46m): capacity {cap.value/1000:.0f}k m3, "
          f"full discharge {e:.0f} ± {half:.0f} GWh")
    print(f"  TPC {tpc.value:.0f} t/cm; 2.2 m draught delta -> {ed:.0f} ± {ehalf:.0f} GWh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
