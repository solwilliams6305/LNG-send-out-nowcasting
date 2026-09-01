"""Cargo-arrival extraction from the physical series (no AIS required).

Mass balance per terminal and gas day:  I_t = I_{t-1} + A_t - S_t - losses
so implied arrivals  A_t = dI_t + S_t (+ small losses folded into noise).

Sources:
  UK  — National Gas publishes storage *inflow* directly (metric="inflow" in
        data/raw/nationalgas_daily.csv): arrival energy with no inversion.
  EU  — implied from ALSI inventory deltas + send-out
        (data/raw/alsi_daily.csv; inventory_gwh preferred, volume x
        ENERGY_PER_M3 as fallback).

A discharge often spans a gas-day boundary, so consecutive above-threshold
days are clustered into one arrival event. Event energies should cluster at
the vessel-class discharge energies in physics.py — that agreement is the
calibration of the filter's jump-size prior, checked by
scripts/arrivals_report.py before AIS data has even accumulated.
"""

from __future__ import annotations

import pandas as pd

from . import config, physics

DAY_THRESHOLD_GWH = 100.0  # a day counts as "receiving" above this
MIN_EVENT_GWH = 150.0      # discard clusters smaller than this (noise, top-ups)


def _cluster(days: pd.DataFrame, source: str) -> list[dict]:
    """Group consecutive receiving days (per terminal) into arrival events."""
    events = []
    for terminal, g in days.groupby("terminal"):
        g = g.sort_values("gas_day").reset_index(drop=True)
        g["gas_day"] = pd.to_datetime(g["gas_day"])
        cluster: list[pd.Series] = []
        for _, row in g.iterrows():
            if cluster and (row.gas_day - cluster[-1].gas_day).days > 1:
                events.append(_event(terminal, cluster, source))
                cluster = []
            cluster.append(row)
        if cluster:
            events.append(_event(terminal, cluster, source))
    return [e for e in events if e["energy_gwh"] >= MIN_EVENT_GWH]


def _event(terminal: str, cluster: list, source: str) -> dict:
    return {
        "terminal": terminal,
        "start_day": cluster[0].gas_day.date().isoformat(),
        "n_days": len(cluster),
        "energy_gwh": round(sum(r.arrival_gwh for r in cluster), 1),
        "source": source,
    }


def uk_arrivals() -> pd.DataFrame:
    ng = pd.read_csv(config.RAW_DIR / "nationalgas_daily.csv")
    inflow = ng[ng.metric == "inflow"].copy()
    inflow["arrival_gwh"] = pd.to_numeric(inflow["value"], errors="coerce") / 1e6
    daily = inflow.groupby(["terminal", "gas_day"], as_index=False)["arrival_gwh"].sum()
    receiving = daily[daily.arrival_gwh >= DAY_THRESHOLD_GWH]
    return pd.DataFrame(_cluster(receiving, "ng_inflow"))


def eu_implied_arrivals() -> pd.DataFrame:
    al = pd.read_csv(config.RAW_DIR / "alsi_daily.csv")
    al = al[al.status.isin(["E", "C"])].copy()
    for c in ("inventory_gwh", "inventory_1e3m3", "send_out_gwh_d"):
        al[c] = pd.to_numeric(al[c], errors="coerce")
    # Prefer GIE's own energy conversion; fall back to volume x central GCV.
    al["inv_gwh"] = al["inventory_gwh"].fillna(
        al["inventory_1e3m3"] * 1e3 * physics.ENERGY_PER_M3.value / 1e3
    )
    daily = (
        al.groupby(["terminal", "gas_day"], as_index=False)
        .agg(inv_gwh=("inv_gwh", "sum"), send_out=("send_out_gwh_d", "sum"))
        .sort_values(["terminal", "gas_day"])
    )
    daily["d_inv"] = daily.groupby("terminal")["inv_gwh"].diff()
    daily["gap_days"] = (
        pd.to_datetime(daily.gas_day).groupby(daily.terminal).diff().dt.days
    )
    daily = daily[daily.gap_days == 1]  # a data gap invalidates the delta
    daily["arrival_gwh"] = daily.d_inv + daily.send_out
    receiving = daily[daily.arrival_gwh >= DAY_THRESHOLD_GWH]
    return pd.DataFrame(_cluster(receiving, "alsi_implied"))


def all_arrivals() -> pd.DataFrame:
    frames = [f for f in (uk_arrivals(), eu_implied_arrivals()) if len(f)]
    out = pd.concat(frames, ignore_index=True).sort_values(["terminal", "start_day"])
    return out.reset_index(drop=True)


def class_reference_lines() -> dict[str, float]:
    """Vessel-class full-discharge energies (GWh) for calibration overlays."""
    out = {}
    for _, _, _, _, cap, half in physics.VESSEL_CLASSES:
        b = physics.Bounded(cap, cap - half, cap + half, "m3")
        e, _ = physics.full_discharge_energy(b)
        out[f"{cap / 1000:.0f}k m3"] = round(e)
    return out
