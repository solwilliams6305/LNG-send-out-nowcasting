"""Read the snapshot archive back into revision paths.

A "revision path" is the trajectory of what a source claimed about one
(series, gas_day) across successive snapshots — the raw material for modelling
the Estimated→Confirmed process. Useful once a couple of weeks of snapshots
exist; harmless to run earlier.
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_snapshots(kind: str) -> pd.DataFrame:
    """Concatenate all snapshot CSVs of one kind ('alsi', 'entsog', 'entsog_hourly')."""
    d = config.SNAPSHOT_DIR / kind
    files = sorted(d.glob("*.csv")) if d.exists() else []
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"])
    return df


def alsi_revision_paths() -> pd.DataFrame:
    """Per (facility, gas_day): first/last reported send-out, revision size, and
    time from first estimate to first Confirmed status."""
    df = load_snapshots("alsi")
    if df.empty:
        return df
    df = df.sort_values("snapshot_utc")

    def summarise(g: pd.DataFrame) -> pd.Series:
        confirmed = g[g["status"] == "C"]
        return pd.Series(
            {
                "n_snapshots": len(g),
                "first_send_out": g["send_out_gwh_d"].iloc[0],
                "last_send_out": g["send_out_gwh_d"].iloc[-1],
                "revision_gwh": g["send_out_gwh_d"].iloc[-1] - g["send_out_gwh_d"].iloc[0],
                "first_status": g["status"].iloc[0],
                "ever_confirmed": bool(len(confirmed)),
                "first_confirmed_at": confirmed["snapshot_utc"].iloc[0] if len(confirmed) else pd.NaT,
                "first_seen_at": g["snapshot_utc"].iloc[0],
            }
        )

    return (
        df.groupby(["terminal", "facility_eic", "gas_day"])
        .apply(summarise, include_groups=False)
        .reset_index()
    )


def entsog_revision_paths() -> pd.DataFrame:
    """Per (terminal, point, gas_day): first/last value and lastUpdate drift."""
    df = load_snapshots("entsog")
    if df.empty:
        return df
    df = df[df["indicator"] == "Physical Flow"].sort_values("snapshot_utc")
    g = df.groupby(["terminal", "operator_key", "point_key", "direction", "gas_day"])
    out = g.agg(
        n_snapshots=("value_kwh", "size"),
        first_value_kwh=("value_kwh", "first"),
        last_value_kwh=("value_kwh", "last"),
        last_update_final=("last_update", "last"),
    ).reset_index()
    out["revision_kwh"] = out["last_value_kwh"] - out["first_value_kwh"]
    return out
