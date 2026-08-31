#!/usr/bin/env python3
"""Sanity/overview chart: daily send-out per core terminal from backfilled ENTSOG data.

Small multiples (one terminal per facet, free y, shared x) — the terminals span
~50-900 GWh/d, so a shared-axis spaghetti plot would flatten the small ones.
Single-hue design: identity is carried by the facet title, not color.
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lng_nowcast import config
from lng_nowcast.terminals import by_slug, tier

# Reference dataviz tokens (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"

WINDOW_DAYS = 365


def main() -> int:
    src = config.RAW_DIR / "entsog_daily_flows.csv"
    if not src.exists():
        print(f"missing {src} — run scripts/backfill_entsog.py first", file=sys.stderr)
        return 1

    df = pd.read_csv(src)
    df = df[df["indicator"] == "Physical Flow"].copy()
    df["gas_day"] = pd.to_datetime(df["gas_day"])
    df["gwh"] = pd.to_numeric(df["value_kwh"], errors="coerce") / 1e6
    cutoff = df["gas_day"].max() - pd.Timedelta(days=WINDOW_DAYS)
    df = df[df["gas_day"] >= cutoff]

    # Sum across point-directions (Mukran reports via two TSOs for one point).
    daily = df.groupby(["terminal", "gas_day"], as_index=False)["gwh"].sum()

    slugs = [t.slug for t in tier("core")]
    order = (
        daily[daily["terminal"].isin(slugs)]
        .groupby("terminal")["gwh"].mean().sort_values(ascending=False).index.tolist()
    )

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for ax, slug in zip(axes.flat, order):
        t = by_slug(slug)
        sub = daily[daily["terminal"] == slug].sort_values("gas_day")
        ma = sub["gwh"].rolling(7, min_periods=3).mean()

        ax.set_facecolor(SURFACE)
        ax.plot(sub["gas_day"], sub["gwh"], color=SERIES, lw=0.8, alpha=0.30)
        ax.plot(sub["gas_day"], ma, color=SERIES, lw=2.0)

        if len(sub):
            last = sub.iloc[-1]
            ax.plot([last["gas_day"]], [last["gwh"]], "o", ms=5, color=SERIES)
            ax.annotate(
                f"{last['gwh']:.0f}",
                (last["gas_day"], last["gwh"]),
                textcoords="offset points", xytext=(6, 0),
                fontsize=9, color=INK, fontweight="bold", va="center",
            )

        ax.set_title(t.name, fontsize=10, color=INK, loc="left", fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)

    fig.suptitle(
        "NW-European LNG send-out — daily physical flow into the grid (GWh/d)",
        fontsize=13, color=INK, fontweight="bold", x=0.01, ha="left",
    )
    pulled = dt.date.today().isoformat()
    fig.text(
        0.01, 0.005,
        f"Thin line: daily · thick: 7-day mean · marker: latest day.  "
        f"UK points (Grain, Milford Haven) publish ~6 days late on ENTSOG.  "
        f"Data: ENTSOG Transparency Platform, pulled {pulled}.",
        fontsize=8, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    config.REPORTS_DIR.mkdir(exist_ok=True)
    out = config.REPORTS_DIR / "sendout_overview.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
