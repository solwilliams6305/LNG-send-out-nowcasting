#!/usr/bin/env python3
"""Extract the 2024->today cargo-arrival dataset and check the jump-size
calibration: do observed event energies cluster at the vessel-class discharge
energies from physics.py?

Writes data/raw/arrival_events.csv and reports/arrival_sizes.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lng_nowcast import arrivals, config

# Reference dataviz tokens (light mode), as in eda_overview.py
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SERIES = "#e1e0d9", "#c3c2b7", "#2a78d6"


def main() -> int:
    ev = arrivals.all_arrivals()
    out = config.RAW_DIR / "arrival_events.csv"
    ev.to_csv(out, index=False)

    print(f"{len(ev)} arrival events -> {out}")
    print("\nper terminal:")
    per = ev.groupby(["terminal", "source"]).agg(
        n=("energy_gwh", "size"),
        median_gwh=("energy_gwh", "median"),
        mean_gwh=("energy_gwh", "mean"),
        max_gwh=("energy_gwh", "max"),
    )
    print(per.round(1).to_string())

    # Grain triangulation: NG inflow (direct) vs nothing to compare EU-side yet;
    # the cross-check that matters now is the class clustering.
    refs = arrivals.class_reference_lines()
    print("\nvessel-class discharge references (GWh):", refs)

    print("\ncluster length distribution (days):",
          ev.n_days.value_counts().sort_index().head(6).to_dict())

    # Long clusters at multi-berth terminals chain several ships (Gate can
    # discharge overlapping cargoes) — restrict the size histogram to short
    # clusters, which are near-certainly single vessels.
    single = ev[ev.n_days <= 3]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist(single.energy_gwh.clip(upper=2600), bins=49, range=(150, 2600),
            color=SERIES, edgecolor=SURFACE, linewidth=0.6)
    for label, e in refs.items():
        if e <= 2600:
            ax.axvline(e, color=INK_2, lw=1, ls="--")
            ax.text(e, ax.get_ylim()[1] * 0.97, f" {label}\n {e} GWh",
                    fontsize=8, color=INK_2, va="top")
    ax.set_title(
        "LNG cargo-arrival event sizes, NW Europe 2024–2026 (clusters ≤ 3 days) — observed vs vessel classes",
        fontsize=12, color=INK, loc="left", fontweight="bold",
    )
    ax.set_xlabel("event energy (GWh)", color=INK_2, fontsize=9)
    ax.set_ylabel("events", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    fig.text(0.01, 0.005,
             "Events: UK direct (National Gas inflow) + EU implied (ALSI mass balance); "
             "clusters > 3 days (multi-ship chains at multi-berth terminals) excluded. "
             "Dashed: full-discharge energy per vessel class (physics.py central values).",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    png = config.REPORTS_DIR / "arrival_sizes.png"
    config.REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(png, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
