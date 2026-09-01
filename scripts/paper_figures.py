#!/usr/bin/env python3
"""Generate the paper's figures into paper/figures/ (300 dpi, light palette):
  fig_nowcast_bands.png — Gate evening nowcast vs truth with 90% bands
  fig_kernel.png        — intraday deviation correlation: data vs Brownian vs fBm
  fig_cover.png         — days-of-cover and EU storage, the stress pair
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lng_nowcast import config

SURFACE, INK, INK2, MUTED = "#ffffff", "#111820", "#4c5a67", "#8a97a3"
GRID, BLUE, AQUA, ORANGE = "#e3e2dc", "#2a78d6", "#0e7d8a", "#c2551f"
FIGDIR = ROOT / "paper" / "figures"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def fig_bands():
    ev = pd.read_csv(config.RAW_DIR / "nowcast_eval.csv")
    g = ev[(ev.terminal == "gate") & (ev.horizon == "H2")].copy()
    g = g[(g.gas_day >= "2026-06-01") & (g.gas_day <= "2026-08-31")]
    x = pd.to_datetime(g.gas_day)
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.fill_between(x, g.q05, g.q95, color=BLUE, alpha=0.18, lw=0, label="90% band")
    ax.plot(x, g.q50, color=BLUE, lw=1.6, label="evening nowcast (median)")
    ax.plot(x, g.truth, color=INK, lw=1.0, ls="--", label="published next morning")
    ax.set_ylabel("GWh/day", fontsize=8, color=INK2)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.set_title("Gate: evening nowcast vs the figure published next morning",
                 fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_nowcast_bands.png", facecolor=SURFACE)


def fig_kernel():
    h = pd.read_csv(config.RAW_DIR / "entsog_hourly_flows.csv")
    h = h[h.indicator == "Physical Flow"].copy()
    h["gwh"] = pd.to_numeric(h.value_kwh, errors="coerce") / 1e6
    local = pd.to_datetime(h.period_from.str[:19])
    sh = local - pd.Timedelta(hours=6)
    h["gas_day"], h["hour"] = sh.dt.strftime("%Y-%m-%d"), sh.dt.hour + 1
    hours = np.arange(1, 24)
    u = hours / 24

    def emp_corr_row(term):
        piv = h[h.terminal == term].pivot_table(index="gas_day", columns="hour",
                                                values="gwh", aggfunc="sum").dropna()
        cum = piv.sort_index(axis=1).cumsum(axis=1)
        S = cum.iloc[:, -1]
        d = (cum.loc[S > 50, hours].div(S[S > 50], axis=0).sub(u, axis=1)).to_numpy()
        return np.corrcoef(d, rowvar=False)[3]  # corr(d(4h), d(u))

    def bb_row(uu):
        s, t = np.minimum(uu[3], uu), np.maximum(uu[3], uu)
        cov = s - uu[3] * uu
        var = uu * (1 - uu)
        return cov / np.sqrt(var[3] * var)

    def fbm_row(uu, H):
        def C(a, b): return 0.5 * (a**(2*H) + b**(2*H) - np.abs(a - b)**(2*H))
        K = np.array([C(uu[3], t) - C(uu[3], 1) * C(t, 1) / C(1, 1) for t in uu])
        V = np.array([C(t, t) - C(t, 1)**2 / C(1, 1) for t in uu])
        return K / np.sqrt(V[3] * V)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, term in zip(axes, ("gate", "zeebrugge")):
        style(ax)
        ax.plot(u, emp_corr_row(term), "o", ms=3, color=INK, label="empirical")
        ax.plot(u, bb_row(u), ls="--", lw=1.2, color=MUTED, label="Brownian bridge")
        ax.plot(u, fbm_row(u, 0.98), lw=1.5, color=BLUE, label="fBm bridge, H=0.98")
        ax.set_title(term, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("day fraction u", fontsize=8, color=INK2)
    axes[0].set_ylabel("corr(d(4h), d(u))", fontsize=8, color=INK2)
    axes[1].legend(fontsize=7, frameon=False, loc="lower center")
    fig.suptitle("Intraday deviations are one persistent ramp, not Brownian noise",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGDIR / "fig_kernel.png", facecolor=SURFACE)


def fig_cover():
    spec = importlib.util.spec_from_file_location("ss", ROOT / "scripts" / "stress_study.py")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    df = ss.build_panel().tail(220)
    x = pd.to_datetime(df.index)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.6), dpi=300, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style(ax)
    axes[0].plot(x, df.cover, color=BLUE, lw=1.5)
    axes[0].axhline(df.cover.median(), color=MUTED, ls="--", lw=1)
    axes[0].set_ylabel("days of cover", fontsize=8, color=INK2)
    axes[0].set_title("NW LNG tank cover and EU storage entering winter 2026",
                      fontsize=9.5, color=INK, loc="left")
    axes[1].plot(x, df.storage, color=AQUA, lw=1.5)
    axes[1].axhline(90, color=ORANGE, ls="--", lw=1)
    axes[1].text(x[2], 90.7, "≈ typical at season end (2024–25)",
                 fontsize=7, color=ORANGE)
    axes[1].set_ylabel("EU storage %", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_cover.png", facecolor=SURFACE)


if __name__ == "__main__":
    FIGDIR.mkdir(exist_ok=True)
    fig_bands()
    fig_kernel()
    fig_cover()
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"{f.name}: {f.stat().st_size/1024:.0f} KB")
