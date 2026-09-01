#!/usr/bin/env python3
"""Walk-forward evaluation of the reference nowcaster.

Per terminal and gas day t, the filter produces posteriors of S_t at growing
information sets:
  H0  day-ahead-ish : history through t-1 + (EU) the ENTSOG nomination for t
  H1  midday        : + cumulative metered flow through gas-hour 4  (EU only)
  H2  evening       : + cumulative through gas-hour 11               (EU only)
scored against the final published daily figure, versus persistence and
raw-nomination baselines. UK terminals run H0 only (no public intraday flow;
NG prevailing nominations are renominated in-day, so using their archived
final value at H0 would leak — checked and excluded).

Outputs data/raw/nowcast_eval.csv and a per-terminal metrics table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lng_nowcast import arrivals, config, conformal
from lng_nowcast.nowcast import ParticleFilter, TerminalModel, coverage

import os

EU = ["gate", "zeebrugge", "dunkerque", "eems"]
UK = ["grain", "south_hook"]
WARMUP_DAYS = 60
# Overridable so extended-window runs (e.g. the event-thread study from
# 2022) don't disturb the paper's canonical evaluation file.
EVAL_START = os.environ.get("LNG_EVAL_START", "2024-06-01")
EVAL_OUT = os.environ.get("LNG_EVAL_OUT", "nowcast_eval.csv")


def eu_panels() -> dict[str, pd.DataFrame]:
    daily = pd.read_csv(config.RAW_DIR / "entsog_daily_flows.csv")
    daily = daily[daily.indicator == "Physical Flow"]
    s = (daily.assign(gwh=pd.to_numeric(daily.value_kwh, errors="coerce") / 1e6)
              .groupby(["terminal", "gas_day"])["gwh"].sum().rename("s_truth"))

    noms = pd.read_csv(config.RAW_DIR / "entsog_daily_flows.csv")  # same file holds Nomination rows
    noms = noms[noms.indicator == "Nomination"]
    nom = (noms.assign(gwh=pd.to_numeric(noms.value_kwh, errors="coerce") / 1e6)
               .groupby(["terminal", "gas_day"])["gwh"].sum().rename("nom"))
    # Leak check: nominations must have been published before the gas day began.
    lu = pd.to_datetime(noms.last_update, errors="coerce", utc=True)
    gd = pd.to_datetime(noms.gas_day, utc=True) + pd.Timedelta(hours=4)  # 06:00 CET ~ 04:00/05:00 UTC
    frac_pre = float((lu < gd).mean())
    print(f"ENTSOG nominations with last_update before gas-day start: {frac_pre:.1%}")

    hourly = pd.read_csv(config.RAW_DIR / "entsog_hourly_flows.csv")
    hourly = hourly[hourly.indicator == "Physical Flow"].copy()
    hourly["gwh"] = pd.to_numeric(hourly.value_kwh, errors="coerce") / 1e6
    # period_from's calendar date mislabels the post-midnight tail of each
    # 06:00-06:00 gas day: derive the true gas day and gas hour (1..24) from
    # the naive local timestamp shifted back six hours.
    local = pd.to_datetime(hourly.period_from.str[:19])
    shifted = local - pd.Timedelta(hours=6)
    hourly["gas_day"] = shifted.dt.strftime("%Y-%m-%d")
    hourly["hour_idx"] = shifted.dt.hour + 1

    # Guard against TSO quirks (found live: NaTran publishes dunkerque hourly
    # ROWS with NaN values — min_count=1 stops all-NaN days becoming "0 flow",
    # which pinned the first evaluation's dunkerque nowcast to zero). The
    # sum-vs-last check also catches any cumulative-convention TSO.
    n_hours = (hourly.dropna(subset=["gwh"]).groupby(["terminal", "gas_day"])["gwh"]
               .size().rename("n_hours"))
    sum24 = hourly.groupby(["terminal", "gas_day"])["gwh"].sum(min_count=1).rename("sum24")
    last24 = (hourly.sort_values("hour_idx").groupby(["terminal", "gas_day"])["gwh"]
              .last().rename("last24"))
    conv = {}
    chk = pd.concat([sum24, last24, n_hours, s], axis=1).dropna()
    chk = chk[chk.n_hours >= 23]
    for t in chk.index.get_level_values("terminal").unique():
        c = chk.xs(t, level="terminal")
        gap_sum = (c.sum24 - c.s_truth).abs().median()
        gap_last = (c.last24 - c.s_truth).abs().median()
        conv[t] = "cumulative" if gap_last < gap_sum else "rate"
        print(f"hourly convention {t}: {conv[t]} "
              f"(median gap: sum {gap_sum:.1f}, last {gap_last:.1f} GWh)")

    def cum_through(h: int) -> pd.Series:
        parts = []
        for t, kind in conv.items():
            g = hourly[(hourly.terminal == t) & (hourly.hour_idx <= h)]
            if kind == "cumulative":
                v = g.dropna(subset=["gwh"]).sort_values("hour_idx").groupby("gas_day")["gwh"].last()
            else:
                v = g.groupby("gas_day")["gwh"].sum(min_count=h)
            parts.append(v.to_frame("v").assign(terminal=t).set_index("terminal", append=True))
        out = pd.concat(parts)["v"]
        return out.reorder_levels(["terminal", "gas_day"]).rename(f"cum{h}")

    cum4, cum11 = cum_through(4), cum_through(11)

    # Bridge parameters (memo U1), fitted on the pre-evaluation window only:
    # mean intraday profile m_u = E[cum_u/S - u] (Zeebrugge ramps
    # systematically) and bridge scale sigma_p from normalized deviations
    # z = (cum_u/S - u - m_u) / sqrt(u(1-u)).
    profile_rel = {}
    for t in conv:
        cc = pd.concat([cum4.xs(t, level="terminal"), cum11.xs(t, level="terminal"),
                        s.xs(t, level="terminal")], axis=1).dropna()
        cc = cc[(cc.s_truth > 5) & (cc.index < EVAL_START)]
        if len(cc) < 60:
            continue
        # sigma_p is fitted per hour: empirically the deviation process at the
        # GTS terminals is *smoother than Brownian* (evening deviations smaller
        # than the bridge kernel extrapolates from the morning), so a pooled
        # scale over-widens the evening channel. Fitting the true covariance
        # kernel of D(u) — OU-bridge or similar — is Solomon's U1 refinement.
        pars = {}
        for h, col in ((4, "cum4"), (11, "cum11")):
            u = h / 24.0
            dev = cc[col] / cc.s_truth - u
            m_u = float(dev.median())
            z = (dev - m_u) / np.sqrt(u * (1 - u))
            # Robust scale: outage/ramp days belong to the regime proposal,
            # not to the everyday intraday-profile noise.
            sp = 1.4826 * float(np.abs(z - z.median()).median())
            pars[h] = (m_u, float(np.clip(sp, 0.01, 1.0)))
        profile_rel[t] = pars
        print(f"bridge {t}: m4 {pars[4][0]:+.4f} sp4 {pars[4][1]:.3f} | "
              f"m11 {pars[11][0]:+.4f} sp11 {pars[11][1]:.3f}")

    alsi = arrivals.eu_daily_panel().set_index(["terminal", "gas_day"])

    def _xs(series, t):
        try:
            return series.xs(t, level="terminal")
        except KeyError:  # terminal absent from this feed (e.g. dunkerque hourly)
            return pd.Series(dtype=float, name=series.name)

    out = {}
    for t in EU:
        p = pd.concat(
            [_xs(s, t), _xs(nom, t), _xs(cum4, t), _xs(cum11, t), _xs(n_hours, t)],
            axis=1,
        )
        a = alsi.xs(t, level="terminal")
        p = p.join(a[["inv_gwh", "arrival_gwh"]], how="left")
        p = p.sort_index()
        # only trust intraday cumulative when the day has complete hourly data
        p.loc[p.n_hours.fillna(0) < 20, ["cum4", "cum11"]] = np.nan
        out[t] = (p, profile_rel.get(t))
    return out


def uk_panels() -> dict[str, pd.DataFrame]:
    ng = pd.read_csv(config.RAW_DIR / "nationalgas_daily.csv")
    ng["gwh"] = pd.to_numeric(ng.value, errors="coerce") / 1e6
    send = (ng[(ng.metric == "send_out") & (ng.maturity == "D+1")]
            .groupby(["terminal", "gas_day"])["gwh"].sum().rename("s_truth"))
    stock = (ng[ng.metric == "stock"].groupby(["terminal", "gas_day"])["gwh"]
             .sum().rename("opening_stock"))
    inflow = arrivals.uk_daily_inflow().set_index(["terminal", "gas_day"])["arrival_gwh"]

    out = {}
    for t in UK:
        p = pd.concat([send.xs(t, level="terminal"), stock.xs(t, level="terminal"),
                       inflow.xs(t, level="terminal")], axis=1).sort_index()
        # opening stock of day t+1 = inventory at end of day t
        p["inv_gwh"] = p.opening_stock.shift(-1)
        p["nom"] = np.nan  # archived prevailing nominations are post-renomination: leaky
        p["cum4"] = np.nan
        p["cum11"] = np.nan
        out[t] = (p, None)
    return out


def run_terminal(name: str, p: pd.DataFrame, horizons: list[str],
                 rels: dict | None = None) -> pd.DataFrame:
    p = p[p.s_truth.notna()].copy()
    if len(p) < 200:
        print(f"{name}: too little data ({len(p)} days), skipping")
        return pd.DataFrame()

    fit = p[p.index < EVAL_START]
    eval_from = EVAL_START
    if len(fit) < 150:
        # Terminal came online after the fit window began (e.g. Eems,
        # commissioned Sep 2022): fit on its first year of data and start
        # its evaluation only after that — still leak-free, per-terminal.
        fit = p.iloc[: min(365, max(len(p) - 60, 0))]
        if len(fit) < 150:
            print(f"{name}: too little history for a leak-free fit, skipping")
            return pd.DataFrame()
        eval_from = str(fit.index[-1])
    model = TerminalModel.from_history(
        name,
        fit.s_truth.to_numpy(float),
        fit.nom.to_numpy(float) if "nom" in fit else np.full(len(fit), np.nan),
        i_max=float(np.nanmax(p.inv_gwh) * 1.15 + 100) if p.inv_gwh.notna().any() else 4000.0,
        arrivals=fit.arrival_gwh.to_numpy(float) if "arrival_gwh" in fit else None,
    )
    print(f"{name}: sigma_S={model.sigma_s:.1f} GWh, lam={model.lam_arrival:.2f}, "
          f"jump={model.jump_mean:.0f}±{model.jump_sd:.0f}, i_max={model.i_max:.0f}")

    i0 = float(p.inv_gwh.dropna().iloc[0]) if p.inv_gwh.notna().any() else model.i_max / 2
    pf = ParticleFilter(model, i0=i0, s0=float(p.s_truth.iloc[0]))

    rows = []
    prev_s = np.nan
    for k, (day, r) in enumerate(p.iterrows()):
        pf.propagate(nom=r.nom if "nom" in r else np.nan)
        views = {"H0": pf.posterior()}
        if "H1" in horizons:
            views["H1"] = pf.posterior({"cum_h": (4, r.cum4, rels[4] if rels else None)})
        if "H2" in horizons:
            views["H2"] = pf.posterior({"cum_h": (11, r.cum11, rels[11] if rels else None)})
        pf.commit({"s_obs": (r.s_truth, model.rel_flow, model.flow_floor),
                   "i_obs": r.inv_gwh if "inv_gwh" in r else None})

        if k >= WARMUP_DAYS and day >= eval_from:
            for h, post in views.items():
                ps = post["S"]
                rows.append({
                    "terminal": name, "gas_day": day, "horizon": h,
                    "mean": ps.mean, "q05": ps.q05, "q25": ps.q25,
                    "q50": ps.q50, "q75": ps.q75, "q95": ps.q95,
                    "truth": r.s_truth, "persist": prev_s,
                    "nom": r.nom if "nom" in r else np.nan,
                })
        prev_s = r.s_truth
    return pd.DataFrame(rows)


def metrics(ev: pd.DataFrame) -> pd.DataFrame:
    """Point nowcast = posterior median (mean is biased at the idle-zero atom).
    Active-day columns restrict to truth > 5 GWh — where nowcasting matters.
    cov*_c columns are the adaptively conformalized bands (memo U4)."""
    ev = conformal.wrap_eval(ev)
    out = []
    for (t, h), g in ev.groupby(["terminal", "horizon"]):
        tr = g.truth.to_numpy(float)
        act = g[g.truth > 5]
        row = {
            "terminal": t, "horizon": h, "days": len(g), "active": len(act),
            "mae_filter": np.nanmean(np.abs(g.q50 - g.truth)),
            "mae_persist": np.nanmean(np.abs(g.persist - g.truth)),
            "cov50": coverage(tr, g.q25.to_numpy(float), g.q75.to_numpy(float)),
            "cov90": coverage(tr, g.q05.to_numpy(float), g.q95.to_numpy(float)),
            "cov50_c": coverage(tr, g.q25_c.to_numpy(float), g.q75_c.to_numpy(float)),
            "cov90_c": coverage(tr, g.q05_c.to_numpy(float), g.q95_c.to_numpy(float)),
            "act_mae_filter": np.nanmean(np.abs(act.q50 - act.truth)) if len(act) else np.nan,
            "act_mae_persist": np.nanmean(np.abs(act.persist - act.truth)) if len(act) else np.nan,
        }
        row["mae_nom"] = np.nanmean(np.abs(g.nom - g.truth)) if g.nom.notna().any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    frames = []
    for t, (p, rels) in eu_panels().items():
        horizons = ["H0", "H1", "H2"] if rels else ["H0"]  # no usable hourly -> H0 only
        frames.append(run_terminal(t, p, horizons, rels))
    for t, (p, rels) in uk_panels().items():
        frames.append(run_terminal(t, p, ["H0"], rels))
    ev = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out = config.RAW_DIR / EVAL_OUT
    ev.to_csv(out, index=False)
    print(f"\n{len(ev)} nowcast rows -> {out}\n")
    m = metrics(ev).sort_values(["terminal", "horizon"])
    pd.set_option("display.width", 200)
    print(m.round(2).to_string(index=False))
    print("\ncov50/cov90 targets: 0.50 / 0.90 — miscalibration is a finding, not a failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
