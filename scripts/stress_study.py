#!/usr/bin/env python3
"""The days-of-cover stress study: does physical LNG cover predict system
stress beyond the market's standard dial (storage fullness %)?

Index (daily, NW Europe = EU core four + UK three):
  T_t     total LNG tank energy (ALSI inv_gwh + NG stocks), GWh
  B_t     total send-out, 7-day smoothed, GWh/d
  cover_t = T_t / B_t              days of tank cover for the LNG system
  arr30_t trailing 30-day mean arrival energy, GWh/d (free-data pipeline proxy;
          the true forward ship pipeline starts accumulating with the AIS
          chokepoint layer and cannot be backfilled)

Targets over the NEXT k days (k = 7, 14):
  sap_spike     max daily UK SAP log-return ≥ its q90
  linepack_low  min NTS closing linepack ≤ its q10
  ttf_spike     max daily TTF front log-return ≥ its q90
Thresholds are full-sample quantiles — disclosed as in-sample event
definitions; predictors at t use information through t only.

Scoring: rank AUC per predictor; logistic (Newton) for storage-only vs
storage+cover — ΔAUC is the "beats the incumbent" number. Caveat printed:
overlapping windows ⇒ effective N ≈ N/k; treat significance informally.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lng_nowcast import arrivals, config

K_HORIZONS = (7, 14)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(score) | np.isnan(y))
    s, t = score[m], y[m].astype(bool)
    if t.sum() < 10 or (~t).sum() < 10:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[t].mean() - (t.sum() + 1) / 2) / (~t).sum())


def logit_fit(X: np.ndarray, y: np.ndarray, iters: int = 50):
    m = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    Xm = np.column_stack([np.ones(m.sum()), X[m]])
    ym = y[m]
    b = np.zeros(Xm.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xm @ b))
        W = p * (1 - p) + 1e-9
        try:
            b += np.linalg.solve(Xm.T @ (Xm * W[:, None]), Xm.T @ (ym - p))
        except np.linalg.LinAlgError:
            break
    p_all = np.full(len(y), np.nan)
    p_all[m] = 1 / (1 + np.exp(-(Xm @ b)))
    return p_all


def build_panel() -> pd.DataFrame:
    al = arrivals.eu_daily_panel()
    eu_core = al[al.terminal.isin(["gate", "eems", "zeebrugge", "dunkerque"])]
    tank_eu = eu_core.groupby("gas_day").inv_gwh.sum()
    burn_eu = eu_core.groupby("gas_day").send_out.sum()

    ng = pd.read_csv(config.RAW_DIR / "nationalgas_daily.csv")
    ng["gwh"] = pd.to_numeric(ng.value, errors="coerce") / 1e6
    tank_uk = ng[ng.metric == "stock"].groupby("gas_day").gwh.sum()
    burn_uk = ng[(ng.metric == "send_out") & (ng.maturity == "D+1")].groupby("gas_day").gwh.sum()

    arr = pd.concat([arrivals.uk_daily_inflow().set_index(["terminal", "gas_day"]).arrival_gwh,
                     al.set_index(["terminal", "gas_day"]).arrival_gwh])
    arr_total = arr.groupby("gas_day").sum()

    si = pd.read_csv(config.RAW_DIR / "stress_inputs.csv")
    si["value"] = pd.to_numeric(si.value, errors="coerce")
    wide = si.pivot_table(index="date", columns="series", values="value")

    px = pd.read_csv(config.RAW_DIR / "prices.csv")
    px["value"] = pd.to_numeric(px.value, errors="coerce")
    sap = px[px.series == "uk_sap_p_kwh"].set_index("date").value
    ttf = px[px.series == "ttf_front_eur_mwh"].set_index("date").value

    df = pd.DataFrame({
        "tank": tank_eu.add(tank_uk, fill_value=np.nan),
        "burn": burn_eu.add(burn_uk, fill_value=np.nan),
        "arr": arr_total,
    })
    df = df.join(wide).join(sap.rename("sap")).join(ttf.rename("ttf"))
    df = df.sort_index()

    df["burn7"] = df.burn.rolling(7, min_periods=4).mean()
    df["cover"] = df.tank / df.burn7.clip(lower=50)
    df["cover_d14"] = df.cover.diff(14)
    df["arr30"] = df.arr.rolling(30, min_periods=15).mean()
    df["storage"] = df.agsi_eu_full_pct
    df["storage_d14"] = df.storage.diff(14)
    df["lp_close"] = df.uk_linepack_close

    df["sap_ret"] = np.log(df.sap).diff() * 100
    df["ttf_ret"] = np.log(df.ttf).diff() * 100
    return df


def targets(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Rare forward events: an extreme daily jump (q97.5) within the next k
    days, or linepack sinking to its q05. Each target masks its own NaNs."""
    out = pd.DataFrame(index=df.index)
    mp = max(2, k // 2)  # tolerate non-trading-day gaps inside the window
    specs = {
        "sap_spike": (df.sap_ret.shift(-1).rolling(k, min_periods=mp).max().shift(-(k - 1)),
                      df.sap_ret.quantile(0.975), True),
        "ttf_spike": (df.ttf_ret.shift(-1).rolling(k, min_periods=mp).max().shift(-(k - 1)),
                      df.ttf_ret.quantile(0.975), True),
        "linepack_low": (df.lp_close.shift(-1).rolling(k, min_periods=mp).min().shift(-(k - 1)),
                         df.lp_close.quantile(0.05), False),
    }
    for name, (fwd, thr, upper) in specs.items():
        y = (fwd >= thr) if upper else (fwd <= thr)
        y = y.astype(float)
        y[fwd.isna()] = np.nan
        out[name] = y
    return out


def main() -> int:
    df = build_panel()
    n_ok = df.cover.notna().sum()
    print(f"panel: {len(df)} days, cover defined on {n_ok}; "
          f"cover now {df.cover.dropna().iloc[-1]:.1f} d "
          f"(median {df.cover.median():.1f}), storage {df.storage.dropna().iloc[-1]:.1f}%\n")

    preds = {"−cover": -df.cover, "−Δ14cover": -df.cover_d14, "−arr30": -df.arr30,
             "−storage%": -df.storage, "−Δ14storage": -df.storage_d14}
    rows = []
    for k in K_HORIZONS:
        tg = targets(df, k)
        for tname in tg.columns:
            y = tg[tname].to_numpy(float)
            row = {"target": tname, "k": k, "events": int(np.nansum(y))}
            for pname, s in preds.items():
                row[f"AUC {pname}"] = auc(s.to_numpy(float), y)
            Xs = np.column_stack([-df.storage.to_numpy(float)])
            Xc = np.column_stack([-df.storage.to_numpy(float), -df.cover.to_numpy(float),
                                  -df.cover_d14.to_numpy(float)])
            row["AUC storage-only"] = auc(logit_fit(Xs, y), y)
            row["AUC + cover"] = auc(logit_fit(Xc, y), y)
            rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print(res.round(3).to_string(index=False))
    res.to_csv(config.RAW_DIR / "stress_study.csv", index=False)
    print("\nCaveats: in-sample event thresholds; overlapping windows (eff. N ≈ N/k); "
          "two winters of data; logistic fitted in-sample (no walk-forward yet) — "
          "treat ΔAUC as descriptive until the out-of-sample rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
