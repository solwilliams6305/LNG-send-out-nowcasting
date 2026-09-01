#!/usr/bin/env python3
"""Walk-forward evaluation of the HSMM Rao-Blackwellized filter (memo U2+U3),
on the identical panels, horizons, and scoring as the bootstrap reference —
so the two tables are directly comparable. 500 particles vs the BPF's 4000.
"""

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from lng_nowcast import config
from lng_nowcast.hsmm_rbpf import HsmmModel, RbParticleFilter

spec = importlib.util.spec_from_file_location("rn", ROOT / "scripts" / "run_nowcast.py")
rn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rn)


def run_terminal_rb(name: str, p: pd.DataFrame, horizons: list[str],
                    rels: dict | None = None,
                    overrides: dict | None = None) -> pd.DataFrame:
    p = p[p.s_truth.notna()].copy()
    if len(p) < 200:
        return pd.DataFrame()
    fit = p[p.index < rn.EVAL_START]
    model = HsmmModel.from_history(
        name, fit.s_truth.to_numpy(float),
        i_max=float(np.nanmax(p.inv_gwh) * 1.15 + 100) if p.inv_gwh.notna().any() else 4000.0,
        arrivals=fit.arrival_gwh.to_numpy(float) if "arrival_gwh" in fit else None,
        **(overrides or {}),
    )
    print(f"{name}: idle D~NB({model.idle_mean:.1f},{model.idle_var:.0f}) "
          f"active D~NB({model.active_mean:.1f},{model.active_var:.0f}) "
          f"sig=({model.sigma_tight:.0f},{model.sigma_broad:.0f}) p_b={model.p_broad:.2f} "
          f"restart~N({model.mu_restart:.0f},{model.sigma_restart:.0f})")

    i0 = float(p.inv_gwh.dropna().iloc[0]) if p.inv_gwh.notna().any() else model.i_max / 2
    pf = RbParticleFilter(model, i0=i0, s0=float(p.s_truth.iloc[0]))

    rows, prev_s = [], np.nan
    for k, (day, r) in enumerate(p.iterrows()):
        pf.propagate()
        views = {"H0": pf.posterior_s()}
        if "H1" in horizons:
            views["H1"] = pf.posterior_s({"cum_h": (4, r.cum4, rels[4] if rels else None)})
        if "H2" in horizons:
            views["H2"] = pf.posterior_s({"cum_h": (11, r.cum11, rels[11] if rels else None)})
        pf.commit({"s_obs": (r.s_truth, model.rel_flow, model.flow_floor),
                   "i_obs": r.inv_gwh if "inv_gwh" in r else None})
        if k >= rn.WARMUP_DAYS and day >= rn.EVAL_START:
            for h, ps in views.items():
                rows.append({"terminal": name, "gas_day": day, "horizon": h,
                             "mean": ps["mean"], "q05": ps["q05"], "q25": ps["q25"],
                             "q50": ps["q50"], "q75": ps["q75"], "q95": ps["q95"],
                             "p_idle": ps["p_idle"],
                             "truth": r.s_truth, "persist": prev_s,
                             "nom": r.nom if "nom" in r else np.nan})
        prev_s = r.s_truth
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    frames = []
    for t, (p, rels) in rn.eu_panels().items():
        horizons = ["H0", "H1", "H2"] if rels else ["H0"]
        frames.append(run_terminal_rb(t, p, horizons, rels))
    for t, (p, rels) in rn.uk_panels().items():
        frames.append(run_terminal_rb(t, p, ["H0"], rels))
    ev = pd.concat([f for f in frames if len(f)], ignore_index=True)
    ev.to_csv(config.RAW_DIR / "nowcast_eval_rb.csv", index=False)
    m = rn.metrics(ev).sort_values(["terminal", "horizon"])
    pd.set_option("display.width", 220)
    print(f"\nRBPF (500 particles), {time.time()-t0:.0f}s wall:")
    print(m.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
