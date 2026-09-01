#!/usr/bin/env python3
"""Feed PGAS posterior means back into the RBPF and re-evaluate (leak-free:
PGAS runs on the pre-evaluation window only, exactly like the moment fits it
replaces). Also reports parameter stability: fit-window vs full-history
posteriors for one baseload and one flickering terminal.

nu is set to 50 in the refit because PGAS fitted a two-component *Gaussian*
mixture — feeding its scales into heavy-t innovations would double-count
tail mass.
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
from lng_nowcast.hsmm_rbpf import HsmmModel
from lng_nowcast.pgas import Theta, run_pgas


def load(modname: str):
    spec = importlib.util.spec_from_file_location(modname, ROOT / "scripts" / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fp = load("fit_pgas")
rb = load("run_nowcast_rb")


def pgas_overrides(terminal: str, y: np.ndarray, sweeps: int = 300) -> dict:
    m = HsmmModel.from_history(terminal, y, i_max=4000)
    th0 = Theta(m.idle_mean, m.idle_var, m.active_mean, m.active_var,
                m.sigma_tight, m.sigma_broad, m.p_broad, m.mu_restart, m.sigma_restart)
    chain = run_pgas(y, th0, sweeps=sweeps, n_part=60)
    burn = chain[sweeps // 3:]

    def pm(k):
        return float(np.mean([getattr(t, k) for t in burn]))

    return {"idle_mean": pm("idle_mean"), "idle_var": pm("idle_var"),
            "active_mean": pm("active_mean"), "active_var": pm("active_var"),
            "sigma_tight": pm("sig_tight"), "sigma_broad": pm("sig_broad"),
            "p_broad": pm("p_broad"), "mu_restart": pm("mu_r"),
            "sigma_restart": pm("sig_r"), "nu": 50.0}


def main() -> int:
    terminals = ["gate", "zeebrugge", "dunkerque", "eems", "grain", "south_hook"]
    overrides = {}
    for t in terminals:
        y = fp.series(t)
        t0 = time.time()
        overrides[t] = pgas_overrides(t, y)
        o = overrides[t]
        print(f"{t}: PGAS in {time.time()-t0:.0f}s -> sig=({o['sigma_tight']:.1f},"
              f"{o['sigma_broad']:.0f}) p_b={o['p_broad']:.2f} "
              f"restart~N({o['mu_restart']:.0f},{o['sigma_restart']:.0f}) "
              f"idle_mean={o['idle_mean']:.1f}")

    frames = []
    for t, (p, rels) in rb.rn.eu_panels().items():
        horizons = ["H0", "H1", "H2"] if rels else ["H0"]
        frames.append(rb.run_terminal_rb(t, p, horizons, rels, overrides=overrides.get(t)))
    for t, (p, rels) in rb.rn.uk_panels().items():
        frames.append(rb.run_terminal_rb(t, p, ["H0"], rels, overrides=overrides.get(t)))
    ev = pd.concat([f for f in frames if len(f)], ignore_index=True)
    ev.to_csv(config.RAW_DIR / "nowcast_eval_rb_pgas.csv", index=False)
    m = rb.rn.metrics(ev).sort_values(["terminal", "horizon"])
    pd.set_option("display.width", 220)
    print("\nRBPF with PGAS-fitted parameters:")
    print(m.round(2).to_string(index=False))

    print("\n=== parameter stability: fit-window (152 d) vs full history (~970 d) ===")
    for t in ("gate", "grain"):
        t0 = time.time()
        full = pgas_overrides(t, fp.series(t, end="2026-09-01"))
        o = overrides[t]
        print(f"{t} ({time.time()-t0:.0f}s): "
              f"sig_tight {o['sigma_tight']:.1f} → {full['sigma_tight']:.1f} | "
              f"sig_broad {o['sigma_broad']:.0f} → {full['sigma_broad']:.0f} | "
              f"p_b {o['p_broad']:.2f} → {full['p_broad']:.2f} | "
              f"restart {o['mu_restart']:.0f} → {full['mu_restart']:.0f} | "
              f"idle_mean {o['idle_mean']:.1f} → {full['idle_mean']:.1f} | "
              f"active_mean {o['active_mean']:.0f} → {full['active_mean']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
