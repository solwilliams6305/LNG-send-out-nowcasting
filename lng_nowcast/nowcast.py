"""Reference state-space nowcaster: daily particle filter over (I_t, S_t).

*** OWNERSHIP NOTE ***
This is deliberately a REFERENCE implementation: transparent choices, no
black-box fitting. The model decisions — the send-out prior, the tail index,
the status-dependent observation noise, the intraday profile — are the
project's mathematical content and are meant to be reworked, defended, and
extended by Solomon. Every default below cites where its number came from.

Model (per terminal, gas-day resolution, energy units GWh):

  state   x_t = (I_t, S_t)   inventory at end of day t; send-out during t
  input   A_t                arrival energy discharged during t (phase 1:
                             observed series — NG inflows / ALSI-implied)

  send-out prior (a decision variable, not physics):
      m_t   = phi * nom_t + (1 - phi) * S_{t-1}     (phi -> 0 if no nomination)
      S_t   = clip(m_t + sigma_S * t_nu, 0, s_max)
      Student-t innovations (nu ~ 4): outages/restarts are the fat tail — the
      revision studies showed exactly this Gaussian-core + rare-break shape.

  inventory:
      I_t = clip(I_{t-1} + A_t - S_t - b*I_{t-1} + eps, 0, i_max)
      b ~ 0.0005/day tank boil-off (mostly recycled into send-out; small).
      The clip is informative: send-out can't exceed tank contents, and a
      full tank forces send-out — the physical coupling the filter exploits.

  observations (whichever the information set contains that day):
      cum_h : cumulative metered flow through gas-hour h (ENTSOG hourly,
              ~2 h publication lag). Flat intraday profile assumed:
              cum_h ~ N(S_t * h/24, (sigma_rel_cum * S_t * h/24 + 2)^2)
              [LNG send-out is contractually flat day-scale; refine later
               from the hourly archive itself.]
      S_obs : a published daily figure ~ N(S_t, (rel * S_t + floor)^2)
              rel per source from the cross-validation study
              (ENTSOG vs NG: 0.2-0.4 %; ALSI similar).
      I_obs : published inventory ~ N(I_t, sigma_inv^2), sigma_inv from the
              ALSI E-vs-C and stock-correction statistics.

  filter: bootstrap PF, systematic resampling at ESS < N/2, small jitter on
  resampled S (particle impoverishment guard on a 2-d state).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TerminalModel:
    name: str
    phi: float = 0.7           # weight on nomination vs persistence
    sigma_s: float = 20.0      # GWh, send-out innovation scale
    nu: float = 4.0            # Student-t dof for send-out innovations
    boiloff: float = 0.0005    # per-day fraction of inventory
    sigma_i_proc: float = 5.0  # GWh, inventory process noise
    i_max: float = 4000.0      # GWh tank capacity
    s_max: float = 800.0       # GWh/d send-out ceiling
    rel_cum: float = 0.02      # relative noise on cumulative intraday flow
    rel_flow: float = 0.004    # relative noise on published daily flows
    flow_floor: float = 1.0    # GWh additive floor on daily-flow noise
    sigma_inv: float = 25.0    # GWh noise on published inventory
    jitter_s: float = 3.0      # GWh post-resample jitter on S
    lam_arrival: float = 0.25  # P(cargo day) — from the arrivals dataset
    jump_mean: float = 1050.0  # GWh — calibrated mode of the event-size study
    jump_sd: float = 300.0
    # Regime-switching proposal: send-out is a decision variable that mostly
    # drifts (tight AR core) but occasionally re-levels by hundreds of GWh
    # (ramp orders, trips, restarts). Without the broad component the proposal
    # has no particles near a re-levelled truth and the filter lags for days —
    # the first UK evaluation demonstrated exactly this failure.
    p_regime: float = 0.06
    sigma_broad: float = 150.0
    n_particles: int = 4000
    seed: int = 7

    @classmethod
    def from_history(cls, name: str, s: np.ndarray, nom: np.ndarray,
                     i_max: float, arrivals: np.ndarray | None = None,
                     **overrides) -> "TerminalModel":
        """Set the send-out innovation scale from realized one-step residuals
        of the blend forecast, and the jump prior from the arrivals dataset —
        moment fits, kept deliberately inspectable."""
        m = cls(name=name, i_max=i_max)
        if arrivals is not None:
            arr = arrivals[~np.isnan(arrivals)]
            days = arr >= 100.0
            if days.sum() >= 10:
                m.lam_arrival = float(days.mean())
                m.jump_mean = float(arr[days].mean())
                m.jump_sd = float(max(arr[days].std(), 100.0))
        prev = s[:-1]
        cur = s[1:]
        nom_cur = nom[1:]
        have_nom = ~np.isnan(nom_cur) & ~np.isnan(cur) & ~np.isnan(prev)
        # Nominations earn their weight: ENTSOG LNG-point nominations proved
        # empirically WORSE than persistence at several terminals (and ~70 %
        # of archived values are post-renomination) — use them only where the
        # fit window shows they help.
        if have_nom.sum() > 100:
            mae_nom = np.nanmean(np.abs(nom_cur[have_nom] - cur[have_nom]))
            mae_per = np.nanmean(np.abs(prev[have_nom] - cur[have_nom]))
            m.phi = 0.5 if mae_nom < 0.9 * mae_per else 0.0
        else:
            m.phi = 0.0
        blend = np.where(~np.isnan(nom_cur) & (m.phi > 0),
                         m.phi * nom_cur + (1 - m.phi) * prev, prev)
        resid = cur - blend
        resid = resid[~np.isnan(resid)]
        if len(resid) > 50:
            mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            # Core scale between the robust and raw spread: pure MAD collapses
            # on flat-running terminals and cannot track re-levelling days.
            m.sigma_s = float(max(mad, 0.4 * np.std(resid), 3.0))
            m.p_regime = float(np.clip(np.mean(np.abs(resid) > 4 * max(mad, 1.0)), 0.02, 0.15))
            m.sigma_broad = float(max(np.quantile(np.abs(resid), 0.98), 4 * m.sigma_s))
        m.s_max = float(np.nanmax(s) * 1.15 + 10)
        m.jitter_s = max(m.sigma_s * 0.15, 1.0)
        for k, v in overrides.items():
            setattr(m, k, v)
        return m


@dataclass
class Posterior:
    mean: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float

    @classmethod
    def of(cls, x: np.ndarray, w: np.ndarray) -> "Posterior":
        order = np.argsort(x)
        cw = np.cumsum(w[order])
        cw /= cw[-1]
        q = np.interp([0.05, 0.25, 0.5, 0.75, 0.95], cw, x[order])
        return cls(float(np.sum(w * x)), *map(float, q))


class ParticleFilter:
    """Usage per gas day (staged assimilation — nowcasts are *views* at
    growing information sets; only commit() updates the state):

        pf.propagate(nom)                       # arrivals sampled from prior
        h0 = pf.posterior()                     # day-ahead nowcast
        h1 = pf.posterior({"cum_h": (4, c4)})   # midday view (EU)
        h2 = pf.posterior({"cum_h": (11, c11)}) # evening view (EU)
        pf.commit({"s_obs": (...), "i_obs": ...})  # end-of-day truth intake
    """

    def __init__(self, model: TerminalModel, i0: float, s0: float):
        self.m = model
        self.rng = np.random.default_rng(model.seed)
        n = model.n_particles
        self.I = np.clip(i0 + self.rng.normal(0, 50, n), 0, model.i_max)
        self.S = np.clip(s0 + self.rng.normal(0, model.sigma_s, n), 0, model.s_max)
        self.w = np.full(n, 1.0 / n)

    def propagate(self, nom: float | None) -> None:
        """One gas-day transition. Arrivals are drawn per particle from the
        calibrated jump prior — the observed arrival series is *not* used
        here, because it is derived from publications that appear only after
        the day being nowcast (it enters via commit()'s i_obs instead)."""
        m, rng, n = self.m, self.rng, self.m.n_particles
        if nom is not None and not np.isnan(nom):
            mean = m.phi * nom + (1 - m.phi) * self.S
        else:
            mean = self.S
        broad = rng.random(n) < m.p_regime
        scale = np.where(broad, m.sigma_broad, m.sigma_s)
        prop = np.clip(mean + rng.standard_t(m.nu, n) * scale, 0, m.s_max)
        # Idle is a sticky discrete state: a shut-in terminal most likely stays
        # at exactly zero; restarts come through the broad regime component.
        idle = self.S < 2.0
        stay_idle = idle & (rng.random(n) < 0.85)
        self.S = np.where(stay_idle, 0.0, prop)

        jump = rng.random(n) < m.lam_arrival
        size = np.clip(rng.normal(m.jump_mean, m.jump_sd, n), 0, None)
        a = np.where(jump, size, 0.0)
        eps = rng.normal(0, m.sigma_i_proc, n)
        self.I = np.clip(self.I + a - self.S - m.boiloff * self.I + eps, 0, m.i_max)

    def _dlogw(self, obs: dict) -> np.ndarray:
        m = self.m
        d = np.zeros(m.n_particles)
        if "cum_h" in obs:
            h, cum, rel = obs["cum_h"] if len(obs["cum_h"]) == 3 else (*obs["cum_h"], None)
            if h > 0 and cum is not None and not np.isnan(cum):
                pred = self.S * h / 24.0
                r = m.rel_cum if rel is None else rel
                sd = r * np.maximum(pred, 1.0) + 2.0
                d += -0.5 * ((cum - pred) / sd) ** 2 - np.log(sd)
        if "s_obs" in obs:
            val, rel, floor = obs["s_obs"]
            if val is not None and not np.isnan(val):
                sd = rel * np.maximum(self.S, 1.0) + floor
                d += -0.5 * ((val - self.S) / sd) ** 2 - np.log(sd)
        if "i_obs" in obs:
            val = obs["i_obs"]
            if val is not None and not np.isnan(val):
                d += -0.5 * ((val - self.I) / m.sigma_inv) ** 2
        return d

    def _weights_with(self, obs: dict | None) -> np.ndarray:
        logw = np.log(self.w + 1e-300)
        if obs:
            logw = logw + self._dlogw(obs)
        logw -= logw.max()
        w = np.exp(logw)
        return w / w.sum()

    def posterior(self, obs: dict | None = None) -> dict[str, Posterior]:
        """Nowcast under the current information set — does NOT mutate state."""
        w = self._weights_with(obs)
        return {"S": Posterior.of(self.S, w), "I": Posterior.of(self.I, w)}

    def commit(self, obs: dict) -> None:
        """Assimilate the day's final observations and resample if depleted."""
        self.w = self._weights_with(obs)
        ess = 1.0 / np.sum(self.w**2)
        if ess >= self.m.n_particles / 2:
            return
        n = self.m.n_particles
        positions = (self.rng.random() + np.arange(n)) / n
        idx = np.searchsorted(np.cumsum(self.w), positions)
        idx = np.clip(idx, 0, n - 1)
        self.I, self.S = self.I[idx], self.S[idx]
        self.S = np.clip(self.S + self.rng.normal(0, self.m.jitter_s, n), 0, self.m.s_max)
        self.w = np.full(n, 1.0 / n)


def coverage(truth: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    ok = (truth >= lo) & (truth <= hi)
    return float(np.mean(ok[~np.isnan(truth)]))
