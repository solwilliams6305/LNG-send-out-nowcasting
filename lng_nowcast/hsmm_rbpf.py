"""U2 + U3: explicit-duration regimes with a Rao-Blackwellized particle filter.

The two upgrades unify. Send-out has a discrete metastable structure —
idle (S = 0 exactly) and active — and *conditional on the discrete path*
everything continuous is linear-Gaussian, so the continuous states are
integrated out exactly with a per-particle Kalman filter.

Discrete path per particle (sampled):
  r_t ∈ {IDLE, ACTIVE}   regime;  a_t ≥ 1  age (days in regime)
  switch hazard  λ_r(a) = P(D = a | D ≥ a)  from a fitted Negative-Binomial
  sojourn law per regime — the explicit-duration (semi-Markov) part. Sojourn
  laws are first-passage-time distributions of the underlying decision
  process; their non-geometric shape is exactly why plain Markov switching
  (constant hazard) misfits long idle spells.
  z_t ∈ {tight, broad}   active-day innovation component,  P(broad) = p_b
  w_t ~ InvGamma(ν/2, ν/2)  latent scale making the innovation Student-t:
        η | z, w ~ N(0, σ_z² w)  ⇒  η ~ t_ν(0, σ_z)  marginally
  J_t ~ Bernoulli(λ_arr), size A_t ~ N⁺(μ_J, σ_J²)   cargo-arrival input

Continuous state x_t = (I_t, S_t)ᵀ, marginalized per particle:
  ACTIVE, staying:   S_t = S_{t-1} + η_t
  ACTIVE, entering:  S_t ~ N(μ_restart, σ_restart²)   (restart level)
  IDLE:              S_t = 0  (deterministic — the boundary state is exact,
                     which is how the truncation problem dissolves)
  always:            I_t = (1-b) I_{t-1} + A_t − S_t + ε_t,  ε ~ N(0, σ_I²)

Observations, all linear in x given the discrete path (R evaluated at the
predicted mean where heteroscedastic — the standard EKF-style approximation):
  s_obs  : H = [0, 1]
  cum_h  : H = [0, u + m_u]   with bridge variance (σ_P Ŝ √(u(1-u)) + 2)²
  i_obs  : H = [1, 0]

Particle weights multiply the *exact* Gaussian predictive likelihoods from
the Kalman recursion — the Rao-Blackwellization. Resampling duplicates
(m, P) sufficient statistics; no jitter is needed because particles
re-diversify through future discrete draws.

Posterior of S_t is a mixture: a point mass at 0 (idle particles) plus
Gaussians N(m_S, P_SS) (active particles); quantiles are computed from
weighted draws of that mixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IDLE, ACTIVE = 0, 1
MAX_AGE = 400  # hazard lookup length; beyond this the hazard is held constant


def _negbin_hazard_impl(mean: float, var: float, max_age: int = MAX_AGE) -> np.ndarray:
    """Discrete hazard λ(a) = P(D=a | D≥a), a = 1.., for a Negative-Binomial
    sojourn on {1,2,...} moment-matched to (mean, var); geometric (constant
    hazard) when var <= mean. NB(r,p) on {0,1,...}: mean r(1-p)/p, var r(1-p)/p²."""
    mean = max(mean, 1.05)
    m0 = mean - 1.0
    v0 = max(var, 1e-6)
    if v0 <= m0 + 1e-9 or m0 <= 0:
        return np.full(max_age, 1.0 / mean)
    p = m0 / v0
    r = m0 * p / (1 - p)
    from math import lgamma, log
    logpmf = np.empty(max_age)
    for k in range(max_age):
        logpmf[k] = (lgamma(k + r) - lgamma(r) - lgamma(k + 1)
                     + r * log(p) + k * log(1 - p))
    pmf = np.exp(logpmf - logpmf.max())
    pmf /= pmf.sum()  # renormalize over the truncated support
    surv = np.concatenate([[1.0], 1.0 - np.cumsum(pmf)[:-1]])
    haz = np.clip(pmf / np.maximum(surv, 1e-12), 1e-4, 0.9999)
    return haz


@dataclass
class HsmmModel:
    name: str
    # sojourn moments (fitted from spell lengths on the training window)
    idle_mean: float = 8.0
    idle_var: float = 120.0
    active_mean: float = 40.0
    active_var: float = 2500.0
    # active-regime innovation mixture (Student-t via latent scale)
    sigma_tight: float = 15.0
    sigma_broad: float = 120.0
    p_broad: float = 0.06
    nu: float = 4.0
    # restart level on idle -> active
    mu_restart: float = 150.0
    sigma_restart: float = 120.0
    # inventory
    boiloff: float = 0.0005
    sigma_i: float = 8.0
    i_max: float = 4000.0
    s_max: float = 800.0
    # arrivals
    lam_arrival: float = 0.25
    jump_mean: float = 1050.0
    jump_sd: float = 300.0
    # observation noise (identified from the cross-source studies; fixed)
    rel_flow: float = 0.004
    flow_floor: float = 1.0
    sigma_inv: float = 25.0
    idle_thresh: float = 2.0  # GWh/d below which a day counts as idle
    n_particles: int = 500
    seed: int = 11
    haz: dict = field(default_factory=dict)

    def __post_init__(self):
        self.haz[IDLE] = _negbin_hazard_impl(self.idle_mean, self.idle_var)
        self.haz[ACTIVE] = _negbin_hazard_impl(self.active_mean, self.active_var)

    # ------------------------------------------------------------------ fit
    @classmethod
    def from_history(cls, name: str, s: np.ndarray, i_max: float,
                     arrivals: np.ndarray | None = None, **overrides) -> "HsmmModel":
        m = cls(name=name, i_max=i_max)
        s = np.asarray(s, float)
        ok = ~np.isnan(s)
        idle = s < m.idle_thresh

        # spell lengths per regime
        spells = {IDLE: [], ACTIVE: []}
        run, cur = 0, None
        for k in range(len(s)):
            if not ok[k]:
                if cur is not None and run:
                    spells[cur].append(run)
                run, cur = 0, None
                continue
            reg = IDLE if idle[k] else ACTIVE
            if reg == cur:
                run += 1
            else:
                if cur is not None and run:
                    spells[cur].append(run)
                cur, run = reg, 1
        if cur is not None and run:
            spells[cur].append(run)
        if len(spells[IDLE]) >= 3:
            m.idle_mean = float(np.mean(spells[IDLE]))
            m.idle_var = float(max(np.var(spells[IDLE]), 1.0))
        if len(spells[ACTIVE]) >= 3:
            m.active_mean = float(np.mean(spells[ACTIVE]))
            m.active_var = float(max(np.var(spells[ACTIVE]), 1.0))

        # active-day innovations (both days active)
        act = (~idle[1:]) & (~idle[:-1]) & ok[1:] & ok[:-1]
        resid = (s[1:] - s[:-1])[act]
        if len(resid) > 50:
            mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            # Same compromise the bootstrap iteration learned: pure MAD
            # collapses on flat-running terminals and ordinary re-levelling
            # days then fall between the mixture components.
            m.sigma_tight = float(max(mad, 0.4 * np.std(resid), 3.0))
            m.p_broad = float(np.clip(np.mean(np.abs(resid) > 4 * max(mad, 1.0)), 0.02, 0.15))
            m.sigma_broad = float(max(np.quantile(np.abs(resid), 0.98), 4 * m.sigma_tight))

        # restart levels: first active day after an idle day
        restarts = s[1:][(idle[:-1]) & (~idle[1:]) & ok[1:] & ok[:-1]]
        if len(restarts) >= 5:
            m.mu_restart = float(np.median(restarts))
            m.sigma_restart = float(max(np.std(restarts), 30.0))

        m.s_max = float(np.nanmax(s) * 1.15 + 10)
        if arrivals is not None:
            arr = arrivals[~np.isnan(arrivals)]
            days = arr >= 100.0
            if days.sum() >= 10:
                m.lam_arrival = float(days.mean())
                m.jump_mean = float(arr[days].mean())
                m.jump_sd = float(max(arr[days].std(), 100.0))
        for k, v in overrides.items():
            setattr(m, k, v)
        m.__post_init__()
        return m


class RbParticleFilter:
    """Rao-Blackwellized HSMM filter. Same staged API as the bootstrap PF:
    propagate() → posterior(obs)/posterior() views → commit(obs)."""

    def __init__(self, model: HsmmModel, i0: float, s0: float):
        self.m = model
        self.rng = np.random.default_rng(model.seed)
        n = model.n_particles
        active0 = s0 >= model.idle_thresh
        self.r = np.full(n, ACTIVE if active0 else IDLE)
        self.a = np.ones(n, dtype=int)
        # Kalman sufficient statistics per particle over x = (I, S)
        self.mI = np.full(n, float(i0))
        self.mS = np.full(n, float(s0) if active0 else 0.0)
        self.PII = np.full(n, 60.0**2)
        self.PSS = np.full(n, (model.sigma_tight * 2.0) ** 2 if active0 else 0.0)
        self.PIS = np.zeros(n)
        self.w = np.full(n, 1.0 / n)

    # ------------------------------------------------------------ propagate
    def propagate(self) -> None:
        m, rng, n = self.m, self.rng, self.m.n_particles

        # regime transition via age-dependent hazard (the semi-Markov step)
        age_idx = np.minimum(self.a - 1, MAX_AGE - 1)
        lam = np.where(self.r == IDLE, m.haz[IDLE][age_idx], m.haz[ACTIVE][age_idx])
        switch = rng.random(n) < lam
        new_r = np.where(switch, 1 - self.r, self.r)
        self.a = np.where(switch, 1, self.a + 1)

        # arrival input, sampled per particle
        jump = rng.random(n) < m.lam_arrival
        A = np.where(jump, np.maximum(rng.normal(m.jump_mean, m.jump_sd, n), 0.0), 0.0)

        # innovation component + latent t-scale for active-staying particles
        z_broad = rng.random(n) < m.p_broad
        sig = np.where(z_broad, m.sigma_broad, m.sigma_tight)
        wscale = 1.0 / rng.gamma(m.nu / 2.0, 2.0 / m.nu, n)  # InvGamma(ν/2, ν/2)
        qS = sig**2 * wscale

        stay_active = (new_r == ACTIVE) & (self.r == ACTIVE)
        enter_active = (new_r == ACTIVE) & (self.r == IDLE)
        idle_now = new_r == IDLE
        d = 1.0 - m.boiloff

        mI, mS = self.mI.copy(), self.mS.copy()
        PII, PSS, PIS = self.PII.copy(), self.PSS.copy(), self.PIS.copy()

        # ACTIVE, staying: x_t = F x_{t-1} + [A,0]' + noise, F = [[d,-1],[0,1]]
        i = stay_active
        nPII = d*d*PII[i] - 2*d*PIS[i] + PSS[i] + qS[i] + m.sigma_i**2
        nPIS = d*PIS[i] - PSS[i] - qS[i]
        nPSS = PSS[i] + qS[i]
        nmI = d*mI[i] - mS[i] + A[i]
        self.mI[i], self.mS[i] = nmI, mS[i]
        self.PII[i], self.PIS[i], self.PSS[i] = nPII, nPIS, nPSS

        # ACTIVE, entering: S_t ~ N(μ_r, σ_r²) independent of the past
        i = enter_active
        self.mS[i] = m.mu_restart
        self.mI[i] = d*mI[i] + A[i] - m.mu_restart
        self.PII[i] = d*d*PII[i] + m.sigma_restart**2 + m.sigma_i**2
        self.PIS[i] = -m.sigma_restart**2
        self.PSS[i] = np.full(i.sum(), m.sigma_restart**2)

        # IDLE (staying or entering): S_t = 0 exactly
        i = idle_now
        self.mI[i] = d*mI[i] + A[i] - 0.0
        self.mS[i] = 0.0
        self.PII[i] = d*d*PII[i] + m.sigma_i**2
        self.PIS[i] = 0.0
        self.PSS[i] = 0.0

        self.r = new_r
        # clamp inventory mean into physical bounds (moment-matching lite)
        np.clip(self.mI, 0.0, m.i_max, out=self.mI)
        np.clip(self.mS, 0.0, m.s_max, out=self.mS)

    # ------------------------------------------------------- obs machinery
    def _obs_list(self, obs: dict) -> list[tuple[float, float, float, float]]:
        """Each observation -> (hI, hS, y, R) with R possibly Ŝ-dependent."""
        m = self.m
        out = []
        if not obs:
            return out
        if "s_obs" in obs:
            val, rel, floor = obs["s_obs"]
            if val is not None and not np.isnan(val):
                out.append((0.0, 1.0, float(val), None))  # R filled per-particle
                self._sobs_pars = (rel, floor)
        if "cum_h" in obs:
            h, cum, prof = obs["cum_h"]
            if h > 0 and cum is not None and not np.isnan(cum):
                u = h / 24.0
                m_u, sigma_p = prof if prof is not None else (0.0, 0.06)
                out.append(("cum", u + m_u, float(cum), (sigma_p, u)))
        if "i_obs" in obs:
            val = obs["i_obs"]
            if val is not None and not np.isnan(val):
                out.append((1.0, 0.0, float(val), m.sigma_inv**2))
        return out

    def _apply_obs(self, obs: dict, commit: bool):
        """Sequential Kalman updates. Returns (logl, mI, mS, PII, PSS, PIS) —
        the *updated* per-particle statistics. With commit=False the filter's
        own state is untouched, but a posterior view must still be built from
        the returned (updated) statistics: in a Rao-Blackwellized filter an
        observation moves every particle's Gaussian, not just its weight."""
        m = self.m
        mI, mS = self.mI, self.mS
        PII, PSS, PIS = self.PII, self.PSS, self.PIS
        if not commit:
            mI, mS = mI.copy(), mS.copy()
            PII, PSS, PIS = PII.copy(), PSS.copy(), PIS.copy()
        logl = np.zeros(m.n_particles)
        for spec in self._obs_list(obs):
            tag, hS, y, Rspec = spec
            if tag == "cum":
                sigma_p, u = Rspec
                R = (sigma_p * np.maximum(mS, 1.0) * np.sqrt(u * (1 - u)) + 2.0) ** 2
                hI = 0.0
            elif Rspec is None:  # s_obs with heteroscedastic R
                rel, floor = self._sobs_pars
                R = (rel * np.maximum(mS, 1.0) + floor) ** 2
                hI, hS = spec[0], spec[1]
            else:
                hI, hS, R = spec[0], spec[1], Rspec
            yhat = hI * mI + hS * mS
            PxH_I = hI * PII + hS * PIS      # Cov(x, Hx) components
            PxH_S = hI * PIS + hS * PSS
            Svar = hI * PxH_I + hS * PxH_S + R
            innov = y - yhat
            logl += -0.5 * (innov**2 / Svar + np.log(2 * np.pi * Svar))
            K_I, K_S = PxH_I / Svar, PxH_S / Svar
            mI = mI + K_I * innov
            mS = mS + K_S * innov
            PII = PII - K_I * PxH_I
            PIS = PIS - K_I * PxH_S
            PSS = PSS - K_S * PxH_S
            # idle particles have S pinned at 0 regardless of algebra
            idle = self.r == IDLE
            mS[idle] = 0.0
            PSS[idle] = 0.0
            PIS[idle] = 0.0
        if commit:
            self.mI, self.mS = mI, mS
            self.PII, self.PSS, self.PIS = PII, PSS, PIS
            np.clip(self.mI, 0.0, m.i_max, out=self.mI)
        return logl, mI, mS, PII, PSS, PIS

    # ------------------------------------------------------------- queries
    def posterior_s(self, obs: dict | None = None, draws_per: int = 4) -> dict:
        """Quantiles of S from the weighted Gaussian-mixture (+ idle atom),
        with the view's observations Kalman-assimilated into each particle."""
        if obs:
            logl, _, mS, _, PSS, _ = self._apply_obs(obs, commit=False)
        else:
            logl = np.zeros(self.m.n_particles)
            mS, PSS = self.mS, self.PSS
        logw = np.log(self.w + 1e-300) + logl
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        sd = np.sqrt(np.maximum(PSS, 0.0))
        draws = self.rng.normal(mS[None, :].repeat(draws_per, 0),
                                sd[None, :].repeat(draws_per, 0))
        draws = np.clip(draws, 0.0, self.m.s_max)
        draws[:, self.r == IDLE] = 0.0
        flat = draws.ravel()
        fw = np.tile(w, draws_per) / draws_per
        order = np.argsort(flat)
        cw = np.cumsum(fw[order])
        cw /= cw[-1]
        q = np.interp([0.05, 0.25, 0.5, 0.75, 0.95], cw, flat[order])
        return {"mean": float(np.sum(fw * flat)), "q05": q[0], "q25": q[1],
                "q50": q[2], "q75": q[3], "q95": q[4],
                "p_idle": float(np.sum(w[self.r == IDLE]))}

    def commit(self, obs: dict) -> None:
        logl, *_ = self._apply_obs(obs, commit=True)
        logw = np.log(self.w + 1e-300) + logl
        logw -= logw.max()
        self.w = np.exp(logw)
        self.w /= self.w.sum()
        n = self.m.n_particles
        if 1.0 / np.sum(self.w**2) < n / 2:
            pos = (self.rng.random() + np.arange(n)) / n
            idx = np.clip(np.searchsorted(np.cumsum(self.w), pos), 0, n - 1)
            for arr in ("r", "a", "mI", "mS", "PII", "PSS", "PIS"):
                setattr(self, arr, getattr(self, arr)[idx].copy())
            self.w = np.full(n, 1.0 / n)
