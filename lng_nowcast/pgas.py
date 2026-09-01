"""U5: Bayesian parameter learning by Particle Gibbs with Ancestor Sampling.

Reference-scope target: the semi-Markov switching model of daily send-out
(the U2 structure), with the published daily figure as a tightly-observed
emission. Latent trajectory per day:  x_t = (r_t, a_t, z_t)
  r_t regime (IDLE/ACTIVE), a_t age, hazards from NB sojourn laws;
  z_t ∈ {tight, broad} innovation component on active-staying days.
Emissions given the observed series y:
  IDLE            y_t ~ N(0, s_eps²)                (s_eps small, fixed)
  ACTIVE entering y_t ~ N(mu_r, sig_r²)             (restart level)
  ACTIVE staying  y_t ~ N(y_{t-1}, sig_{z_t}²)      (switching regression)

Gibbs sweep (Lindsten–Jordan–Schön 2014):
  1. x_{1:T} | θ, y  by conditional SMC with ancestor sampling — the
     reference trajectory is retained as particle 0 and, at every step, its
     ancestor is resampled with weights  w_{t-1}^{(i)} · f(x_t^ref | x_{t-1}^{(i)}).
     Because staying-active emissions condition on the *observed* y_{t-1},
     the emission factor is identical across candidate ancestors and drops
     out: the AS weight is the discrete transition probability alone.
  2. θ | x_{1:T}, y  by conjugate/MH blocks:
     sig_tight², sig_broad² ~ inverse-gamma posteriors from their residuals;
     p_broad ~ Beta; (mu_r, sig_r²) ~ Normal–inverse-gamma from restarts;
     NB sojourn moments by random-walk MH on log(mean-1), log(var) against
     the completed-spell likelihood.

Extending x_t with latent (S_t, I_t) — intraday settings, revision-era data —
is the same machinery with the U3 Kalman inside (Rao-Blackwellized particle
Gibbs); this module is the demonstrable core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hsmm_rbpf import ACTIVE, IDLE, MAX_AGE, _negbin_hazard_impl


@dataclass
class Theta:
    idle_mean: float
    idle_var: float
    active_mean: float
    active_var: float
    sig_tight: float
    sig_broad: float
    p_broad: float
    mu_r: float
    sig_r: float

    def hazards(self):
        return {IDLE: _negbin_hazard_impl(self.idle_mean, self.idle_var),
                ACTIVE: _negbin_hazard_impl(self.active_mean, self.active_var)}


S_EPS = 1.5  # GWh: how close to zero an idle day's published figure sits


def _emission_logpdf(y_t, y_prev, r, entering, z, th: Theta):
    if r == IDLE:
        s = S_EPS
        mu = 0.0
    elif entering:
        s, mu = th.sig_r, th.mu_r
    else:
        s = th.sig_broad if z == 1 else th.sig_tight
        mu = y_prev
    return -0.5 * ((y_t - mu) / s) ** 2 - np.log(s)


def csmc_as(y: np.ndarray, th: Theta, ref: dict | None, n_part: int = 60,
            rng: np.random.Generator | None = None) -> dict:
    """One conditional-SMC-with-ancestor-sampling pass; returns a sampled
    trajectory {r, a, z, entering}. With ref=None runs plain SMC (init)."""
    rng = rng or np.random.default_rng()
    T = len(y)
    haz = th.hazards()
    R = np.empty((T, n_part), dtype=np.int8)
    A_ = np.empty((T, n_part), dtype=np.int32)
    Z = np.zeros((T, n_part), dtype=np.int8)
    E = np.zeros((T, n_part), dtype=bool)
    anc = np.empty((T, n_part), dtype=np.int32)
    logw = np.zeros(n_part)

    # t = 0: regime from the first observation, age 1
    r0 = IDLE if abs(y[0]) < 2.0 else ACTIVE
    R[0] = r0
    A_[0] = 1
    E[0] = r0 == ACTIVE
    if ref is not None:
        R[0, 0], A_[0, 0], Z[0, 0], E[0, 0] = ref["r"][0], ref["a"][0], ref["z"][0], ref["e"][0]

    for t in range(1, T):
        w = np.exp(logw - logw.max())
        w /= w.sum()
        # multinomial ancestors for particles 1..N-1
        anc_t = rng.choice(n_part, size=n_part, p=w)
        if ref is not None:
            # ancestor-sample the reference's ancestor (index 0 slot)
            lam_prev = np.array([haz[R[t - 1, i]][min(A_[t - 1, i] - 1, MAX_AGE - 1)]
                                 for i in range(n_part)])
            rr, aa, zz = ref["r"][t], ref["a"][t], ref["z"][t]
            if aa > 1:  # ref stayed: ancestor must match regime & age
                match = (R[t - 1] == rr) & (A_[t - 1] == aa - 1)
                trans = np.where(match, 1 - lam_prev, 0.0)
            else:       # ref switched: ancestor must be the other regime
                trans = np.where(R[t - 1] == 1 - rr, lam_prev, 0.0)
            if rr == ACTIVE and aa > 1:
                trans = trans * (th.p_broad if zz == 1 else 1 - th.p_broad)
            asw = w * trans
            anc_t[0] = rng.choice(n_part, p=asw / asw.sum()) if asw.sum() > 0 else int(np.argmax(w))
        anc[t] = anc_t

        r_prev = R[t - 1, anc_t]
        a_prev = A_[t - 1, anc_t]
        lam = np.array([haz[r_prev[i]][min(a_prev[i] - 1, MAX_AGE - 1)] for i in range(n_part)])
        switch = rng.random(n_part) < lam
        r_new = np.where(switch, 1 - r_prev, r_prev)
        a_new = np.where(switch, 1, a_prev + 1)
        z_new = (rng.random(n_part) < th.p_broad).astype(np.int8)
        e_new = switch & (r_new == ACTIVE)
        if ref is not None:
            r_new[0], a_new[0], z_new[0], e_new[0] = ref["r"][t], ref["a"][t], ref["z"][t], ref["e"][t]
        R[t], A_[t], Z[t], E[t] = r_new, a_new, z_new, e_new

        logw = np.array([_emission_logpdf(y[t], y[t - 1], R[t, i], E[t, i], Z[t, i], th)
                         for i in range(n_part)])

    # backward trace of one trajectory
    w = np.exp(logw - logw.max())
    w /= w.sum()
    k = int(rng.choice(n_part, p=w))
    out = {"r": np.empty(T, np.int8), "a": np.empty(T, np.int32),
           "z": np.empty(T, np.int8), "e": np.empty(T, bool)}
    for t in range(T - 1, -1, -1):
        out["r"][t], out["a"][t] = R[t, k], A_[t, k]
        out["z"][t], out["e"][t] = Z[t, k], E[t, k]
        k = anc[t, k] if t > 0 else k
    return out


def _spells(r: np.ndarray) -> dict:
    sp = {IDLE: [], ACTIVE: []}
    run, cur = 0, None
    for x in r:
        if x == cur:
            run += 1
        else:
            if cur is not None:
                sp[cur].append(run)
            cur, run = int(x), 1
    if cur is not None:
        sp[cur].append(run)
    return sp


def _nb_loglik(spells: list[int], mean: float, var: float) -> float:
    if not spells:
        return 0.0
    haz = _negbin_hazard_impl(mean, var)
    lam = np.clip(haz, 1e-6, 1 - 1e-6)
    logsurv = np.cumsum(np.log(1 - lam))
    ll = 0.0
    for d in spells:
        d = min(d, MAX_AGE - 1)
        ll += np.log(lam[d - 1]) + (logsurv[d - 2] if d >= 2 else 0.0)
    return float(ll)


def gibbs_theta(traj: dict, y: np.ndarray, th: Theta, rng) -> Theta:
    r, z, e = traj["r"], traj["z"], traj["e"]
    stay = (r == ACTIVE) & ~e
    stay[0] = False
    resid = y[stay] - y[np.flatnonzero(stay) - 1]
    broad = z[stay] == 1

    def ig_draw(res, prior_a=2.0, prior_b=200.0):
        n = len(res)
        a = prior_a + n / 2
        b = prior_b + 0.5 * float(np.sum(res**2))
        return float(np.sqrt(b / rng.gamma(a)))

    sig_t = ig_draw(resid[~broad]) if (~broad).sum() > 2 else th.sig_tight
    sig_b = ig_draw(resid[broad], prior_b=5e4) if broad.sum() > 2 else th.sig_broad
    if sig_b < 2 * sig_t:  # keep the components identified
        sig_b = 2 * sig_t
    p_b = float(rng.beta(1 + broad.sum(), 9 + (~broad).sum()))

    restarts = y[e]
    if len(restarts) > 2:
        mu_r = float(rng.normal(restarts.mean(), restarts.std() / np.sqrt(len(restarts)) + 1e-3))
        sig_r = ig_draw(restarts - mu_r, prior_b=1e4)
    else:
        mu_r, sig_r = th.mu_r, th.sig_r

    sp = _spells(r)
    new = Theta(th.idle_mean, th.idle_var, th.active_mean, th.active_var,
                sig_t, sig_b, p_b, mu_r, max(sig_r, 20.0))
    # RW-MH on sojourn moments per regime
    for reg, mkey, vkey in ((IDLE, "idle_mean", "idle_var"),
                            (ACTIVE, "active_mean", "active_var")):
        mean0, var0 = getattr(new, mkey), getattr(new, vkey)
        cur_ll = _nb_loglik(sp[reg], mean0, var0)
        for _ in range(5):
            mp = float(np.exp(np.log(mean0 - 1 + 1e-3) + rng.normal(0, 0.15)) + 1)
            vp = float(np.exp(np.log(var0) + rng.normal(0, 0.25)))
            prop_ll = _nb_loglik(sp[reg], mp, vp)
            if np.log(rng.random()) < prop_ll - cur_ll:
                mean0, var0, cur_ll = mp, vp, prop_ll
        setattr(new, mkey, mean0)
        setattr(new, vkey, var0)
    return new


def run_pgas(y: np.ndarray, th0: Theta, sweeps: int = 300, n_part: int = 60,
             seed: int = 3) -> list[Theta]:
    rng = np.random.default_rng(seed)
    th = th0
    traj = csmc_as(y, th, ref=None, n_part=n_part, rng=rng)
    chain = []
    for _ in range(sweeps):
        traj = csmc_as(y, th, ref=traj, n_part=n_part, rng=rng)
        th = gibbs_theta(traj, y, th, rng)
        chain.append(th)
    return chain
