"""Support code for the cost-of-linearization walkthrough.

The production posterior in this project runs through *exact* engines — particle/SMC over
the true emission, Euler–Maruyama over the true nonlinear drift — because a linearized
Gaussian surrogate silently biases any *reported* result (the linearization-is-init-only
policy in AGENTS.md). But before you pay for that expensive machinery, a cheaper question
is worth asking: *for this model, in this regime, how much would a linear/Gaussian
approximation actually cost me?* This module prices that, using only simulation and
O(T) Gaussian filtering — it never fits the nonlinear model.

The yardstick is the **simulated truth**. We treat a toy nonlinear state-space model as
ground truth, generate data from it, fit a *relaxed* candidate with the cheapest valid
Gaussian filter, and check where the known simulated truth lands inside the candidate's
distributions. If the candidate is faithful the truth lands uniformly (calibrated); the
*way* it departs from uniform names the cost. Sweep the regime knob (swing amplitude) and
you watch "linear is fine" hold, then break.

The toy: a damped pendulum, discrete time, state z = (theta, omega).

    theta' = theta + DT * omega
    omega' = omega - DT * OMEGA0_SQ * sin(theta) - DT * gamma * omega + process_noise
    y      = sin(theta) + obs_noise                       (a folding, nonlinear readout)

with process/obs noise drawn Student-t (heavy-tailed). Three orthogonal relaxation knobs
each turn one piece linear/Gaussian — sin(theta) -> theta in the dynamics, sin(theta) ->
theta in the readout, Student-t -> variance-matched Gaussian in the noise. Relax all three
and the model is an exact LGSSM (exact Kalman filter); keep any nonlinear piece and the
candidate is fit with the EKF (still O(T)). Three static parameters are inferred — the
process-noise scale q, the observation-noise scale r, and the damping gamma — on a small
log-space grid, so the whole cross-generation experiment vectorises and 100s of refits run
in well under a second.

Two diagnostics fall out, and they are not equally trustworthy:

* the **predictive PIT** of a held-out future observation — the trusted reliability
  readout, because its target is an observable (no pseudo-true ambiguity);
* the **parameter fractional rank** (the posterior CDF at the true value, per shared
  parameter) — the rough/confounded probe, which under model relaxation carries
  pseudo-true projection bias on top of dispersion, so its shape must not be over-read.

Both are calibrated against simultaneous null bands (Säilynoja, Bürkner & Vehtari 2022,
arXiv 2103.10522). The PIT screen certifies *honesty*, not *accuracy*; to price the
sharpness a calibrated-but-vague linear fit leaves on the table, we also score against the
**oracle** predictive p(y | true state, true params) — free because the data is simulated —
with a proper rule (CRPS). This module is self-contained (no cross-notebook imports).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from jax.scipy.stats import norm as jnorm
from plotly.subplots import make_subplots
from scipy import stats as sstats

type GridShape = tuple[int, int, int]

jax.config.update("jax_enable_x64", True)

# ── Calibration palette (self-contained; shared convention across the SBC-style notebooks) ──
C_UNIFORM = "#111827"  # the uniform target / reference
C_PASS = "#059669"  # calibrated — green
C_OVER = "#dc2626"  # overconfident / heavy-tailed — red
C_UNDER = "#2563eb"  # underconfident / over-dispersed — blue
C_BIAS = "#ea580c"  # biased / heavy-tail knob — orange
C_BAND = "rgba(148,163,184,0.30)"  # simultaneous null-band fill (slate)

# ── Fixed model constants (everything not inferred) ──────────────────────────────
DT = 0.1  # discrete step
OMEGA0_SQ = 1.0  # squared natural frequency
NU = 2.3  # Student-t degrees of freedom (very heavy tails; variance finite since NU > 2)
P0 = jnp.array([1e-3, 1e-3])  # known initial-state covariance (released from a known angle)
T = 30  # trajectory length
H = 5  # holdout / forecast horizon (the last H steps are forecast, not fitted)

# ── Priors on the three shared static params, in log space: log(param) ~ N(MU, SIG) ──
#    order: q (process-noise sd), r (obs-noise sd), gamma (damping)
MU = jnp.array([np.log(0.12), np.log(0.15), np.log(0.15)])
SIG = jnp.array([0.5, 0.4, 0.5])
PARAM_NAMES = ("q", "r", "γ")

# ── Grid + experiment defaults ───────────────────────────────────────────────────
GRID_K = 3.2  # grid spans MU ± GRID_K·SIG per param
GRID_DEFAULT = (15, 41, 15)  # r is best-identified, so it gets the fine axis (param-rank figures)
GRID_HEATMAP = (11, 21, 11)  # the predictive PIT is grid-robust, so the sweep uses a coarse grid
S_DEFAULT = 800  # cross-generation replicates for the calibration figures
S_HEATMAP = 400  # replicates per heatmap / curve cell
BINS_DEFAULT = 20
GATE_AMP = 0.6  # a moderate swing for the gate / floor panels
GALLERY_AMP = 1.3  # where all three single-knob signatures are visible at once
NSIM_BAND = 2000  # Monte-Carlo replicates calibrating a simultaneous band
SEED = 7

# ── Regime sweep axis (initial swing amplitude, radians) ─────────────────────────
AMPS = (0.2, 0.6, 1.0, 1.5, 2.0, 2.6)


@dataclass(frozen=True)
class Cfg:
    """Which spots the candidate treats as linear / Gaussian (True = relaxed).

    Four orthogonal model spots: the two nonlinearities (dynamics drift, sensor readout) and
    the two noise distributions (process, observation). The *filter* used on a config is a
    separate axis (KF / EKF / UKF / PF), passed explicitly where it matters."""

    dyn_linear: bool
    obs_linear: bool
    proc_gauss: bool
    obs_gauss: bool


# Named configs used throughout
KF = Cfg(True, True, True, True)  # fully linear-Gaussian -> the candidate is an exact Kalman filter
FULL = Cfg(
    False, False, False, False
)  # the realistic truth: nonlinear drift + readout, heavy noise
BEST = Cfg(False, False, True, True)  # cheapest *faithful* Gaussian filter: EKF over the true funcs

# Each truth relaxes exactly ONE of the four spots away from the fully-linear-Gaussian KF
# candidate, so its single-knob panel isolates that spot's cost (the KF candidate stays exact).
TRUTH_DYN = Cfg(False, True, True, True)  # only the restoring force is nonlinear
TRUTH_OBS = Cfg(True, False, True, True)  # only the readout folds
TRUTH_PROC = Cfg(True, True, False, True)  # only the process noise is heavy-tailed
TRUTH_OBSNOISE = Cfg(True, True, True, False)  # only the observation noise is heavy-tailed

# Candidate strategies for the regime heatmap (truth held at FULL)
HEATMAP_CANDS = (
    ("fully-linear KF", KF),
    ("EKF · keep dynamics", Cfg(False, True, True, True)),
    ("EKF · keep readout", Cfg(True, False, True, True)),
    ("EKF · keep both", BEST),
)

# ── Semantic palette for the knobs (built on the shared calibration colours) ─────
C_DYN = C_UNDER  # dynamics knob — blue
C_OBSK = C_OVER  # readout knob — red (the dominant cost)
C_NOISE = C_BIAS  # process-noise knob — orange
C_OBSNOISE = "#7c3aed"  # observation-noise knob — purple
# colour per heatmap candidate (red = worst fully-linear … green = best keep-both)
HEATMAP_COLORS = (C_OBSK, C_NOISE, C_DYN, C_PASS)


# ── Model pieces (the pendulum), parameterised by the relaxation knobs ───────────
def trans_mean(z, gamma, dyn_linear):
    th, om = z[0], z[1]
    g = th if dyn_linear else jnp.sin(th)
    return jnp.array([th + DT * om, om - DT * OMEGA0_SQ * g - DT * gamma * om])


def trans_jac(z, gamma, dyn_linear):
    th = z[0]
    gp = 1.0 if dyn_linear else jnp.cos(th)
    return jnp.array([[1.0, DT], [-DT * OMEGA0_SQ * gp, 1.0 - DT * gamma]])


def obs_mean(th, obs_linear):
    return th if obs_linear else jnp.sin(th)


def obs_jac(th, obs_linear):
    return 1.0 if obs_linear else jnp.cos(th)


# ── Simulate the (possibly nonlinear / Student-t) truth ──────────────────────────
def simulate(key, q, r, gamma, amp, cfg: Cfg):
    kp, ko, ki = jax.random.split(key, 3)
    # standardise Student-t to unit variance so q, r stay the noise *sd* (variance-matched)
    s = np.sqrt((NU - 2.0) / NU)
    wp = (jax.random.normal(kp, (T,)) if cfg.proc_gauss else jax.random.t(kp, NU, (T,)) * s) * q
    wo = (jax.random.normal(ko, (T,)) if cfg.obs_gauss else jax.random.t(ko, NU, (T,)) * s) * r

    def step(z, w):
        zn = trans_mean(z, gamma, cfg.dyn_linear) + jnp.array([0.0, w])
        return zn, zn

    z0 = jnp.array([amp, 0.0]) + jnp.sqrt(P0) * jax.random.normal(ki, (2,))
    _, zs = jax.lax.scan(step, z0, wp[1:])
    zfull = jnp.concatenate([z0[None, :], zs])  # [T, 2] — full latent state, both θ and ω
    y = obs_mean(zfull[:, 0], cfg.obs_linear) + wo
    return y, zfull


# ── The cheapest valid Gaussian filter (EKF; exactly the Kalman filter when linear) ──
def filter_forecast(params, amp, y, cfg: Cfg):
    """Marginal log-likelihood over the fit window plus the H-step-ahead forecast (mean, var)."""
    q, r, gamma = params
    Q = jnp.array([[0.0, 0.0], [0.0, q * q]])
    R = r * r
    eye = jnp.eye(2)

    def update(m, P, yt):
        hp = obs_jac(m[0], cfg.obs_linear)
        Hh = jnp.array([hp, 0.0])
        yhat = obs_mean(m[0], cfg.obs_linear)
        S = hp * hp * P[0, 0] + R
        K = (P @ Hh) / S
        innov = yt - yhat
        m2 = m + K * innov
        ImKH = eye - jnp.outer(K, Hh)
        P2 = ImKH @ P @ ImKH.T + jnp.outer(K, K) * R  # Joseph form keeps P symmetric PSD
        ll = -0.5 * (jnp.log(2 * jnp.pi * S) + innov * innov / S)
        return m2, P2, ll

    def predict(m, P):
        F = trans_jac(m, gamma, cfg.dyn_linear)
        return trans_mean(m, gamma, cfg.dyn_linear), F @ P @ F.T + Q

    Tfit = T - H
    m, P = jnp.array([amp, 0.0]), jnp.diag(P0)
    m, P, ll0 = update(m, P, y[0])  # first obs folds into the initial prior (no predict)

    def fit_step(carry, yt):
        m, P, acc = carry
        m, P = predict(m, P)
        m, P, ll = update(m, P, yt)
        return (m, P, acc + ll), None

    (m, P, acc), _ = jax.lax.scan(fit_step, (m, P, ll0), y[1:Tfit])

    def fwd(carry, _):  # predict-only roll-forward across the holdout
        m, P = carry
        return predict(m, P), None

    (m, P), _ = jax.lax.scan(fwd, (m, P), None, length=H)
    hp = obs_jac(m[0], cfg.obs_linear)
    yhat_f = obs_mean(m[0], cfg.obs_linear)
    S_f = hp * hp * P[0, 0] + R
    return acc, yhat_f, S_f


_sim_batch = jax.jit(jax.vmap(simulate, in_axes=(0, 0, 0, 0, None, None)), static_argnums=(5,))
# filter over the param grid (inner vmap) and over replicates (outer vmap), one jit
_ff_batch = jax.jit(
    jax.vmap(
        jax.vmap(filter_forecast, in_axes=(0, None, None, None)), in_axes=(None, None, 0, None)
    ),
    static_argnums=(3,),
)


def _make_grid(grid):
    axes = [
        np.linspace(float(MU[i] - GRID_K * SIG[i]), float(MU[i] + GRID_K * SIG[i]), grid[i])
        for i in range(3)
    ]
    g1, g2, g3 = np.meshgrid(*axes, indexing="ij")
    u_flat = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)  # log-space
    params = np.exp(u_flat)
    logprior = sum(-0.5 * ((u_flat[:, i] - float(MU[i])) / float(SIG[i])) ** 2 for i in range(3))
    return axes, jnp.asarray(params), jnp.asarray(logprior)


@lru_cache(maxsize=512)
def crossgen(
    truth_cfg: Cfg,
    cand_cfg: Cfg,
    amp: float,
    n_sims: int,
    grid: GridShape,
    seed: int,
):
    """One cross-generation experiment: simulate from truth_cfg, fit cand_cfg, return PITs.

    Returns ``(param_pits[n_sims, 3], pred_pits[n_sims])``. ``param_pits[:, d]`` is the
    candidate's marginal posterior CDF for parameter ``d`` evaluated at the true value (the
    fractional rank); ``pred_pits`` is the candidate's predictive CDF at the held-out future
    observation ``y[T-1]``. Both are Uniform(0, 1) iff the candidate is calibrated.
    """
    axes, params, logprior = _make_grid(grid)
    key = jax.random.PRNGKey(seed)
    tkey, skey = jax.random.split(key)
    u_true = MU + SIG * jax.random.normal(tkey, (n_sims, 3))  # draw truths from the prior
    p_true = jnp.exp(u_true)
    sim_keys = jax.random.split(skey, n_sims)
    ys, _ = _sim_batch(sim_keys, p_true[:, 0], p_true[:, 1], p_true[:, 2], amp, truth_cfg)
    ll, yhat_f, S_f = _ff_batch(params, amp, ys, cand_cfg)  # each [n_sims, Ngrid]
    w = np.asarray(jax.nn.softmax(ll + logprior[None, :], axis=1))  # grid posterior weights

    ystar = np.asarray(ys[:, T - 1])
    cdf = np.asarray(jnorm.cdf((ystar[:, None] - yhat_f) / jnp.sqrt(S_f)))
    pred_pits = np.sum(w * cdf, axis=1)  # mixture predictive CDF at the truth

    param_pits = np.zeros((n_sims, 3))
    w3 = w.reshape(n_sims, *grid)
    u_true_np = np.asarray(u_true)
    for d in range(3):
        marg = w3.sum(axis=tuple(j + 1 for j in range(3) if j != d))  # [n_sims, G_d]
        marg = marg / marg.sum(axis=1, keepdims=True)
        fmid = np.cumsum(marg, axis=1) - 0.5 * marg  # midpoint plotting position
        for i in range(n_sims):
            param_pits[i, d] = np.clip(np.interp(u_true_np[i, d], axes[d], fmid[i]), 0.0, 1.0)
    return param_pits, pred_pits


# ── The oracle predictive + proper (CRPS) scoring ────────────────────────────────
# Calibration (the PIT screen above) certifies a forecaster is honest *relative to its own
# information* — it is blind to a model that launders unresolved signal into predictive width
# and stays calibrated by being vague. Because we *simulated* the data we know the true state
# x_t and true params θ, so we can write the **oracle** predictive p(y_{t+H} | x_t, θ) — the
# best any forecaster could do — with no inference, and score both it and the candidate with a
# proper, sharpness-aware rule (the closed-form Gaussian CRPS, on each predictive's mean and
# spread). The gap CRPS_candidate − CRPS_oracle is the accuracy loss the PIT cannot see.
M_SCORE = 500  # oracle forecast samples per replicate (for its predictive mean/spread + the KDE)


def _oracle_forecast(key, zb, q, r, gamma, cfg: Cfg, M):
    """H-step-ahead forecast samples from the *true* boundary states under the true model.

    Flat-batched over (replicate × sample): zb is [n, 2], q/r/gamma are [n], and we roll the
    true nonlinear dynamics H steps on [n, M] state arrays. Returns [n, M] forecast obs.
    (A scan-in-vmap over the same work was the runtime bottleneck.)"""
    n = zb.shape[0]
    kp, ko = jax.random.split(key)
    s = np.sqrt((NU - 2.0) / NU)
    wp = (
        jax.random.normal(kp, (n, M, H)) if cfg.proc_gauss else jax.random.t(kp, NU, (n, M, H)) * s
    ) * q[:, None, None]
    wo = (jax.random.normal(ko, (n, M)) if cfg.obs_gauss else jax.random.t(ko, NU, (n, M)) * s) * r[
        :, None
    ]
    th = jnp.broadcast_to(zb[:, None, 0], (n, M))
    om = jnp.broadcast_to(zb[:, None, 1], (n, M))
    g = gamma[:, None]
    for k in range(H):
        gg = th if cfg.dyn_linear else jnp.sin(th)
        th, om = th + DT * om, om - DT * OMEGA0_SQ * gg - DT * g * om + wp[:, :, k]
    return (th if cfg.obs_linear else jnp.sin(th)) + wo


_oracle_forecast_j = jax.jit(_oracle_forecast, static_argnums=(5, 6))


def _crps_gaussian(mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of a Gaussian predictive N(mu, sd²) at observation y, vectorised
    (Gneiting & Raftery 2007). Lower is better; it penalises predictive *width* as well as
    miscentring — the property the PIT lacks.

        CRPS = sd · [ z(2Φ(z) − 1) + 2φ(z) − 1/√π ],   z = (y − mu)/sd

    Built from scipy's Normal (no hand-rolled estimator); verified against the known value
    CRPS(N(0,1), 0) = 0.23369."""
    sd = np.maximum(sd, 1e-9)
    z = (y - mu) / sd
    return sd * (z * (2 * sstats.norm.cdf(z) - 1) + 2 * sstats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def _crps_ensemble(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of an ensemble (sample) predictive — the proper score for the particle filter,
    whose forecast can be non-Gaussian (e.g. bimodal under a folding readout) and is thus not
    captured by a mean/sd summary. Sorted O(M log M) form of E|X−y| − ½E|X−X'|; verified to
    agree with the closed-form Gaussian CRPS on Gaussian samples."""
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    i = np.arange(m)
    return np.mean(np.abs(s - y[:, None]), axis=1) - (1.0 / m**2) * np.sum(
        (2 * i + 1 - m) * s, axis=1
    )


class _Scored(NamedTuple):
    pred_pits: np.ndarray  # [n] candidate predictive PIT (the calibration screen)
    crps_cand: np.ndarray  # [n] candidate CRPS (lower = sharper & accurate)
    crps_oracle: np.ndarray  # [n] oracle CRPS — the achievable floor
    width: np.ndarray  # [n] candidate predictive sd / oracle predictive sd
    y_grid: np.ndarray  # for the example panel: a y axis to draw densities on
    cand_dens: np.ndarray  # candidate mixture density of the example replicate
    oracle_dens: np.ndarray  # oracle predictive density of the example replicate (KDE, cosmetic)
    y_real: float  # the realized future obs of the example replicate
    ex_width: float  # the example replicate's width ratio


def _scored(
    truth: Cfg,
    cand: Cfg,
    amp: float,
    n_sims: int,
    grid: GridShape,
    seed: int,
    M: int,
) -> _Scored:
    _axes, params, logprior = _make_grid(grid)
    base = jax.random.PRNGKey(seed)
    tkey, skey = jax.random.split(base)  # same split as crossgen -> identical truths & PITs
    okey = jax.random.fold_in(base, 1)
    u_true = MU + SIG * jax.random.normal(tkey, (n_sims, 3))
    p_true = jnp.exp(u_true)
    ys, zfull = _sim_batch(
        jax.random.split(skey, n_sims), p_true[:, 0], p_true[:, 1], p_true[:, 2], amp, truth
    )
    ll, yhat_f, S_f = _ff_batch(params, amp, ys, cand)
    w = np.asarray(jax.nn.softmax(ll + logprior[None, :], axis=1))
    yh, sv = np.asarray(yhat_f), np.asarray(S_f)
    ystar = np.asarray(ys[:, T - 1])
    pred_pits = np.sum(w * np.asarray(jnorm.cdf((ystar[:, None] - yh) / np.sqrt(sv))), axis=1)

    # candidate predictive = grid mixture of Gaussians -> exact moments, no sampling needed
    mean_c = np.sum(w * yh, axis=1)
    sd_c = np.sqrt(np.maximum(np.sum(w * (sv + yh**2), axis=1) - mean_c**2, 1e-12))
    # oracle predictive = forward simulation from the TRUE boundary state under the true model
    zb = zfull[:, T - H - 1, :]
    osamp = np.asarray(
        _oracle_forecast_j(okey, zb, p_true[:, 0], p_true[:, 1], p_true[:, 2], truth, M)
    )
    mean_o, sd_o = osamp.mean(axis=1), osamp.std(axis=1)
    crps_c = _crps_gaussian(mean_c, sd_c, ystar)
    crps_o = _crps_gaussian(mean_o, sd_o, ystar)
    width = sd_c / sd_o

    # a representative "calibrated but wide" replicate: high width, realized y comfortably inside
    inside = (pred_pits > 0.15) & (pred_pits < 0.85)
    pick = np.where(inside)[0]
    ex = (
        int(pick[np.argmin(np.abs(width[pick] - np.quantile(width[pick], 0.7)))])
        if pick.size
        else 0
    )
    lo = float(min(osamp[ex].min(), mean_c[ex] - 4 * sd_c[ex]))
    hi = float(max(osamp[ex].max(), mean_c[ex] + 4 * sd_c[ex]))
    y_grid = np.linspace(lo, hi, 300)
    cand_dens = np.sum(
        w[ex][:, None]
        * np.exp(-0.5 * (y_grid[None, :] - yh[ex][:, None]) ** 2 / sv[ex][:, None])
        / np.sqrt(2 * np.pi * sv[ex][:, None]),
        axis=0,
    )
    oracle_dens = sstats.gaussian_kde(osamp[ex])(y_grid)  # cosmetic; CRPS itself is bandwidth-free
    return _Scored(
        pred_pits,
        crps_c,
        crps_o,
        width,
        y_grid,
        cand_dens,
        oracle_dens,
        float(ystar[ex]),
        float(width[ex]),
    )


@lru_cache(maxsize=256)
def crossgen_scored(
    truth: Cfg,
    cand: Cfg,
    amp: float,
    n_sims: int,
    grid: GridShape,
    seed: int,
    M: int,
) -> _Scored:
    return _scored(truth, cand, amp, n_sims, grid, seed, M)


# ── PIT calibration: simultaneous bands, χ² score, shape verdict ─────────────────
_PIT_HIST_BAND_CACHE: dict[
    tuple[str, int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = {}
_PIT_ECDF_BAND_CACHE: dict[
    tuple[str, int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = {}


def _simultaneous_envelope(
    stats_sim: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise bounds tuned to one γ so their *joint* coverage is 1 − alpha (Säilynoja,
    Bürkner & Vehtari 2022, arXiv 2103.10522). ``stats_sim`` is [n_sim, n_point], one row per
    simulated null dataset; bisect γ until the per-point γ/2 … 1−γ/2 quantile band contains a
    whole null dataset exactly 1 − alpha of the time, which accounts for the dependence between
    points that a naive pointwise interval ignores."""
    n_sim = stats_sim.shape[0]
    col = np.sort(stats_sim, axis=0)

    def at(gamma: float) -> tuple[np.ndarray, np.ndarray]:
        lo = max(int(np.floor(gamma / 2 * n_sim)) - 1, 0)
        hi = min(int(np.ceil((1 - gamma / 2) * n_sim)) - 1, n_sim - 1)
        return col[lo], col[hi]

    def coverage(gamma: float) -> float:
        lo, hi = at(gamma)
        return float(np.mean(np.all((stats_sim >= lo) & (stats_sim <= hi), axis=1)))

    glo, ghi = 1e-4, alpha
    for _ in range(40):
        gm = 0.5 * (glo + ghi)
        if coverage(gm) > 1 - alpha:
            glo = gm
        else:
            ghi = gm
    return at(0.5 * (glo + ghi))


def pit_hist_band(S: int, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin simultaneous 95% band for a histogram of S Uniform(0, 1) PITs."""
    key = ("hist", S, bins)
    if key not in _PIT_HIST_BAND_CACHE:
        u = np.random.default_rng(20240701).random((NSIM_BAND, S))
        idx = np.minimum((u * bins).astype(int), bins - 1)
        counts = np.zeros((NSIM_BAND, bins), dtype=int)
        rows = np.repeat(np.arange(NSIM_BAND), S)
        np.add.at(counts, (rows, idx.ravel()), 1)
        lo, hi = _simultaneous_envelope(counts)
        _PIT_HIST_BAND_CACHE[key] = (lo, hi, np.linspace(0.0, 1.0, bins + 1))
    return _PIT_HIST_BAND_CACHE[key]


def _ecdf_at(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(samples), grid, side="right") / samples.size


def pit_ecdf_band(S: int, n_eval: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simultaneous 95% band for the ECDF of S Uniform(0, 1) PITs, on a fixed grid."""
    key = ("ecdf", S, n_eval)
    if key not in _PIT_ECDF_BAND_CACHE:
        grid = np.linspace(0.0, 1.0, n_eval)
        rng = np.random.default_rng(20240702)
        curves = np.stack([_ecdf_at(rng.random(S), grid) for _ in range(NSIM_BAND)])
        lo, hi = _simultaneous_envelope(curves)
        _PIT_ECDF_BAND_CACHE[key] = (grid, lo, hi)
    return _PIT_ECDF_BAND_CACHE[key]


def pit_ecdf_diff(
    pits: np.ndarray, S: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ECDF-minus-uniform of the PITs, with the (also centred) simultaneous null band."""
    grid, lo, hi = pit_ecdf_band(S)
    return grid, _ecdf_at(pits, grid) - grid, lo - grid, hi - grid


def calib_chi2(pits: np.ndarray, bins: int = BINS_DEFAULT) -> float:
    """χ² distance from uniform of a PIT histogram (≈ bins-1 when calibrated)."""
    counts, _ = np.histogram(pits, bins=bins, range=(0.0, 1.0))
    expected = pits.size / bins
    return float(np.sum((counts - expected) ** 2 / expected))


def diagnose_pit(pits: np.ndarray, S: int, bins: int) -> tuple[str, str]:
    """A plain-language read of a PIT histogram and the colour to render it in."""
    lo, hi, edges = pit_hist_band(S, bins)
    counts, _ = np.histogram(pits, bins=edges)
    if not np.any((counts < lo) | (counts > hi)):
        return "flat — within the band, calibrated ✓", C_PASS
    x = np.linspace(-1.0, 1.0, bins)
    trend = float(np.polyfit(x, counts, 1)[0]) * bins
    half = bins // 2
    curv = float(counts[0] + counts[-1] - counts[half - 1] - counts[half])
    if abs(curv) >= abs(trend):
        if curv > 0:
            return "∪ — mass at both ends: overconfident / heavy-tailed", C_OVER
        return "∩ dome — too dispersed: underconfident", C_UNDER
    if trend < 0:
        return "↘ slope — biased: candidate sits above the truth", C_BIAS
    return "↗ slope — biased: candidate sits below the truth", C_BIAS


# ── Plotting helpers ─────────────────────────────────────────────────────────────
def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, title: str, height: int = 460, width: int = 940) -> go.Figure:
    fig.update_layout(
        title=f"<b>{title}</b>",
        height=height,
        width=width,
        margin=dict(t=70, b=60, l=66, r=26),
        template="plotly_white",
        legend=dict(
            x=0.99, y=0.99, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.78)"
        ),
        bargap=0.04,
    )
    return fig


def _add_pit_hist(
    fig: go.Figure,
    pits: np.ndarray,
    S: int,
    bins: int,
    color: str,
    *,
    row: int | None = None,
    col: int | None = None,
    show_legend: bool = False,
) -> None:
    lo, hi, edges = pit_hist_band(S, bins)
    counts, _ = np.histogram(pits, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    where = dict(row=row, col=col) if row is not None else {}
    fig.add_trace(
        go.Scatter(
            x=centers, y=lo, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False
        ),
        **where,
    )
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=hi,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=C_BAND,
            name="95% simultaneous band",
            legendgroup="band",
            showlegend=show_legend,
            hoverinfo="skip",
        ),
        **where,
    )
    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=(edges[1] - edges[0]) * 0.92,
            marker=dict(color=color, line=dict(width=0)),
            name="PIT counts",
            legendgroup="bars",
            showlegend=show_legend,
            hoverinfo="skip",
        ),
        **where,
    )
    fig.add_hline(y=S / bins, line=dict(color=C_UNIFORM, width=1.4, dash="dot"), **(where or {}))


# ── Figure 1: what the three relaxation knobs change ─────────────────────────────
def fig_relaxations() -> go.Figure:
    """The three knobs, drawn. Each turns one nonlinear/heavy-tailed piece into its
    linear/Gaussian stand-in: the restoring torque sin(θ)→θ, the readout sin(θ)→θ, and the
    Student-t noise → a variance-matched Gaussian. The first two agree only for small θ; the
    third agrees in the bulk but never in the tails."""
    th = np.linspace(-np.pi, np.pi, 400)
    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.075,
        subplot_titles=(
            "dynamics: restoring torque",
            "measurement: sensor readout",
            "noise: density (log scale)",
        ),
    )
    # dynamics: -ω0² sin θ vs -ω0² θ
    fig.add_trace(
        go.Scatter(
            x=th,
            y=-OMEGA0_SQ * np.sin(th),
            line=dict(color=C_DYN, width=2.6),
            name="truth  −ω₀²·sin θ",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=th,
            y=-OMEGA0_SQ * th,
            line=dict(color=C_UNIFORM, width=1.8, dash="dash"),
            name="relaxed  −ω₀²·θ",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    # measurement: sin θ vs θ
    fig.add_trace(
        go.Scatter(
            x=th,
            y=np.sin(th),
            line=dict(color=C_OBSK, width=2.6),
            name="truth  y = sin θ",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=th,
            y=th,
            line=dict(color=C_UNIFORM, width=1.8, dash="dash"),
            name="relaxed  y = θ",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    # noise: standardised Student-t vs standard normal density
    e = np.linspace(-6, 6, 400)
    s = np.sqrt((NU - 2.0) / NU)
    t_pdf = sstats.t.pdf(e / s, NU) / s  # standardised to unit variance
    g_pdf = np.exp(-0.5 * e**2) / np.sqrt(2 * np.pi)
    fig.add_trace(
        go.Scatter(
            x=e,
            y=t_pdf,
            line=dict(color=C_NOISE, width=2.6),
            name=f"truth  Student-t(ν={NU:.0f})",
            hoverinfo="skip",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=e,
            y=g_pdf,
            line=dict(color=C_UNIFORM, width=1.8, dash="dash"),
            name="relaxed  Gaussian",
            hoverinfo="skip",
        ),
        row=1,
        col=3,
    )
    fig.update_yaxes(type="log", range=[-3.2, 0.0], row=1, col=3)
    fig.update_xaxes(title_text="θ (rad)", row=1, col=1)
    fig.update_xaxes(title_text="θ (rad)", row=1, col=2)
    fig.update_xaxes(title_text="noise (in sd)", row=1, col=3)
    for c in (1, 2):
        fig.add_vrect(x0=-0.5, x1=0.5, fillcolor="rgba(5,150,105,0.07)", line_width=0, row=1, col=c)
    fig.update_annotations(font_size=13)
    return _layout(
        fig,
        "Three knobs, each turning one piece linear/Gaussian (green = small-angle agreement)",
        height=380,
        width=1020,
    )


# ── Figure 2: the model and its regime knob ──────────────────────────────────────
def fig_trajectories(amp: float) -> go.Figure:
    """The pendulum at one swing amplitude: the true (nonlinear) state and readout against
    their small-angle stand-ins. At small amplitude the dashed relaxations sit on top of the
    truth; wind the amplitude up and the restoring force weakens (sin θ < θ) and the readout
    *folds* (sin θ turns over while θ keeps climbing) — the regime where linear stops being
    free."""
    steps = np.arange(T)
    gamma_med = float(np.exp(MU[2]))

    def roll(dyn_linear):
        z = np.array([amp, 0.0])
        out = [z[0]]
        for _ in range(T - 1):
            th, om = z
            g = th if dyn_linear else np.sin(th)
            z = np.array([th + DT * om, om - DT * OMEGA0_SQ * g - DT * gamma_med * om])
            out.append(z[0])
        return np.array(out)

    th_true, th_lin = roll(False), roll(True)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=("hidden state θ(t)", "what the sensor reports, y(t)"),
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=th_true,
            line=dict(color=C_DYN, width=2.6),
            name="θ — true (sin) dynamics",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=th_lin,
            line=dict(color=C_UNIFORM, width=1.8, dash="dash"),
            name="θ — small-angle dynamics",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=np.sin(th_true),
            line=dict(color=C_OBSK, width=2.6),
            name="y = sin θ — true readout",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=th_true,
            line=dict(color=C_UNIFORM, width=1.8, dash="dash"),
            name="y = θ — linear readout",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(220,38,38,0.05)", line_width=0, row=2, col=1)
    fig.update_xaxes(title_text="time step t", row=2, col=1)
    fig.update_yaxes(title_text="θ (rad)", row=1, col=1)
    fig.update_yaxes(title_text="y", row=2, col=1)
    fig.update_annotations(font_size=13)
    return _layout(
        fig, f"The pendulum truth at swing amplitude θ₀ = {amp:.2f} rad", height=520, width=940
    )


# ── Figure 3: the SBC gate ───────────────────────────────────────────────────────
def fig_gate(S: int = S_DEFAULT, bins: int = BINS_DEFAULT) -> go.Figure:
    """Truth == candidate == fully-linear-Gaussian (the exact Kalman filter), so there is no
    model mismatch and no linearization error. Every PIT must be uniform — the three
    parameter fractional ranks and the predictive PIT all sit inside the band. This is the
    calibrated reference the cost panels are read against; it also validates the grid posterior
    and the whole cross-generation harness."""
    pp, dp = crossgen(KF, KF, GATE_AMP, S, GRID_DEFAULT, SEED)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"param {PARAM_NAMES[0]} — fractional rank",
            f"param {PARAM_NAMES[1]} — fractional rank",
            f"param {PARAM_NAMES[2]} — fractional rank",
            "predictive PIT (held-out forecast)",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )
    _add_pit_hist(fig, pp[:, 0], S, bins, C_PASS, row=1, col=1)
    _add_pit_hist(fig, pp[:, 1], S, bins, C_PASS, row=1, col=2)
    _add_pit_hist(fig, pp[:, 2], S, bins, C_PASS, row=2, col=1)
    _add_pit_hist(fig, dp, S, bins, C_PASS, row=2, col=2, show_legend=True)
    fig.update_annotations(font=dict(size=12))
    return _layout(
        fig,
        f"The gate: an exact filter on its own model — every PIT uniform ({S} replicates)",
        height=560,
        width=940,
    )


# ── Figure 4: the dictionary of costs (predictive PIT per single knob) ───────────
def fig_cost_gallery(
    amp: float = GALLERY_AMP, S: int = S_DEFAULT, bins: int = BINS_DEFAULT
) -> go.Figure:
    """Relax exactly one of the four spots in the truth away from the fully-linear-Gaussian
    Kalman candidate and read the predictive PIT. The gate (left) is flat; each spot writes
    its own signature. The two **nonlinearity** spots tilt the histogram — the folding readout
    violently (χ² in the hundreds), the small-angle dynamics mildly. The two **noise** spots
    are subtler and diffuse: the Gaussian filter inflates its estimated noise scale to absorb
    the heavy-tailed shocks, so χ² registers a real cost a KS statistic or per-bin band can
    miss. The χ² score (uniform ≈ bins-1) headlines each panel."""
    cases = (
        ("gate (exact KF)", KF, C_PASS),
        ("dynamics → sin θ", TRUTH_DYN, C_DYN),
        ("readout → sin θ", TRUTH_OBS, C_OBSK),
        ("process noise → t", TRUTH_PROC, C_NOISE),
        ("obs noise → t", TRUTH_OBSNOISE, C_OBSNOISE),
    )
    titles = []
    pits = []
    for label, truth, _color in cases:
        _, dp = crossgen(truth, KF, amp, S, GRID_DEFAULT, SEED)
        pits.append(dp)
        titles.append(f"{label}<br>χ² = {calib_chi2(dp, bins):.0f}")
    fig = make_subplots(rows=1, cols=5, subplot_titles=titles, horizontal_spacing=0.028)
    for c, ((_label, _truth, color), dp) in enumerate(zip(cases, pits, strict=True), start=1):
        _add_pit_hist(fig, dp, S, bins, color, row=1, col=c, show_legend=(c == 1))
        fig.update_xaxes(title_text="predictive PIT", row=1, col=c)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_annotations(font=dict(size=11))
    return _layout(
        fig.update_layout(title_text=None),
        f"The cost of relaxing each single spot at swing θ₀ = {amp:.2f}  (uniform χ² ≈ {bins - 1})",
        height=420,
        width=1160,
    )


# ── Figure 5: binning-free ECDF view of the same costs ───────────────────────────
def fig_ecdf_costs(amp: float = GALLERY_AMP, S: int = S_DEFAULT) -> go.Figure:
    """The same five configurations as ECDF-minus-uniform curves — no bin choice to second
    guess. Calibration is the flat line at 0 inside the grey band; a slope-shaped excursion is
    a bias, an S-shaped one a mis-scaled spread. The folding readout leaves the band by the
    widest margin; the two noise spots wander only slightly."""
    cases = (
        ("gate (exact KF)", KF, C_PASS, "solid"),
        ("dynamics → sin θ", TRUTH_DYN, C_DYN, "dot"),
        ("readout → sin θ", TRUTH_OBS, C_OBSK, "dashdot"),
        ("process noise → t", TRUTH_PROC, C_NOISE, "dash"),
        ("obs noise → t", TRUTH_OBSNOISE, C_OBSNOISE, "longdash"),
    )
    fig = go.Figure()
    grid, lo, hi = pit_ecdf_band(S)
    fig.add_trace(
        go.Scatter(
            x=grid, y=lo, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=hi,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=C_BAND,
            name="95% band (calibrated)",
            hoverinfo="skip",
        )
    )
    for label, truth, color, dash in cases:
        _, dp = crossgen(truth, KF, amp, S, GRID_DEFAULT, SEED)
        _, diff, _, _ = pit_ecdf_diff(dp, S)
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=diff,
                mode="lines",
                line=dict(color=color, width=2.4, dash=dash),
                name=label,
                hoverinfo="skip",
            )
        )
    fig.add_hline(y=0.0, line=dict(color=C_UNIFORM, width=1.2, dash="dot"))
    fig.update_xaxes(title_text="predictive PIT")
    fig.update_yaxes(title_text="ECDF − uniform")
    return _layout(
        fig,
        f"Binning-free: each relaxation bends the predictive ECDF its own way (θ₀ = {amp:.2f})",
        height=460,
        width=900,
    )


# ── Figure 6: rough probe vs trusted readout ─────────────────────────────────────
def fig_rough_vs_trusted(
    amp: float = 1.5, S: int = S_DEFAULT, bins: int = BINS_DEFAULT
) -> go.Figure:
    """The full nonlinear/Student-t truth fitted by the fully-linear Kalman candidate. The
    three parameter fractional ranks (top) are the *rough probe*: under model relaxation the
    candidate estimates a pseudo-true projection of each parameter, so their shapes mix bias
    and dispersion and must not be over-read. The predictive PIT (bottom) is the *trusted
    readout*: its target is an observable, so its departure from uniform is an honest measure
    of forecast unreliability."""
    pp, dp = crossgen(FULL, KF, amp, S, GRID_DEFAULT, SEED)
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{}, {}, {}], [{"colspan": 3}, None, None]],
        subplot_titles=(
            f"param {PARAM_NAMES[0]}  (rough)",
            f"param {PARAM_NAMES[1]}  (rough)",
            f"param {PARAM_NAMES[2]}  (rough)",
            f"predictive PIT — trusted readout   (χ² = {calib_chi2(dp, bins):.0f})",
        ),
        row_heights=[0.46, 0.54],
        horizontal_spacing=0.07,
        vertical_spacing=0.17,
    )
    for d in range(3):
        _add_pit_hist(fig, pp[:, d], S, bins, C_UNDER, row=1, col=d + 1)
    verdict, color = diagnose_pit(dp, S, bins)
    _add_pit_hist(fig, dp, S, bins, color, row=2, col=1, show_legend=True)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_xaxes(title_text="predictive PIT", row=2, col=1)
    fig.update_annotations(font=dict(size=12))
    return _layout(
        fig,
        f"Full nonlinear truth, fully-linear candidate (θ₀ = {amp:.2f}) — {verdict}",
        height=560,
        width=940,
    )


# ── Figure 7: the EKF's own approximation floor ──────────────────────────────────
def fig_ekf_floor(amp: float = GATE_AMP, S: int = S_DEFAULT, bins: int = BINS_DEFAULT) -> go.Figure:
    """A subtlety the gate hides. Keep the nonlinear pieces and filter them with the EKF, with
    truth == candidate (still no model mismatch): the only error left is the EKF's own
    linearization. On the left, the exact Kalman gate stays flat at any amplitude; on the
    right, the EKF on its *own* generative model is flat when swings are gentle but develops a
    mild non-uniformity as they grow — that residual is the filter-approximation cost, to be
    surfaced, not fixed."""
    _, dp_kf = crossgen(KF, KF, amp, S, GRID_DEFAULT, SEED)
    _, dp_ekf = crossgen(BEST, BEST, amp, S, GRID_DEFAULT, SEED)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"exact Kalman, truth = candidate<br>χ² = {calib_chi2(dp_kf, bins):.0f}",
            f"EKF, truth = candidate (nonlinear)<br>χ² = {calib_chi2(dp_ekf, bins):.0f}",
        ),
        horizontal_spacing=0.08,
    )
    _add_pit_hist(fig, dp_kf, S, bins, C_PASS, row=1, col=1)
    v2, c2 = diagnose_pit(dp_ekf, S, bins)
    _add_pit_hist(fig, dp_ekf, S, bins, c2, row=1, col=2, show_legend=True)
    for c in (1, 2):
        fig.update_xaxes(title_text="predictive PIT", row=1, col=c)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_annotations(font=dict(size=12))
    return _layout(
        fig,
        f"No model mismatch — only the filter differs (θ₀ = {amp:.2f}):  {v2}",
        height=430,
        width=940,
    )


# ── Figure 8: the regime cost map (hero) ─────────────────────────────────────────
def _heatmap_matrix(S: int) -> np.ndarray:
    return np.array(
        [
            [calib_chi2(crossgen(FULL, c, a, S, GRID_HEATMAP, SEED)[1]) for a in AMPS]
            for _label, c in HEATMAP_CANDS
        ]
    )


def fig_regime_heatmap(S: int = S_HEATMAP) -> go.Figure:
    """The hero. Truth is the full nonlinear/Student-t pendulum throughout. Rows are candidate
    filtering strategies, columns are swing amplitude; each cell is the predictive-PIT χ² (log
    colour, annotated). Two things jump out: every strategy is cheap at small amplitude (left
    edge), and the rows separate by *which* piece they keep nonlinear — keeping the folding
    readout (bottom two rows) stays near the calibrated floor at every amplitude, while
    linearizing it (top two rows) explodes. Linearizing the dynamics alone barely helps,
    because the readout is the load-bearing nonlinearity."""
    mat = _heatmap_matrix(S)
    labels = [lab for lab, _c in HEATMAP_CANDS]
    fig = go.Figure(
        go.Heatmap(
            z=np.log10(mat),
            x=[f"{a:.1f}" for a in AMPS],
            y=labels,
            text=[[f"{v:.0f}" for v in row] for row in mat],
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorscale="YlOrRd",
            colorbar=dict(title="log₁₀ χ²", tickvals=[1, 2, 3], ticktext=["10", "100", "1000"]),
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title_text="swing amplitude θ₀ (rad)  →  more nonlinear")
    fig.update_yaxes(autorange="reversed")
    return _layout(
        fig,
        f"Cost map: predictive-PIT χ² by candidate × regime  (uniform ≈ {BINS_DEFAULT - 1}; {S} replicates/cell)",
        height=430,
        width=900,
    )


# ── Figure 9: the cost curves ────────────────────────────────────────────────────
def fig_regime_curves(S: int = S_HEATMAP) -> go.Figure:
    """The same numbers as lines, which makes the trend and the ordering explicit. The dotted
    line is the 95% uniform threshold; below it a candidate is indistinguishable from
    calibrated. The two readout-keeping strategies hug the floor across the whole sweep; the
    two readout-linearizing ones climb steeply once the swing leaves the small-angle regime."""
    mat = _heatmap_matrix(S)
    fig = go.Figure()
    for (label, _c), row, color in zip(HEATMAP_CANDS, mat, HEATMAP_COLORS, strict=True):
        fig.add_trace(
            go.Scatter(
                x=list(AMPS),
                y=row,
                mode="lines+markers",
                line=dict(color=color, width=2.4),
                marker=dict(size=7),
                name=label,
                hoverinfo="skip",
            )
        )
    crit = float(sstats.chi2.ppf(0.95, BINS_DEFAULT - 1))
    fig.add_hline(
        y=crit,
        line=dict(color=C_UNIFORM, width=1.4, dash="dot"),
        annotation_text="95% uniform threshold",
        annotation_position="top left",
    )
    fig.update_yaxes(title_text="predictive-PIT χ²", type="log")
    fig.update_xaxes(title_text="swing amplitude θ₀ (rad)")
    return _layout(
        fig,
        "When does linear stop being free? Cost vs regime, per candidate",
        height=460,
        width=900,
    )


# ── Figure 10: calibrated, and still vague (the PIT blind spot) ──────────────────
def fig_calibrated_but_vague(amp: float = 1.3, S: int = S_HEATMAP) -> go.Figure:
    """The blind spot. The candidate keeps the folding readout but linearizes the dynamics —
    so it is **PIT-calibrated** (middle panel: flat, in band), yet it cannot resolve the state
    structure and quietly inflates its fitted noise scale to stay honest. Left: one such
    forecast — the candidate predictive (red) is far wider than the oracle (green, the best
    achievable given the true state and params), though both comfortably cover the realized
    value, so neither PIT is extreme. Right: this is systematic — *every* forecast is wider
    than the oracle (the ratio piles up well above 1). Calibration certifies honesty; it says
    nothing about this wasted sharpness."""
    cand = TRUTH_OBS  # Cfg(True, False, True): keep the readout, linearize the dynamics
    sc = crossgen_scored(FULL, cand, amp, S, GRID_HEATMAP, SEED, M_SCORE)
    chi2 = calib_chi2(sc.pred_pits)
    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.4, 0.3, 0.3],
        subplot_titles=(
            f"one forecast: {sc.ex_width:.1f}× too wide",
            f"predictive PIT (χ² = {chi2:.0f})",
            f"width ratio over {S} forecasts",
        ),
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=sc.y_grid,
            y=sc.oracle_dens,
            mode="lines",
            line=dict(color=C_PASS, width=2.4),
            fill="tozeroy",
            fillcolor=_rgba(C_PASS, 0.12),
            name="oracle predictive",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sc.y_grid,
            y=sc.cand_dens,
            mode="lines",
            line=dict(color=C_OBSK, width=2.4),
            fill="tozeroy",
            fillcolor=_rgba(C_OBSK, 0.10),
            name="candidate predictive",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=sc.y_real, line=dict(color=C_NOISE, width=2.0, dash="dash"), row=1, col=1)
    _add_pit_hist(fig, sc.pred_pits, S, BINS_DEFAULT, C_OBSK, row=1, col=2)
    fig.add_trace(
        go.Histogram(
            x=sc.width, marker=dict(color=C_OBSK), nbinsx=40, showlegend=False, hoverinfo="skip"
        ),
        row=1,
        col=3,
    )
    fig.add_vline(x=1.0, line=dict(color=C_PASS, width=1.6, dash="dot"), row=1, col=3)
    fig.add_vline(x=float(np.mean(sc.width)), line=dict(color=C_OBSK, width=1.6), row=1, col=3)
    fig.update_xaxes(title_text="future observation y", row=1, col=1)
    fig.update_xaxes(title_text="predictive PIT", row=1, col=2)
    fig.update_xaxes(title_text="candidate sd / oracle sd", row=1, col=3)
    fig.update_annotations(font=dict(size=12))
    return _layout(
        fig,
        f"Calibrated, and still vague — keep-readout candidate at θ₀ = {amp:.2f}",
        height=420,
        width=1040,
    )


# ── Figure 11: the oracle score sees what calibration cannot ─────────────────────
def fig_oracle_screen(S: int = S_HEATMAP) -> go.Figure:
    """Two screens over the same sweep (truth = full nonlinear pendulum). **Top — calibration**:
    predictive-PIT χ² per candidate, with the 95% threshold; the readout-keeping strategies stay
    in the band (PIT says "fine"). **Bottom — accuracy**: the candidate CRPS against the oracle
    floor (black, the best achievable). Every candidate sits *above* the floor — sharpness that
    calibration rates as perfect — and the readout-linearizers peel away as the swing grows. The
    gap to the floor overstates what a real nonlinear fit could recover (the oracle also knows the
    true state), so a small flat gap means linear is genuinely fine, while a widening gap is the
    trigger to spend on the exact engine."""
    scores = {
        lab_: [crossgen_scored(FULL, c, a, S, GRID_HEATMAP, SEED, M_SCORE) for a in AMPS]
        for lab_, c in HEATMAP_CANDS
    }
    oracle_floor = [
        float(np.mean(scores["fully-linear KF"][i].crps_oracle)) for i in range(len(AMPS))
    ]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "the cheap screen: predictive-PIT χ² (calibration)",
            "the accuracy screen: CRPS vs the oracle floor (lower = sharper)",
        ),
    )
    for (label, _c), color in zip(HEATMAP_CANDS, HEATMAP_COLORS, strict=True):
        chis = [calib_chi2(scores[label][i].pred_pits) for i in range(len(AMPS))]
        crps = [float(np.mean(scores[label][i].crps_cand)) for i in range(len(AMPS))]
        fig.add_trace(
            go.Scatter(
                x=list(AMPS),
                y=chis,
                mode="lines+markers",
                line=dict(color=color, width=2.2),
                marker=dict(size=6),
                name=label,
                legendgroup=label,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(AMPS),
                y=crps,
                mode="lines+markers",
                line=dict(color=color, width=2.2),
                marker=dict(size=6),
                name=label,
                legendgroup=label,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
    fig.add_hline(
        y=float(sstats.chi2.ppf(0.95, BINS_DEFAULT - 1)),
        line=dict(color=C_UNIFORM, width=1.4, dash="dot"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(AMPS),
            y=oracle_floor,
            mode="lines",
            line=dict(color=C_UNIFORM, width=2.0, dash="dash"),
            name="oracle floor",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="PIT χ²", type="log", row=1, col=1)
    fig.update_yaxes(title_text="CRPS", row=2, col=1)
    fig.update_xaxes(title_text="swing amplitude θ₀ (rad)", row=2, col=1)
    fig.update_annotations(font=dict(size=12))
    return _layout(
        fig,
        "Calibration says honest; the oracle score says whether it is sharp",
        height=620,
        width=920,
    )


# ── The filter axis: KF / EKF / UKF / PF (the "states" / inference spot) ─────────
# The knobs above relax the *model*; this axis fixes the full nonlinear model (with the true
# params, like the oracle) and varies only HOW the posterior over the latent state is computed:
#   KF  — linearize the model to fully-linear, then the exact Kalman filter
#   EKF — keep the model, linearize each step at the mean (first-order Jacobian)
#   UKF — keep the model, match moments through deterministic sigma points (no Jacobian)
#   PF  — keep the model, represent the state belief by a particle cloud (no Gaussian assumption)
# On a linear-Gaussian model KF == EKF == UKF == PF (all exact); their spread on the nonlinear
# model is the cost of the Gaussian-state approximation — the spot the particle filter removes.
N_PF = 1500  # particles for the bootstrap filter


def _ukf_forecast(params, amp, y, cfg: Cfg):
    """Unscented (sigma-point) Kalman filter; returns the H-step forecast (mean, var).
    Exact for linear-Gaussian models — validated == the exact KF — so any EKF↔UKF gap is purely
    how each linearization handles curvature."""
    q, r, gamma = params
    Q = jnp.array([[0.0, 0.0], [0.0, q * q]])
    R = r * r
    n = 2
    lam = 3.0 - n  # κ = 3 − n with α = 1 ⇒ λ = κ
    nl = n + lam
    wm = jnp.array([lam / nl, *([1.0 / (2 * nl)] * (2 * n))])
    wc = wm.at[0].set(lam / nl + 2.0)  # α = 1, β = 2
    eye = jnp.eye(n)

    def sigma(m, P):
        chol = jnp.linalg.cholesky(nl * (0.5 * (P + P.T)) + 1e-9 * eye)
        return jnp.concatenate([m[None, :], m[None, :] + chol.T, m[None, :] - chol.T], axis=0)

    def predict(m, P):
        prop = jax.vmap(lambda z: trans_mean(z, gamma, cfg.dyn_linear))(sigma(m, P))
        mp = wm @ prop
        d = prop - mp
        return mp, (wc[:, None] * d).T @ d + Q

    def obs_moments(m, P):
        pts = sigma(m, P)
        ys = obs_mean(pts[:, 0], cfg.obs_linear)
        yh = wm @ ys
        dy = ys - yh
        return pts, yh, jnp.sum(wc * dy * dy) + R, (wc[:, None] * (pts - m)).T @ dy

    def update(m, P, yt):
        _pts, yh, S, Pxy = obs_moments(m, P)
        K = Pxy / S
        return m + K * (yt - yh), P - S * jnp.outer(K, K)

    Tfit = T - H
    m, P = jnp.array([amp, 0.0]), jnp.diag(P0)
    m, P = update(m, P, y[0])

    def step(carry, yt):
        m, P = predict(*carry)
        return update(m, P, yt), None

    (m, P), _ = jax.lax.scan(step, (m, P), y[1:Tfit])
    (m, P), _ = jax.lax.scan(lambda c, _: (predict(*c), None), (m, P), None, length=H)
    _pts, yhat_f, S_f, _ = obs_moments(m, P)
    return yhat_f, S_f


def _pf_forecast(key, params, amp, y, cfg: Cfg, N):
    """Bootstrap particle filter over the (true-param) model; returns N H-step-ahead forecast
    samples. The exact state filter (as N→∞): no Gaussian-belief assumption, so it keeps a
    multimodal state posterior when the folding readout induces one. Validated against the KF
    forecast on linear-Gaussian models."""
    q, r, gamma = params
    ki, kfit, kfwd, kobs = (
        jax.random.fold_in(key, 0),
        jax.random.fold_in(key, 1),
        jax.random.fold_in(key, 2),
        jax.random.fold_in(key, 3),
    )
    x = jnp.array([amp, 0.0]) + jnp.sqrt(P0) * jax.random.normal(ki, (N, 2))
    Tfit = T - H

    def logw(x, yt):
        return -0.5 * (yt - obs_mean(x[:, 0], cfg.obs_linear)) ** 2 / (r * r)

    def resample(k, x, lw):
        u = (jax.random.uniform(k) + jnp.arange(N)) / N
        idx = jnp.searchsorted(jnp.cumsum(jax.nn.softmax(lw)), u)
        return x[jnp.clip(idx, 0, N - 1)]

    fit_keys = jax.random.split(kfit, Tfit)
    x = resample(fit_keys[0], x, logw(x, y[0]))

    def step(x, inp):
        yt, k = inp
        kp, kr = jax.random.split(k)
        xn = jax.vmap(lambda z: trans_mean(z, gamma, cfg.dyn_linear))(x)
        xn = xn.at[:, 1].add(jax.random.normal(kp, (N,)) * q)
        return resample(kr, xn, logw(xn, yt)), None

    x, _ = jax.lax.scan(step, x, (y[1:Tfit], fit_keys[1:Tfit]))

    def fwd(x, k):
        xn = jax.vmap(lambda z: trans_mean(z, gamma, cfg.dyn_linear))(x)
        return xn.at[:, 1].add(jax.random.normal(k, (N,)) * q), None

    x, _ = jax.lax.scan(fwd, x, jax.random.split(kfwd, H))
    return obs_mean(x[:, 0], cfg.obs_linear) + jax.random.normal(kobs, (N,)) * r


_kf_batch = jax.jit(jax.vmap(filter_forecast, in_axes=(0, None, 0, None)), static_argnums=(3,))
_ukf_batch = jax.jit(jax.vmap(_ukf_forecast, in_axes=(0, None, 0, None)), static_argnums=(3,))
_pf_batch = jax.jit(
    jax.vmap(_pf_forecast, in_axes=(0, 0, None, 0, None, None)), static_argnums=(4, 5)
)

# colour per filter (worst → best): KF red, EKF orange, UKF blue, PF green; oracle = grey
FILTER_COLORS = {"KF": C_OBSK, "EKF": C_NOISE, "UKF": C_DYN, "PF": C_PASS, "oracle": C_UNIFORM}


@lru_cache(maxsize=64)
def filter_compare(
    amp: float,
    n_sims: int,
    seed: int,
    n_pf: int,
    m_or: int,
) -> dict[str, float]:
    """Forecast CRPS for KF/EKF/UKF/PF + the oracle floor, all at the TRUE params on a full
    nonlinear, Gaussian-noise truth (so only the state-inference differs). Lower = sharper."""
    truth = BEST  # nonlinear dynamics + readout, Gaussian noise (so noise is not a confound)
    base = jax.random.PRNGKey(seed)
    tkey, skey, pkey, okey = jax.random.split(base, 4)
    p_true = jnp.exp(MU + SIG * jax.random.normal(tkey, (n_sims, 3)))
    q, rr, g = p_true[:, 0], p_true[:, 1], p_true[:, 2]
    ys, zfull = _sim_batch(jax.random.split(skey, n_sims), q, rr, g, amp, truth)
    ystar = np.asarray(ys[:, T - 1])

    kf_y, kf_s = _kf_batch(p_true, amp, ys, KF)[1:]
    ekf_y, ekf_s = _kf_batch(p_true, amp, ys, BEST)[1:]
    ukf_y, ukf_s = _ukf_batch(p_true, amp, ys, BEST)
    pf = np.asarray(_pf_batch(jax.random.split(pkey, n_sims), p_true, amp, ys, BEST, n_pf))
    orc = np.asarray(_oracle_forecast_j(okey, zfull[:, T - H - 1, :], q, rr, g, truth, m_or))
    return {
        "KF": float(_crps_gaussian(np.asarray(kf_y), np.sqrt(np.asarray(kf_s)), ystar).mean()),
        "EKF": float(_crps_gaussian(np.asarray(ekf_y), np.sqrt(np.asarray(ekf_s)), ystar).mean()),
        "UKF": float(_crps_gaussian(np.asarray(ukf_y), np.sqrt(np.asarray(ukf_s)), ystar).mean()),
        "PF": float(_crps_ensemble(pf, ystar).mean()),
        "oracle": float(_crps_ensemble(orc, ystar).mean()),
    }


# ── Figure 12: the filter axis — Gaussian-state cost vs the exact particle filter ─
def fig_filter_comparison(n_sims: int = 250) -> go.Figure:
    """The fourth spot: holding the full nonlinear model and the TRUE parameters fixed, vary
    only the state-inference engine. KF (linearize the whole model) is worst; EKF and UKF keep
    the model but force a single-Gaussian state belief; the PF keeps the exact (sample) state
    posterior. The PF→Gaussian-filter gap is the cost of the Gaussian-state approximation — the
    one spot the other knobs never touch — and the PF→oracle gap is the irreducible uncertainty
    of not knowing the state. Forecast CRPS (lower = sharper); ν makes no difference here since
    the truth's noise is Gaussian, so this isolates state representation alone."""
    rows = {k: [] for k in FILTER_COLORS}
    for a in AMPS:
        res = filter_compare(a, n_sims, SEED, N_PF, M_SCORE)
        for k in FILTER_COLORS:
            rows[k].append(res[k])
    fig = go.Figure()
    for k in ("KF", "EKF", "UKF", "PF"):
        fig.add_trace(
            go.Scatter(
                x=list(AMPS),
                y=rows[k],
                mode="lines+markers",
                line=dict(color=FILTER_COLORS[k], width=2.4),
                marker=dict(size=7),
                name=k,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(AMPS),
            y=rows["oracle"],
            mode="lines",
            line=dict(color=FILTER_COLORS["oracle"], width=2.0, dash="dash"),
            name="oracle floor",
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title_text="swing amplitude θ₀ (rad)  →  more nonlinear")
    fig.update_yaxes(title_text="forecast CRPS (lower = sharper)")
    return _layout(
        fig,
        "The state-inference spot: Gaussian filters vs the exact particle filter",
        height=470,
        width=900,
    )
