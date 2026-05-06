"""Samplers for pedagogical visualisation on ``synthetic_posteriors`` targets.

Each sampler exposes a ``run_*(target, config) -> SamplerTrace`` entry point that
returns a uniform record the notebook can plot without knowing what sampler
produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blackjax.vi.pathfinder as pathfinder
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from synthetic_posteriors.targets import Target

Array = jax.Array


@dataclass(frozen=True)
class SamplerTrace:
    """Uniform plotting record.

    - ``positions`` (N, D): flat array of all points to plot.
    - ``stage`` (N,): integer stage index for the plotted sampling path.
    - ``killed`` (N,) bool: ``True`` if this point was aborted (divergent / out-of-support).
    - ``connect``: draw a polyline between consecutive points in ``positions``.
    - ``stage_label``: axis label for the colour gradient.
    - ``summary``: one-line string for a footer annotation.
    """

    positions: Array
    stage: Array
    killed: Array
    connect: bool
    stage_label: str
    summary: str

# ─── Pathfinder (variational) ────────────────────────────────────────────────


@dataclass(frozen=True)
class PathfinderConfig:
    """Config for BlackJAX Pathfinder.

    - ``maxiter``: L-BFGS iteration budget — Pathfinder evaluates one Gaussian
      approximation per iterate and picks the best ELBO.
    - ``num_elbo_samples``: MC samples used to estimate each iterate's ELBO.
    - ``num_draws``: importance-resampled draws produced from the chosen iterate.
    """

    maxiter: int = 30
    maxcor: int = 10
    num_elbo_samples: int = 200
    num_draws: int = 180
    initial_position: tuple[float, ...] = (0.0, 0.0)
    seed: int = 0


@dataclass(frozen=True)
class PathfinderTrace:
    """Pedagogic record exposing Pathfinder's L-BFGS path + final draws.

    - ``path_positions`` (M, D): iterates along the L-BFGS trajectory (iter 0 = init).
    - ``path_elbo`` (M,): ELBO of each iterate's local Gaussian approximation.
      Non-finite entries mark iterates where the inverse-Hessian estimate became
      degenerate (e.g. after convergence) — Pathfinder discards these.
    - ``best_iter``: argmax over the finite ELBOs — the iterate whose Gaussian
      is used to produce the final draws.
    - ``draws`` (N, D): samples from the selected Gaussian approximation.
    - ``summary``: one-line footer string.
    """

    path_positions: Array
    path_elbo: Array
    best_iter: int
    draws: Array
    summary: str


def run_pathfinder(target: Target, config: PathfinderConfig | None = None) -> PathfinderTrace:
    cfg = config or PathfinderConfig()

    def log_prob(x: Array) -> Array:
        return target.log_prob(x)

    key = jax.random.PRNGKey(cfg.seed)
    fit_key, draw_key = jax.random.split(key)
    init = jnp.asarray(cfg.initial_position, dtype=jnp.float64)

    state, info = pathfinder.approximate(
        fit_key,
        log_prob,
        init,
        num_samples=cfg.num_elbo_samples,
        maxiter=cfg.maxiter,
        maxcor=cfg.maxcor,
    )
    path_positions = jnp.asarray(info.path.position)
    path_elbo = jnp.asarray(info.path.elbo)

    finite = jnp.isfinite(path_elbo)
    masked = jnp.where(finite, path_elbo, -jnp.inf)
    best_iter = int(jnp.argmax(masked))

    draws, _logq = pathfinder.sample(draw_key, state, num_samples=cfg.num_draws)

    finite_count = int(finite.sum())
    best_elbo = float(path_elbo[best_iter])
    summary = (
        f"Pathfinder · L-BFGS iters={path_positions.shape[0] - 1} "
        f"(finite-ELBO={finite_count}) · best iter={best_iter} "
        f"(ELBO={best_elbo:.2f}) · draws={cfg.num_draws}"
    )
    return PathfinderTrace(
        path_positions=path_positions,
        path_elbo=path_elbo,
        best_iter=best_iter,
        draws=draws,
        summary=summary,
    )


def run_pathfinder_sampler(
    target: Target, config: PathfinderConfig | None = None
) -> SamplerTrace:
    """SamplerTrace-returning wrapper so Pathfinder can be used in the gallery swap.

    Positions are the final importance-resampled draws (iid), stage is a dummy
    index, ``connect=False``. The summary preserves Pathfinder's ELBO diagnostics.
    """
    trace = run_pathfinder(target, config)
    n = trace.draws.shape[0]
    return SamplerTrace(
        positions=trace.draws,
        stage=jnp.arange(n),
        killed=jnp.zeros((n,), dtype=bool),
        connect=False,
        stage_label="draw index",
        summary=trace.summary,
    )


# ─── Multi-path Pathfinder (multi-start with importance resampling) ──────────


@dataclass(frozen=True)
class MultiPathfinderConfig:
    """Config for multi-path Pathfinder (Zhang et al. 2022, Algorithm 4).

    Runs ``num_paths`` independent L-BFGS paths from overdispersed initial
    positions, then mixes the resulting Gaussian approximations via Pareto-
    smoothed importance resampling (PSIR).

    - ``init_scale`` / ``init_center``: initial positions are drawn from
      ``N(init_center, init_scale² · I)`` — dispersion controls how widely the
      paths explore.
    - ``draws_per_path``: candidate pool size before resampling = ``num_paths *
      draws_per_path``.
    - ``num_resampled``: final draw count after PSIR.
    """

    num_paths: int = 8
    maxiter: int = 30
    maxcor: int = 10
    num_elbo_samples: int = 200
    draws_per_path: int = 80
    num_resampled: int = 180
    init_scale: float = 2.0
    init_center: tuple[float, ...] = (0.0, 0.0)
    seed: int = 0


@dataclass(frozen=True)
class MultiPathfinderTrace:
    """Pedagogic record for multi-path Pathfinder.

    - ``path_positions`` (K, M+1, D): L-BFGS iterates per path.
    - ``path_elbo`` (K, M+1): ELBO per path x iter.
    - ``best_iter`` (K,): argmax-ELBO iter per path.
    - ``path_inits`` (K, D): the overdispersed starting positions.
    - ``path_best_elbo`` (K,): selected-iterate ELBO per path (the mixture weight input).
    - ``candidate_positions`` (K·M, D): all per-path draws before resampling.
    - ``candidate_path_id`` (K·M,): which path each candidate came from.
    - ``candidate_log_weight`` (K·M,): importance ratios ``log p(x) - log q_mix(x)``
      under the equal-weight mixture of the K selected Gaussians.
    - ``draws`` (N, D): PSIR-resampled output.
    - ``pareto_k``: PSIS khat diagnostic (``> 0.7`` ⇒ unreliable).
    """

    path_positions: Array
    path_elbo: Array
    best_iter: Array
    path_inits: Array
    path_best_elbo: Array
    candidate_positions: Array
    candidate_path_id: Array
    candidate_log_weight: Array
    draws: Array
    pareto_k: float
    summary: str


def _state_gaussian(state) -> tuple[Array, Array]:
    """Recover (mu, Sigma) of the Gaussian approximation at a Pathfinder iterate.

    Follows ``bfgs_sample``'s parametrisation: phi = mu + T @ u with
    ``u ~ N(0, I)``, giving Sigma = T @ T.T.
    """
    alpha, beta, gamma = state.alpha, state.beta, state.gamma
    position, grad = state.position, state.grad_position
    D = position.shape[0]
    Q, R = jnp.linalg.qr(jnp.diag(jnp.sqrt(1.0 / alpha)) @ beta)
    Id_k = jnp.eye(R.shape[0])
    L = jnp.linalg.cholesky(Id_k + R @ gamma @ R.T)
    T = jnp.diag(jnp.sqrt(alpha)) @ (Q @ (L - Id_k) @ Q.T + jnp.eye(D))
    Sigma = T @ T.T
    mu = position + jnp.diag(alpha) @ grad + beta @ gamma @ beta.T @ grad
    return mu, Sigma


def _gaussian_logpdf(x: Array, mu: Array, Sigma: Array) -> Array:
    """Log N(x | mu, Sigma) with a small jitter for stability."""
    D = mu.shape[0]
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-10 * jnp.eye(D)
    L = jnp.linalg.cholesky(Sigma)
    diff = x - mu
    solved = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (jnp.sum(solved * solved) + logdet + D * jnp.log(2.0 * jnp.pi))


def _psis_log_weights(log_w: Array) -> tuple[Array, float]:
    """Pareto-smooth top-tail importance log-weights via ArviZ's ``psislw``.

    Returns the (normalised) smoothed log-weights plus the generalised-Pareto
    shape parameter (``khat``) — values ``> 0.7`` indicate that resampling is
    unreliable even after smoothing (tail too heavy).
    """
    from arviz.stats import psislw
    lw = np.asarray(log_w, dtype=np.float64)
    finite = np.isfinite(lw)
    if finite.sum() < 5:
        return jnp.asarray(lw), float("nan")
    lw_clean = np.where(finite, lw, -1e30)  # ArviZ rejects -inf directly
    smoothed, k_hat = psislw(lw_clean)
    # Restore -inf at originally non-finite positions so they get zero weight.
    smoothed = np.where(finite, smoothed, -np.inf)
    return jnp.asarray(smoothed), float(k_hat)


def run_multipath_pathfinder(
    target: Target, config: MultiPathfinderConfig | None = None
) -> MultiPathfinderTrace:
    cfg = config or MultiPathfinderConfig()

    def log_prob(x: Array) -> Array:
        return target.log_prob(x)

    D = len(cfg.init_center)
    key = jax.random.PRNGKey(cfg.seed)
    init_key, fit_base, draw_base, resample_key = jax.random.split(key, 4)
    inits = jnp.asarray(cfg.init_center) + cfg.init_scale * jax.random.normal(
        init_key, (cfg.num_paths, D)
    )
    fit_keys = jax.random.split(fit_base, cfg.num_paths)
    draw_keys = jax.random.split(draw_base, cfg.num_paths)

    paths_pos: list[Array] = []
    paths_elbo: list[Array] = []
    best_iters: list[int] = []
    best_elbos: list[float] = []
    states: list = []
    draws_list: list[Array] = []

    for k in range(cfg.num_paths):
        state, info = pathfinder.approximate(
            fit_keys[k], log_prob, inits[k],
            num_samples=cfg.num_elbo_samples,
            maxiter=cfg.maxiter, maxcor=cfg.maxcor,
        )
        path_pos = jnp.asarray(info.path.position)
        path_elbo = jnp.asarray(info.path.elbo)
        masked = jnp.where(jnp.isfinite(path_elbo), path_elbo, -jnp.inf)
        b = int(jnp.argmax(masked))
        paths_pos.append(path_pos)
        paths_elbo.append(path_elbo)
        best_iters.append(b)
        best_elbos.append(float(path_elbo[b]))
        states.append(state)
        draws_k, _logq = pathfinder.sample(
            draw_keys[k], state, num_samples=cfg.draws_per_path
        )
        draws_list.append(draws_k)

    # Recover (mu_k, Sigma_k) for each selected Gaussian.
    gaussians = [_state_gaussian(s) for s in states]
    candidates = jnp.concatenate(draws_list, axis=0)  # (K*M, D)
    path_ids = jnp.repeat(jnp.arange(cfg.num_paths), cfg.draws_per_path)

    # log p(x) for each candidate
    log_p = jax.vmap(log_prob)(candidates)
    # log q_mix(x) = logmeanexp_k log q_k(x) (equal-weight mixture)
    log_qs = jnp.stack(
        [jax.vmap(lambda x, mu=mu, S=S: _gaussian_logpdf(x, mu, S))(candidates)
         for (mu, S) in gaussians],
        axis=0,
    )  # (K, K*M)
    log_q_mix = jax.scipy.special.logsumexp(log_qs, axis=0) - jnp.log(cfg.num_paths)

    raw_log_w = log_p - log_q_mix
    # Guard non-finite candidates (out-of-support or numerical issues).
    finite = jnp.isfinite(raw_log_w)
    clean_log_w = jnp.where(finite, raw_log_w, -jnp.inf)

    smoothed_log_w, k_hat = _psis_log_weights(clean_log_w)
    # Normalise to probabilities for resampling.
    logz = jax.scipy.special.logsumexp(smoothed_log_w)
    probs = jnp.exp(smoothed_log_w - logz)
    probs = jnp.where(jnp.isfinite(probs), probs, 0.0)
    probs = probs / jnp.sum(probs)

    resample_idx = jax.random.choice(
        resample_key, candidates.shape[0], shape=(cfg.num_resampled,),
        replace=True, p=probs,
    )
    draws = candidates[resample_idx]

    reached = int(jnp.sum(jnp.isfinite(jnp.asarray(best_elbos))))
    summary = (
        f"Multi-Pathfinder · K={cfg.num_paths} paths (init σ={cfg.init_scale}) "
        f"· finite-best={reached}/{cfg.num_paths} "
        f"· candidates={candidates.shape[0]} · PSIR→{cfg.num_resampled} "
        f"· k̂={k_hat:.2f}"
    )

    return MultiPathfinderTrace(
        path_positions=jnp.stack(paths_pos, axis=0),
        path_elbo=jnp.stack(paths_elbo, axis=0),
        best_iter=jnp.asarray(best_iters),
        path_inits=inits,
        path_best_elbo=jnp.asarray(best_elbos),
        candidate_positions=candidates,
        candidate_path_id=path_ids,
        candidate_log_weight=smoothed_log_w,
        draws=draws,
        pareto_k=float(k_hat),
        summary=summary,
    )


def run_multipath_pathfinder_sampler(
    target: Target, config: MultiPathfinderConfig | None = None
) -> SamplerTrace:
    """SamplerTrace-returning wrapper for the gallery swap.

    Positions are the final PSIR draws, stage is a dummy index.
    """
    trace = run_multipath_pathfinder(target, config)
    n = trace.draws.shape[0]
    return SamplerTrace(
        positions=trace.draws,
        stage=jnp.arange(n),
        killed=jnp.zeros((n,), dtype=bool),
        connect=False,
        stage_label="draw index",
        summary=trace.summary,
    )
