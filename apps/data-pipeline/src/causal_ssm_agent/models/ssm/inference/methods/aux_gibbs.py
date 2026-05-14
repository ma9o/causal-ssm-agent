"""Auxiliary Gibbs sampler: blocked aux-Kalman latent + MALA parameter."""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from causal_ssm_agent.models.ssm.inference.methods.map import (
    _build_map_laplace_bundle,
    fit_map,
)
from causal_ssm_agent.models.ssm.inference.methods.scipy_pathfinder import (
    ScipyPathfinderResult,
    scipy_pathfinder,
)
from causal_ssm_agent.models.ssm.inference.shared import _filter_public_samples
from causal_ssm_agent.models.ssm.inference.trajectory_mcmc import (
    AuxGibbsMCMCResult,
    build_auxiliary_kalman_bundle,
    build_auxiliary_kalman_latent_kernel,
    build_mala_parameter_kernel,
    initialize_ieks_latents,
    initialize_particle_smoother_latents,
    run_aux_gibbs,
)
from causal_ssm_agent.models.ssm.inference.types import InferenceResult
from causal_ssm_agent.models.ssm.inference.utils import extract_constrained_samples

logger = logging.getLogger(__name__)


def _phase_elapsed(t0: float) -> float:
    return time.monotonic() - t0


# Sites the output-sensitivity check typically flags as having zero Jacobian at
# prior draws — Pathfinder's Gaussian approximation can't meaningfully
# initialise them, so they inherit the prior-median flat-layout value plus a
# small per-chain jitter. See Corenflos & Särkkä docstring block in the
# slow-test benchmark discussion for the literature pattern this implements.
_WEAKLY_IDENTIFIED_SITE_NAMES: tuple[str, ...] = ("obs_df",)


@dataclass(frozen=True)
class AuxGibbsPathfinderCache:
    """Reusable Pathfinder outputs for aux_gibbs initialisation."""

    initial_positions_override: jnp.ndarray
    parameter_preconditioner_chol: jnp.ndarray
    diagnostics: dict[str, Any]


@functools.partial(jax.jit, static_argnames=("runtime_log_posterior_fn",))
def _pathfinder_value_and_grad_runtime(
    z: jnp.ndarray,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    runtime_log_posterior_fn,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jax.value_and_grad(
        lambda z_arg: runtime_log_posterior_fn(
            z_arg,
            observations,
            times,
            latent_mode_init=None,
        )
    )(z)


def _flat_indices_for_sites(
    flat_example: jnp.ndarray,
    unravel_fn,
    site_names: tuple[str, ...],
) -> list[int]:
    """Return flat indices in the aux-Kalman layout that belong to the given sites."""
    dim = int(flat_example.shape[0])
    site_for_idx: list[str | None] = [None] * dim
    for idx in range(dim):
        onehot = np.zeros(dim, dtype=np.float64)
        onehot[idx] = 1.0
        unraveled = unravel_fn(jnp.asarray(onehot, dtype=flat_example.dtype))
        for name, value in unraveled.items():
            if np.any(np.abs(np.asarray(value)) > 1e-10):
                site_for_idx[idx] = name
                break
    return [idx for idx, name in enumerate(site_for_idx) if name is not None and name in site_names]


def _laplace_preconditioner_chol_from_map_result(
    map_result: InferenceResult, jitter: float = 1e-6
) -> jnp.ndarray:
    """Build a MALA preconditioner Cholesky from a ``fit_map`` Laplace cov."""
    covariance = np.asarray(map_result.diagnostics["parameter_covariance"])
    covariance = 0.5 * (covariance + covariance.T)
    covariance = covariance + jitter * np.eye(covariance.shape[0], dtype=covariance.dtype)
    return jnp.asarray(np.linalg.cholesky(covariance))


def _pathfinder_preconditioner_chol_from_state(
    pathfinder_state: ScipyPathfinderResult, jitter: float = 1e-6
) -> jnp.ndarray:
    """Return the MALA preconditioner Cholesky carried by a scipy Pathfinder result.

    ``scipy_pathfinder`` already computes a PSD lower-triangular Cholesky of
    the L-BFGS quasi-Newton inverse-Hessian at the best-ELBO iterate (with a
    jitter addend for numerical safety). Re-symmetrising here would require
    reconstructing the full covariance and re-factorising; instead we take
    the chol directly and cast to the sampler dtype.
    """
    del jitter  # jitter already applied inside scipy_pathfinder
    return jnp.asarray(pathfinder_state.chol)


def _run_pathfinder_approximation(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    trace_key: jnp.ndarray,
    pathfinder_key: jnp.ndarray,
    reparam,
    n_ieks_iters: int,
    num_elbo_samples: int,
    maxiter: int,
    n_pathfinder_starts: int = 2,
    pathfinder_parallel_workers: int | None = None,
    init_scale: float = 0.1,
) -> tuple[ScipyPathfinderResult, dict[str, Any]]:
    """Multi-start scipy-driven Pathfinder; returns best-ELBO Gaussian + diagnostics.

    Uses :func:`scipy_pathfinder` under the hood. Compared to
    ``blackjax.vi.pathfinder`` the whole L-BFGS optimisation + per-iterate
    ELBO evaluation happens in Python/numpy; only the log-posterior + grad
    is jit-compiled (same compile cost as ``fit_map``). Avoids blackjax's
    ``lax.while_loop``-driven graph explosion that took ~1700 s to compile
    and 25 GiB HBM at T=1000 on A100.

    Start positions are ``n_pathfinder_starts`` independent Gaussian
    perturbations of the bundle's ``flat_example`` (typically the prior
    median), scaled by ``init_scale``. The RNG seed is derived from
    ``pathfinder_key`` so repeat runs are reproducible.
    """
    if n_pathfinder_starts < 1:
        raise ValueError("n_pathfinder_starts must be >= 1.")
    backend = (
        model.make_likelihood_backend()
        if model.likelihood == "kalman"
        else model.make_laplace_backend(n_ieks_iters)
    )
    laplace_bundle = _build_map_laplace_bundle(
        model, observations, times, trace_key, backend, reparam
    )
    runtime_log_post_fn = laplace_bundle["log_posterior_fn"]
    flat_example = laplace_bundle["flat_example"]
    flat_dtype = flat_example.dtype

    def log_post_and_grad_np(x: np.ndarray) -> tuple[float, np.ndarray]:
        xj = jnp.asarray(x, dtype=flat_dtype)
        fx, gx = _pathfinder_value_and_grad_runtime(
            xj,
            observations,
            times,
            runtime_log_posterior_fn=runtime_log_post_fn,
        )
        return float(fx), np.asarray(jax.device_get(gx), dtype=np.float64)

    # Derive a reproducible int seed from the JAX PRNG key.
    seed_int = int(jax.device_get(random.randint(pathfinder_key, (), 0, 2**31 - 1)))
    rng = np.random.default_rng(seed_int)
    base_np = np.asarray(jax.device_get(flat_example), dtype=np.float64)
    x0_starts = [
        base_np + float(init_scale) * rng.standard_normal(base_np.shape)
        for _ in range(int(n_pathfinder_starts))
    ]

    result = scipy_pathfinder(
        log_post_and_grad_np,
        x0_starts,
        maxiter=int(maxiter),
        elbo_samples=int(num_elbo_samples),
        seed=seed_int,
        parallel_workers=pathfinder_parallel_workers,
    )
    diagnostics = {
        "n_pathfinder_starts": int(n_pathfinder_starts),
        "n_pathfinder_starts_finite": int(result.diagnostics["n_starts_finite"]),
        "pathfinder_parallel_workers": int(result.diagnostics["parallel_workers"]),
        "best_pathfinder_elbo": float(result.best_elbo),
        "pathfinder_elbo": float(result.best_elbo),  # backwards-compat
        "pathfinder_elbo_min": float(result.diagnostics["elbo_min"]),
        "pathfinder_elbo_max": float(result.diagnostics["elbo_max"]),
        "pathfinder_elbo_spread": float(result.diagnostics["elbo_spread"]),
        "pathfinder_maxiter": int(result.diagnostics["maxiter"]),
        "pathfinder_lbfgs_memory": int(result.diagnostics["lbfgs_memory"]),
        "pathfinder_elbo_samples": int(result.diagnostics["elbo_samples"]),
        "pathfinder_per_start": result.diagnostics["per_start"],
    }
    return result, diagnostics


def _pathfinder_init_positions(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    trace_key: jnp.ndarray,
    pathfinder_key: jnp.ndarray,
    sample_key: jnp.ndarray,
    reparam,
    n_ieks_iters: int,
    num_chains: int,
    num_elbo_samples: int,
    maxiter: int,
    dtype,
    n_pathfinder_starts: int = 2,
    pathfinder_parallel_workers: int | None = None,
    pathfinder_init_scale: float | None = None,
    aux_bundle: dict[str, Any] | None = None,
    weakly_identified_sites: tuple[str, ...] = _WEAKLY_IDENTIFIED_SITE_NAMES,
    prior_release_scale: float = 0.05,
    release_jitter_key: jnp.ndarray | None = None,
    best_state: Any | None = None,
    pathfinder_diagnostics: dict[str, Any] | None = None,
) -> tuple[jnp.ndarray, dict[str, Any], Any]:
    """Run Pathfinder on the IEKS-marginal log-posterior for theta.

    Returns ``(init_positions_per_chain, diagnostics, best_state)``. The
    laplace bundle uses the same ``_discover_sites`` + ``ravel_pytree`` layout
    as ``build_auxiliary_kalman_bundle``, so the flat positions are directly
    consumable by :func:`run_aux_gibbs`.

    When ``n_pathfinder_starts > 1`` this runs K independent Pathfinder
    approximations, picks the one with the highest ELBO, and samples the
    ``num_chains`` initial positions from that top-ELBO approximation.

    ``pathfinder_init_scale``:
        * ``None`` (default) — sample ``num_chains`` positions from the fitted
          scipy Pathfinder Gaussian: ``x = mean + chol @ zeta``.
        * ``float`` — take Pathfinder's best-ELBO ``mean`` as the common centre
          and perturb each chain with ``pathfinder_init_scale * randn``.
          Works around occasional over-wide Gaussian covariance that scatters
          chains into distant basins on ill-conditioned posteriors.

    Per-parameter init: when ``aux_bundle`` is provided, flat indices that
    belong to ``weakly_identified_sites`` are overridden with the prior-median
    value (``aux_bundle["flat_example"]``) plus ``prior_release_scale * randn``
    per chain. This is the literature-standard pattern for sites the output
    sensitivity check flags as having zero Jacobian at prior draws (e.g.
    Student-t ``obs_df``): Pathfinder's Gaussian approximation can't
    meaningfully initialise them, so they start at the prior mode and let the
    parameter-MALA kernel explore from there.
    """
    if best_state is None or pathfinder_diagnostics is None:
        best_state, pathfinder_diagnostics = _run_pathfinder_approximation(
            model,
            observations,
            times,
            trace_key=trace_key,
            pathfinder_key=pathfinder_key,
            reparam=reparam,
            n_ieks_iters=n_ieks_iters,
            num_elbo_samples=num_elbo_samples,
            maxiter=maxiter,
            n_pathfinder_starts=n_pathfinder_starts,
            pathfinder_parallel_workers=pathfinder_parallel_workers,
        )
    mean_np = np.asarray(best_state.mean, dtype=np.float64)
    chol_np = np.asarray(best_state.chol, dtype=np.float64)
    p_dim = mean_np.shape[0]
    zeta = random.normal(sample_key, (num_chains, p_dim), dtype=dtype)
    zeta_np = np.asarray(jax.device_get(zeta), dtype=np.float64)
    if pathfinder_init_scale is None:
        # Draw from Pathfinder's fitted Gaussian: x = mean + chol @ zeta.
        positions_np = mean_np[None, :] + zeta_np @ chol_np.T
        sampling_mode = "pathfinder_gaussian"
    else:
        # Scaled isotropic perturbation around the mode — useful when the
        # fitted covariance is suspiciously wide.
        positions_np = mean_np[None, :] + float(pathfinder_init_scale) * zeta_np
        sampling_mode = "mode_plus_scaled_normal"
    positions = jnp.asarray(positions_np, dtype=dtype)
    if not bool(jax.device_get(jnp.all(jnp.isfinite(positions)))):
        raise RuntimeError("Pathfinder returned non-finite chain-init positions for aux_gibbs.")

    prior_site_indices: list[int] = []
    if aux_bundle is not None and weakly_identified_sites:
        prior_site_indices = _flat_indices_for_sites(
            aux_bundle["flat_example"],
            aux_bundle["unravel_fn"],
            weakly_identified_sites,
        )
        if prior_site_indices:
            flat_example = jnp.asarray(aux_bundle["flat_example"], dtype=dtype)
            dim = int(flat_example.shape[0])
            mask = np.zeros(dim, dtype=bool)
            for idx in prior_site_indices:
                mask[idx] = True
            mask_j = jnp.asarray(mask)
            jitter_key = (
                release_jitter_key
                if release_jitter_key is not None
                else random.fold_in(sample_key, 0xC0FFEE)
            )
            noise = random.normal(jitter_key, (num_chains, dim), dtype=dtype)
            prior_values = flat_example[None, :] + float(prior_release_scale) * noise
            positions = jnp.where(mask_j[None, :], prior_values, positions)

    diag = {
        "init_method": "pathfinder",
        "pathfinder_sampling_mode": sampling_mode,
        "pathfinder_init_scale": pathfinder_init_scale,
        **pathfinder_diagnostics,
        "prior_released_site_names": list(weakly_identified_sites) if prior_site_indices else [],
        "prior_released_site_indices": prior_site_indices,
        "prior_release_scale": float(prior_release_scale) if prior_site_indices else 0.0,
    }
    return positions, diag, best_state


def build_aux_gibbs_pathfinder_cache(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_chains: int,
    seed: int,
    reparam,
    n_ieks_iters: int = 6,
    pathfinder_num_elbo_samples: int = 20,
    pathfinder_maxiter: int = 20,
    n_pathfinder_starts: int = 2,
    pathfinder_parallel_workers: int | None = None,
    pathfinder_init_scale: float | None = None,
    prior_release_scale: float = 0.05,
) -> AuxGibbsPathfinderCache:
    """Build reusable aux_gibbs Pathfinder init positions and MALA preconditioner."""
    base_key = random.PRNGKey(seed)
    trace_key, pathfinder_key, pf_sample_key, release_key = random.split(base_key, 4)
    bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=trace_key,
        reparam=reparam,
    )
    pathfinder_state, pathfinder_diagnostics = _run_pathfinder_approximation(
        model,
        observations,
        times,
        trace_key=trace_key,
        pathfinder_key=pathfinder_key,
        reparam=reparam,
        n_ieks_iters=n_ieks_iters,
        num_elbo_samples=pathfinder_num_elbo_samples,
        maxiter=pathfinder_maxiter,
        n_pathfinder_starts=n_pathfinder_starts,
        pathfinder_parallel_workers=pathfinder_parallel_workers,
    )
    init_positions, init_diagnostics, pathfinder_state = _pathfinder_init_positions(
        model,
        observations,
        times,
        trace_key=trace_key,
        pathfinder_key=pathfinder_key,
        sample_key=pf_sample_key,
        reparam=reparam,
        n_ieks_iters=n_ieks_iters,
        num_chains=num_chains,
        num_elbo_samples=pathfinder_num_elbo_samples,
        maxiter=pathfinder_maxiter,
        dtype=bundle["flat_example"].dtype,
        n_pathfinder_starts=n_pathfinder_starts,
        pathfinder_parallel_workers=pathfinder_parallel_workers,
        pathfinder_init_scale=pathfinder_init_scale,
        aux_bundle=bundle,
        prior_release_scale=prior_release_scale,
        release_jitter_key=release_key,
        best_state=pathfinder_state,
        pathfinder_diagnostics=pathfinder_diagnostics,
    )
    preconditioner_chol = _pathfinder_preconditioner_chol_from_state(pathfinder_state)
    return AuxGibbsPathfinderCache(
        initial_positions_override=init_positions,
        parameter_preconditioner_chol=preconditioner_chol,
        diagnostics={
            **init_diagnostics,
            "auto_preconditioner": True,
            "auto_preconditioner_method": "pathfinder",
            "auto_preconditioner_device": jax.default_backend(),
            "auto_preconditioner_n_pathfinder_starts": int(
                pathfinder_diagnostics["n_pathfinder_starts"]
            ),
            "auto_preconditioner_n_pathfinder_starts_finite": int(
                pathfinder_diagnostics["n_pathfinder_starts_finite"]
            ),
            "auto_preconditioner_best_pathfinder_elbo": float(
                pathfinder_diagnostics["best_pathfinder_elbo"]
            ),
            "auto_preconditioner_pathfinder_elbo_spread": float(
                pathfinder_diagnostics["pathfinder_elbo_spread"]
            ),
        },
    )


def fit_aux_gibbs(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_warmup: int = 2500,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    latent_kernel: str = "kalman",
    latent_proposal_family: str = "eq8",
    latent_delta: float = 1e-3,
    latent_target_accept: float = 0.5,
    parameter_kernel: str = "mala",
    param_step_size: float = 0.02,
    param_target_accept: float = 0.57,
    adaptation_rate: float = 0.05,
    init_scale: float = 0.05,
    retain_latent_paths: bool = False,
    compute_latent_posterior_summary: bool = True,
    adaptation_scheme: str = "dual_averaging",
    init_method: str = "pathfinder",
    latent_init_method: str = "ieks",
    latent_init_num_particles: int = 64,
    latent_init_guidance: str = "bffg",
    n_ieks_iters: int = 6,
    pathfinder_num_elbo_samples: int = 20,
    pathfinder_maxiter: int = 20,
    n_pathfinder_starts: int = 2,
    pathfinder_parallel_workers: int | None = None,
    pathfinder_init_scale: float | None = None,
    parallel_filter: bool = True,
    latent_delta_profile: str = "scalar",
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    auto_preconditioner_maxiter: int = 200,
    auto_preconditioner_method: str = "pathfinder",
    initial_positions_override: jnp.ndarray | None = None,
    emit_per_t_log_alpha: bool = False,
    reparam=None,
    **_kwargs,
) -> InferenceResult:
    """Fit an SSM with blocked aux-Kalman/MALA MCMC.

    ``parallel_filter`` toggles the Corenflos/Särkkä O(log T) associative
    Kalman filter and RTS sampler used inside the auxiliary-Kalman latent
    step. Turning it off falls back to a plain O(T) sequential ``lax.scan``
    filter (identical predict/update math) — useful for benchmarking or for
    very short trajectories where the per-step constant of the associative
    scan is larger than the gain from log-depth parallelism.

    ``latent_delta_profile`` selects how the latent step size δ is distributed
    across time steps, addressing the heterogeneously-informative-observation
    pathology described in Corenflos & Särkkä §4.4. Supported values:

    * ``"scalar"`` — a single global δ adapted by ``adaptation_scheme`` (the
      shipping default; closest to the plain eq-8 sampler).
    * ``"T_minus_one_third"`` — a single scalar δ fixed at
      ``latent_delta * T**(-1/3)`` and held frozen (Remark 3.1's worst-case
      MALA rate bound; expects ``adaptation_scheme="simple"`` +
      ``adaptation_rate=0`` to actually hold the scale).
    * ``"informativeness"`` — a per-time-step δ_t ∝ 1 / n_observed_t, rescaled
      so the mean matches ``latent_delta``. Slots with many observed channels
      get a smaller δ_t; slots with none (missing / between-anchor interval
      summaries) get a larger one, so the global accept probability is no
      longer dominated by the most informative time step.

    ``latent_proposal_family`` selects the latent auxiliary-Kalman proposal:

    * ``"eq8"`` — the reparametrised auxiliary variable used by the original
      implementation in this repo, with TULAc-tamed observation gradients.
    * ``"eq10_11"`` — the non-reparametrised Corenflos & Särkkä (2025,
      eq. 10/11) auxiliary-variable/LGSSM construction, with the raw
      observation gradient replaced by the same TULAc-tamed gradient used in
      ``"eq8"``. This is a target-preserving proposal modification, not the
      literal raw-gradient Eq. 10/11 kernel.

    Auto-preconditioner: when ``parameter_preconditioner_chol`` is ``None``,
    ``auto_preconditioner_method`` selects how the MALA preconditioner is
    built. The default ``"pathfinder"`` reuses the same best-ELBO Pathfinder
    Gaussian approximation built for per-chain initialisation — so only one
    L-BFGS-style fit is driven, not two.

    * ``"pathfinder"`` (default) — reuse Pathfinder's fitted Gaussian
      approximation and pass its covariance Cholesky directly to the
      parameter-MALA kernel.
    * ``"map"`` — run a separate internal MAP+IEKS fit and use the
      L-BFGS-B inverse-Hessian approximation. ``auto_preconditioner_maxiter``
      controls the inner optimiser budget. Heavier than ``"pathfinder"``
      because it is a full second L-BFGS on the same objective; kept for
      callers that need exact Laplace covariance at the posterior mode.

    Provide a precomputed Cholesky to skip the auto-preconditioner step
    entirely.

    Per-parameter init (default ``init_method="pathfinder"``): Pathfinder's
    Gaussian approximation initialises well-identified flat indices; sites
    flagged as weakly-identified (e.g. Student-t ``obs_df``) instead inherit
    the prior-median value plus a small per-chain jitter. The literature-
    standard "MAP/Pathfinder for well-identified parameters, prior mean for
    variance/df parameters" pattern, applied at flat-index granularity.
    """
    if latent_kernel != "kalman":
        raise ValueError(
            f"Unsupported aux_gibbs latent kernel {latent_kernel!r}. Supported: 'kalman'."
        )
    if latent_proposal_family not in {"eq8", "eq10_11"}:
        raise ValueError(
            f"Unsupported aux_gibbs latent proposal family {latent_proposal_family!r}. "
            "Supported: 'eq8' or 'eq10_11'."
        )
    if parameter_kernel != "mala":
        raise ValueError(
            f"Unsupported aux_gibbs parameter kernel {parameter_kernel!r}. Supported: 'mala'."
        )
    if init_method not in {"random", "pathfinder"}:
        raise ValueError(
            f"Unsupported aux_gibbs init_method {init_method!r}. "
            "Supported: 'random' or 'pathfinder'."
        )
    if latent_init_method not in {"predictive", "particle_smoother", "ieks"}:
        raise ValueError(
            f"Unsupported aux_gibbs latent_init_method {latent_init_method!r}. "
            "Supported: 'predictive', 'particle_smoother', or 'ieks'."
        )
    if latent_init_guidance not in {"bootstrap", "bffg"}:
        raise ValueError(
            f"Unsupported aux_gibbs latent_init_guidance {latent_init_guidance!r}. "
            "Supported: 'bootstrap' or 'bffg'."
        )
    if latent_delta_profile not in {"scalar", "T_minus_one_third", "informativeness"}:
        raise ValueError(
            f"Unsupported latent_delta_profile {latent_delta_profile!r}. "
            "Supported: 'scalar', 'T_minus_one_third', 'informativeness'."
        )
    if auto_preconditioner_method not in {"map", "pathfinder"}:
        raise ValueError(
            f"Unsupported auto_preconditioner_method {auto_preconditioner_method!r}. "
            "Supported: 'map' or 'pathfinder'."
        )

    overall_t0 = time.monotonic()
    T_time_log = int(observations.shape[0])
    n_manifest_log = int(observations.shape[1]) if observations.ndim >= 2 else 0
    logger.info(
        "aux_gibbs entry: chains=%d warmup=%d samples=%d T=%d n_manifest=%d "
        "init_method=%s parallel_filter=%s latent_kernel=%s parameter_kernel=%s "
        "auto_preconditioner_method=%s",
        num_chains,
        num_warmup,
        num_samples,
        T_time_log,
        n_manifest_log,
        init_method,
        parallel_filter,
        latent_kernel,
        parameter_kernel,
        auto_preconditioner_method,
    )

    base_key = random.PRNGKey(seed)
    trace_key, pathfinder_key, pf_sample_key, release_key, latent_init_key = random.split(
        base_key, 5
    )

    phase_t0 = time.monotonic()
    logger.info("phase 1/6: building auxiliary Kalman bundle...")
    bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=trace_key,
        reparam=reparam,
    )
    logger.info(
        "phase 1/6: bundle ready in %.1fs (dim=%d, public_sites=%d)",
        _phase_elapsed(phase_t0),
        int(bundle["flat_example"].shape[0]),
        len(bundle.get("public_sites", [])),
    )

    init_positions = None
    init_diagnostics: dict[str, Any] = {"init_method": init_method}
    shared_pathfinder_state: Any | None = None
    shared_pathfinder_diagnostics: dict[str, Any] | None = None
    phase_t0 = time.monotonic()
    if initial_positions_override is not None:
        init_positions = jnp.asarray(initial_positions_override, dtype=bundle["flat_example"].dtype)
        if init_positions.shape != (num_chains, int(bundle["flat_example"].shape[0])):
            raise ValueError(
                "initial_positions_override must have shape (num_chains, dim); got "
                f"{init_positions.shape}"
            )
        init_diagnostics = {"init_method": "user_provided"}
        logger.info(
            "phase 2/6: init positions = user_provided override (%.1fs)",
            _phase_elapsed(phase_t0),
        )
    elif init_method == "pathfinder":
        logger.info(
            "phase 2/6: running scipy_pathfinder warmup (n_starts=%d, maxiter=%d, "
            "n_ieks_iters=%d, elbo_samples=%d, parallel_workers=%s)...",
            n_pathfinder_starts,
            pathfinder_maxiter,
            n_ieks_iters,
            pathfinder_num_elbo_samples,
            pathfinder_parallel_workers if pathfinder_parallel_workers is not None else "starts",
        )
        if parameter_preconditioner_chol is None and auto_preconditioner_method == "pathfinder":
            shared_pathfinder_state, shared_pathfinder_diagnostics = _run_pathfinder_approximation(
                model,
                observations,
                times,
                trace_key=trace_key,
                pathfinder_key=pathfinder_key,
                reparam=reparam,
                n_ieks_iters=n_ieks_iters,
                num_elbo_samples=pathfinder_num_elbo_samples,
                maxiter=pathfinder_maxiter,
                n_pathfinder_starts=n_pathfinder_starts,
                pathfinder_parallel_workers=pathfinder_parallel_workers,
            )
        init_positions, init_diagnostics, shared_pathfinder_state = _pathfinder_init_positions(
            model,
            observations,
            times,
            trace_key=trace_key,
            pathfinder_key=pathfinder_key,
            sample_key=pf_sample_key,
            reparam=reparam,
            n_ieks_iters=n_ieks_iters,
            num_chains=num_chains,
            num_elbo_samples=pathfinder_num_elbo_samples,
            maxiter=pathfinder_maxiter,
            dtype=bundle["flat_example"].dtype,
            n_pathfinder_starts=n_pathfinder_starts,
            pathfinder_parallel_workers=pathfinder_parallel_workers,
            pathfinder_init_scale=pathfinder_init_scale,
            aux_bundle=bundle,
            release_jitter_key=release_key,
            best_state=shared_pathfinder_state,
            pathfinder_diagnostics=shared_pathfinder_diagnostics,
        )
        best_elbo_log = (
            shared_pathfinder_diagnostics.get("best_pathfinder_elbo")
            if shared_pathfinder_diagnostics
            else None
        )
        elbo_spread_log = (
            shared_pathfinder_diagnostics.get("pathfinder_elbo_spread")
            if shared_pathfinder_diagnostics
            else None
        )
        logger.info(
            "phase 2/6: pathfinder init complete in %.1fs (best_elbo=%s, elbo_spread=%s, "
            "n_starts_finite=%s, sampling_mode=%s)",
            _phase_elapsed(phase_t0),
            f"{best_elbo_log:.2f}" if isinstance(best_elbo_log, (int, float)) else "n/a",
            f"{elbo_spread_log:.2f}" if isinstance(elbo_spread_log, (int, float)) else "n/a",
            (
                shared_pathfinder_diagnostics.get("n_pathfinder_starts_finite")
                if shared_pathfinder_diagnostics
                else "n/a"
            ),
            init_diagnostics.get("init_sampling_mode") or init_diagnostics.get("init_method"),
        )
    else:
        logger.info(
            "phase 2/6: init_method=%s — skipped pathfinder warmup (%.1fs)",
            init_method,
            _phase_elapsed(phase_t0),
        )

    initial_latent_trajectories = None
    if latent_init_method in {"particle_smoother", "ieks"}:
        phase_t0 = time.monotonic()
        if init_positions is None:
            init_keys = random.split(pf_sample_key, num_chains)
            init_noise = jax.vmap(
                lambda key: random.normal(
                    key,
                    bundle["flat_example"].shape,
                    dtype=bundle["flat_example"].dtype,
                )
            )(init_keys)
            init_positions = bundle["flat_example"][None, ...] + init_scale * init_noise
            init_diagnostics = {
                **init_diagnostics,
                "init_method": "random",
                "random_init_scale": float(init_scale),
            }
        seed_int = int(jax.device_get(random.randint(latent_init_key, (), 0, 2**31 - 1)))
        if latent_init_method == "particle_smoother":
            logger.info(
                "phase 2b/6: particle-smoother latent init (particles=%d, guidance=%s)...",
                latent_init_num_particles,
                latent_init_guidance,
            )
            initial_latent_trajectories, latent_init_diagnostics = (
                initialize_particle_smoother_latents(
                    bundle,
                    init_positions,
                    seed=seed_int,
                    num_particles=latent_init_num_particles,
                    guidance=latent_init_guidance,
                )
            )
        else:
            logger.info(
                "phase 2b/6: IEKS latent init (n_ieks_iters=%d)...",
                n_ieks_iters,
            )
            initial_latent_trajectories, latent_init_diagnostics = initialize_ieks_latents(
                bundle,
                init_positions,
                model=model,
                seed=seed_int,
                n_ieks_iters=n_ieks_iters,
                reparam=reparam,
                trace_key=trace_key,
            )
        init_diagnostics = {**init_diagnostics, **latent_init_diagnostics}
        logger.info(
            "phase 2b/6: %s latent init complete in %.1fs",
            latent_init_method,
            _phase_elapsed(phase_t0),
        )
    else:
        init_diagnostics = {
            **init_diagnostics,
            "latent_init_method": "predictive",
        }

    # Auto-build the Laplace preconditioner when the caller did not provide
    # one. The MAP path remains the default; the Pathfinder path reuses the
    # same best-state Gaussian approximation used for initialisation.
    preconditioner_diagnostics: dict[str, Any] = {}
    phase_t0 = time.monotonic()
    if parameter_preconditioner_chol is None:
        logger.info(
            "phase 3/6: building MALA preconditioner via %s...",
            auto_preconditioner_method,
        )
        if auto_preconditioner_method == "pathfinder":
            if shared_pathfinder_state is None or shared_pathfinder_diagnostics is None:
                shared_pathfinder_state, shared_pathfinder_diagnostics = (
                    _run_pathfinder_approximation(
                        model,
                        observations,
                        times,
                        trace_key=trace_key,
                        pathfinder_key=pathfinder_key,
                        reparam=reparam,
                        n_ieks_iters=n_ieks_iters,
                        num_elbo_samples=pathfinder_num_elbo_samples,
                        maxiter=pathfinder_maxiter,
                        n_pathfinder_starts=n_pathfinder_starts,
                        pathfinder_parallel_workers=pathfinder_parallel_workers,
                    )
                )
            parameter_preconditioner_chol = _pathfinder_preconditioner_chol_from_state(
                shared_pathfinder_state
            )
            parameter_preconditioner_chol = jax.device_put(parameter_preconditioner_chol)
            preconditioner_diagnostics = {
                "auto_preconditioner": True,
                "auto_preconditioner_method": "pathfinder",
                "auto_preconditioner_device": jax.default_backend(),
                "auto_preconditioner_n_pathfinder_starts": int(
                    shared_pathfinder_diagnostics["n_pathfinder_starts"]
                ),
                "auto_preconditioner_n_pathfinder_starts_finite": int(
                    shared_pathfinder_diagnostics["n_pathfinder_starts_finite"]
                ),
                "auto_preconditioner_best_pathfinder_elbo": float(
                    shared_pathfinder_diagnostics["best_pathfinder_elbo"]
                ),
                "auto_preconditioner_pathfinder_elbo_spread": float(
                    shared_pathfinder_diagnostics["pathfinder_elbo_spread"]
                ),
            }
        else:
            map_result = fit_map(
                model,
                observations,
                times,
                num_samples=1,
                seed=seed,
                n_ieks_iters=n_ieks_iters,
                maxiter=auto_preconditioner_maxiter,
                parameter_covariance_method="optimizer_hess_inv",
                reparam=reparam,
            )
            parameter_preconditioner_chol = _laplace_preconditioner_chol_from_map_result(map_result)
            preconditioner_diagnostics = {
                "auto_preconditioner": True,
                "auto_preconditioner_method": "map",
                "auto_preconditioner_maxiter": int(auto_preconditioner_maxiter),
            }
        logger.info(
            "phase 3/6: preconditioner ready in %.1fs (method=%s)",
            _phase_elapsed(phase_t0),
            auto_preconditioner_method,
        )
    else:
        preconditioner_diagnostics = {"auto_preconditioner": False}
        logger.info(
            "phase 3/6: preconditioner = caller-provided (skipped, %.1fs)",
            _phase_elapsed(phase_t0),
        )

    # Resolve the δ profile. "scalar" keeps a single global δ; the other two
    # variants freeze per-time step sizes — for them, the simple exponential
    # adapter at rate 0 is the "don't touch it" path.
    effective_delta = latent_delta
    delta_profile: jnp.ndarray | None = None
    T_time = int(observations.shape[0])
    if latent_delta_profile == "T_minus_one_third":
        effective_delta = float(latent_delta) * float(T_time) ** (-1.0 / 3.0)
    elif latent_delta_profile == "informativeness":
        obs_count = jnp.sum(~jnp.isnan(observations), axis=tuple(range(1, observations.ndim)))
        info_t = jnp.maximum(obs_count.astype(jnp.float32), 1.0)
        raw_weights = 1.0 / info_t
        normalised = raw_weights / jnp.mean(raw_weights)
        delta_profile = float(latent_delta) * normalised
    phase_t0 = time.monotonic()
    logger.info(
        "phase 4/6: building latent + parameter kernel specs (parallel_filter=%s, "
        "latent_proposal_family=%s, latent_delta_profile=%s, effective_delta=%.4g, "
        "param_step_size=%.4g)...",
        parallel_filter,
        latent_proposal_family,
        latent_delta_profile,
        float(effective_delta),
        float(param_step_size),
    )
    latent_kernel_spec = build_auxiliary_kalman_latent_kernel(
        bundle,
        delta=effective_delta,
        target_accept=latent_target_accept,
        proposal_family=latent_proposal_family,
        parallel=parallel_filter,
        delta_profile=delta_profile,
        emit_per_t_log_alpha=emit_per_t_log_alpha,
    )
    parameter_kernel_spec = build_mala_parameter_kernel(
        bundle,
        step_size=param_step_size,
        target_accept=param_target_accept,
        preconditioner_chol=parameter_preconditioner_chol,
    )
    logger.info("phase 4/6: kernel specs ready in %.1fs", _phase_elapsed(phase_t0))

    phase_t0 = time.monotonic()
    logger.info(
        "phase 5/6: starting MCMC kernel — first call triggers JAX JIT compile of the "
        "parallel-Kalman scan + MALA step (this can take 1-5 min on a fresh container) "
        "before any sampling iteration begins...",
    )
    run_result = run_aux_gibbs(
        bundle,
        latent_kernel=latent_kernel_spec,
        parameter_kernel=parameter_kernel_spec,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        adaptation_rate=adaptation_rate,
        init_scale=init_scale,
        retain_latent_paths=retain_latent_paths,
        adaptation_scheme=adaptation_scheme,
        init_positions=init_positions,
        initial_latent_trajectories=initial_latent_trajectories,
        emit_per_t_log_alpha=emit_per_t_log_alpha,
        compute_latent_posterior_summary=compute_latent_posterior_summary,
    )
    latent_acc = float(jnp.mean(run_result["chain_extra_fields"]["latent_accept_prob"]))
    param_acc = float(jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"]))

    def _scalar_or_summary(value: Any) -> str:
        # Latent / parameter step sizes can be scalar OR per-time-step / per-chain
        # arrays depending on the adaptation scheme. Render either flavor compactly.
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            try:
                arr = np.asarray(value, dtype=float)
                return f"mean={float(arr.mean()):.4g}, shape={tuple(arr.shape)}"
            except (TypeError, ValueError):
                return repr(value)[:64]

    logger.info(
        "phase 5/6: MCMC kernel complete in %.1fs (latent_acc=%.3f, param_acc=%.3f, "
        "final_latent_delta=%s, final_param_step_size=%s)",
        _phase_elapsed(phase_t0),
        latent_acc,
        param_acc,
        _scalar_or_summary(run_result["final_latent_delta"]),
        _scalar_or_summary(run_result["final_param_step_size"]),
    )

    phase_t0 = time.monotonic()
    logger.info("phase 6/6: extracting + filtering public posterior samples...")
    flat_particles = run_result["grouped_positions"].reshape((-1, bundle["dim"]))
    constrained_samples = extract_constrained_samples(
        flat_particles,
        bundle["site_info"],
        bundle["unravel_fn"],
        model.spec,
        reparam=reparam,
        model=model,
        observations=observations,
        times=times,
    )
    public_samples = _filter_public_samples(constrained_samples, bundle["public_sites"])
    grouped_public_samples = {
        name: values.reshape((num_chains, num_samples, *values.shape[1:]))
        for name, values in public_samples.items()
    }
    mcmc = AuxGibbsMCMCResult(
        chain_samples=grouped_public_samples,
        chain_extra_fields=run_result["chain_extra_fields"],
        num_chains=num_chains,
        num_samples=num_samples,
    )
    diagnostics = {
        "mcmc": mcmc,
        "public_sites": sorted(bundle["public_sites"]),
        "likelihood_backend": model.make_likelihood_backend(),
        "aux_gibbs": {
            "latent_kernel": latent_kernel,
            "latent_proposal_family": latent_proposal_family,
            "parameter_kernel": parameter_kernel,
            "adaptation_scheme": adaptation_scheme,
            "parallel_filter": parallel_filter,
            "latent_delta_profile": latent_delta_profile,
            "latent_accept_rate": float(
                jnp.mean(run_result["chain_extra_fields"]["latent_accept_prob"])
            ),
            "parameter_accept_rate": float(
                jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"])
            ),
            "final_latent_delta": run_result["final_latent_delta"],
            "final_param_step_size": run_result["final_param_step_size"],
            "chain_post_warmup_complete_log_posterior_mean": jax.device_get(
                run_result["post_warmup_complete_log_posterior_mean"]
            ).tolist(),
            **init_diagnostics,
            **preconditioner_diagnostics,
        },
        "latent_posterior_summary": run_result["latent_posterior_summary"],
        "chain_complete_log_posterior_history": run_result["complete_log_posterior_history"],
    }
    if run_result["latent_paths"] is not None:
        diagnostics["latent_paths"] = run_result["latent_paths"]

    logger.info(
        "phase 6/6: posterior extraction complete in %.1fs (n_public_sites=%d, "
        "draws_per_chain=%d). aux_gibbs total: %.1fs",
        _phase_elapsed(phase_t0),
        len(grouped_public_samples),
        num_samples,
        _phase_elapsed(overall_t0),
    )

    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="aux_gibbs",
        diagnostics=diagnostics,
    )
