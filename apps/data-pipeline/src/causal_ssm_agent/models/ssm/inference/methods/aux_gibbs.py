"""Auxiliary Gibbs sampler: blocked aux-Kalman latent + MALA parameter."""

from __future__ import annotations

from typing import Any

import blackjax.vi.pathfinder as pathfinder
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from causal_ssm_agent.models.ssm.inference.methods.map import (
    _build_laplace_em_bundle,
    fit_map,
)
from causal_ssm_agent.models.ssm.inference.shared import _filter_public_samples
from causal_ssm_agent.models.ssm.inference.trajectory_mcmc import (
    AuxGibbsMCMCResult,
    build_auxiliary_kalman_bundle,
    build_auxiliary_kalman_latent_kernel,
    build_mala_parameter_kernel,
    run_aux_gibbs,
)
from causal_ssm_agent.models.ssm.inference.types import InferenceResult
from causal_ssm_agent.models.ssm.inference.utils import extract_constrained_samples

# Sites the output-sensitivity check typically flags as having zero Jacobian at
# prior draws — Pathfinder's Gaussian approximation can't meaningfully
# initialise them, so they inherit the prior-median flat-layout value plus a
# small per-chain jitter. See Corenflos & Särkkä docstring block in the
# slow-test benchmark discussion for the literature pattern this implements.
_WEAKLY_IDENTIFIED_SITE_NAMES: tuple[str, ...] = ("obs_df",)


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
    pathfinder_state: Any,
    *,
    low_rank_scale: float = 1.0,
    jitter: float = 1e-6,
) -> jnp.ndarray:
    """Build a MALA preconditioner Cholesky from a Pathfinder Gaussian."""
    alpha = np.asarray(pathfinder_state.alpha, dtype=np.float64)
    beta = np.asarray(pathfinder_state.beta, dtype=np.float64)
    gamma = np.asarray(pathfinder_state.gamma, dtype=np.float64)
    covariance = np.diag(alpha)
    if beta.size:
        covariance = covariance + float(low_rank_scale) * (beta @ gamma @ beta.T)
    covariance = 0.5 * (covariance + covariance.T)
    covariance = covariance + jitter * np.eye(covariance.shape[0], dtype=covariance.dtype)
    return jnp.asarray(np.linalg.cholesky(covariance))


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
) -> tuple[Any, dict[str, Any]]:
    """Run multi-start Pathfinder and return the highest-ELBO approximation."""
    if n_pathfinder_starts < 1:
        raise ValueError("n_pathfinder_starts must be >= 1.")
    backend = (
        model.make_likelihood_backend()
        if model.likelihood == "kalman"
        else model.make_laplace_backend(n_ieks_iters)
    )
    laplace_bundle = _build_laplace_em_bundle(
        model, observations, times, trace_key, backend, reparam
    )
    start_keys = random.split(pathfinder_key, n_pathfinder_starts)
    states: list[Any] = []
    elbos: list[float] = []
    for start_key in start_keys:
        state_k, _ = pathfinder.approximate(
            start_key,
            laplace_bundle["log_posterior_fn"],
            laplace_bundle["flat_example"],
            num_samples=num_elbo_samples,
            maxiter=maxiter,
        )
        elbo_k = float(jax.device_get(state_k.elbo))
        if not bool(jax.device_get(jnp.all(jnp.isfinite(state_k.position)))):
            continue
        if not jnp.isfinite(elbo_k):
            continue
        states.append(state_k)
        elbos.append(elbo_k)
    if not states:
        raise RuntimeError(
            "All pathfinder starts produced non-finite ELBO or position; "
            "cannot build the aux_gibbs Pathfinder approximation."
        )
    best_idx = int(max(range(len(elbos)), key=lambda i: elbos[i]))
    diagnostics = {
        "n_pathfinder_starts": n_pathfinder_starts,
        "n_pathfinder_starts_finite": len(states),
        "best_pathfinder_elbo": elbos[best_idx],
        "pathfinder_elbo": elbos[best_idx],  # backwards-compat with single-start key
        "pathfinder_elbo_min": min(elbos),
        "pathfinder_elbo_max": max(elbos),
        "pathfinder_elbo_spread": max(elbos) - min(elbos),
        "pathfinder_elbos": elbos,
    }
    return states[best_idx], diagnostics


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
        * ``None`` (default) — sample ``num_chains`` positions from Pathfinder's
          own Gaussian approximation via :func:`blackjax.pathfinder.sample`.
        * ``float`` — take Pathfinder's mode ``state.position`` as the common
          centre and perturb each chain with ``pathfinder_init_scale * randn``.
          Works around Pathfinder's occasional over-wide Gaussian covariance
          that scatters chains into distant basins on ill-conditioned
          posteriors.

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
        )
    if pathfinder_init_scale is None:
        positions, _log_q = pathfinder.sample(sample_key, best_state, num_samples=num_chains)
        sampling_mode = "pathfinder_gaussian"
    else:
        noise = random.normal(sample_key, (num_chains, best_state.position.shape[0]), dtype=dtype)
        positions = best_state.position[None, :] + float(pathfinder_init_scale) * noise
        sampling_mode = "mode_plus_scaled_normal"
    positions = jnp.asarray(positions, dtype=dtype)
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
    adaptation_scheme: str = "dual_averaging",
    init_method: str = "pathfinder",
    n_ieks_iters: int = 6,
    pathfinder_num_elbo_samples: int = 20,
    pathfinder_maxiter: int = 20,
    n_pathfinder_starts: int = 2,
    pathfinder_init_scale: float | None = None,
    pathfinder_preconditioner_low_rank_scale: float = 1.0,
    parallel_filter: bool = True,
    latent_delta_profile: str = "scalar",
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    auto_preconditioner_maxiter: int = 200,
    auto_preconditioner_method: str = "map",
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
      implementation in this repo.
    * ``"eq10_11"`` — the standard non-reparametrised auxiliary proposal from
      Corenflos & Särkkä (2025, eq. 10/11).

    Auto-preconditioner: when ``parameter_preconditioner_chol`` is ``None``,
    ``auto_preconditioner_method`` selects how the MALA preconditioner is
    built:

    * ``"map"`` — run the existing internal MAP+IEKS fit and use the
      L-BFGS-B inverse-Hessian approximation. ``auto_preconditioner_maxiter``
      controls the inner optimiser budget.
    * ``"pathfinder"`` — reuse Pathfinder's fitted Gaussian approximation
      and pass its covariance Cholesky directly to the parameter-MALA kernel.
      ``pathfinder_preconditioner_low_rank_scale`` scales Pathfinder's
      low-rank covariance correction ``beta @ gamma @ beta.T`` before the
      Cholesky is formed; values in ``[0, 1]`` shrink toward Pathfinder's
      diagonal approximation.

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
    if not 0.0 <= float(pathfinder_preconditioner_low_rank_scale) <= 1.0:
        raise ValueError(
            "pathfinder_preconditioner_low_rank_scale must be in [0, 1]; got "
            f"{pathfinder_preconditioner_low_rank_scale!r}."
        )

    base_key = random.PRNGKey(seed)
    trace_key, pathfinder_key, pf_sample_key, release_key = random.split(base_key, 4)
    bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=trace_key,
        reparam=reparam,
    )

    init_positions = None
    init_diagnostics: dict[str, Any] = {"init_method": init_method}
    shared_pathfinder_state: Any | None = None
    shared_pathfinder_diagnostics: dict[str, Any] | None = None
    if initial_positions_override is not None:
        init_positions = jnp.asarray(initial_positions_override, dtype=bundle["flat_example"].dtype)
        if init_positions.shape != (num_chains, int(bundle["flat_example"].shape[0])):
            raise ValueError(
                "initial_positions_override must have shape (num_chains, dim); got "
                f"{init_positions.shape}"
            )
        init_diagnostics = {"init_method": "user_provided"}
    elif init_method == "pathfinder":
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
            pathfinder_init_scale=pathfinder_init_scale,
            aux_bundle=bundle,
            release_jitter_key=release_key,
            best_state=shared_pathfinder_state,
            pathfinder_diagnostics=shared_pathfinder_diagnostics,
        )

    # Auto-build the Laplace preconditioner when the caller did not provide
    # one. The MAP path remains the default; the Pathfinder path reuses the
    # same best-state Gaussian approximation used for initialisation.
    preconditioner_diagnostics: dict[str, Any] = {}
    if parameter_preconditioner_chol is None:
        if auto_preconditioner_method == "pathfinder":
            if shared_pathfinder_state is None or shared_pathfinder_diagnostics is None:
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
                )
            parameter_preconditioner_chol = _pathfinder_preconditioner_chol_from_state(
                shared_pathfinder_state,
                low_rank_scale=pathfinder_preconditioner_low_rank_scale,
            )
            parameter_preconditioner_chol = jax.device_put(parameter_preconditioner_chol)
            preconditioner_diagnostics = {
                "auto_preconditioner": True,
                "auto_preconditioner_method": "pathfinder",
                "auto_preconditioner_device": jax.default_backend(),
                "pathfinder_preconditioner_low_rank_scale": float(
                    pathfinder_preconditioner_low_rank_scale
                ),
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
            cpu_device = jax.devices("cpu")[0]
            with jax.default_device(cpu_device):
                map_result = fit_map(
                    model,
                    jax.device_put(observations, cpu_device),
                    jax.device_put(times, cpu_device),
                    num_samples=1,
                    seed=seed,
                    n_ieks_iters=n_ieks_iters,
                    maxiter=auto_preconditioner_maxiter,
                    parameter_covariance_method="optimizer_hess_inv",
                    reparam=reparam,
                )
            parameter_preconditioner_chol = _laplace_preconditioner_chol_from_map_result(
                map_result
            )
            # Place the Cholesky on the default (GPU, when present) device so the
            # MALA kernel's preconditioner·grad and preconditioner·noise multiplies
            # stay accelerator-local in the hot loop.
            parameter_preconditioner_chol = jax.device_put(parameter_preconditioner_chol)
            preconditioner_diagnostics = {
                "auto_preconditioner": True,
                "auto_preconditioner_method": "map",
                "auto_preconditioner_maxiter": int(auto_preconditioner_maxiter),
                "auto_preconditioner_device": "cpu",
            }
    else:
        preconditioner_diagnostics = {"auto_preconditioner": False}

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
        emit_per_t_log_alpha=emit_per_t_log_alpha,
    )
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

    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="aux_gibbs",
        diagnostics=diagnostics,
    )
