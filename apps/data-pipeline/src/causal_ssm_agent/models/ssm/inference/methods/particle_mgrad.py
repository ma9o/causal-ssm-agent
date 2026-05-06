"""Particle-mGRAD latent sampler with MALA parameter updates."""

from __future__ import annotations

from typing import Any

import blackjax.vi.pathfinder as pathfinder
import jax
import jax.numpy as jnp
import jax.random as random

from causal_ssm_agent.models.ssm.inference.methods.map import _build_map_laplace_bundle
from causal_ssm_agent.models.ssm.inference.shared import _filter_public_samples
from causal_ssm_agent.models.ssm.inference.trajectory_mcmc import (
    AuxGibbsMCMCResult,
    build_auxiliary_kalman_bundle,
    build_mala_parameter_kernel,
    build_particle_mgrad_latent_kernel,
    run_aux_gibbs,
)
from causal_ssm_agent.models.ssm.inference.types import InferenceResult
from causal_ssm_agent.models.ssm.inference.utils import extract_constrained_samples


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
    n_pathfinder_starts: int = 1,
    pathfinder_init_scale: float | None = None,
    method_label: str = "particle_mgrad",
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Run Pathfinder on the IEKS-marginal log-posterior for theta.

    See the aux_gibbs twin for the multi-start (``n_pathfinder_starts > 1``)
    semantics — same behaviour and diagnostics here.

    ``pathfinder_init_scale``:
        * ``None`` (default) — sample ``num_chains`` positions from Pathfinder's
          own Gaussian approximation via :func:`blackjax.pathfinder.sample`.
          This is the classical Pathfinder workflow; on ill-conditioned
          posteriors the Gaussian covariance can be dramatically wider than
          the target, scattering chains across distant basins.
        * ``float`` — take Pathfinder's mode ``state.position`` as the common
          centre and perturb each chain with ``pathfinder_init_scale * randn``.
          Works around the over-wide-Gaussian failure mode by initialising
          all chains in a controlled neighbourhood of the ELBO optimum.
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

    def _log_posterior_for_dataset(z):
        return laplace_bundle["log_posterior_fn"](
            z,
            observations,
            times,
            latent_mode_init=None,
        )

    start_keys = random.split(pathfinder_key, n_pathfinder_starts)
    states: list[Any] = []
    elbos: list[float] = []
    for start_key in start_keys:
        state_k, _ = pathfinder.approximate(
            start_key,
            _log_posterior_for_dataset,
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
            f"cannot seed {method_label} chains."
        )
    best_idx = int(max(range(len(elbos)), key=lambda i: elbos[i]))
    best_state = states[best_idx]
    if pathfinder_init_scale is None:
        positions, _log_q = pathfinder.sample(sample_key, best_state, num_samples=num_chains)
        sampling_mode = "pathfinder_gaussian"
    else:
        noise = random.normal(sample_key, (num_chains, best_state.position.shape[0]), dtype=dtype)
        positions = best_state.position[None, :] + float(pathfinder_init_scale) * noise
        sampling_mode = "mode_plus_scaled_normal"
    positions = jnp.asarray(positions, dtype=dtype)
    if not bool(jax.device_get(jnp.all(jnp.isfinite(positions)))):
        raise RuntimeError(
            f"Pathfinder returned non-finite chain-init positions for {method_label}."
        )
    diagnostics = {
        "init_method": "pathfinder",
        "pathfinder_sampling_mode": sampling_mode,
        "pathfinder_init_scale": pathfinder_init_scale,
        "n_pathfinder_starts": n_pathfinder_starts,
        "n_pathfinder_starts_finite": len(states),
        "best_pathfinder_elbo": elbos[best_idx],
        "pathfinder_elbo": elbos[best_idx],
        "pathfinder_elbo_min": min(elbos),
        "pathfinder_elbo_max": max(elbos),
        "pathfinder_elbo_spread": max(elbos) - min(elbos),
        "pathfinder_elbos": elbos,
    }
    return positions, diagnostics


def _fit_particle_latent_mcmc(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    method_name: str,
    diagnostics_key: str,
    num_particles: int,
    particle_count_diagnostic_key: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    latent_delta: float,
    latent_delta_min: float | None,
    latent_delta_max: float | None,
    latent_target_accept: float,
    backward_sampling: bool,
    parameter_preconditioner_chol: jnp.ndarray | None,
    parameter_kernel: str,
    param_step_size: float,
    param_target_accept: float,
    adaptation_rate: float,
    init_scale: float,
    retain_latent_paths: bool,
    compute_latent_posterior_summary: bool,
    adaptation_scheme: str,
    init_method: str,
    n_ieks_iters: int,
    pathfinder_num_elbo_samples: int,
    pathfinder_maxiter: int,
    n_pathfinder_starts: int,
    pathfinder_init_scale: float | None,
    initial_positions_override: jnp.ndarray | None,
    reparam,
) -> InferenceResult:
    if parameter_kernel != "mala":
        raise ValueError(
            f"Unsupported {method_name} parameter kernel {parameter_kernel!r}. Supported: 'mala'."
        )
    if init_method not in {"random", "pathfinder"}:
        raise ValueError(
            f"Unsupported {method_name} init_method {init_method!r}. "
            "Supported: 'random' or 'pathfinder'."
        )

    base_key = random.PRNGKey(seed)
    trace_key, pathfinder_key, pf_sample_key = random.split(base_key, 3)
    bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=trace_key,
        reparam=reparam,
    )
    latent_kernel_spec = build_particle_mgrad_latent_kernel(
        bundle,
        delta=latent_delta,
        target_accept=latent_target_accept,
        num_particles=num_particles,
        backward_sampling=backward_sampling,
        min_scale=latent_delta_min,
        max_scale=latent_delta_max,
    )

    parameter_kernel_spec = build_mala_parameter_kernel(
        bundle,
        step_size=param_step_size,
        target_accept=param_target_accept,
        preconditioner_chol=parameter_preconditioner_chol,
    )

    init_positions = None
    init_diagnostics: dict[str, Any] = {"init_method": init_method}
    if initial_positions_override is not None:
        init_positions = jnp.asarray(initial_positions_override, dtype=bundle["flat_example"].dtype)
        if init_positions.shape != (num_chains, int(bundle["flat_example"].shape[0])):
            raise ValueError(
                "initial_positions_override must have shape (num_chains, dim); got "
                f"{init_positions.shape}"
            )
        init_diagnostics = {"init_method": "user_provided"}
    elif init_method == "pathfinder":
        init_positions, init_diagnostics = _pathfinder_init_positions(
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
            method_label=method_name,
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
        compute_latent_posterior_summary=compute_latent_posterior_summary,
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
        backend=method_name,
    )
    kernel_diagnostics = {
        "latent_kernel": "particle_mgrad",
        "parameter_kernel": parameter_kernel,
        particle_count_diagnostic_key: num_particles,
        "backward_sampling": backward_sampling,
        "adaptation_scheme": adaptation_scheme,
        "latent_update_fraction": float(
            jnp.mean(run_result["chain_extra_fields"]["latent_accept_prob"])
        ),
        "parameter_accept_rate": float(
            jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"])
        ),
        "latent_delta_min": latent_delta_min,
        "latent_delta_max": latent_delta_max,
        "final_latent_delta": run_result["final_latent_delta"],
        "final_param_step_size": run_result["final_param_step_size"],
        "chain_post_warmup_complete_log_posterior_mean": jax.device_get(
            run_result["post_warmup_complete_log_posterior_mean"]
        ).tolist(),
        **init_diagnostics,
    }
    diagnostics = {
        "mcmc": mcmc,
        "public_sites": sorted(bundle["public_sites"]),
        "likelihood_backend": model.make_likelihood_backend(),
        diagnostics_key: kernel_diagnostics,
        "latent_posterior_summary": run_result["latent_posterior_summary"],
        "chain_complete_log_posterior_history": run_result["complete_log_posterior_history"],
    }
    if run_result["latent_paths"] is not None:
        diagnostics["latent_paths"] = run_result["latent_paths"]

    return InferenceResult(
        _samples=mcmc.get_samples(),
        method=method_name,
        diagnostics=diagnostics,
    )


def fit_particle_mgrad(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    latent_delta: float = 0.2,
    latent_delta_min: float | None = None,
    latent_delta_max: float | None = None,
    latent_target_accept: float = 0.5,
    n_particles: int = 25,
    backward_sampling: bool = True,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    parameter_kernel: str = "mala",
    param_step_size: float = 0.05,
    param_target_accept: float = 0.57,
    adaptation_rate: float = 0.05,
    init_scale: float = 0.05,
    retain_latent_paths: bool = False,
    compute_latent_posterior_summary: bool = True,
    adaptation_scheme: str = "simple",
    init_method: str = "random",
    n_ieks_iters: int = 6,
    pathfinder_num_elbo_samples: int = 20,
    pathfinder_maxiter: int = 20,
    n_pathfinder_starts: int = 1,
    pathfinder_init_scale: float | None = None,
    initial_positions_override: jnp.ndarray | None = None,
    reparam=None,
    **_kwargs,
) -> InferenceResult:
    """Fit an SSM with the Particle-mGRAD latent kernel and MALA parameter updates."""
    return _fit_particle_latent_mcmc(
        model,
        observations,
        times,
        method_name="particle_mgrad",
        diagnostics_key="particle_mgrad",
        num_particles=n_particles,
        particle_count_diagnostic_key="n_particles",
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        latent_delta=latent_delta,
        latent_delta_min=latent_delta_min,
        latent_delta_max=latent_delta_max,
        latent_target_accept=latent_target_accept,
        backward_sampling=backward_sampling,
        parameter_preconditioner_chol=parameter_preconditioner_chol,
        parameter_kernel=parameter_kernel,
        param_step_size=param_step_size,
        param_target_accept=param_target_accept,
        adaptation_rate=adaptation_rate,
        init_scale=init_scale,
        retain_latent_paths=retain_latent_paths,
        compute_latent_posterior_summary=compute_latent_posterior_summary,
        adaptation_scheme=adaptation_scheme,
        init_method=init_method,
        n_ieks_iters=n_ieks_iters,
        pathfinder_num_elbo_samples=pathfinder_num_elbo_samples,
        pathfinder_maxiter=pathfinder_maxiter,
        n_pathfinder_starts=n_pathfinder_starts,
        pathfinder_init_scale=pathfinder_init_scale,
        initial_positions_override=initial_positions_override,
        reparam=reparam,
    )
