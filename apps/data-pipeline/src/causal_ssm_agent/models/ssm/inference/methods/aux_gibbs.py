"""Auxiliary Gibbs sampler: blocked eq-8 aux-Kalman latent + MALA parameter."""

from __future__ import annotations

from typing import Any

import blackjax.vi.pathfinder as pathfinder
import jax
import jax.numpy as jnp
import jax.random as random

from causal_ssm_agent.models.ssm.inference.methods.map import _build_laplace_em_bundle
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
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Run Pathfinder on the IEKS-marginal log-posterior for theta.

    Returns (init_positions_per_chain, diagnostics). The laplace bundle uses
    the same ``_discover_sites`` + ``ravel_pytree`` layout as
    ``build_auxiliary_kalman_bundle``, so the flat positions are directly
    consumable by :func:`run_aux_gibbs`.
    """
    backend = (
        model.make_likelihood_backend()
        if model.likelihood == "kalman"
        else model.make_laplace_backend(n_ieks_iters)
    )
    laplace_bundle = _build_laplace_em_bundle(
        model, observations, times, trace_key, backend, reparam
    )
    state, _ = pathfinder.approximate(
        pathfinder_key,
        laplace_bundle["log_posterior_fn"],
        laplace_bundle["flat_example"],
        num_samples=num_elbo_samples,
        maxiter=maxiter,
    )
    if not bool(jax.device_get(jnp.all(jnp.isfinite(state.position)))):
        raise RuntimeError("Pathfinder returned a non-finite mode for aux_gibbs init.")
    if not bool(jax.device_get(jnp.isfinite(state.elbo))):
        raise RuntimeError("Pathfinder returned a non-finite ELBO for aux_gibbs init.")
    positions, _log_q = pathfinder.sample(sample_key, state, num_samples=num_chains)
    positions = jnp.asarray(positions, dtype=dtype)
    if not bool(jax.device_get(jnp.all(jnp.isfinite(positions)))):
        raise RuntimeError("Pathfinder returned non-finite chain-init positions for aux_gibbs.")
    diag = {
        "init_method": "pathfinder",
        "pathfinder_elbo": float(jax.device_get(state.elbo)),
    }
    return positions, diag


def fit_aux_gibbs(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    latent_kernel: str = "kalman",
    latent_delta: float = 0.2,
    latent_target_accept: float = 0.5,
    parameter_kernel: str = "mala",
    param_step_size: float = 0.05,
    param_target_accept: float = 0.57,
    adaptation_rate: float = 0.05,
    init_scale: float = 0.05,
    retain_latent_paths: bool = False,
    adaptation_scheme: str = "dual_averaging",
    init_method: str = "pathfinder",
    n_ieks_iters: int = 6,
    pathfinder_num_elbo_samples: int = 20,
    pathfinder_maxiter: int = 20,
    reparam=None,
    **_kwargs,
) -> InferenceResult:
    """Fit an SSM with blocked aux-Kalman/MALA MCMC (eq-8 reparametrisation)."""
    if latent_kernel != "kalman":
        raise ValueError(
            f"Unsupported aux_gibbs latent kernel {latent_kernel!r}. Supported: 'kalman'."
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

    base_key = random.PRNGKey(seed)
    trace_key, pathfinder_key, pf_sample_key = random.split(base_key, 3)
    bundle = build_auxiliary_kalman_bundle(
        model,
        observations,
        times,
        trace_key=trace_key,
        reparam=reparam,
    )
    latent_kernel_spec = build_auxiliary_kalman_latent_kernel(
        bundle,
        delta=latent_delta,
        target_accept=latent_target_accept,
    )
    parameter_kernel_spec = build_mala_parameter_kernel(
        bundle,
        step_size=param_step_size,
        target_accept=param_target_accept,
    )
    init_positions = None
    init_diagnostics: dict[str, Any] = {"init_method": init_method}
    if init_method == "pathfinder":
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
            "parameter_kernel": parameter_kernel,
            "adaptation_scheme": adaptation_scheme,
            "latent_accept_rate": float(
                jnp.mean(run_result["chain_extra_fields"]["latent_accept_prob"])
            ),
            "parameter_accept_rate": float(
                jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"])
            ),
            "final_latent_delta": run_result["final_latent_delta"],
            "final_param_step_size": run_result["final_param_step_size"],
            **init_diagnostics,
        },
        "latent_posterior_summary": run_result["latent_posterior_summary"],
    }
    if run_result["latent_paths"] is not None:
        diagnostics["latent_paths"] = run_result["latent_paths"]

    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="aux_gibbs",
        diagnostics=diagnostics,
    )
