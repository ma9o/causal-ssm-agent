"""Shared parameter warmup wrapper for particle MCMC methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.ssm.inference.warmup.parameter_warmup import (
    DEFAULT_PRIOR_RELEASED_SITE_NAMES,
    prepare_parameter_warmup,
)

if TYPE_CHECKING:
    import jax.numpy as jnp

    from nof1_causal_lab.models.ssm.inference.bundle import ParticleRuntimeBundle


def prepare_pmcmc_parameter_warmup(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    bundle: ParticleRuntimeBundle,
    method_label: str,
    phase_label: str,
    trace_key: jnp.ndarray,
    pathfinder_key: jnp.ndarray,
    sample_key: jnp.ndarray,
    reparam,
    seed: int,
    n_ieks_iters: int,
    num_chains: int,
    init_method: str,
    initial_positions_override: jnp.ndarray | None,
    init_scale: float,
    parameter_preconditioner_chol: jnp.ndarray | None,
    auto_preconditioner_method: str,
    auto_preconditioner_maxiter: int,
    pathfinder_num_elbo_samples: int,
    pathfinder_maxiter: int,
    n_pathfinder_starts: int,
    pathfinder_parallel_workers: int | None,
    pathfinder_init_scale: float | None,
):
    return prepare_parameter_warmup(
        model,
        observations,
        times,
        bundle=bundle.cached,
        method_label=method_label,
        phase_label=phase_label,
        trace_key=trace_key,
        pathfinder_key=pathfinder_key,
        sample_key=sample_key,
        reparam=reparam,
        seed=seed,
        n_ieks_iters=n_ieks_iters,
        num_chains=num_chains,
        init_method=init_method,
        initial_positions_override=initial_positions_override,
        init_scale=init_scale,
        parameter_preconditioner_chol=parameter_preconditioner_chol,
        auto_preconditioner_method=auto_preconditioner_method,
        auto_preconditioner_maxiter=auto_preconditioner_maxiter,
        pathfinder_num_elbo_samples=pathfinder_num_elbo_samples,
        pathfinder_maxiter=pathfinder_maxiter,
        n_pathfinder_starts=n_pathfinder_starts,
        pathfinder_parallel_workers=pathfinder_parallel_workers,
        pathfinder_init_scale=pathfinder_init_scale,
        prior_released_sites=DEFAULT_PRIOR_RELEASED_SITE_NAMES,
    )
