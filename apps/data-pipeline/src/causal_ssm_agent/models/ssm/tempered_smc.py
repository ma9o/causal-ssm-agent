"""Tempered SMC with preconditioned HMC/MALA mutations (fit_tempered_smc).

Thin wrapper around run_tempered_smc() in tempered_core.py.
See tempered_core.py for algorithmic details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from causal_ssm_agent.models.ssm.tempered_core import run_tempered_smc

if TYPE_CHECKING:
    import jax.numpy as jnp

    from causal_ssm_agent.models.ssm.inference import InferenceResult


def fit_tempered_smc(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    n_outer: int = 100,
    n_csmc_particles: int = 20,
    n_mh_steps: int = 10,
    param_step_size: float = 0.1,
    n_warmup: int | None = None,
    target_accept: float | None = None,
    seed: int = 0,
    adaptive_tempering: bool = True,
    target_ess_ratio: float = 0.5,
    waste_free: bool = False,
    n_leapfrog: int = 5,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM parameters via tempered SMC with preconditioned HMC mutations."""
    return run_tempered_smc(
        model,
        observations,
        times,
        n_outer=n_outer,
        n_csmc_particles=n_csmc_particles,
        n_mh_steps=n_mh_steps,
        param_step_size=param_step_size,
        n_warmup=n_warmup,
        target_accept=target_accept,
        seed=seed,
        adaptive_tempering=adaptive_tempering,
        target_ess_ratio=target_ess_ratio,
        waste_free=waste_free,
        n_leapfrog=n_leapfrog,
        method_name="tempered_smc",
        likelihood_backend=model.make_likelihood_backend(),
        print_prefix="Tempered SMC",
    )
