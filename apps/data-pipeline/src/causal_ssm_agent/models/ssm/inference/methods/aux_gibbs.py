"""Auxiliary Gibbs sampler: blocked eq-8 aux-Kalman latent + MALA parameter."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as random

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

    trace_key = random.PRNGKey(seed)
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
            "latent_accept_rate": float(
                jnp.mean(run_result["chain_extra_fields"]["latent_accept_prob"])
            ),
            "parameter_accept_rate": float(
                jnp.mean(run_result["chain_extra_fields"]["parameter_accept_prob"])
            ),
            "final_latent_delta": run_result["final_latent_delta"],
            "final_param_step_size": run_result["final_param_step_size"],
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
