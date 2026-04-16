"""NUTS (HMC) inference backend for SSM models."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax.random as random
from numpyro.infer import MCMC, NUTS, init_to_median

from causal_ssm_agent.models.ssm.inference.shared import (
    _apply_reparam,
    _filter_public_samples,
    _trace_public_sites,
)
from causal_ssm_agent.models.ssm.inference.types import InferenceResult

if TYPE_CHECKING:
    import jax.numpy as jnp

    from causal_ssm_agent.models.ssm.model import SSMModel


def fit_nuts(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    dense_mass: bool = False,
    target_accept_prob: float = 0.85,
    max_tree_depth: int = 8,
    n_ieks_iters: int = 5,
    reparam=None,
    **kwargs: Any,
) -> InferenceResult:
    """Fit using NUTS (HMC).

    For Kalman-eligible models (all Gaussian + identity link), uses the exact
    Kalman marginal likelihood. For non-Gaussian models, uses the IEKS/Laplace
    approximate marginal likelihood — the IEKS marginalizes latent states,
    then NUTS samples the parameter posterior.

    Args:
        model: SSMModel instance
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        num_warmup: Number of warmup samples
        num_samples: Number of posterior samples
        num_chains: Number of MCMC chains
        seed: Random seed
        dense_mass: Use dense mass matrix
        target_accept_prob: Target acceptance probability
        max_tree_depth: Max tree depth
        n_ieks_iters: IEKS Newton iterations for Laplace backend (non-Gaussian only)
        reparam: Optional reparameterization config (Strategy, dict, or None)
        **kwargs: Additional MCMC arguments

    Returns:
        InferenceResult with NUTS samples
    """
    if model.likelihood == "kalman":
        backend = model.make_likelihood_backend()
    else:
        backend = model.make_laplace_backend(n_ieks_iters)

    base_model_fn = functools.partial(model.model, likelihood_backend=backend)
    public_sites = _trace_public_sites(base_model_fn, observations, times)
    model_fn = _apply_reparam(base_model_fn, reparam)
    kernel = NUTS(
        model_fn,
        init_strategy=init_to_median(num_samples=15),
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
        regularize_mass_matrix=True,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        jit_model_args=False,
        **kwargs,
    )

    rng_key = random.PRNGKey(seed)
    mcmc.run(
        rng_key,
        observations,
        times,
        extra_fields=("diverging", "num_steps", "accept_prob", "energy"),
    )

    samples = _filter_public_samples(mcmc.get_samples(), public_sites)

    return InferenceResult(
        _samples=samples,
        method="nuts",
        diagnostics={"mcmc": mcmc, "public_sites": sorted(public_sites)},
    )
