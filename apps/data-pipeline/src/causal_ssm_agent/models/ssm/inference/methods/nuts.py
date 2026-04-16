"""NUTS (HMC) inference backend for SSM models."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import blackjax.vi.pathfinder as pathfinder
import jax
import jax.random as random
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value

from causal_ssm_agent.models.ssm.inference.methods.map import _build_laplace_em_bundle
from causal_ssm_agent.models.ssm.inference.shared import (
    _apply_reparam,
    _filter_public_samples,
    _trace_public_sites,
)
from causal_ssm_agent.models.ssm.inference.types import InferenceResult

if TYPE_CHECKING:
    import jax.numpy as jnp

    from causal_ssm_agent.models.ssm.model import SSMModel


_PATHFINDER_NUM_ELBO_SAMPLES = 25
_PATHFINDER_MAXITER = 15


def _build_pathfinder_init_strategy(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    likelihood_backend,
    reparam,
    trace_key,
    pathfinder_key,
) -> tuple[Any, dict[str, Any]]:
    """Build a constrained NumPyro init strategy from a Pathfinder mode."""
    bundle = _build_laplace_em_bundle(
        model,
        observations,
        times,
        trace_key,
        likelihood_backend,
        reparam,
    )
    state, _ = pathfinder.approximate(
        pathfinder_key,
        bundle["log_posterior_fn"],
        bundle["flat_example"],
        num_samples=_PATHFINDER_NUM_ELBO_SAMPLES,
        maxiter=_PATHFINDER_MAXITER,
    )
    position = state.position.astype(bundle["flat_example"].dtype)
    if not bool(jax.device_get(jax.numpy.all(jax.numpy.isfinite(position)))):
        raise RuntimeError("Pathfinder returned a non-finite initialization vector")
    if not bool(jax.device_get(jax.numpy.isfinite(state.elbo))):
        raise RuntimeError("Pathfinder returned a non-finite ELBO")

    unconstrained = bundle["unravel_fn"](position)
    init_values = {
        name: bundle["site_info"][name]["transform"](unconstrained[name])
        for name in bundle["site_info"]
    }
    return init_to_value(values=init_values), {
        "init_method": "pathfinder",
        "pathfinder_elbo": float(jax.device_get(state.elbo)),
    }


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
    init_diagnostics: dict[str, Any]
    rng_key = random.PRNGKey(seed)
    if model.likelihood == "kalman":
        backend = model.make_likelihood_backend()
        init_strategy = init_to_median(num_samples=15)
        init_diagnostics = {"init_method": "median"}
        mcmc_key = rng_key
    else:
        backend = model.make_laplace_backend(n_ieks_iters)
        trace_key, pathfinder_key, mcmc_key = random.split(rng_key, 3)
        init_strategy, init_diagnostics = _build_pathfinder_init_strategy(
            model,
            observations,
            times,
            likelihood_backend=backend,
            reparam=reparam,
            trace_key=trace_key,
            pathfinder_key=pathfinder_key,
        )

    base_model_fn = functools.partial(model.model, likelihood_backend=backend)
    public_sites = _trace_public_sites(base_model_fn, observations, times)
    model_fn = _apply_reparam(base_model_fn, reparam)
    kernel = NUTS(
        model_fn,
        init_strategy=init_strategy,
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
        chain_method="vectorized",
        jit_model_args=False,
        **kwargs,
    )

    mcmc.run(
        mcmc_key,
        observations,
        times,
        extra_fields=("diverging", "num_steps", "accept_prob", "energy"),
    )

    samples = _filter_public_samples(mcmc.get_samples(), public_sites)

    return InferenceResult(
        _samples=samples,
        method="nuts",
        diagnostics={
            "mcmc": mcmc,
            "public_sites": sorted(public_sites),
            **init_diagnostics,
        },
    )
