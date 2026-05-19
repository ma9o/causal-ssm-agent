"""Inference backends for SSM models.

Separates inference from model definition. SSMModel defines the probabilistic
model; this module provides fit() to run inference with the supported backends.

Available methods:
- PIT Particle-mGRAD: blocked complete-data updates with PIT dSMC Particle-mGRAD
  latent proposals.
- Auxiliary Kalman MCMC: blocked complete-data updates with auxiliary Kalman latent proposals.
- MAP: L-BFGS mode finding + Laplace Gaussian parameter posterior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from numpyro import handlers

from nof1_causal_lab.models.ssm.autoreparam import AutoReparam
from nof1_causal_lab.models.ssm.inference.shared import (
    select_default_method as select_default_method,
)
from nof1_causal_lab.models.ssm.inference.types import (
    FittedArtifact as FittedArtifact,
)
from nof1_causal_lab.models.ssm.inference.types import (  # noqa: TC001
    InferenceMethod,
    InferenceResult,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMModel

# Sentinel for "use AutoReparam with method-appropriate centering".
_AUTO_REPARAM = object()


def _resolve_reparam(reparam, method: InferenceMethod):
    """Resolve _AUTO_REPARAM sentinel to a concrete AutoReparam config."""
    del method
    if reparam is not _AUTO_REPARAM:
        return reparam
    return AutoReparam(centered=0.0)  # fully decentered


def fit(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    method: InferenceMethod = "aux_kalman_mcmc",
    reparam=_AUTO_REPARAM,
    **kwargs: Any,
) -> InferenceResult:
    """Fit an SSM using the specified inference method.

    Args:
        model: SSMModel instance defining the probabilistic model
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        method: Inference method: "pit_particle_mgrad", "aux_kalman_mcmc", or "map".
        reparam: Reparameterization config. Can be:
            - ``_AUTO_REPARAM`` (default): Uses ``AutoReparam`` with method-appropriate
              centering.
            - A ``Strategy`` instance (e.g., ``AutoReparam(centered=0.0)``)
            - A dict mapping site names to ``Reparam`` instances
            - None: no reparameterization
        **kwargs: Method-specific arguments

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    reparam = _resolve_reparam(reparam, method)
    if method == "pit_particle_mgrad":
        from nof1_causal_lab.models.ssm.inference.methods.pit_particle_mgrad import (
            fit_pit_particle_mgrad,
        )

        return fit_pit_particle_mgrad(model, observations, times, reparam=reparam, **kwargs)
    if method == "aux_kalman_mcmc":
        from nof1_causal_lab.models.ssm.inference.methods.aux_kalman_mcmc import fit_aux_kalman_mcmc

        return fit_aux_kalman_mcmc(model, observations, times, reparam=reparam, **kwargs)
    if method == "map":
        from nof1_causal_lab.models.ssm.inference.methods.map import fit_map

        return fit_map(model, observations, times, reparam=reparam, **kwargs)
    raise ValueError(
        f"Unknown inference method: {method!r}. Use 'pit_particle_mgrad', 'aux_kalman_mcmc', or 'map'."
    )


def prior_predictive(
    model: SSMModel,
    times: jnp.ndarray,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample from the prior predictive distribution.

    Args:
        model: SSMModel instance
        times: (T,) time points
        num_samples: Number of prior samples
        seed: Random seed

    Returns:
        Dict of prior predictive samples
    """
    from nof1_causal_lab.models.ssm.prior_predictive_runtime import (
        sample_prior_predictive_from_runtime,
    )

    return sample_prior_predictive_from_runtime(
        model.spec,
        model.get_prior_runtime_bundle(),
        times,
        num_samples=num_samples,
        seed=seed,
    )


def _eval_model(
    model_fn,
    params_dict: dict[str, jnp.ndarray],
    observations: jnp.ndarray,
    times: jnp.ndarray,
) -> tuple[Any, Any]:
    """Evaluate model with substituted params. Returns (log_likelihood, log_prior).

    Uses numpyro.handlers to substitute parameter values and trace the model,
    computing log_prior + log_likelihood without any code duplication.

    Args:
        model_fn: NumPyro model function
        params_dict: Parameter values to substitute
        observations: Observed data
        times: Time points

    Returns:
        Tuple of (log_likelihood, log_prior)
    """
    with handlers.seed(rng_seed=0), handlers.substitute(data=params_dict):
        trace = handlers.trace(model_fn).get_trace(observations, times)

    log_lik = 0.0
    log_prior = 0.0
    for name, site in trace.items():
        if site["type"] == "sample":
            if name == "log_likelihood":
                # Factor site: fn is Unit with log_factor attribute
                log_lik = site["fn"].log_factor
            elif not site.get("is_observed", False):
                log_prior = log_prior + jnp.sum(site["fn"].log_prob(site["value"]))

    return log_lik, log_prior
