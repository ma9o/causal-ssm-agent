"""Inference backends for SSM models.

Separates inference from model definition. SSMModel defines the probabilistic
model; this module provides fit() to run inference with the supported backends.

Method:
- Marginalized Particle Gibbs: collapsed joint parameter/trajectory updates
  using posterior-mixture conditional SMC, with selectable latent smoother
  (plain CSMC, particle-aMALA, or particle-mGRAD).
- Particle marginal Metropolis-Hastings: parameter-space pseudo-marginal MH
  using the same discretized runtime bundle and bootstrap particle likelihood
  estimator.
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
from nof1_causal_lab.models.ssm.preflight import (
    validate_observations_for_fit as validate_observations_for_fit,
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
    method: InferenceMethod = "marginal_particle_gibbs",
    reparam=_AUTO_REPARAM,
    **kwargs: Any,
) -> InferenceResult:
    """Fit an SSM using the specified inference method.

    Args:
        model: SSMModel instance defining the probabilistic model
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        method: Inference method.
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
    validate_observations_for_fit(model, observations)
    reparam = _resolve_reparam(reparam, method)
    if method == "marginal_particle_gibbs":
        from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs import (
            fit_marginal_particle_gibbs,
        )

        return fit_marginal_particle_gibbs(model, observations, times, reparam=reparam, **kwargs)
    if method == "particle_marginal_mh":
        from nof1_causal_lab.models.ssm.inference.methods.particle_marginal_mh import (
            fit_particle_marginal_mh,
        )

        return fit_particle_marginal_mh(model, observations, times, reparam=reparam, **kwargs)
    raise ValueError(
        "Unknown inference method: "
        f"{method!r}. Use 'marginal_particle_gibbs' or 'particle_marginal_mh'."
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
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
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
