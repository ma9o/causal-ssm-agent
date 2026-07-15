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

from typing import TYPE_CHECKING, Unpack

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
    import jax.numpy as jnp

    from nof1_causal_lab.models.ssm.model import SSMModel
    from nof1_causal_lab.sampler_config import MarginalParticleGibbsOptions

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
    **kwargs: Unpack[MarginalParticleGibbsOptions],
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
    raise ValueError(f"Unknown inference method: {method!r}. Use 'marginal_particle_gibbs'.")


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
