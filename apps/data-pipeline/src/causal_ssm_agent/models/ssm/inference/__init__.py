"""Inference backends for SSM models.

Separates inference from model definition. SSMModel defines the probabilistic
model; this module provides fit() to run inference with different backends.

Auto-routing (method="auto", the default) always selects NUTS. NUTS
auto-selects the state marginalization backend: exact Kalman filter for
linear Gaussian models, IEKS/Laplace for non-Gaussian emissions.

Available methods:
- Auxiliary Gibbs: blocked complete-data updates with auxiliary Kalman latent proposals.
- NUTS: HMC-based sampling with Kalman or Laplace state marginalization.
- MAP: L-BFGS mode finding + Laplace Gaussian parameter posterior.
- SVI: Fast approximate posterior via ELBO optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from numpyro import handlers

from causal_ssm_agent.models.ssm.autoreparam import AutoReparam
from causal_ssm_agent.models.ssm.inference.shared import (
    _apply_reparam as _apply_reparam,
)
from causal_ssm_agent.models.ssm.inference.shared import (
    select_default_method as select_default_method,
)
from causal_ssm_agent.models.ssm.inference.structure import plan_inference_structure
from causal_ssm_agent.models.ssm.inference.types import (
    FittedArtifact as FittedArtifact,
)
from causal_ssm_agent.models.ssm.inference.types import (  # noqa: TC001
    InferenceMethod,
    InferenceResult,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMModel

# Sentinel for "use AutoReparam with method-appropriate centering".
_AUTO_REPARAM = object()

_AUTO_METHOD_CONFIG_KEYS: dict[str, str] = {
    "svi": "svi_config",
    "nuts": "nuts_config",
    "map": "smc_config",
}


def _resolve_reparam(reparam, method: InferenceMethod):
    """Resolve _AUTO_REPARAM sentinel to a concrete AutoReparam config."""
    if reparam is not _AUTO_REPARAM:
        return reparam
    # SVI benefits from learnable centering; all other methods use fixed NCP.
    if method == "svi":
        return AutoReparam()  # centered=None → learnable via numpyro.param
    return AutoReparam(centered=0.0)  # fully decentered


def _resolve_auto_method_kwargs(
    method: InferenceMethod,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge backend-specific config blocks emitted by ``method='auto'``."""
    nuts_config = kwargs.get("nuts_config") or {}
    smc_config = kwargs.get("smc_config") or {}
    resolved = {
        key: value
        for key, value in kwargs.items()
        if key not in {"svi_config", "nuts_config", "smc_config"}
    }
    if method == "nuts":
        # n_ieks_iters controls the Laplace backend for non-Gaussian models.
        n_ieks_iters = nuts_config.get("n_ieks_iters") or smc_config.get("n_ieks_iters")
        merged = {**nuts_config, **resolved}
        if n_ieks_iters is not None:
            merged.setdefault("n_ieks_iters", n_ieks_iters)
        return merged
    if method == "map":
        n_ieks_iters = smc_config.get("n_ieks_iters")
        if n_ieks_iters is not None:
            resolved["n_ieks_iters"] = n_ieks_iters
        return resolved
    if method == "svi":
        svi_config = kwargs.get("svi_config") or {}
        return {**svi_config, **resolved}
    config_key = _AUTO_METHOD_CONFIG_KEYS.get(method)
    if config_key is None:
        return resolved
    method_config = kwargs.get(config_key) or {}
    return {**method_config, **resolved}


def fit(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    method: InferenceMethod = "auto",
    reparam=_AUTO_REPARAM,
    **kwargs: Any,
) -> InferenceResult:
    """Fit an SSM using the specified inference method.

    Args:
        model: SSMModel instance defining the probabilistic model
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        method: Inference method - "auto" (always NUTS, default),
            "aux_gibbs", "nuts", "map", or "svi"
        reparam: Reparameterization config. Can be:
            - ``_AUTO_REPARAM`` (default): Uses ``AutoReparam`` with method-appropriate
              centering (learnable for SVI, fully decentered for MCMC/SMC).
            - A ``Strategy`` instance (e.g., ``AutoReparam(centered=0.0)``)
            - A dict mapping site names to ``Reparam`` instances
            - None: no reparameterization
        **kwargs: Method-specific arguments

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    if method == "auto":
        inference_structure = plan_inference_structure(
            model.spec,
            likelihood=model.likelihood,
            observation_support=getattr(model, "observation_support", None),
            n_timepoints=int(times.shape[0]),
        )
        method = inference_structure.resolved_method
        kwargs = _resolve_auto_method_kwargs(method, kwargs)

    reparam = _resolve_reparam(reparam, method)
    if method == "aux_gibbs":
        from causal_ssm_agent.models.ssm.inference.methods.aux_gibbs import fit_aux_gibbs

        return fit_aux_gibbs(model, observations, times, reparam=reparam, **kwargs)
    if method == "nuts":
        from causal_ssm_agent.models.ssm.inference.methods.nuts import fit_nuts

        return fit_nuts(model, observations, times, reparam=reparam, **kwargs)
    if method == "svi":
        from causal_ssm_agent.models.ssm.inference.methods.svi import fit_svi

        return fit_svi(model, observations, times, reparam=reparam, **kwargs)
    if method == "map":
        from causal_ssm_agent.models.ssm.inference.methods.map import fit_map

        return fit_map(model, observations, times, reparam=reparam, **kwargs)
    raise ValueError(
        f"Unknown inference method: {method!r}. Use 'auto', 'aux_gibbs', 'nuts', 'map', or 'svi'."
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
    from causal_ssm_agent.models.ssm.prior_predictive_runtime import (
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
