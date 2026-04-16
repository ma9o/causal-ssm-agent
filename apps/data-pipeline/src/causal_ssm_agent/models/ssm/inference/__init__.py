"""Inference backends for SSM models.

Separates inference from model definition. SSMModel defines the probabilistic
model; this module provides fit() to run inference with different backends.

Auto-routing (method="auto", the default) always selects NUTS. NUTS
auto-selects the state marginalization backend: exact Kalman filter for
linear Gaussian models, IEKS/Laplace for non-Gaussian emissions.

Available methods:
- NUTS: HMC-based sampling with Kalman or Laplace state marginalization.
- MAP: L-BFGS mode finding + Laplace Gaussian parameter posterior.
- SVI: Fast approximate posterior via ELBO optimization.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import jax.random as random
from jax import tree_util
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoNormal
from numpyro.optim import ClippedAdam

from causal_ssm_agent.models.ssm.autoreparam import AutoReparam
from causal_ssm_agent.models.ssm.inference.shared import (
    _filter_public_samples,
    _trace_public_sites,
)
from causal_ssm_agent.models.ssm.inference.shared import (
    select_default_method as select_default_method,
)
from causal_ssm_agent.models.ssm.inference.structure import plan_inference_structure
from causal_ssm_agent.models.ssm.inference.types import (
    FittedArtifact as FittedArtifact,
)
from causal_ssm_agent.models.ssm.inference.types import (
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


def _all_numeric_leaves_finite(tree: Any) -> bool:
    """Return ``True`` when every numeric leaf in a pytree is finite."""
    for leaf in tree_util.tree_leaves(tree):
        arr = jnp.asarray(leaf)
        if arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
            continue
        if not bool(jnp.all(jnp.isfinite(arr))):
            return False
    return True


def _apply_reparam(model_fn, reparam_config):
    """Wrap a model function with reparameterization if config is provided.

    Args:
        model_fn: A NumPyro model function.
        reparam_config: A dict, callable (Strategy), or None.

    Returns:
        The model function, possibly wrapped with handlers.reparam.
    """
    if reparam_config is None:
        return model_fn
    return handlers.reparam(model_fn, config=reparam_config)


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
            "nuts", "map", or "svi"
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
    if method == "nuts":
        return _fit_nuts(model, observations, times, reparam=reparam, **kwargs)
    if method == "svi":
        return _fit_svi(model, observations, times, reparam=reparam, **kwargs)
    if method == "map":
        from causal_ssm_agent.models.ssm.inference.methods.laplace_em import fit_laplace_em

        return fit_laplace_em(model, observations, times, reparam=reparam, **kwargs)
    raise ValueError(
        f"Unknown inference method: {method!r}. "
        "Use 'auto', 'nuts', 'map', or 'svi'."
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


def _fit_nuts(
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


def _fit_svi(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    guide_type: str = "mvn",
    num_steps: int = 5000,
    num_samples: int = 1000,
    learning_rate: float = 0.01,
    init_scale: float = 0.01,
    seed: int = 0,
    reparam=None,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit using Stochastic Variational Inference.

    Uses AutoGuide to learn an approximate posterior. numpyro.factor() sites
    are handled automatically - the guide only models latent sample sites.

    Args:
        model: SSMModel instance
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        guide_type: Guide family - "normal", "mvn", or "delta"
        num_steps: Number of SVI optimization steps
        num_samples: Number of posterior samples to draw after fitting
        learning_rate: Adam learning rate
        seed: Random seed
        reparam: Optional reparameterization config (Strategy, dict, or None)
        **kwargs: Ignored

    Returns:
        InferenceResult with approximate posterior samples
    """
    guide_cls = {
        "normal": AutoNormal,
        "mvn": AutoMultivariateNormal,
        "delta": AutoDelta,
    }[guide_type]
    base_model_fn = functools.partial(
        model.model,
        likelihood_backend=model.make_likelihood_backend(),
    )
    public_sites = _trace_public_sites(base_model_fn, observations, times)
    model_fn = _apply_reparam(base_model_fn, reparam)
    guide_kwargs = {"init_loc_fn": init_to_median(num_samples=15)}
    if guide_type != "delta":
        guide_kwargs["init_scale"] = init_scale
    guide = guide_cls(model_fn, **guide_kwargs)
    optimizer = ClippedAdam(step_size=learning_rate)
    svi = SVI(model_fn, guide, optimizer, Trace_ELBO())

    rng_key = random.PRNGKey(seed)
    rng_key, init_key, sample_key = random.split(rng_key, 3)
    svi_state = svi.init(init_key, observations, times)

    losses = []
    for step in range(num_steps):
        svi_state, loss = svi.update(
            svi_state,
            observations,
            times,
            forward_mode_differentiation=False,
        )
        if not bool(jnp.isfinite(loss)):
            raise FloatingPointError(f"SVI produced non-finite losses at step {step + 1}")
        losses.append(loss)

    svi_losses = (
        jnp.asarray(losses, dtype=jnp.float64) if losses else jnp.empty((0,), dtype=jnp.float64)
    )
    svi_params = svi.get_params(svi_state)

    if not _all_numeric_leaves_finite(svi_params):
        raise FloatingPointError("SVI produced non-finite guide parameters")

    # Draw posterior samples from the fitted guide
    predictive = Predictive(
        model_fn,
        guide=guide,
        params=svi_params,
        num_samples=num_samples,
    )
    raw_samples = predictive(sample_key, observations, times)

    samples = _filter_public_samples(raw_samples, public_sites)
    if not _all_numeric_leaves_finite(samples):
        raise FloatingPointError("SVI produced non-finite posterior samples")

    return InferenceResult(
        _samples=samples,
        method="svi",
        diagnostics={"losses": svi_losses, "params": svi_params},
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
