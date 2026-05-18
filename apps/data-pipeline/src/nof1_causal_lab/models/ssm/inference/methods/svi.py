"""SVI (Stochastic Variational Inference) backend for SSM models."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import jax.random as random
from jax import tree_util
from numpyro.infer import SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoNormal
from numpyro.optim import ClippedAdam

from nof1_causal_lab.models.ssm.inference.shared import (
    _apply_reparam,
    _filter_public_samples,
    _trace_public_sites,
)
from nof1_causal_lab.models.ssm.inference.types import InferenceResult

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMModel


def _all_numeric_leaves_finite(tree: Any) -> bool:
    """Return ``True`` when every numeric leaf in a pytree is finite."""
    for leaf in tree_util.tree_leaves(tree):
        arr = jnp.asarray(leaf)
        if arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
            continue
        if not bool(jnp.all(jnp.isfinite(arr))):
            return False
    return True


def fit_svi(
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
