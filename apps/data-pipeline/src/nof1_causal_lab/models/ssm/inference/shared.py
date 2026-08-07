"""Shared inference helpers that do not own concrete backend implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from numpyro import handlers

from nof1_causal_lab.models.ssm.constants import INTERNAL_DIAGNOSTIC_SITES

if TYPE_CHECKING:
    import jax.numpy as jnp

    from nof1_causal_lab.models.ssm.inference.types import InferenceMethod
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

logger = logging.getLogger(__name__)


def select_default_method(
    spec: SSMSpec,
    observation_support: ObservationSupportRuntime | None = None,
    n_timepoints: int | None = None,
) -> InferenceMethod:
    """Select the default inference method based on model structure."""
    from nof1_causal_lab.models.ssm.execution.planning import plan_inference_structure

    inference_structure = plan_inference_structure(
        spec,
        observation_support=observation_support,
        n_timepoints=n_timepoints,
    )
    logger.info(
        "Inference routing: resolved_method=%s structural_backend=%s",
        inference_structure.resolved_method,
        inference_structure.structural_backend,
    )
    return inference_structure.resolved_method


def _trace_public_sites(
    model_fn,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    exclude: set[str] | None = None,
) -> set[str]:
    """Trace a model once and return user-facing sample/deterministic site names."""
    excluded = set(INTERNAL_DIAGNOSTIC_SITES)
    if exclude is not None:
        excluded.update(exclude)

    with handlers.seed(rng_seed=0):
        trace = handlers.trace(model_fn).get_trace(observations, times)

    return {
        name
        for name, site in trace.items()
        if site["type"] in ("sample", "deterministic")
        and not site.get("is_observed", False)
        and name not in excluded
    }


def _filter_public_samples(
    samples: dict[str, jnp.ndarray], public_sites: set[str]
) -> dict[str, jnp.ndarray]:
    """Drop internal handler sites, keeping only original model outputs."""
    return {name: values for name, values in samples.items() if name in public_sites}
