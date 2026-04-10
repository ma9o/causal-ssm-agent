"""Cached runtime context for pre-fit parametric diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.inference.utils import _build_runtime_eval_fns_from_registry
from causal_ssm_agent.models.ssm.parameterization import (
    SiteRuntimeBundle,
    build_site_runtime_bundle,
)

from .observation_moments import (
    _predict_observation_moments,
    _predict_observation_row_scales,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from causal_ssm_agent.models.ssm.model import SSMModel, SSMSpec
    from causal_ssm_agent.models.ssm.structure_runtime import SSMStructureRuntime

logger = get_prefect_logger(__name__)

_PARAMETRIC_ID_CONTEXT_CACHE_MAXSIZE = 8


@dataclass(frozen=True)
class ParametricIdContext:
    """Reusable topology-dependent runtime state for parametric diagnostics."""

    cache_key: tuple[str, ...]
    spec: SSMSpec
    structure_runtime: SSMStructureRuntime
    site_runtime: SiteRuntimeBundle
    predict_moments_fn: Callable
    jacobian_fn: Callable
    row_scales_fn: Callable
    log_lik_fn: Callable
    log_prior_unc_fn: Callable

    @property
    def registry(self):
        return self.site_runtime.registry

    @property
    def transforms(self):
        return self.site_runtime.transforms

    @property
    def flat_dim(self):
        return self.site_runtime.flat_dim

    @property
    def unravel_fn(self):
        return self.site_runtime.unravel_fn

    @property
    def param_names(self):
        return self.site_runtime.param_names

    @property
    def site_shapes(self):
        return self.site_runtime.site_shapes

    @property
    def scalar_names(self):
        return self.site_runtime.scalar_names

    @property
    def param_index(self):
        return self.site_runtime.param_index


_PARAMETRIC_ID_CONTEXT_CACHE: OrderedDict[tuple[str, ...], ParametricIdContext] = OrderedDict()


def _normalize_cache_value(value: Any):
    """Convert spec/backend metadata into a stable JSON-serializable form."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jnp.ndarray):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, dict):
        return {
            str(key): _normalize_cache_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(item) for item in value]
    return repr(value)


def _parametric_id_context_key(model: SSMModel) -> tuple[str, ...]:
    """Build a process-local cache key for topology-stable diagnostic sweeps."""
    spec_payload = {
        field_name: _normalize_cache_value(field_value)
        for field_name, field_value in vars(model.spec).items()
    }
    spec_fingerprint = hashlib.sha256(
        json.dumps(spec_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    observation_support = getattr(model, "observation_support", None)
    support_payload = (
        None
        if observation_support is None
        else {
            field_name: _normalize_cache_value(field_value)
            for field_name, field_value in vars(observation_support).items()
        }
    )
    support_fingerprint = hashlib.sha256(
        json.dumps(support_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    pf_key = tuple(str(int(value)) for value in np.asarray(model.pf_key).reshape(-1))
    return (
        "parametric-id",
        spec_fingerprint,
        support_fingerprint,
        str(model.likelihood),
        str(model.n_particles),
        "reparam:none",
        *pf_key,
    )


def clear_stage4b_sweep_context_cache() -> None:
    """Clear the process-local diagnostic context cache."""
    _PARAMETRIC_ID_CONTEXT_CACHE.clear()


def get_stage4b_sweep_context(model: SSMModel) -> ParametricIdContext:
    """Build or reuse a topology-keyed diagnostic runtime context."""
    cache_key = _parametric_id_context_key(model)
    cached = _PARAMETRIC_ID_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        _PARAMETRIC_ID_CONTEXT_CACHE.move_to_end(cache_key)
        return cached

    structure_runtime = model.structure_runtime
    site_runtime = build_site_runtime_bundle(model.spec, structure_runtime)
    backend = model.make_likelihood_backend()
    log_lik_fn, log_prior_unc_fn = _build_runtime_eval_fns_from_registry(
        model.spec,
        site_runtime.registry,
        site_runtime.unravel_fn,
        site_runtime.transforms,
        structure_runtime,
        backend,
    )

    def _predict(z_flat, times):
        return _predict_observation_moments(
            z_flat,
            site_runtime.unravel_fn,
            site_runtime.transforms,
            model.spec,
            times,
            structure_runtime=structure_runtime,
            observation_support=getattr(model, "observation_support", None),
            registry=site_runtime.registry,
        )

    def _row_scales(z_flat, times):
        return _predict_observation_row_scales(
            z_flat,
            site_runtime.unravel_fn,
            site_runtime.transforms,
            model.spec,
            times,
            structure_runtime=structure_runtime,
            observation_support=getattr(model, "observation_support", None),
            registry=site_runtime.registry,
        )

    context = ParametricIdContext(
        cache_key=cache_key,
        spec=model.spec,
        structure_runtime=structure_runtime,
        site_runtime=site_runtime,
        predict_moments_fn=_predict,
        jacobian_fn=jax.jit(jax.jacfwd(_predict, argnums=0)),
        row_scales_fn=jax.jit(_row_scales),
        log_lik_fn=log_lik_fn,
        log_prior_unc_fn=log_prior_unc_fn,
    )
    _PARAMETRIC_ID_CONTEXT_CACHE[cache_key] = context
    _PARAMETRIC_ID_CONTEXT_CACHE.move_to_end(cache_key)
    while len(_PARAMETRIC_ID_CONTEXT_CACHE) > _PARAMETRIC_ID_CONTEXT_CACHE_MAXSIZE:
        _PARAMETRIC_ID_CONTEXT_CACHE.popitem(last=False)
    return context
