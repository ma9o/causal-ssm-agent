"""Compilation and serialization helpers for executable SSM artifacts."""

from __future__ import annotations

from dataclasses import fields
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.models.ssm import SSMPriors, SSMSpec
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction, ModelSpec

if TYPE_CHECKING:
    import polars as pl

    from causal_ssm_agent.workers.schemas_prior import PriorProposal

CompiledSSMArtifact = dict[str, Any]

_SPEC_ARRAY_FIELDS = {
    "drift",
    "diffusion",
    "cint",
    "lambda_mat",
    "manifest_means",
    "manifest_var",
    "t0_means",
    "t0_var",
}
_SPEC_BOOL_ARRAY_FIELDS = {
    "drift_mask",
    "lambda_mask",
    "time_invariant_mask",
}
_SPEC_ENUM_FIELDS = {
    "diffusion_dist": DistributionFamily,
    "manifest_dist": DistributionFamily,
    "manifest_link": LinkFunction,
}
_SPEC_ENUM_LIST_FIELDS = {
    "diffusion_dists": DistributionFamily,
    "manifest_dists": DistributionFamily,
    "manifest_links": LinkFunction,
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def serialize_ssm_spec(spec: SSMSpec) -> dict[str, Any]:
    """Convert an SSMSpec into a JSON-serializable payload."""
    return {field.name: _to_jsonable(getattr(spec, field.name)) for field in fields(SSMSpec)}


def serialize_ssm_priors(priors: SSMPriors) -> dict[str, Any]:
    """Convert SSMPriors into a JSON-serializable payload."""
    return {field.name: _to_jsonable(getattr(priors, field.name)) for field in fields(SSMPriors)}


def deserialize_ssm_spec(payload: dict[str, Any]) -> SSMSpec:
    """Restore an SSMSpec from a serialized artifact."""
    kwargs: dict[str, Any] = {}

    for key, value in payload.items():
        if key in _SPEC_ARRAY_FIELDS and isinstance(value, list):
            kwargs[key] = jnp.asarray(value, dtype=jnp.float32)
        elif key in _SPEC_BOOL_ARRAY_FIELDS and value is not None:
            kwargs[key] = np.asarray(value, dtype=bool)
        elif key in _SPEC_ENUM_FIELDS and value is not None:
            kwargs[key] = _SPEC_ENUM_FIELDS[key](value)
        elif key in _SPEC_ENUM_LIST_FIELDS and value is not None:
            enum_cls = _SPEC_ENUM_LIST_FIELDS[key]
            kwargs[key] = [enum_cls(item) for item in value]
        else:
            kwargs[key] = value

    return SSMSpec(**kwargs)


def deserialize_ssm_priors(payload: dict[str, Any]) -> SSMPriors:
    """Restore SSMPriors from a serialized artifact."""
    return SSMPriors(**payload)


def trial_compile_model_spec(
    model_spec: ModelSpec | dict,
    causal_spec: dict | None = None,
) -> str | None:
    """Try compiling a ModelSpec with default priors to catch structural errors early.

    Returns None on success, or an error message string on failure.
    """
    from causal_ssm_agent.orchestrator.schemas_model import ModelSpec as ModelSpecCls
    from causal_ssm_agent.orchestrator.schemas_model import ParameterSpec
    from causal_ssm_agent.workers.prior_research import get_default_prior

    spec_obj = (
        ModelSpecCls.model_validate(model_spec) if isinstance(model_spec, dict) else model_spec
    )

    default_priors: dict[str, dict] = {}
    for param in spec_obj.parameters:
        ps = param if isinstance(param, ParameterSpec) else ParameterSpec.model_validate(param)
        default_priors[ps.name] = get_default_prior(ps).model_dump()

    try:
        compile_ssm_artifact(model_spec, default_priors, causal_spec=causal_spec)
    except Exception as e:
        return str(e)
    return None


def compile_ssm_artifact(
    model_spec: ModelSpec | dict,
    priors: dict[str, PriorProposal] | dict[str, dict],
    causal_spec: dict | None = None,
) -> CompiledSSMArtifact:
    """Compile user-facing specs into an executable, serializable SSM artifact."""
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

    builder = SSMModelBuilder(model_spec=model_spec, priors=priors, causal_spec=causal_spec)
    spec, ssm_priors = builder.compile_inputs()

    return {
        "schema_version": 1,
        "spec": serialize_ssm_spec(spec),
        "priors": serialize_ssm_priors(ssm_priors),
    }


def make_builder_from_compiled_artifact(
    compiled_ssm: CompiledSSMArtifact,
    *,
    model_config: dict | None = None,
    sampler_config: dict | None = None,
):
    """Instantiate an SSMModelBuilder directly from a compiled artifact."""
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

    spec = deserialize_ssm_spec(compiled_ssm["spec"])
    priors = deserialize_ssm_priors(compiled_ssm["priors"])
    return SSMModelBuilder(
        ssm_spec=spec,
        ssm_priors=priors,
        model_config=model_config,
        sampler_config=sampler_config,
    )


def build_compiled_ssm_builder(
    compiled_ssm: CompiledSSMArtifact,
    raw_data: pl.DataFrame,
    *,
    model_config: dict | None = None,
    sampler_config: dict | None = None,
):
    """Build a ready-to-fit SSMModelBuilder from a compiled artifact."""
    from causal_ssm_agent.utils.data import pivot_to_wide

    if raw_data.is_empty():
        raise ValueError("Cannot build SSM model from empty data")

    builder = make_builder_from_compiled_artifact(
        compiled_ssm,
        model_config=model_config,
        sampler_config=sampler_config,
    )
    X = pivot_to_wide(raw_data)
    builder.build_model(X)
    return builder
