"""Compilation and serialization helpers for executable SSM artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.models.ssm import SSMSpec
from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionFamily,
    LinkFunction,
    ModelSpec,
    ParameterRole,
)
from causal_ssm_agent.workers.schemas_prior import PriorProposal

if TYPE_CHECKING:
    import polars as pl

    from causal_ssm_agent.orchestrator.schemas import LatentModel, MeasurementModel

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


def _normalize_measurement_instruction(text: str) -> str:
    """Normalize free-text measurement instructions for duplicate checks."""
    return " ".join(text.lower().split())


def _collect_measurement_compile_errors(
    measurement: MeasurementModel,
    latent: LatentModel,
) -> list[str]:
    """Collect deterministic measurement checks best handled at compile time."""
    from causal_ssm_agent.orchestrator.schemas import check_semantic_collisions

    errors: list[str] = []

    if not measurement.indicators:
        errors.append("Measurement model must include at least one indicator.")
        return errors

    outcome_names = [construct.name for construct in latent.constructs if construct.is_outcome]
    for outcome_name in outcome_names:
        if not measurement.get_indicators_for_construct(outcome_name):
            errors.append(f"Outcome construct '{outcome_name}' must have at least one indicator.")

    duplicate_groups: dict[tuple[str, str, str, str, tuple[str, ...]], list[str]] = defaultdict(
        list
    )
    for indicator in measurement.indicators:
        collisions = check_semantic_collisions(indicator.how_to_measure, indicator.aggregation)
        for warning in collisions:
            errors.append(f"Indicator '{indicator.name}': {warning}")

        duplicate_key = (
            indicator.construct_name,
            _normalize_measurement_instruction(indicator.how_to_measure),
            indicator.measurement_dtype,
            indicator.aggregation,
            tuple(indicator.ordinal_levels or ()),
        )
        duplicate_groups[duplicate_key].append(indicator.name)

    for duplicate_key, indicator_names in duplicate_groups.items():
        if len(indicator_names) < 2:
            continue

        construct_name = duplicate_key[0]
        joined_names = ", ".join(sorted(indicator_names))
        errors.append(
            f"Construct '{construct_name}' has duplicate indicator operationalizations: "
            f"{joined_names}. Each indicator should add distinct measurement information."
        )

    return errors


def validate_measurement_model_for_compilation(
    measurement_model: dict,
    latent_model: LatentModel | dict,
) -> tuple[MeasurementModel | None, list[str]]:
    """Validate measurement output against schema and compile-time constraints."""
    from causal_ssm_agent.orchestrator.schemas import LatentModel as LatentModelCls
    from causal_ssm_agent.orchestrator.schemas import validate_measurement_model

    latent = (
        LatentModelCls.model_validate(latent_model)
        if isinstance(latent_model, dict)
        else latent_model
    )
    measurement, errors = validate_measurement_model(measurement_model, latent)
    if measurement is None:
        return None, errors

    compile_errors = _collect_measurement_compile_errors(measurement, latent)
    if compile_errors:
        return None, compile_errors

    return measurement, []


def trial_compile_measurement_model(
    measurement_model: MeasurementModel | dict,
    latent_model: LatentModel | dict,
) -> str | None:
    """Try compiling a measurement model and return a feedback string on failure."""
    measurement_data = (
        measurement_model.model_dump(mode="json")
        if hasattr(measurement_model, "model_dump")
        else measurement_model
    )
    _, errors = validate_measurement_model_for_compilation(measurement_data, latent_model)
    if errors:
        return "\n".join(errors)
    return None


def _collect_model_spec_compile_errors(
    model_spec: ModelSpec,
    causal_spec: dict | None = None,
) -> list[str]:
    """Collect deterministic ModelSpec checks that the compiler owns."""
    errors: list[str] = []
    n_manifest = len(model_spec.likelihoods)

    if causal_spec is not None:
        from causal_ssm_agent.utils.causal_spec import get_constructs

        constructs = get_constructs(causal_spec)
        if not constructs:
            errors.append("causal_spec.latent.constructs is empty")
            return errors

        n_latent = len(constructs)
        if n_manifest < n_latent:
            errors.append(
                "Loading matrix is rank-deficient: "
                f"n_manifest ({n_manifest}) < n_latent ({n_latent})."
            )
        return errors

    ar_params = [p for p in model_spec.parameters if p.role == ParameterRole.AR_COEFFICIENT]
    if not ar_params:
        errors.append(
            "No AR_COEFFICIENT parameters found in ModelSpec; "
            "cannot infer latent dimensionality without causal_spec."
        )
        return errors

    n_latent = len(ar_params)
    if n_manifest < n_latent:
        errors.append(
            "Loading matrix is rank-deficient: "
            f"n_manifest ({n_manifest}) < inferred n_latent ({n_latent})."
        )

    return errors


def validate_model_spec_for_compilation(
    model_spec: ModelSpec | dict,
    causal_spec: dict | None = None,
) -> tuple[ModelSpec | None, list[str]]:
    """Validate model-spec schema/domain rules plus compiler-owned invariants."""
    from causal_ssm_agent.orchestrator.schemas_model import validate_model_spec_dict

    indicators = None
    if causal_spec is not None:
        from causal_ssm_agent.utils.causal_spec import get_indicators

        indicators = get_indicators(causal_spec)

    model_spec_data = (
        model_spec.model_dump(mode="json") if isinstance(model_spec, ModelSpec) else model_spec
    )

    spec_obj, errors = validate_model_spec_dict(model_spec_data, indicators=indicators)
    if spec_obj is None:
        return None, errors

    compile_errors = _collect_model_spec_compile_errors(spec_obj, causal_spec=causal_spec)
    if compile_errors:
        return None, compile_errors

    return spec_obj, []


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
    from causal_ssm_agent.models.ssm.parameterization import compile_prior_semantics
    from causal_ssm_agent.models.ssm_builder import compile_ssm_inputs

    validated_model_spec, errors = validate_model_spec_for_compilation(
        model_spec, causal_spec=causal_spec
    )
    if errors:
        raise ValueError("ModelSpec failed compiler validation:\n" + "\n".join(errors))

    assert validated_model_spec is not None
    raw_priors: dict[str, dict] = {
        key: value.model_dump() if isinstance(value, PriorProposal) else value
        for key, value in priors.items()
    }
    spec, ssm_priors, parameter_bindings = compile_ssm_inputs(
        validated_model_spec,
        raw_priors,
        causal_spec=causal_spec,
    )

    return {
        "schema_version": 1,
        "spec": serialize_ssm_spec(spec),
        "compiled_prior_semantics": compile_prior_semantics(spec, ssm_priors),
        "parameter_bindings": parameter_bindings,
    }


def _reconstruct_priors_from_compiled_semantics(
    compiled_ssm: CompiledSSMArtifact,
):
    """Reconstruct builder priors from the canonical compiled semantics block."""
    from causal_ssm_agent.models.ssm.parameterization import (
        load_prior_runtime_bundle,
        reconstruct_ssm_priors,
    )

    semantics = compiled_ssm.get("compiled_prior_semantics")
    if semantics is None:
        raise ValueError(
            "Compiled artifact is missing required 'compiled_prior_semantics'. "
            "Recompile the artifact with the current compiler."
        )

    missing_keys = [key for key in ("site_registry", "prior_state") if key not in semantics]
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Compiled artifact has incomplete 'compiled_prior_semantics': missing {missing}."
        )

    bundle = load_prior_runtime_bundle(semantics)
    return reconstruct_ssm_priors(bundle.registry, bundle.prior_state)


def make_builder_from_compiled_artifact(
    compiled_ssm: CompiledSSMArtifact,
    *,
    model_config: dict | None = None,
    sampler_config: dict | None = None,
):
    """Instantiate an SSMModelBuilder directly from a compiled artifact.

    Reads builder priors from ``compiled_prior_semantics``, which is now the
    only supported cross-stage prior representation.
    """
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

    spec = deserialize_ssm_spec(compiled_ssm["spec"])
    priors = _reconstruct_priors_from_compiled_semantics(compiled_ssm)

    return SSMModelBuilder(
        ssm_spec=spec,
        ssm_priors=priors,
        compiled_prior_semantics=compiled_ssm.get("compiled_prior_semantics"),
        model_config=model_config,
        sampler_config=sampler_config,
        parameter_bindings=list(compiled_ssm.get("parameter_bindings", []) or []),
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
