"""Compilation and serialization helpers for executable SSM artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import fields
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.models.ssm import SSMSpec
from causal_ssm_agent.models.ssm_compilation_common import dump_prior_payloads
from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionFamily,
    LinkFunction,
    ModelSpec,
    ParameterRole,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from causal_ssm_agent.orchestrator.schemas import LatentModel, MeasurementModel
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
    "t0_correlation_mask",
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


def collect_estimation_projection_compile_errors(
    causal_spec: dict,
    *,
    manifest_names: Sequence[str] | None = None,
) -> list[str]:
    """Validate that the retained estimation projection can be compiled.

    Under the current compiler/runtime, every retained estimation state must be
    supported by at least one manifest channel, and the loading matrix must be
    able to reach full column rank.
    """
    from causal_ssm_agent.utils.causal_spec import (
        get_estimation_state_order,
        get_indicators,
    )

    errors: list[str] = []
    try:
        latent_states = get_estimation_state_order(causal_spec)
    except ValueError as exc:
        return [str(exc)]
    if not latent_states:
        return ["causal_spec.estimation.state_order is empty"]

    indicators = get_indicators(causal_spec)
    indicator_lookup = {
        indicator["name"]: indicator
        for indicator in indicators
        if isinstance(indicator, dict) and isinstance(indicator.get("name"), str)
    }
    used_manifests = (
        list(manifest_names) if manifest_names is not None else list(indicator_lookup.keys())
    )
    covered_state_counts = Counter(
        indicator.get("construct_name")
        for manifest_name in used_manifests
        if isinstance((indicator := indicator_lookup.get(manifest_name)), dict)
        and isinstance(indicator.get("construct_name"), str)
    )
    uncovered_states = sorted(
        state for state in latent_states if covered_state_counts.get(state, 0) == 0
    )
    if uncovered_states:
        errors.append(
            "Retained estimation states have no measurement indicators: "
            f"{uncovered_states}. Add proxy indicators for these constructs or "
            "exclude them from the executable estimation projection."
        )

    n_manifest = len(used_manifests)
    n_latent = len(latent_states)
    if n_manifest < n_latent:
        errors.append(
            f"Loading matrix is rank-deficient: n_manifest ({n_manifest}) < n_latent ({n_latent})."
        )

    return errors


def _collect_model_spec_compile_errors(
    model_spec: ModelSpec,
    causal_spec: dict | None = None,
) -> list[str]:
    """Collect deterministic ModelSpec checks that the compiler owns."""
    errors: list[str] = []
    if causal_spec is not None:
        manifest_names = [likelihood.variable for likelihood in model_spec.likelihoods]
        return collect_estimation_projection_compile_errors(
            causal_spec,
            manifest_names=manifest_names,
        )

    n_manifest = len(model_spec.likelihoods)

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


def _compile_validated_ssm_artifact(
    validated_model_spec: ModelSpec,
    raw_priors: dict[str, dict],
    *,
    causal_spec: dict | None = None,
) -> CompiledSSMArtifact:
    """Compile an already-validated ``ModelSpec`` into a serialized SSM artifact."""
    from causal_ssm_agent.models.ssm.parameterization import compile_prior_semantics
    from causal_ssm_agent.models.ssm_compilation import compile_ssm_inputs

    spec, ssm_priors, parameter_bindings, compile_diagnostics = compile_ssm_inputs(
        validated_model_spec,
        raw_priors,
        causal_spec=causal_spec,
    )

    return {
        "schema_version": 1,
        "spec": serialize_ssm_spec(spec),
        "compiled_prior_semantics": compile_prior_semantics(spec, ssm_priors),
        "parameter_bindings": parameter_bindings,
        "compile_diagnostics": compile_diagnostics,
    }


def trial_compile_model_spec(
    model_spec: ModelSpec | dict,
    causal_spec: dict | None = None,
) -> str | None:
    """Try compiling a ModelSpec with default priors to catch structural errors early.

    Returns None on success, or an error message string on failure.
    """
    from causal_ssm_agent.workers.prior_research import get_default_prior

    validated_model_spec, errors = validate_model_spec_for_compilation(
        model_spec,
        causal_spec=causal_spec,
    )
    if errors:
        return "ModelSpec failed compiler validation:\n" + "\n".join(errors)

    default_priors: dict[str, dict] = {}
    assert validated_model_spec is not None
    for parameter in validated_model_spec.parameters:
        default_priors[parameter.name] = get_default_prior(parameter).model_dump()

    try:
        _compile_validated_ssm_artifact(
            validated_model_spec,
            default_priors,
            causal_spec=causal_spec,
        )
    except Exception as e:
        return str(e)
    return None


def compile_ssm_artifact(
    model_spec: ModelSpec | dict,
    priors: dict[str, PriorProposal] | dict[str, dict],
    causal_spec: dict | None = None,
) -> CompiledSSMArtifact:
    """Compile user-facing specs into an executable, serializable SSM artifact."""
    validated_model_spec, errors = validate_model_spec_for_compilation(
        model_spec, causal_spec=causal_spec
    )
    if errors:
        raise ValueError("ModelSpec failed compiler validation:\n" + "\n".join(errors))

    assert validated_model_spec is not None
    raw_priors = dump_prior_payloads(priors)
    return _compile_validated_ssm_artifact(
        validated_model_spec,
        raw_priors,
        causal_spec=causal_spec,
    )


def _extract_serialized_prior_value(
    params: dict[str, Any],
    key: str,
    flat_index: int,
) -> float:
    """Read one scalar parameter value from serialized compiled prior semantics."""
    if key not in params:
        raise ValueError(f"Compiled prior state is missing required key {key!r}")

    values = np.asarray(params[key], dtype=float).ravel()
    if values.size == 0:
        raise ValueError(f"Compiled prior state key {key!r} is empty")
    if values.size == 1:
        return float(values[0])
    if flat_index < 0 or flat_index >= values.size:
        raise ValueError(
            f"Compiled prior index {flat_index} is out of bounds for {key!r} with size {values.size}"
        )
    return float(values[flat_index])


def _compiled_distribution_for_site(
    site,
    params: dict[str, Any],
    flat_index: int,
) -> tuple[str, dict[str, float]]:
    """Convert one compiled site element back to a user-facing distribution row."""
    from causal_ssm_agent.distributions import (
        PriorRuntimeKind,
        get_positive_runtime_kind_from_index,
        get_real_runtime_kind_from_index,
    )
    from causal_ssm_agent.models.ssm.parameterization import SupportClass

    if site.support == SupportClass.REAL:
        family = int(_extract_serialized_prior_value(params, "family", flat_index))
        runtime_kind = get_real_runtime_kind_from_index(family)
        if runtime_kind == PriorRuntimeKind.UNIFORM:
            return "Uniform", {
                "lower": _extract_serialized_prior_value(params, "low", flat_index),
                "upper": _extract_serialized_prior_value(params, "high", flat_index),
            }
        if "low" in params and "high" in params:
            return "TruncatedNormal", {
                "mu": _extract_serialized_prior_value(params, "loc", flat_index),
                "sigma": _extract_serialized_prior_value(params, "scale", flat_index),
                "lower": _extract_serialized_prior_value(params, "low", flat_index),
                "upper": _extract_serialized_prior_value(params, "high", flat_index),
            }
        return "Normal", {
            "mu": _extract_serialized_prior_value(params, "loc", flat_index),
            "sigma": _extract_serialized_prior_value(params, "scale", flat_index),
        }

    family = int(_extract_serialized_prior_value(params, "family", flat_index))
    runtime_kind = get_positive_runtime_kind_from_index(family)
    if runtime_kind == PriorRuntimeKind.HALF_NORMAL:
        return "HalfNormal", {
            "sigma": _extract_serialized_prior_value(params, "scale", flat_index),
        }
    if runtime_kind == PriorRuntimeKind.GAMMA:
        return "Gamma", {
            "concentration": _extract_serialized_prior_value(params, "concentration", flat_index),
            "rate": _extract_serialized_prior_value(params, "rate", flat_index),
        }
    if runtime_kind == PriorRuntimeKind.LOG_NORMAL:
        return "LogNormal", {
            "mu": _extract_serialized_prior_value(params, "loc", flat_index),
            "sigma": _extract_serialized_prior_value(params, "scale", flat_index),
        }
    if runtime_kind == PriorRuntimeKind.EXPONENTIAL:
        return "Exponential", {
            "rate": _extract_serialized_prior_value(params, "rate", flat_index),
        }

    raise ValueError(f"Unsupported compiled positive-support prior family index {family}")


def _normalize_authored_prior_payload(
    parameter: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate authored prior metadata and force the canonical parameter name."""
    from causal_ssm_agent.workers.schemas_prior import PriorProposal

    normalized = dict(payload)
    normalized["parameter"] = parameter
    return PriorProposal.model_validate(normalized).model_dump(mode="json")


def _build_compiled_parameter_prior(
    *,
    parameter: str,
    binding: dict[str, Any],
    site_by_name: dict[str, Any],
    prior_state: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a generic public prior row from one compiler binding."""
    from causal_ssm_agent.workers.schemas_prior import PriorProposal

    site_name = str(binding["site_name"])
    flat_index = int(binding["flat_index"])
    site = site_by_name.get(site_name)
    if site is None:
        raise ValueError(f"Compiled artifact is missing site registry entry for {site_name!r}")

    params = prior_state.get(site_name)
    if not isinstance(params, dict):
        raise ValueError(f"Compiled artifact is missing prior state for site {site_name!r}")

    distribution, distribution_params = _compiled_distribution_for_site(site, params, flat_index)
    return PriorProposal(
        parameter=parameter,
        distribution=distribution,
        params=distribution_params,
        sources=[],
        reasoning=f"Compiler-resolved prior for {parameter}.",
    ).model_dump(mode="json")


def _merge_resolved_prior_metadata(
    resolved: dict[str, Any],
    authored: dict[str, Any],
) -> dict[str, Any]:
    """Overlay authored metadata onto a compiler-resolved prior row."""
    merged = dict(resolved)
    for field in (
        "distribution",
        "params",
        "sources",
        "reasoning",
        "reference_interval_days",
        "density_points",
    ):
        if field in authored:
            merged[field] = authored[field]
    return _normalize_authored_prior_payload(str(merged["parameter"]), merged)


def _resolve_latent_names(
    compiled_ssm: CompiledSSMArtifact,
    *,
    expected: int,
) -> list[str]:
    """Resolve latent state names from the compiled spec with safe fallbacks."""
    spec_payload = compiled_ssm.get("spec")
    latent_names = []
    if isinstance(spec_payload, dict):
        latent_names = [str(name) for name in spec_payload.get("latent_names") or [] if name]

    if len(latent_names) >= expected:
        return latent_names[:expected]
    return latent_names + [f"latent_{idx}" for idx in range(len(latent_names), expected)]


def _build_compiled_initial_state_priors(
    compiled_ssm: CompiledSSMArtifact,
    *,
    site_by_field: dict[str, Any],
    prior_state: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose implicit initial-state compiler defaults as public prior rows."""
    from causal_ssm_agent.workers.schemas_prior import PriorProposal

    mean_site = site_by_field.get("t0_means")
    sd_site = site_by_field.get("t0_var_diag")
    if mean_site is None or sd_site is None:
        return []

    mean_params = prior_state.get(mean_site.name)
    sd_params = prior_state.get(sd_site.name)
    if not isinstance(mean_params, dict) or not isinstance(sd_params, dict):
        return []

    n_latent = int(np.prod(mean_site.shape)) if mean_site.shape else 1
    latent_names = _resolve_latent_names(compiled_ssm, expected=n_latent)

    rows: list[dict[str, Any]] = []
    for index, latent_name in enumerate(latent_names):
        rows.append(
            PriorProposal(
                parameter=f"t0_mean_{latent_name}",
                distribution="Normal",
                params={
                    "mu": _extract_serialized_prior_value(mean_params, "loc", index),
                    "sigma": _extract_serialized_prior_value(mean_params, "scale", index),
                },
                sources=[],
                reasoning=(
                    "Default weakly informative prior for the initial state mean of "
                    f"{latent_name.replace('_', ' ')}."
                ),
            ).model_dump(mode="json")
        )
    for index, latent_name in enumerate(latent_names):
        rows.append(
            PriorProposal(
                parameter=f"t0_sd_{latent_name}",
                distribution="HalfNormal",
                params={"sigma": _extract_serialized_prior_value(sd_params, "scale", index)},
                sources=[],
                reasoning=(
                    "Default weakly informative prior for the initial state standard deviation of "
                    f"{latent_name.replace('_', ' ')}."
                ),
            ).model_dump(mode="json")
        )
    return rows


def resolve_prior_proposals(
    compiled_ssm: CompiledSSMArtifact,
    *,
    authored_priors: dict[str, PriorProposal] | dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    """Build canonical public prior rows from a compiled artifact.

    The compiler owns membership, ordering of bound parameters, and implicit
    defaults. Authored prior metadata is overlaid when available because some
    semantic priors (for example DT-scale Beta priors on persistence) are
    intentionally lossy after compilation to the executable CT representation.
    """
    from causal_ssm_agent.models.ssm.parameterization import load_prior_runtime_bundle

    semantics = compiled_ssm.get("compiled_prior_semantics")
    if not isinstance(semantics, dict):
        raise ValueError("Compiled artifact is missing required 'compiled_prior_semantics'")

    bundle = load_prior_runtime_bundle(semantics)
    site_by_name = {site.name: site for site in bundle.registry}
    site_by_field = {site.priors_field: site for site in bundle.registry if site.priors_field}
    binding_by_parameter = {
        str(binding["parameter"]): dict(binding)
        for binding in list(compiled_ssm.get("parameter_bindings", []) or [])
        if isinstance(binding, dict) and "parameter" in binding
    }
    authored_payloads = dump_prior_payloads(authored_priors)

    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()

    for parameter, authored_payload in authored_payloads.items():
        binding = binding_by_parameter.get(parameter)
        if binding is None:
            resolved.append(_normalize_authored_prior_payload(parameter, authored_payload))
        else:
            compiled_row = _build_compiled_parameter_prior(
                parameter=parameter,
                binding=binding,
                site_by_name=site_by_name,
                prior_state=bundle.prior_state,
            )
            resolved.append(_merge_resolved_prior_metadata(compiled_row, authored_payload))
        seen.add(parameter)

    for binding in list(compiled_ssm.get("parameter_bindings", []) or []):
        if not isinstance(binding, dict):
            continue
        parameter = str(binding.get("parameter") or "")
        if not parameter or parameter in seen:
            continue
        resolved.append(
            _build_compiled_parameter_prior(
                parameter=parameter,
                binding=binding,
                site_by_name=site_by_name,
                prior_state=bundle.prior_state,
            )
        )
        seen.add(parameter)

    resolved.extend(
        _build_compiled_initial_state_priors(
            compiled_ssm,
            site_by_field=site_by_field,
            prior_state=bundle.prior_state,
        )
    )
    return resolved


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


def _reconstruct_ssm_inputs_from_artifact(
    compiled_ssm: CompiledSSMArtifact,
):
    """Reconstruct executable SSM inputs from a compiled artifact."""
    spec = deserialize_ssm_spec(compiled_ssm["spec"])
    priors = _reconstruct_priors_from_compiled_semantics(compiled_ssm)
    return spec, priors


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

    spec, priors = _reconstruct_ssm_inputs_from_artifact(compiled_ssm)

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
    wide_data: pl.DataFrame,
    *,
    model_config: dict | None = None,
    sampler_config: dict | None = None,
):
    """Build a ready-to-fit SSMModelBuilder from a compiled artifact."""
    if wide_data.is_empty():
        raise ValueError("Cannot build SSM model from empty data")

    builder = make_builder_from_compiled_artifact(
        compiled_ssm,
        model_config=model_config,
        sampler_config=sampler_config,
    )
    builder.build_model(wide_data)
    return builder
