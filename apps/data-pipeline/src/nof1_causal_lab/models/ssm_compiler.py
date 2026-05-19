"""Compilation and serialization helpers for executable SSM artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import fields
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

from nof1_causal_lab.artifacts.latent_model import LatentModel
from nof1_causal_lab.artifacts.measurement_model import (
    MeasurementModel,
    check_semantic_collisions,
    validate_measurement_model,
)
from nof1_causal_lab.artifacts.model_spec import (
    DistributionFamily,
    LinkFunction,
    ModelSpec,
    ParameterRole,
    validate_model_spec_dict,
)
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.ssm import SSMSpec
from nof1_causal_lab.models.ssm_compilation_common import dump_prior_payloads

logger = get_prefect_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from nof1_causal_lab.workers.schemas_prior import PriorProposal

CompiledSSMArtifact = dict[str, Any]

_SPEC_ARRAY_FIELDS = {
    "drift",
    "diffusion_chol",
    "cint",
    "input_effect",
    "static_state_sds",
    "static_factor_loadings",
    "lambda_mat",
    "manifest_means",
    "manifest_chol",
    "t0_means",
    "t0_chol",
}
_SPEC_BOOL_ARRAY_FIELDS = {
    "drift_diag_mask",
    "drift_offdiag_mask",
    "cint_mask",
    "input_effect_mask",
    "static_state_sd_mask",
    "lambda_mask",
    "diffusion_chol_mask",
    "manifest_means_mask",
    "manifest_chol_diag_mask",
    "t0_means_mask",
    "t0_chol_diag_mask",
    "t0_correlation_mask",
    "time_invariant_mask",
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


def serialize_edge_lag_days(
    edge_lag_days: dict[tuple[int, int], float],
) -> list[dict[str, float | int]]:
    """Convert edge-lag metadata into a JSON-serializable payload."""
    return [
        {
            "effect_idx": int(effect_idx),
            "cause_idx": int(cause_idx),
            "lag_days": float(lag_days),
        }
        for (effect_idx, cause_idx), lag_days in sorted(edge_lag_days.items())
    ]


def deserialize_edge_lag_days(payload: Any) -> dict[tuple[int, int], float]:
    """Restore serialized edge-lag metadata from a compiled artifact payload."""
    if not isinstance(payload, list):
        raise ValueError(
            "Compiled artifact is missing required 'edge_lag_days'. Recompile the artifact."
        )

    edge_lag_days: dict[tuple[int, int], float] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Compiled artifact 'edge_lag_days' entries must be JSON objects.")
        effect_idx = entry.get("effect_idx")
        cause_idx = entry.get("cause_idx")
        lag_days = entry.get("lag_days")
        if not isinstance(effect_idx, int) or not isinstance(cause_idx, int):
            raise ValueError(
                "Compiled artifact 'edge_lag_days' entries must include integer effect_idx "
                "and cause_idx values."
            )
        if not isinstance(lag_days, int | float):
            raise ValueError(
                "Compiled artifact 'edge_lag_days' entries must include numeric lag_days values."
            )
        edge_lag_days[(effect_idx, cause_idx)] = float(lag_days)
    return edge_lag_days


def deserialize_ssm_spec(payload: dict[str, Any]) -> SSMSpec:
    """Restore an SSMSpec from a serialized artifact."""
    legacy_scalar_family_fields = {"diffusion_dist", "manifest_dist"} & set(payload)
    if legacy_scalar_family_fields:
        removed = ", ".join(sorted(legacy_scalar_family_fields))
        raise ValueError(
            f"Legacy SSMSpec payload contains removed scalar family fields: {removed}. "
            "Regenerate the compiled SSM artifact."
        )

    required_mask_fields = {"drift_diag_mask", "drift_offdiag_mask", "lambda_mask"} - set(payload)
    if required_mask_fields:
        missing = ", ".join(sorted(required_mask_fields))
        raise ValueError(
            "Serialized SSMSpec payload is missing required structural masks: "
            f"{missing}. Regenerate the compiled SSM artifact."
        )
    required_template_fields = {
        "drift",
        "cint",
        "input_effect",
        "static_state_sds",
        "static_factor_loadings",
        "lambda_mat",
        "diffusion_chol",
        "manifest_means",
        "manifest_chol",
        "t0_means",
        "t0_chol",
    } - set(payload)
    if required_template_fields:
        missing = ", ".join(sorted(required_template_fields))
        raise ValueError(
            "Serialized SSMSpec payload is missing required matrix templates: "
            f"{missing}. "
            "Regenerate the compiled SSM artifact."
        )
    required_compiled_mask_fields = {
        "cint_mask",
        "input_effect_mask",
        "static_state_sd_mask",
        "diffusion_chol_mask",
        "manifest_means_mask",
        "manifest_chol_diag_mask",
        "t0_means_mask",
        "t0_chol_diag_mask",
        "t0_correlation_mask",
    } - set(payload)
    if required_compiled_mask_fields:
        missing = ", ".join(sorted(required_compiled_mask_fields))
        raise ValueError(
            "Serialized SSMSpec payload is missing required compiled masks: "
            f"{missing}. "
            "Regenerate the compiled SSM artifact."
        )
    legacy_matrix_mode_fields = [
        key
        for key in ("drift", "diffusion_chol", "manifest_chol", "t0_chol")
        if isinstance(payload.get(key), str)
    ]
    if legacy_matrix_mode_fields:
        removed = ", ".join(sorted(legacy_matrix_mode_fields))
        raise ValueError(
            f"Legacy SSMSpec payload contains removed matrix mode fields: {removed}. "
            "Regenerate the compiled SSM artifact."
        )

    kwargs: dict[str, Any] = {}

    for key, value in payload.items():
        if key in _SPEC_ARRAY_FIELDS and isinstance(value, list):
            kwargs[key] = jnp.asarray(value, dtype=jnp.float64)
        elif key in _SPEC_BOOL_ARRAY_FIELDS and value is not None:
            kwargs[key] = np.asarray(value, dtype=bool)
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
    latent = (
        LatentModel.model_validate(latent_model) if isinstance(latent_model, dict) else latent_model
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
        if isinstance(measurement_model, BaseModel)
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
    from nof1_causal_lab.utils.causal_spec import (
        get_estimation_state_order,
        get_manifest_indicators,
    )

    errors: list[str] = []
    try:
        latent_states = get_estimation_state_order(causal_spec)
    except ValueError as exc:
        return [str(exc)]
    if not latent_states:
        return ["causal_spec.estimation.state_order is empty"]

    indicators = get_manifest_indicators(causal_spec)
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
    indicators = None
    if causal_spec is not None:
        from nof1_causal_lab.utils.causal_spec import get_manifest_indicators

        indicators = get_manifest_indicators(causal_spec)

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
    from nof1_causal_lab.models.ssm.parameterization import compile_prior_semantics
    from nof1_causal_lab.models.ssm_compilation import compile_ssm_inputs_from_model_spec

    spec, ssm_priors, parameter_bindings, compile_diagnostics, edge_lag_days = (
        compile_ssm_inputs_from_model_spec(
            validated_model_spec,
            raw_priors,
            causal_spec=causal_spec,
        )
    )

    return {
        "schema_version": 1,
        "spec": serialize_ssm_spec(spec),
        "edge_lag_days": serialize_edge_lag_days(edge_lag_days),
        "compiled_prior_semantics": compile_prior_semantics(spec, ssm_priors),
        "parameter_bindings": parameter_bindings,
        "compile_diagnostics": [
            diagnostic.model_dump(mode="json") for diagnostic in compile_diagnostics
        ],
    }


def compile_ssm_artifact_with_default_priors(
    model_spec: ModelSpec | dict,
    causal_spec: dict | None = None,
) -> CompiledSSMArtifact:
    """Compile a ModelSpec using compiler-owned default priors for warmup paths."""
    from nof1_causal_lab.workers.prior_research import get_default_prior

    validated_model_spec, errors = validate_model_spec_for_compilation(
        model_spec,
        causal_spec=causal_spec,
    )
    if errors:
        raise ValueError("ModelSpec failed compiler validation:\n" + "\n".join(errors))

    assert validated_model_spec is not None
    default_priors = {
        parameter.name: get_default_prior(parameter).model_dump()
        for parameter in validated_model_spec.parameters
    }
    return _compile_validated_ssm_artifact(
        validated_model_spec,
        default_priors,
        causal_spec=causal_spec,
    )


def trial_compile_model_spec(
    model_spec: ModelSpec | dict,
    causal_spec: dict | None = None,
) -> str | None:
    """Try compiling a ModelSpec with default priors to catch structural errors early.

    Returns None on success, or an error message string on failure.
    """
    try:
        compile_ssm_artifact_with_default_priors(model_spec, causal_spec=causal_spec)
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
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
    from nof1_causal_lab.distributions import (
        PriorDistributionFamily,
        get_positive_runtime_kind_from_index,
        get_real_runtime_kind_from_index,
    )
    from nof1_causal_lab.models.ssm.parameterization import SupportClass

    if site.support in {SupportClass.REAL, SupportClass.CORRELATION}:
        family = int(_extract_serialized_prior_value(params, "family", flat_index))
        prior_family = get_real_runtime_kind_from_index(family)
        base_params = {
            "mu": _extract_serialized_prior_value(params, "loc", flat_index),
            "sigma": _extract_serialized_prior_value(params, "scale", flat_index),
        }
        bounded_params = {
            **base_params,
            "lower": _extract_serialized_prior_value(params, "low", flat_index),
            "upper": _extract_serialized_prior_value(params, "high", flat_index),
        }
        if prior_family == PriorDistributionFamily.NORMAL:
            return "Normal", base_params
        if prior_family == PriorDistributionFamily.UNIFORM:
            return "Uniform", {
                "lower": bounded_params["lower"],
                "upper": bounded_params["upper"],
            }
        if prior_family == PriorDistributionFamily.TRUNCATED_NORMAL:
            return "TruncatedNormal", bounded_params
        raise ValueError(f"Unsupported compiled real-support prior family index {family}")

    family = int(_extract_serialized_prior_value(params, "family", flat_index))
    prior_family = get_positive_runtime_kind_from_index(family)
    if prior_family == PriorDistributionFamily.HALF_NORMAL:
        return "HalfNormal", {
            "sigma": _extract_serialized_prior_value(params, "scale", flat_index),
        }
    if prior_family == PriorDistributionFamily.GAMMA:
        return "Gamma", {
            "concentration": _extract_serialized_prior_value(params, "concentration", flat_index),
            "rate": _extract_serialized_prior_value(params, "rate", flat_index),
        }
    if prior_family == PriorDistributionFamily.LOG_NORMAL:
        return "LogNormal", {
            "mu": _extract_serialized_prior_value(params, "loc", flat_index),
            "sigma": _extract_serialized_prior_value(params, "scale", flat_index),
        }
    if prior_family == PriorDistributionFamily.EXPONENTIAL:
        return "Exponential", {
            "rate": _extract_serialized_prior_value(params, "rate", flat_index),
        }
    if prior_family == PriorDistributionFamily.DELTA:
        return "Delta", {
            "value": _extract_serialized_prior_value(params, "value", flat_index),
        }

    raise ValueError(f"Unsupported compiled positive-support prior family index {family}")


def _normalize_authored_prior_payload(
    parameter: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate authored prior metadata and force the canonical parameter name."""
    from nof1_causal_lab.workers.schemas_prior import PriorProposal

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
    from nof1_causal_lab.workers.schemas_prior import PriorProposal

    site_name = str(binding["site_name"])
    flat_index = int(binding["flat_index"])
    site = site_by_name.get(site_name)
    if site is None:
        raise ValueError(f"Compiled artifact is missing site registry entry for {site_name!r}")

    params = prior_state.get(site_name)
    if not isinstance(params, dict):
        raise ValueError(f"Compiled artifact is missing prior state for site {site_name!r}")

    distribution, distribution_params = _compiled_distribution_for_site(site, params, flat_index)
    if parameter.startswith("t0_mean_"):
        reasoning = (
            "Compiler-resolved prior for the initial state mean of "
            f"{parameter.removeprefix('t0_mean_').replace('_', ' ')}."
        )
    elif parameter.startswith("t0_sd_"):
        reasoning = (
            "Compiler-resolved prior for the initial state standard deviation of "
            f"{parameter.removeprefix('t0_sd_').replace('_', ' ')}."
        )
    else:
        reasoning = f"Compiler-resolved prior for {parameter}."
    return PriorProposal(
        parameter=parameter,
        distribution=PriorDistributionFamily(distribution),
        params=distribution_params,
        sources=[],
        reasoning=reasoning,
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
    """Resolve latent state names from the compiled spec."""
    spec_payload = compiled_ssm.get("spec")
    latent_names: list[str] = []
    if isinstance(spec_payload, dict):
        latent_names = [str(name) for name in spec_payload.get("latent_names") or [] if name]

    if len(latent_names) < expected:
        raise ValueError(
            f"Compiled SSM declares {expected} latent states but spec.latent_names "
            f"provides only {len(latent_names)} ({latent_names!r}). "
            "Rebuild the compiled artifact with complete latent_names."
        )
    return latent_names[:expected]


def _build_compiled_initial_state_priors(
    compiled_ssm: CompiledSSMArtifact,
    *,
    site_by_field: dict[str, Any],
    prior_state: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose implicit initial-state compiler defaults as public prior rows."""
    from nof1_causal_lab.workers.schemas_prior import PriorProposal

    mean_site = site_by_field.get("t0_means")
    sd_site = site_by_field.get("t0_var_diag")
    if mean_site is None and sd_site is None:
        return []
    if mean_site is None or sd_site is None:
        logger.warning("Missing mean/sd sites for initial-state prior binding; skipping")
        return []

    mean_params = prior_state.get(mean_site.name)
    sd_params = prior_state.get(sd_site.name)
    if not isinstance(mean_params, dict) or not isinstance(sd_params, dict):
        logger.warning("Missing prior state for initial-state sites; skipping")
        return []

    n_latent = int(np.prod(mean_site.shape)) if mean_site.shape else 1
    latent_names = _resolve_latent_names(compiled_ssm, expected=n_latent)

    rows: list[dict[str, Any]] = []
    for index, latent_name in enumerate(latent_names):
        rows.append(
            PriorProposal(
                parameter=f"t0_mean_{latent_name}",
                distribution=PriorDistributionFamily.NORMAL,
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
                distribution=PriorDistributionFamily.HALF_NORMAL,
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
    from nof1_causal_lab.models.ssm.parameterization import load_prior_runtime_bundle

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

    for row in _build_compiled_initial_state_priors(
        compiled_ssm,
        site_by_field=site_by_field,
        prior_state=bundle.prior_state,
    ):
        parameter = str(row.get("parameter") or "")
        if not parameter or parameter in seen:
            continue
        resolved.append(row)
        seen.add(parameter)
    return resolved


def make_builder_from_compiled_artifact(
    compiled_ssm: CompiledSSMArtifact,
    *,
    sampler_config: dict | None = None,
):
    """Instantiate an SSMModelBuilder directly from a compiled artifact.

    Uses ``compiled_prior_semantics`` as the canonical runtime prior state for
    compiled artifacts.
    """
    from nof1_causal_lab.models.ssm.parameterization import load_prior_runtime_bundle
    from nof1_causal_lab.models.ssm_builder import SSMModelBuilder

    spec = deserialize_ssm_spec(compiled_ssm["spec"])
    semantics = compiled_ssm.get("compiled_prior_semantics")
    if not isinstance(semantics, dict):
        raise ValueError(
            "Compiled artifact is missing required 'compiled_prior_semantics'. "
            "Recompile the artifact with the current compiler."
        )
    prior_runtime_bundle = load_prior_runtime_bundle(semantics)

    return SSMModelBuilder(
        ssm_spec=spec,
        compiled_prior_semantics=semantics,
        prior_runtime_bundle=prior_runtime_bundle,
        sampler_config=sampler_config,
        parameter_bindings=list(compiled_ssm.get("parameter_bindings", []) or []),
    )


def build_compiled_ssm_builder(
    compiled_ssm: CompiledSSMArtifact,
    wide_data: pl.DataFrame,
    *,
    sampler_config: dict | None = None,
):
    """Build a ready-to-fit SSMModelBuilder from a compiled artifact."""
    if wide_data.is_empty():
        raise ValueError("Cannot build SSM model from empty data")

    builder = make_builder_from_compiled_artifact(
        compiled_ssm,
        sampler_config=sampler_config,
    )
    builder.build_model(wide_data)
    return builder
