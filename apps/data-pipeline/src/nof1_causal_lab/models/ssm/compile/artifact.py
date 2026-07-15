"""Compilation and serialization helpers for executable SSM artifacts."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.latent_structure import LatentStructure
from nof1_causal_lab.artifacts.measurement_structure import (
    MeasurementStructure,
    check_semantic_collisions,
    validate_measurement_structure,
)
from nof1_causal_lab.artifacts.statistical_model_spec import (
    ParameterRole,
    StatisticalModelSpec,
    validate_statistical_model_spec_dict,
)
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.models.ssm.compile.common import dump_prior_payloads
from nof1_causal_lab.models.ssm.compile.contracts import (
    CompiledParameterBinding,
    CompiledSSMArtifact,
    SerializedEdgeLag,
    SerializedSSMSpec,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from nof1_causal_lab.models.ssm import SSMSpec
    from nof1_causal_lab.workers.schemas_prior import PriorProposal


def _to_jsonable(value: Any) -> Any:
    from nof1_causal_lab.models.ssm.dynamics.serialization import dynamics_spec_to_dict
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    if isinstance(value, DynamicsSpec):
        return _to_jsonable(dynamics_spec_to_dict(value))
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def serialize_ssm_spec(spec: SSMSpec) -> SerializedSSMSpec:
    """Convert an SSMSpec into a JSON-serializable payload.

    Emits the same block-spec shape used by ``SSMSpec`` in memory. The
    serialized artifact has no flat parameter fields; supports and templates
    live under the block that owns them.
    """
    from nof1_causal_lab.models.ssm.dynamics.serialization import dynamics_spec_to_dict

    payload = {
        "n_latent": spec.n_latent,
        "n_manifest": spec.n_manifest,
        "dynamics_spec": _to_jsonable(dynamics_spec_to_dict(spec.dynamics_spec)),
        "diffusion_block": _to_jsonable(spec.diffusion_block),
        "lambda_block": _to_jsonable(spec.lambda_block),
        "manifest_means_block": _to_jsonable(spec.manifest_means_block),
        "manifest_chol_block": _to_jsonable(spec.manifest_chol_block),
        "t0_means_block": _to_jsonable(spec.t0_means_block),
        "t0_chol_block": _to_jsonable(spec.t0_chol_block),
        "input_effect_block": _to_jsonable(spec.input_effect_block),
        "static_state_sd_block": _to_jsonable(spec.static_state_sd_block),
        "static_factor_loadings": _to_jsonable(spec.static_factor_loadings),
        "diffusion_dists": _to_jsonable(spec.diffusion_dists),
        "manifest_dists": _to_jsonable(spec.manifest_dists),
        "manifest_level_counts": _to_jsonable(spec.manifest_level_counts),
        "manifest_links": _to_jsonable(spec.manifest_links),
        "manifest_standardized": _to_jsonable(spec.manifest_standardized),
        "latent_names": _to_jsonable(spec.latent_names),
        "manifest_names": _to_jsonable(spec.manifest_names),
        "input_names": _to_jsonable(spec.input_names),
        "input_source_indicators": _to_jsonable(spec.input_source_indicators),
        "input_scales": _to_jsonable(spec.input_scales),
        "input_missing_policies": _to_jsonable(spec.input_missing_policies),
        "static_factor_names": _to_jsonable(spec.static_factor_names),
        "initialization_policy": spec.initialization_policy,
        "observation_intercept_policy": spec.observation_intercept_policy,
    }
    return SerializedSSMSpec.model_validate(payload)


def serialize_edge_lag_days(
    edge_lag_days: dict[tuple[int, int], float],
) -> list[SerializedEdgeLag]:
    """Convert edge-lag metadata into a JSON-serializable payload."""
    return [
        SerializedEdgeLag(
            effect_idx=int(effect_idx),
            cause_idx=int(cause_idx),
            lag_days=float(lag_days),
        )
        for (effect_idx, cause_idx), lag_days in sorted(edge_lag_days.items())
    ]


def deserialize_ssm_spec(payload: SerializedSSMSpec) -> SSMSpec:
    """Restore an SSMSpec from a serialized artifact."""
    from nof1_causal_lab.models.ssm.dynamics.serialization import dynamics_spec_from_dict
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.structure import (
        DiffusionBlockSpec,
        ManifestCholBlockSpec,
        SparseMatrixBlockSpec,
        SparseVectorBlockSpec,
        T0CholBlockSpec,
    )

    def _bool_array(block: dict[str, Any], key: str) -> np.ndarray:
        return np.asarray(block[key], dtype=bool)

    def _float_array(block: dict[str, Any], key: str) -> jnp.ndarray:
        return jnp.asarray(block[key], dtype=jnp.float32)

    def _optional_bool_array(block: dict[str, Any], key: str) -> np.ndarray | None:
        value = block.get(key)
        return None if value is None else np.asarray(value, dtype=bool)

    def _diffusion_block(block: dict[str, Any]) -> DiffusionBlockSpec:
        return DiffusionBlockSpec(
            n_latent=int(block["n_latent"]),
            diffusion_chol_support=_bool_array(block, "diffusion_chol_support"),
            diffusion_chol_template=_float_array(block, "diffusion_chol_template"),
            time_invariant_mask=_optional_bool_array(block, "time_invariant_mask"),
        )

    def _sparse_matrix_block(block: dict[str, Any]) -> SparseMatrixBlockSpec:
        return SparseMatrixBlockSpec(
            n_rows=int(block["n_rows"]),
            n_cols=int(block["n_cols"]),
            free_support=_bool_array(block, "free_support"),
            template=_float_array(block, "template"),
            free_site_name=str(block["free_site_name"]),
            det_site_name=str(block["det_site_name"]),
            support=SupportClass(block["support"]),
            site_kind=SiteKind(block["site_kind"]),
            assembly_group=str(block["assembly_group"]),
            fixed_spec_field=str(block["fixed_spec_field"]),
            priors_field=str(block["priors_field"]),
        )

    def _sparse_vector_block(block: dict[str, Any]) -> SparseVectorBlockSpec:
        return SparseVectorBlockSpec(
            n=int(block["n"]),
            free_support=_bool_array(block, "free_support"),
            template=_float_array(block, "template"),
            free_site_name=str(block["free_site_name"]),
            det_site_name=str(block["det_site_name"]),
            support=SupportClass(block["support"]),
            site_kind=SiteKind(block["site_kind"]),
            assembly_group=str(block["assembly_group"]),
            fixed_spec_field=str(block["fixed_spec_field"]),
            priors_field=str(block["priors_field"]),
        )

    def _manifest_chol_block(block: dict[str, Any]) -> ManifestCholBlockSpec:
        return ManifestCholBlockSpec(
            n_manifest=int(block["n_manifest"]),
            diag_support=_bool_array(block, "diag_support"),
            template=_float_array(block, "template"),
        )

    def _t0_chol_block(block: dict[str, Any]) -> T0CholBlockSpec:
        return T0CholBlockSpec(
            n_latent=int(block["n_latent"]),
            diag_support=_bool_array(block, "diag_support"),
            correlation_support=_bool_array(block, "correlation_support"),
            template=_float_array(block, "template"),
        )

    return SSMSpec(
        n_latent=payload.n_latent,
        n_manifest=payload.n_manifest,
        dynamics_spec=dynamics_spec_from_dict(payload.dynamics_spec),
        diffusion_block=_diffusion_block(payload.diffusion_block),
        lambda_block=_sparse_matrix_block(payload.lambda_block),
        manifest_means_block=_sparse_vector_block(payload.manifest_means_block),
        manifest_chol_block=_manifest_chol_block(payload.manifest_chol_block),
        t0_means_block=_sparse_vector_block(payload.t0_means_block),
        t0_chol_block=_t0_chol_block(payload.t0_chol_block),
        input_effect_block=_sparse_matrix_block(payload.input_effect_block),
        static_state_sd_block=_sparse_vector_block(payload.static_state_sd_block),
        static_factor_loadings=jnp.asarray(payload.static_factor_loadings, dtype=jnp.float32),
        diffusion_dists=list(payload.diffusion_dists),
        manifest_dists=list(payload.manifest_dists),
        manifest_level_counts=payload.manifest_level_counts,
        manifest_links=payload.manifest_links,
        manifest_standardized=payload.manifest_standardized,
        latent_names=payload.latent_names,
        manifest_names=payload.manifest_names,
        input_names=payload.input_names,
        input_source_indicators=payload.input_source_indicators,
        input_scales=payload.input_scales,
        input_missing_policies=(
            [str(policy) for policy in payload.input_missing_policies]
            if payload.input_missing_policies is not None
            else None
        ),
        static_factor_names=payload.static_factor_names,
        initialization_policy=payload.initialization_policy,
        observation_intercept_policy=payload.observation_intercept_policy,
    )


def _normalize_measurement_instruction(text: str) -> str:
    """Normalize free-text measurement instructions for duplicate checks."""
    return " ".join(text.lower().split())


def _collect_measurement_compile_errors(
    measurement: MeasurementStructure,
    latent: LatentStructure,
) -> list[str]:
    """Collect deterministic measurement checks best handled at compile time."""
    errors: list[str] = []

    if not measurement.indicators:
        errors.append("Measurement structure must include at least one indicator.")
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


def validate_measurement_structure_for_compilation(
    measurement_structure: dict,
    latent_structure: LatentStructure | dict,
) -> tuple[MeasurementStructure | None, list[str]]:
    """Validate measurement output against schema and compile-time constraints."""
    latent = (
        LatentStructure.model_validate(latent_structure)
        if isinstance(latent_structure, dict)
        else latent_structure
    )
    measurement, errors = validate_measurement_structure(measurement_structure, latent)
    if measurement is None:
        return None, errors

    compile_errors = _collect_measurement_compile_errors(measurement, latent)
    if compile_errors:
        return None, compile_errors

    return measurement, []


def collect_estimation_projection_compile_errors(
    causal_design: dict,
    *,
    manifest_names: Sequence[str] | None = None,
) -> list[str]:
    """Validate that the retained estimation projection can be compiled.

    Under the current compiler/runtime, every retained estimation state must be
    supported by at least one manifest channel, and the loading matrix must be
    able to reach full column rank.
    """
    from nof1_causal_lab.utils.causal_design import (
        get_estimation_state_order,
        get_manifest_indicators,
    )

    errors: list[str] = []
    try:
        latent_states = get_estimation_state_order(causal_design)
    except ValueError as exc:
        return [str(exc)]
    if not latent_states:
        return ["causal_design.estimation.state_order is empty"]

    indicators = get_manifest_indicators(causal_design)
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


def _collect_statistical_model_spec_compile_errors(
    statistical_model_spec: StatisticalModelSpec,
    causal_design: dict | None = None,
) -> list[str]:
    """Collect deterministic StatisticalModelSpec checks that the compiler owns."""
    errors: list[str] = []
    if causal_design is not None:
        manifest_names = [likelihood.variable for likelihood in statistical_model_spec.likelihoods]
        return collect_estimation_projection_compile_errors(
            causal_design,
            manifest_names=manifest_names,
        )

    n_manifest = len(statistical_model_spec.likelihoods)

    ar_params = [
        p for p in statistical_model_spec.parameters if p.role == ParameterRole.AR_COEFFICIENT
    ]
    if not ar_params:
        errors.append(
            "No AR_COEFFICIENT parameters found in StatisticalModelSpec; "
            "cannot infer latent dimensionality without causal_design."
        )
        return errors

    n_latent = len(ar_params)
    if n_manifest < n_latent:
        errors.append(
            "Loading matrix is rank-deficient: "
            f"n_manifest ({n_manifest}) < inferred n_latent ({n_latent})."
        )

    return errors


def validate_statistical_model_spec_for_compilation(
    statistical_model_spec: StatisticalModelSpec | dict,
    causal_design: dict | None = None,
) -> tuple[StatisticalModelSpec | None, list[str]]:
    """Validate statistical-model-spec schema/domain rules plus compiler-owned invariants."""
    indicators = None
    if causal_design is not None:
        from nof1_causal_lab.utils.causal_design import get_manifest_indicators

        indicators = get_manifest_indicators(causal_design)

    statistical_model_spec_data = (
        statistical_model_spec.model_dump(mode="json")
        if isinstance(statistical_model_spec, StatisticalModelSpec)
        else statistical_model_spec
    )

    spec_obj, errors = validate_statistical_model_spec_dict(
        statistical_model_spec_data, indicators=indicators
    )
    if spec_obj is None:
        return None, errors

    compile_errors = _collect_statistical_model_spec_compile_errors(
        spec_obj, causal_design=causal_design
    )
    if compile_errors:
        return None, compile_errors

    return spec_obj, []


def _compile_validated_ssm_artifact(
    validated_statistical_model_spec: StatisticalModelSpec,
    raw_priors: dict[str, dict],
    *,
    causal_design: dict | None = None,
) -> CompiledSSMArtifact:
    """Compile an already-validated ``StatisticalModelSpec`` into a serialized SSM artifact."""
    from nof1_causal_lab.models.ssm.compile.inputs import (
        compile_ssm_inputs_from_statistical_model_spec,
    )
    from nof1_causal_lab.models.ssm.parameterization import compile_prior_semantics

    spec, prior_registry, parameter_bindings, compile_diagnostics, edge_lag_days = (
        compile_ssm_inputs_from_statistical_model_spec(
            validated_statistical_model_spec,
            raw_priors,
            causal_design=causal_design,
        )
    )

    return CompiledSSMArtifact(
        schema_version=1,
        spec=serialize_ssm_spec(spec),
        edge_lag_days=serialize_edge_lag_days(edge_lag_days),
        compiled_prior_semantics=compile_prior_semantics(spec, prior_registry),
        parameter_bindings=parameter_bindings,
        compile_diagnostics=compile_diagnostics,
    )


def compile_ssm_artifact_with_default_priors(
    statistical_model_spec: StatisticalModelSpec | dict,
    causal_design: dict | None = None,
) -> CompiledSSMArtifact:
    """Compile a StatisticalModelSpec using compiler-owned default priors for warmup paths."""
    from nof1_causal_lab.workers.prior_research import get_default_prior

    validated_statistical_model_spec, errors = validate_statistical_model_spec_for_compilation(
        statistical_model_spec,
        causal_design=causal_design,
    )
    if errors:
        raise ValueError("StatisticalModelSpec failed compiler validation:\n" + "\n".join(errors))

    assert validated_statistical_model_spec is not None
    default_priors = {
        parameter.name: get_default_prior(parameter).model_dump()
        for parameter in validated_statistical_model_spec.parameters
    }
    return _compile_validated_ssm_artifact(
        validated_statistical_model_spec,
        default_priors,
        causal_design=causal_design,
    )


def trial_compile_statistical_model_spec(
    statistical_model_spec: StatisticalModelSpec | dict,
    causal_design: dict | None = None,
) -> str | None:
    """Try compiling a StatisticalModelSpec with default priors to catch structural errors early.

    Returns None on success, or an error message string on failure.
    """
    try:
        compile_ssm_artifact_with_default_priors(
            statistical_model_spec, causal_design=causal_design
        )
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        return str(e)
    return None


def compile_ssm_artifact(
    statistical_model_spec: StatisticalModelSpec | dict,
    priors: dict[str, PriorProposal] | dict[str, dict],
    causal_design: dict | None = None,
) -> CompiledSSMArtifact:
    """Compile user-facing specs into an executable, serializable SSM artifact."""
    validated_statistical_model_spec, errors = validate_statistical_model_spec_for_compilation(
        statistical_model_spec, causal_design=causal_design
    )
    if errors:
        raise ValueError("StatisticalModelSpec failed compiler validation:\n" + "\n".join(errors))

    assert validated_statistical_model_spec is not None
    raw_priors = dump_prior_payloads(priors)
    return _compile_validated_ssm_artifact(
        validated_statistical_model_spec,
        raw_priors,
        causal_design=causal_design,
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
    binding: CompiledParameterBinding,
    site_by_name: dict[str, Any],
    prior_state: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a generic public prior row from one compiler binding."""
    from nof1_causal_lab.workers.schemas_prior import PriorProposal, prior_params_model

    site_name = binding.site_name
    flat_index = binding.flat_index
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
    prior_distribution = PriorDistributionFamily(distribution)
    return PriorProposal(
        parameter=parameter,
        distribution=prior_distribution,
        params=prior_params_model(prior_distribution, distribution_params),
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
    latent_names = list(compiled_ssm.spec.latent_names or [])

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
    from nof1_causal_lab.workers.schemas_prior import (
        LocationScalePriorParams,
        PriorProposal,
        ScalePriorParams,
    )

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
                params=LocationScalePriorParams(
                    mu=_extract_serialized_prior_value(mean_params, "loc", index),
                    sigma=_extract_serialized_prior_value(mean_params, "scale", index),
                ),
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
                params=ScalePriorParams(
                    sigma=_extract_serialized_prior_value(sd_params, "scale", index)
                ),
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

    bundle = load_prior_runtime_bundle(compiled_ssm.compiled_prior_semantics)
    site_by_name = {site.name: site for site in bundle.site_runtime.registry}
    site_by_field = {
        site.priors_field: site for site in bundle.site_runtime.registry if site.priors_field
    }
    binding_by_parameter = {
        binding.parameter: binding for binding in compiled_ssm.parameter_bindings
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

    for binding in compiled_ssm.parameter_bindings:
        parameter = binding.parameter
        if parameter in seen:
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


def build_model_from_compiled_artifact(
    compiled_ssm: CompiledSSMArtifact,
    wide_data: pl.DataFrame,
):
    """Build a live ``SSMModel`` from a compiled artifact and wide data."""
    if wide_data.is_empty():
        raise ValueError("Cannot build SSM model from empty data")

    from nof1_causal_lab.models.ssm.parameterization import load_prior_runtime_bundle
    from nof1_causal_lab.models.ssm.runtime import build_ssm_model

    spec = deserialize_ssm_spec(compiled_ssm.spec)
    semantics = compiled_ssm.compiled_prior_semantics
    prior_runtime_bundle = load_prior_runtime_bundle(semantics)
    return build_ssm_model(
        wide_data,
        ssm_spec=spec,
        compiled_prior_semantics=semantics,
        prior_runtime_bundle=prior_runtime_bundle,
        parameter_bindings=compiled_ssm.parameter_bindings,
    )
