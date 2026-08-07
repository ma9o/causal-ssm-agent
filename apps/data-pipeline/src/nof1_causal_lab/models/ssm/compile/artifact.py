"""Compilation and serialization helpers for executable SSM artifacts."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from nof1_causal_lab.artifacts.measurement_structure import (
    MeasurementStructure,
    check_semantic_collisions,
    validate_measurement_structure,
)
from nof1_causal_lab.artifacts.prior import (
    ExecutablePrior,
    LocationScalePriorParams,
    PriorPlan,
    ScalePriorParams,
    prior_params_model,
)
from nof1_causal_lab.artifacts.statistical_model_spec import (
    ParameterRole,
    StatisticalModelSpec,
)
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.ssm.compile.contracts import (
    CompiledParameterBinding,
    CompiledSSMArtifact,
    CompiledStructure,
    SerializedEdgeLag,
    SerializedSSMSpec,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nof1_causal_lab.artifacts.latent_structure import LatentStructure
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.models.ssm import SSMSpec


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
        "manifest_cat_anchor": _to_jsonable(spec.manifest_cat_anchor),
        "latent_names": _to_jsonable(spec.latent_names),
        "manifest_names": _to_jsonable(spec.manifest_names),
        "input_names": _to_jsonable(spec.input_names),
        "input_source_indicators": _to_jsonable(spec.input_source_indicators),
        "input_scales": _to_jsonable(spec.input_scales),
        "input_missing_policies": _to_jsonable(spec.input_missing_policies),
        "input_lagged": _to_jsonable(spec.input_lagged),
        "static_factor_names": _to_jsonable(spec.static_factor_names),
    }
    return SerializedSSMSpec.model_validate(payload)


def serialize_edge_lag_days(
    edge_lag_days: dict[tuple[int, int], float],
    structural_plan: StructuralPlan,
) -> list[SerializedEdgeLag]:
    """Convert edge-lag metadata into a JSON-serializable payload."""
    from nof1_causal_lab.utils.structural_plan import get_edges, get_state_names

    state_index = {name: index for index, name in enumerate(get_state_names(structural_plan))}
    source_id_by_target = {
        (state_index[str(edge["effect"])], state_index[str(edge["cause"])]): str(edge["source_id"])
        for edge in get_edges(structural_plan)
        if edge["effect"] in state_index and edge["cause"] in state_index
    }
    missing_sources = sorted(set(edge_lag_days) - set(source_id_by_target))
    if missing_sources:
        raise ValueError(
            f"Compiled edge lags have no StructuralPlan source binding: {missing_sources}."
        )
    return [
        SerializedEdgeLag(
            source_id=source_id_by_target[(effect_idx, cause_idx)],
            effect_idx=int(effect_idx),
            cause_idx=int(cause_idx),
            lag_days=float(lag_days),
        )
        for (effect_idx, cause_idx), lag_days in sorted(edge_lag_days.items())
    ]


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
    measurement_structure: UncheckedJsonObject,
    latent_structure: LatentStructure,
) -> tuple[MeasurementStructure | None, list[str]]:
    """Validate measurement output against schema and compile-time constraints."""
    measurement, errors = validate_measurement_structure(
        measurement_structure,
        latent_structure,
    )
    if measurement is None:
        return None, errors

    compile_errors = _collect_measurement_compile_errors(measurement, latent_structure)
    if compile_errors:
        return None, compile_errors

    return measurement, []


def collect_structural_plan_compile_errors(
    structural_plan: StructuralPlan,
    *,
    manifest_names: Sequence[str] | None = None,
) -> list[str]:
    """Validate that the retained executable structure can be compiled.

    Under the current compiler/runtime, every retained state must be
    supported by at least one manifest channel, and the loading matrix must be
    able to reach full column rank.
    """
    from nof1_causal_lab.utils.structural_plan import (
        get_manifest_indicators,
        get_state_names,
    )

    errors: list[str] = []
    try:
        latent_states = get_state_names(structural_plan)
    except ValueError as exc:
        return [str(exc)]
    if not latent_states:
        return ["structural_plan.state_order is empty"]

    indicators = get_manifest_indicators(structural_plan)
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
            "Retained states have no measurement indicators: "
            f"{uncovered_states}. Add proxy indicators for these constructs or "
            "exclude them from the executable structural plan."
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
    structural_plan: StructuralPlan | None = None,
) -> list[str]:
    """Collect deterministic StatisticalModelSpec checks that the compiler owns."""
    errors: list[str] = []
    if structural_plan is not None:
        manifest_names = [likelihood.variable for likelihood in statistical_model_spec.likelihoods]
        return collect_structural_plan_compile_errors(
            structural_plan,
            manifest_names=manifest_names,
        )

    n_manifest = len(statistical_model_spec.likelihoods)

    ar_params = [
        p for p in statistical_model_spec.parameters if p.role == ParameterRole.AR_COEFFICIENT
    ]
    if not ar_params:
        errors.append(
            "No AR_COEFFICIENT parameters found in StatisticalModelSpec; "
            "cannot infer latent dimensionality without structural_plan."
        )
        return errors

    n_latent = len(ar_params)
    if n_manifest < n_latent:
        errors.append(
            "Loading matrix is rank-deficient: "
            f"n_manifest ({n_manifest}) < inferred n_latent ({n_latent})."
        )

    return errors


def _compile_validated_ssm_artifact(
    validated_statistical_model_spec: StatisticalModelSpec,
    prior_plan: PriorPlan,
    *,
    structural_plan: StructuralPlan,
) -> CompiledSSMArtifact:
    """Compile an already-validated ``StatisticalModelSpec`` into a serialized SSM artifact."""
    from nof1_causal_lab.models.ssm.compile.inputs import (
        compile_ssm_inputs_from_statistical_model_spec,
    )
    from nof1_causal_lab.models.ssm.compile.structural import (
        compile_anchor_certificates,
        compile_structural_bindings,
    )
    from nof1_causal_lab.models.ssm.parameterization import compile_prior_semantics

    spec, prior_registry, parameter_bindings, compile_diagnostics, edge_lag_days = (
        compile_ssm_inputs_from_statistical_model_spec(
            validated_statistical_model_spec,
            prior_plan,
            structural_plan=structural_plan,
        )
    )

    structure = CompiledStructure(
        spec=serialize_ssm_spec(spec),
        edge_lag_days=serialize_edge_lag_days(edge_lag_days, structural_plan),
        bindings=compile_structural_bindings(spec, structural_plan),
        anchor_certificates=compile_anchor_certificates(spec, structural_plan),
    )
    return CompiledSSMArtifact(
        schema_version=2,
        structure=structure,
        compiled_prior_semantics=compile_prior_semantics(spec, prior_registry),
        parameter_bindings=parameter_bindings,
        compile_diagnostics=compile_diagnostics,
    )


def compile_ssm_artifact(
    statistical_model_spec: StatisticalModelSpec,
    prior_plan: PriorPlan,
    structural_plan: StructuralPlan,
) -> CompiledSSMArtifact:
    """Compile user-facing specs into an executable, serializable SSM artifact."""
    errors = _collect_statistical_model_spec_compile_errors(
        statistical_model_spec,
        structural_plan=structural_plan,
    )
    if errors:
        raise ValueError("StatisticalModelSpec failed compiler validation:\n" + "\n".join(errors))

    return _compile_validated_ssm_artifact(
        statistical_model_spec,
        prior_plan,
        structural_plan=structural_plan,
    )


def _extract_serialized_prior_value(
    params: UncheckedJsonObject,
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
    params: UncheckedJsonObject,
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


def _build_compiled_parameter_prior(
    *,
    parameter: str,
    binding: CompiledParameterBinding,
    site_by_name: UncheckedJsonObject,
    prior_state: dict[str, UncheckedJsonObject],
) -> ExecutablePrior:
    """Build one authoring-scale prior row from a compiler binding."""
    site_name = binding.site_name
    flat_index = binding.flat_index
    site = site_by_name.get(site_name)
    if site is None:
        raise ValueError(f"Compiled artifact is missing site registry entry for {site_name!r}")

    params = prior_state.get(site_name)
    if not isinstance(params, dict):
        raise ValueError(f"Compiled artifact is missing prior state for site {site_name!r}")

    distribution, distribution_params = _compiled_distribution_for_site(site, params, flat_index)
    prior_distribution = PriorDistributionFamily(distribution)
    return ExecutablePrior(
        parameter=parameter,
        distribution=prior_distribution,
        params=prior_params_model(prior_distribution, distribution_params),
    )


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
    site_by_field: UncheckedJsonObject,
    prior_state: dict[str, UncheckedJsonObject],
) -> list[ExecutablePrior]:
    """Expose implicit initial-state compiler defaults as executable priors."""
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

    rows: list[ExecutablePrior] = []
    for index, latent_name in enumerate(latent_names):
        rows.append(
            ExecutablePrior(
                parameter=f"t0_mean_{latent_name}",
                distribution=PriorDistributionFamily.NORMAL,
                params=LocationScalePriorParams(
                    mu=_extract_serialized_prior_value(mean_params, "loc", index),
                    sigma=_extract_serialized_prior_value(mean_params, "scale", index),
                ),
            )
        )
    for index, latent_name in enumerate(latent_names):
        rows.append(
            ExecutablePrior(
                parameter=f"t0_sd_{latent_name}",
                distribution=PriorDistributionFamily.HALF_NORMAL,
                params=ScalePriorParams(
                    sigma=_extract_serialized_prior_value(sd_params, "scale", index)
                ),
            )
        )
    return rows


def resolve_executable_priors(
    compiled_ssm: CompiledSSMArtifact,
    *,
    authored_plan: PriorPlan | None = None,
) -> list[ExecutablePrior]:
    """Build canonical executable prior rows from a compiled artifact.

    The compiler owns membership, ordering of bound parameters, and implicit
    defaults. Authoring-scale priors are retained when available because some
    semantic priors (for example DT-scale Beta priors on persistence) are
    intentionally lossy after compilation to the executable CT representation.
    """
    from nof1_causal_lab.models.ssm.parameterization import load_prior_runtime_bundle

    bundle = load_prior_runtime_bundle(compiled_ssm.compiled_prior_semantics)
    site_by_name = {site.name: site for site in bundle.site_runtime.registry}
    site_by_field = {
        site.priors_field: site for site in bundle.site_runtime.registry if site.priors_field
    }
    authored_priors = authored_plan.priors if authored_plan is not None else {}
    resolved: list[ExecutablePrior] = []
    seen: set[str] = set()

    for parameter, authored_prior in authored_priors.items():
        resolved.append(authored_prior)
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
        parameter = row.parameter
        if parameter in seen:
            continue
        resolved.append(row)
        seen.add(parameter)
    return resolved
