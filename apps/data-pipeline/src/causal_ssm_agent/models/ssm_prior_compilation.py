"""Pure prior-compilation and binding stages for SSM compilation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from causal_ssm_agent.artifacts.duration import parse_duration_to_hours
from causal_ssm_agent.artifacts.model_spec import ModelSpec, ParameterRole
from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.compilation_errors import AggregatedCompileError
from causal_ssm_agent.models.ssm.inference.targets.base import NUMERICAL_EPSILON
from causal_ssm_agent.models.ssm.priors import SSMPriors
from causal_ssm_agent.models.ssm.structure_runtime import SSMStructureRuntime
from causal_ssm_agent.models.ssm_compilation_common import (
    SAMPLE_SITE_FOR_PRIOR_FIELD,
    PriorIndexMaps,
    build_array_prior_payload,
    empty_prior_index_maps,
    normalize_prior_params,
)
from causal_ssm_agent.models.ssm_prior_indexing import build_prior_index_maps
from causal_ssm_agent.models.ssm_spec_translation import get_construct_dt_days
from causal_ssm_agent.workers.schemas_prior import (
    PriorPathologyCertificate,
    PriorValidationResult,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec

logger = get_prefect_logger("causal_ssm_agent.models.ssm_compilation")
CompileDiagnostic = PriorValidationResult
PriorFailureStage = Literal[
    "compiled_parameters",
    "latent_dynamics",
    "observation_mean",
    "observation_sample",
    "support_violation",
    "model_build",
    "prior_sampling",
    "unknown",
]


class PriorCompilationError(AggregatedCompileError):
    """Aggregate independent prior-compilation failures into one exception."""

    header = "Prior compilation failed"


def _iter_offdiag_positions(ssm_spec: SSMSpec) -> list[tuple[int, int]]:
    return list(SSMStructureRuntime(ssm_spec).offdiag_positions)


def _drift_parameter_name(
    ssm_spec: SSMSpec,
    effect_idx: int,
    cause_idx: int,
) -> tuple[str, str, str]:
    cause_name = (
        ssm_spec.latent_names[cause_idx] if ssm_spec.latent_names else f"latent_{cause_idx}"
    )
    effect_name = (
        ssm_spec.latent_names[effect_idx] if ssm_spec.latent_names else f"latent_{effect_idx}"
    )
    return f"beta_{cause_name}_{effect_name}", cause_name, effect_name


def _resolve_model_clock_interval_days(causal_spec: dict | None) -> float | None:
    """Resolve the declared model clock interval without silently defaulting to 1 day."""
    if causal_spec is None:
        return None

    model_clock = (
        causal_spec.get("measurement", {}).get("model_clock")
        if isinstance(causal_spec, dict)
        else getattr(getattr(causal_spec, "measurement", None), "model_clock", None)
    )
    if not model_clock:
        return None

    try:
        interval_days = parse_duration_to_hours(model_clock) / 24.0
    except ValueError as exc:
        raise ValueError(
            "causal_spec.measurement.model_clock must parse to a positive interval to "
            "compile cross-lag priors without explicit reference_interval_days."
        ) from exc

    if interval_days <= 0:
        raise ValueError(
            "causal_spec.measurement.model_clock must resolve to a positive interval to "
            "compile cross-lag priors."
        )
    return interval_days


def _resolve_cross_lag_interval_days(
    *,
    param_name: str,
    prior_spec: dict[str, Any],
    flat_index: int,
    structure_runtime: SSMStructureRuntime,
    ssm_spec: SSMSpec,
    edge_lag_days: dict[tuple[int, int], float] | None,
    causal_spec: dict | None,
) -> float:
    """Resolve a positive authoring interval for cross-lag priors."""
    ref_days = prior_spec.get("reference_interval_days")
    if ref_days is not None:
        interval_days = float(ref_days)
        if interval_days <= 0:
            raise ValueError(
                f"Cross-lag prior '{param_name}' must set reference_interval_days to a "
                f"positive value, got {interval_days:.3g}."
            )
        return interval_days

    if flat_index >= structure_runtime.n_drift_offdiag:
        raise ValueError(
            f"Cross-lag prior '{param_name}' resolved to invalid flat index {flat_index}."
        )

    effect_idx, cause_idx = structure_runtime.offdiag_positions[flat_index]
    lag_days = (edge_lag_days or {}).get((effect_idx, cause_idx))
    if lag_days is not None:
        interval_days = float(lag_days)
        if interval_days <= 0:
            raise ValueError(
                f"Cross-lag prior '{param_name}' maps to non-positive edge lag {interval_days:.3g}."
            )
        return interval_days

    effect_name = (
        ssm_spec.latent_names[effect_idx] if ssm_spec.latent_names else f"latent_{effect_idx}"
    )
    interval_days = _resolve_model_clock_interval_days(causal_spec)
    if interval_days is not None:
        return interval_days

    raise ValueError(
        f"Cross-lag prior '{param_name}' could not resolve an authoring interval. "
        "Set reference_interval_days explicitly, or compile with edge_lag_days / "
        f"causal_spec model_clock metadata for effect '{effect_name}'."
    )


def _format_interval_days(days: float) -> str:
    """Render a positive day interval for diagnostics."""
    return f"{float(days):.1f}d"


def _collect_source_intervals(prior_spec: dict[str, Any]) -> list[float]:
    """Extract positive `study_interval_days` metadata from prior sources."""
    intervals: list[float] = []
    for source in prior_spec.get("sources") or []:
        if not isinstance(source, dict):
            continue
        value = source.get("study_interval_days")
        try:
            days = float(value)
        except (TypeError, ValueError):
            continue
        if days > 0:
            intervals.append(days)
    return sorted(intervals)


def _interval_ratio(lhs: float, rhs: float) -> float:
    """Return the larger/smaller ratio for two positive intervals."""
    return max(lhs, rhs) / max(min(lhs, rhs), NUMERICAL_EPSILON)


def _compile_warning(
    *,
    code: str,
    parameter: str,
    issue: str,
    suggested_adjustment: str,
    compiled_site_name: str | None = None,
    compiled_flat_index: int | None = None,
    failure_stage: PriorFailureStage | None = None,
    pathology_certificate: PriorPathologyCertificate | None = None,
) -> CompileDiagnostic:
    """Build a typed non-fatal compile diagnostic."""
    return CompileDiagnostic(
        parameter=parameter,
        is_valid=True,
        code=code,
        origin="compile",
        severity="warning",
        issue=issue,
        suggested_adjustment=suggested_adjustment,
        related_parameters=[parameter],
        compiled_site_name=compiled_site_name,
        compiled_flat_index=compiled_flat_index,
        failure_stage=failure_stage,
        pathology_certificate=pathology_certificate,
    )


def collect_interval_provenance_warnings(
    ssm_spec: SSMSpec,
    *,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    raw_priors: dict[str, dict] | None = None,
) -> list[CompileDiagnostic]:
    """Collect deterministic interval-authoring diagnostics for lagged drift priors."""
    edge_lags = edge_lag_days or {}
    if not edge_lags:
        return []

    positions = _iter_offdiag_positions(ssm_spec)
    warnings: list[CompileDiagnostic] = []

    for effect_idx, cause_idx in positions:
        if (effect_idx, cause_idx) not in edge_lags:
            continue

        parameter_name, cause_name, effect_name = _drift_parameter_name(
            ssm_spec,
            effect_idx,
            cause_idx,
        )
        expected_lag_days = edge_lags[(effect_idx, cause_idx)]
        prior_spec = (raw_priors or {}).get(parameter_name) or {}
        ref_days = prior_spec.get("reference_interval_days")
        source_intervals = _collect_source_intervals(prior_spec)
        if not source_intervals:
            continue

        unique_source_intervals = sorted(dict.fromkeys(source_intervals))
        if (
            len(unique_source_intervals) > 1
            and _interval_ratio(unique_source_intervals[0], unique_source_intervals[-1]) > 2.0
        ):
            rendered = ", ".join(_format_interval_days(days) for days in unique_source_intervals)
            warnings.append(
                _compile_warning(
                    code="interval_sources_mixed",
                    parameter=parameter_name,
                    issue=(
                        f"Cited sources for {cause_name}->{effect_name} mix materially different "
                        f"study intervals ({rendered}), so the authored interval provenance is weak."
                    ),
                    suggested_adjustment=(
                        "Use sources measured on a comparable interval when possible, or explain "
                        "which interval the prior is expressed on with `reference_interval_days`."
                    ),
                )
            )

        source_interval_days = unique_source_intervals[0]
        if ref_days is None and _interval_ratio(source_interval_days, expected_lag_days) > 2.0:
            warnings.append(
                _compile_warning(
                    code="interval_reference_missing",
                    parameter=parameter_name,
                    issue=(
                        f"`reference_interval_days` is omitted, so {parameter_name} is being "
                        f"interpreted on the default model interval ({_format_interval_days(expected_lag_days)}), "
                        f"but the cited evidence for {cause_name}->{effect_name} is on "
                        f"{_format_interval_days(source_interval_days)}."
                    ),
                    suggested_adjustment=(
                        "If the authored effect is meant to be on the source study interval, set "
                        "`reference_interval_days` to that interval. Otherwise explain why a "
                        "daily-scale prior is appropriate."
                    ),
                )
            )
        elif ref_days is not None:
            try:
                authored_interval_days = float(ref_days)
            except (TypeError, ValueError):
                continue
            if (
                authored_interval_days > 0
                and _interval_ratio(authored_interval_days, source_interval_days) > 2.0
            ):
                warnings.append(
                    _compile_warning(
                        code="interval_reference_mismatch",
                        parameter=parameter_name,
                        issue=(
                            f"{parameter_name} is authored on "
                            f"{_format_interval_days(authored_interval_days)} via `reference_interval_days`, "
                            f"but the cited evidence for {cause_name}->{effect_name} is on "
                            f"{_format_interval_days(source_interval_days)}."
                        ),
                        suggested_adjustment=(
                            "Confirm that the prior was intentionally rescaled to the authored interval, "
                            "or align `reference_interval_days` with the evidence interval."
                        ),
                    )
                )

    return warnings


def collect_compile_diagnostics(
    ssm_spec: SSMSpec,
    *,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    raw_priors: dict[str, dict] | None = None,
    ssm_priors: SSMPriors | None = None,
) -> list[CompileDiagnostic]:
    """Collect structured compiler diagnostics for downstream consumers."""
    diagnostics = collect_interval_provenance_warnings(
        ssm_spec,
        edge_lag_days=edge_lag_days,
        raw_priors=raw_priors,
    )
    if ssm_priors is not None:
        diagnostics.extend(collect_first_order_approximation_warnings(ssm_priors))
    return diagnostics


def _log_compile_diagnostics(diagnostics: list[CompileDiagnostic]) -> None:
    for issue in diagnostics:
        logger.warning("%s: %s", issue.parameter, issue.issue)


def collect_first_order_approximation_warnings(
    ssm_priors: SSMPriors,
) -> list[CompileDiagnostic]:
    """Return typed warnings when the first-order DT->CT approximation looks weak."""
    diag_prior = ssm_priors.drift_diag
    offdiag_prior = ssm_priors.drift_offdiag
    if diag_prior is None or offdiag_prior is None:
        return []

    diag_mu = diag_prior.get("mu")
    offdiag_mu = offdiag_prior.get("mu")
    if diag_mu is None or offdiag_mu is None:
        return []

    if isinstance(diag_mu, (int, float)):
        diag_mu = [diag_mu]
    if isinstance(offdiag_mu, (int, float)):
        offdiag_mu = [offdiag_mu]
    if not diag_mu or not offdiag_mu:
        return []

    min_diag = min(abs(float(value)) for value in diag_mu)
    if min_diag < NUMERICAL_EPSILON:
        return []

    warnings: list[CompileDiagnostic] = []
    for idx, offdiag_value in enumerate(offdiag_mu):
        ratio = abs(float(offdiag_value)) / min_diag
        if ratio <= 0.2:
            continue
        warnings.append(
            _compile_warning(
                code="dt_ct_approximation_warning",
                parameter="drift_offdiag",
                issue=(
                    "First-order DT->CT approximation may be inaccurate: "
                    f"off-diagonal drift[{idx}] magnitude ({abs(float(offdiag_value)):.3f}) is "
                    f"{ratio * 100:.0f}% of minimum diagonal magnitude ({min_diag:.3f})."
                ),
                suggested_adjustment=(
                    "Consider a shorter reference interval or elicit priors directly on CT rates."
                ),
                compiled_site_name="drift_offdiag_free",
                compiled_flat_index=idx,
                failure_stage="compiled_parameters",
                pathology_certificate=PriorPathologyCertificate(
                    kind="dt_ct_approximation",
                    primary_score=ratio,
                ),
            )
        )
    return warnings


def _collect_role_lookup(model_spec: ModelSpec | dict | None) -> dict[str, ParameterRole]:
    role_by_name: dict[str, ParameterRole] = {}
    spec_obj: ModelSpec | None = None
    if isinstance(model_spec, dict) and model_spec.get("parameters"):
        spec_obj = ModelSpec.model_validate(model_spec)
    elif isinstance(model_spec, ModelSpec):
        spec_obj = model_spec

    if spec_obj is None:
        return role_by_name

    for parameter in spec_obj.parameters:
        role_by_name[parameter.name] = parameter.role
    return role_by_name


def _append_structured_prior(
    per_element: dict[str, list[tuple[int, dict[str, float | int]]]],
    attr: str,
    idx: int,
    normalized: dict[str, float | int],
) -> None:
    per_element.setdefault(attr, []).append((idx, normalized))


def _coerce_initial_state_correlation_prior(
    normalized: dict[str, float | int],
) -> dict[str, float | int]:
    """Interpret authored initial-state priors on the correlation scale."""
    coerced = dict(normalized)
    lower = max(float(coerced.get("lower", -1.0)), -1.0)
    upper = min(float(coerced.get("upper", 1.0)), 1.0)
    if lower >= upper:
        raise ValueError(
            "Initial-state correlation priors must have bounds within [-1, 1] "
            f"with lower < upper; got lower={lower}, upper={upper}."
        )
    coerced["lower"] = lower
    coerced["upper"] = upper
    return coerced


def compile_priors(
    raw_priors: dict[str, dict],
    model_spec: ModelSpec | dict | None,
    ssm_spec: SSMSpec | None,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    causal_spec: dict | None = None,
) -> tuple[SSMPriors, PriorIndexMaps, list[CompileDiagnostic]]:
    """Compile prior proposals into ``SSMPriors`` with explicit index maps."""
    ssm_priors = SSMPriors()
    role_by_name = _collect_role_lookup(model_spec)
    per_element: dict[str, list[tuple[int, dict[str, float | int]]]] = {}
    has_model_spec = model_spec is not None and (
        not isinstance(model_spec, dict) or bool(model_spec)
    )
    resolved_model_spec = model_spec if has_model_spec else None
    if resolved_model_spec is not None and ssm_spec is not None:
        index_maps = build_prior_index_maps(ssm_spec, resolved_model_spec, causal_spec=causal_spec)
    elif raw_priors and not has_model_spec:
        raise ValueError(
            "compile_priors() requires model_spec when compiling semantic prior proposals."
        )
    elif raw_priors and ssm_spec is None:
        raise ValueError(
            "compile_priors() requires a translated SSMSpec when compiling semantic prior "
            "proposals."
        )
    else:
        index_maps = empty_prior_index_maps()
    (
        offdiag_param_index,
        lambda_param_index,
        diag_param_index,
        diffusion_diag_param_index,
        diffusion_offdiag_param_index,
        t0_offdiag_param_index,
        t0_mean_param_index,
        t0_sd_param_index,
        manifest_mean_param_index,
        manifest_var_param_index,
        cint_param_index,
        static_state_sd_param_index,
        observation_site_param_index,
    ) = index_maps
    structure_runtime = SSMStructureRuntime(ssm_spec) if ssm_spec is not None else None
    errors: list[str] = []

    for param_name, prior_spec in raw_priors.items():
        try:
            distribution = prior_spec.get("distribution", "Normal")
            normalized = normalize_prior_params(distribution, prior_spec.get("params", {}))

            if param_name in diag_param_index:
                attr, idx = diag_param_index[param_name]
                construct_name = param_name.removeprefix("rho_").removeprefix("ar_")
                ref_days = prior_spec.get("reference_interval_days")
                dt = (
                    float(ref_days)
                    if ref_days is not None and ref_days > 0
                    else get_construct_dt_days(causal_spec, construct_name)
                )
                param_errors: list[str] = []
                lower = normalized.get("lower")
                upper = normalized.get("upper")
                if lower is not None and float(lower) < 0.0:
                    param_errors.append(
                        f"AR prior '{param_name}' must be on the DT persistence scale in [0, 1], "
                        f"but lower bound is {float(lower):.3g}"
                    )
                if upper is not None and float(upper) > 1.0:
                    param_errors.append(
                        f"AR prior '{param_name}' must be on the DT persistence scale in [0, 1], "
                        f"but upper bound is {float(upper):.3g}"
                    )

                mu_ar = float(normalized.get("mu", 0.5))
                if not 0.0 < mu_ar < 1.0:
                    param_errors.append(
                        f"AR prior '{param_name}' must have DT persistence mean in (0, 1), got {mu_ar:.3g}"
                    )
                if param_errors:
                    errors.extend(param_errors)
                    continue

                mu_ar = min(max(mu_ar, 0.001), 0.999)
                sigma_ar = normalized.get("sigma", 0.2)
                _append_structured_prior(
                    per_element,
                    attr,
                    idx,
                    {"mu": -math.log(mu_ar) / dt, "sigma": sigma_ar / (mu_ar * dt)},
                )
                continue

            if param_name in offdiag_param_index:
                attr, idx = offdiag_param_index[param_name]
                if structure_runtime is None or ssm_spec is None:
                    raise ValueError(
                        "Cross-lag prior compilation requires a translated SSMSpec runtime."
                    )
                dt = _resolve_cross_lag_interval_days(
                    param_name=param_name,
                    prior_spec=prior_spec,
                    flat_index=idx,
                    structure_runtime=structure_runtime,
                    ssm_spec=ssm_spec,
                    edge_lag_days=edge_lag_days,
                    causal_spec=causal_spec,
                )
                _append_structured_prior(
                    per_element,
                    attr,
                    idx,
                    {
                        "mu": normalized.get("mu", 0.0) / dt,
                        "sigma": normalized.get("sigma", 0.5) / dt,
                    },
                )
                continue

            if param_name in lambda_param_index:
                attr, idx = lambda_param_index[param_name]
                _append_structured_prior(per_element, attr, idx, normalized)
                continue

            if param_name in cint_param_index:
                attr, idx = cint_param_index[param_name]
                _append_structured_prior(per_element, attr, idx, normalized)
                continue

            if param_name in static_state_sd_param_index:
                attr, idx = static_state_sd_param_index[param_name]
                _append_structured_prior(per_element, attr, idx, normalized)
                continue

            if param_name in manifest_mean_param_index:
                attr, idx = manifest_mean_param_index[param_name]
                _append_structured_prior(per_element, attr, idx, normalized)
                continue

            if param_name in manifest_var_param_index:
                attr, idx = manifest_var_param_index[param_name]
                _append_structured_prior(per_element, attr, idx, normalized)
                continue

            if param_name in diffusion_diag_param_index:
                attr, idx = diffusion_diag_param_index[param_name]
                _append_structured_prior(per_element, attr, idx, normalized)
                continue

            if param_name in t0_mean_param_index:
                attr, idx = t0_mean_param_index[param_name]
                _append_structured_prior(per_element, attr, idx, normalized)
                continue

            if param_name in t0_sd_param_index:
                attr, idx = t0_sd_param_index[param_name]
                _append_structured_prior(per_element, attr, idx, normalized)
                continue

            if param_name in diffusion_offdiag_param_index:
                attr, idx = diffusion_offdiag_param_index[param_name]
                _append_structured_prior(per_element, attr, idx, normalized)
                continue

            if param_name in t0_offdiag_param_index:
                attr, idx = t0_offdiag_param_index[param_name]
                _append_structured_prior(
                    per_element,
                    attr,
                    idx,
                    _coerce_initial_state_correlation_prior(normalized),
                )
                continue

            if param_name in observation_site_param_index:
                attr, _idx = observation_site_param_index[param_name]
                setattr(ssm_priors, attr, dict(normalized))
                continue

            role = role_by_name.get(param_name)
            if role is not None:
                errors.append(
                    f"Prior {param_name!r} with role {role.value!r} could not be structurally "
                    "bound to the compiled SSM. Compile priors with a translated SSMSpec that "
                    "matches the ModelSpec."
                )
                continue

            errors.append(
                f"Prior {param_name!r} does not correspond to any parameter in ModelSpec."
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue

    if errors:
        raise PriorCompilationError(errors)

    for attr, entries in per_element.items():
        current = getattr(ssm_priors, attr)
        result = build_array_prior_payload(attr, entries, current, ssm_spec)
        setattr(ssm_priors, attr, result)

    diagnostics: list[CompileDiagnostic] = []
    if ssm_spec is not None:
        diagnostics = collect_compile_diagnostics(
            ssm_spec,
            edge_lag_days=edge_lag_days,
            raw_priors=raw_priors,
            ssm_priors=ssm_priors,
        )
        _log_compile_diagnostics(diagnostics)

    return ssm_priors, index_maps, diagnostics


def bind_parameters(index_maps: PriorIndexMaps) -> list[dict[str, Any]]:
    """Map semantic parameter names to NumPyro sample sites."""
    (
        offdiag_index,
        lambda_index,
        diag_index,
        diffusion_diag_index,
        diffusion_offdiag_index,
        t0_offdiag_index,
        t0_mean_index,
        t0_sd_index,
        manifest_mean_index,
        manifest_var_index,
        cint_index,
        static_state_sd_index,
        observation_site_index,
    ) = index_maps

    bindings: list[dict[str, Any]] = []
    ordered_maps = (
        diag_index,
        offdiag_index,
        diffusion_diag_index,
        cint_index,
        static_state_sd_index,
        t0_mean_index,
        t0_sd_index,
        diffusion_offdiag_index,
        t0_offdiag_index,
        lambda_index,
        manifest_mean_index,
        manifest_var_index,
        observation_site_index,
    )
    for mapping in ordered_maps:
        for param_name, (prior_field, flat_index) in sorted(mapping.items()):
            sample_site = SAMPLE_SITE_FOR_PRIOR_FIELD.get(prior_field)
            if sample_site is None:
                continue
            bindings.append(
                {
                    "parameter": param_name,
                    "site_name": sample_site,
                    "flat_index": flat_index,
                }
            )

    bindings.sort(key=lambda entry: str(entry["parameter"]))
    return bindings
