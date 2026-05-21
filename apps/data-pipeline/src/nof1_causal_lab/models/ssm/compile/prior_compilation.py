"""Pure prior-compilation and binding stages for SSM compilation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.linalg

from nof1_causal_lab.artifacts.duration import parse_duration_to_hours
from nof1_causal_lab.artifacts.model_spec import ModelSpec, ParameterRole
from nof1_causal_lab.distributions import (
    PriorDistributionFamily,
    get_positive_runtime_family_index,
    get_real_runtime_family_index,
)
from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.ssm.compile.common import (
    axis_names_with_fallback,
    normalize_prior_params,
    resolve_scalar_parameter_name,
)
from nof1_causal_lab.models.ssm.compile.prior_indexing import (
    SemanticBindingRegistry,
    build_semantic_prior_bindings,
    empty_prior_bindings,
)
from nof1_causal_lab.models.ssm.compile.spec_translation import get_construct_dt_days
from nof1_causal_lab.models.ssm.inference.targets.base import NUMERICAL_EPSILON
from nof1_causal_lab.models.ssm.parameter_layout import SSMParameterLayout
from nof1_causal_lab.models.ssm.parameterization import SupportClass, build_site_registry
from nof1_causal_lab.models.ssm.priors import (
    PriorRegistry,
    PriorSpec,
    default_prior_for_descriptor,
    default_prior_registry,
    prior_spec_from_normalized_params,
    prior_spec_to_normalized_params,
)
from nof1_causal_lab.models.ssm.structure.sites import (
    PriorAuthoringTransform,
    SiteDescriptor,
    SiteKind,
    site_size,
)
from nof1_causal_lab.workers.schemas_prior import (
    PriorPathologyCertificate,
    PriorValidationResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.models.ssm.model import SSMSpec

logger = get_prefect_logger("nof1_causal_lab.models.ssm.compile.inputs")
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
_LOGM_IMAG_TOL = 1e-8
_LOGM_RELATIVE_DEVIATION_WARNING_THRESHOLD = 0.2

_NONDEGENERATE_TOL = 1e-12

_DEGENERATE_PRIOR_PREAMBLE = (
    "Stage 4 priors must have strictly positive variance. Zero-width priors assert "
    "the parameter's value with infinite certainty, which is a structural claim "
    "rather than a Bayesian belief. Legitimate fixed-value cases (identification "
    "fixings, baseline policies) belong on the structural surface — the skeleton "
    "parameter list or model-spec policy toggles — not on the prior surface."
)


def _validate_nondegenerate_prior(
    parameter: str,
    distribution: PriorDistributionFamily | str,
    raw_params: dict[str, Any],
) -> list[str]:
    """Reject Stage-4-authored priors with zero variance or explicit point masses.

    The validator runs on the LLM-authored (raw) params dict so error messages
    reference what was actually authored. Family-specific positivity checks cover
    every distribution that ``normalize_prior_params`` accepts.
    """
    family_str = (
        distribution.value
        if isinstance(distribution, PriorDistributionFamily)
        else str(distribution)
    )
    family_lower = family_str.lower().replace("-", "_")
    issues: list[str] = []

    def _err(detail: str, suggestion: str = "") -> str:
        message = f"Prior {parameter!r} ({family_str}): {detail}. {_DEGENERATE_PRIOR_PREAMBLE}"
        if suggestion:
            message += " " + suggestion
        return message

    if family_lower in {"delta", "dirac"}:
        return [
            _err(
                "Delta/point-mass priors are not supported on the prior surface",
                "If a parameter must be fixed, change the structural surface instead.",
            )
        ]

    if "value" in raw_params:
        issues.append(
            _err(
                "explicit 'value' fields are not supported on the prior surface",
                "If you intended to fix the parameter, use the structural surface.",
            )
        )

    sigma = raw_params.get("sigma")
    lower = raw_params.get("lower")
    upper = raw_params.get("upper")
    alpha = raw_params.get("alpha")
    beta = raw_params.get("beta")
    concentration = raw_params.get("concentration")
    rate = raw_params.get("rate")

    sigma_families = {
        "normal",
        "truncated_normal",
        "truncatednormal",
        "half_normal",
        "halfnormal",
        "log_normal",
        "lognormal",
    }
    if family_lower in sigma_families and sigma is not None and float(sigma) <= _NONDEGENERATE_TOL:
        issues.append(_err(f"sigma={float(sigma):.3g} must be strictly positive"))

    bound_families = {"uniform", "truncated_normal", "truncatednormal"}
    if (
        family_lower in bound_families
        and lower is not None
        and upper is not None
        and float(upper) - float(lower) <= _NONDEGENERATE_TOL
    ):
        issues.append(
            _err(
                f"support [{float(lower):.4g}, {float(upper):.4g}] has zero width "
                f"(lower must be strictly less than upper)",
                f"For tight belief near {float(lower):.4g}, author a Beta or "
                "Gamma with small but positive sd, or widen the bounds.",
            )
        )

    if family_lower == "beta":
        if alpha is not None and float(alpha) <= 0.0:
            issues.append(_err(f"alpha={float(alpha):.3g} must be strictly positive"))
        if beta is not None and float(beta) <= 0.0:
            issues.append(_err(f"beta={float(beta):.3g} must be strictly positive"))

    if family_lower == "gamma":
        if concentration is not None and float(concentration) <= 0.0:
            issues.append(
                _err(f"concentration={float(concentration):.3g} must be strictly positive")
            )
        if rate is not None and float(rate) <= 0.0:
            issues.append(_err(f"rate={float(rate):.3g} must be strictly positive"))

    if family_lower == "exponential" and rate is not None and float(rate) <= 0.0:
        issues.append(_err(f"rate={float(rate):.3g} must be strictly positive"))

    return issues


class PriorCompilationError(AggregatedCompileError):
    """Aggregate independent prior-compilation failures into one exception."""

    header = "Prior compilation failed"


def _iter_offdiag_positions(ssm_spec: SSMSpec) -> list[tuple[int, int]]:
    return list(SSMParameterLayout.from_spec(ssm_spec).offdiag_positions)


def _structural_dense_component(ssm_spec: SSMSpec):
    from nof1_causal_lab.models.ssm.dynamics.composite import StructuralDenseLinearSpec

    for component in ssm_spec.drift_spec.components:
        if isinstance(component, StructuralDenseLinearSpec):
            return component
    raise TypeError("Matrix-log drift diagnostics require a StructuralDenseLinearSpec component.")


def _drift_parameter_name(
    ssm_spec: SSMSpec,
    effect_idx: int,
    cause_idx: int,
    *,
    parameter_layout: SSMParameterLayout | None = None,
) -> tuple[str, str, str]:
    if not ssm_spec.latent_names:
        raise ValueError(
            "SSMSpec.latent_names is empty; cross-lag parameter names require explicit "
            "latent_names on the translated SSMSpec."
        )
    runtime = parameter_layout or SSMParameterLayout.from_spec(ssm_spec)
    flat_idx = runtime.offdiag_index.get((effect_idx, cause_idx))
    if flat_idx is None:
        raise ValueError(f"No drift_offdiag entry at latent pair ({effect_idx}, {cause_idx}).")
    offdiag_site = runtime.site_by_kind(SiteKind.DRIFT_OFFDIAG)
    if offdiag_site is None:
        raise ValueError("No active drift off-diagonal sample site.")
    name = resolve_scalar_parameter_name(ssm_spec, runtime, offdiag_site.name, flat_idx)
    if name is None:
        raise ValueError(
            f"resolve_scalar_parameter_name failed for {offdiag_site.name}[{flat_idx}]."
        )
    cause_name = ssm_spec.latent_names[cause_idx]
    effect_name = ssm_spec.latent_names[effect_idx]
    return name, cause_name, effect_name


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
    parameter_layout: SSMParameterLayout,
    ssm_spec: SSMSpec,
    edge_lag_days: dict[tuple[int, int], float] | None,
    causal_spec: dict | None,
    effect_idx: int | None = None,
    cause_idx: int | None = None,
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

    if effect_idx is None or cause_idx is None:
        if flat_index >= parameter_layout.n_drift_offdiag:
            raise ValueError(
                f"Cross-lag prior '{param_name}' resolved to invalid flat index {flat_index}."
            )
        effect_idx, cause_idx = parameter_layout.offdiag_positions[flat_index]

    if effect_idx is None or cause_idx is None:
        raise ValueError(f"Cross-lag prior '{param_name}' is missing effect/cause metadata.")

    lag_days = (edge_lag_days or {}).get((effect_idx, cause_idx))
    if lag_days is not None:
        interval_days = float(lag_days)
        if interval_days <= 0:
            raise ValueError(
                f"Cross-lag prior '{param_name}' maps to non-positive edge lag {interval_days:.3g}."
            )
        return interval_days

    if not ssm_spec.latent_names:
        raise ValueError(
            f"Cross-lag prior '{param_name}' cannot resolve effect name: "
            "SSMSpec.latent_names is empty."
        )
    effect_name = ssm_spec.latent_names[effect_idx]
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
    prior_registry: PriorRegistry | None = None,
    offdiag_interval_days: dict[int, float] | None = None,
) -> list[CompileDiagnostic]:
    """Collect structured compiler diagnostics for downstream consumers."""
    diagnostics = collect_interval_provenance_warnings(
        ssm_spec,
        edge_lag_days=edge_lag_days,
        raw_priors=raw_priors,
    )
    if prior_registry is not None:
        diagnostics.extend(
            collect_first_order_approximation_warnings(
                prior_registry,
                ssm_spec=ssm_spec,
                edge_lag_days=edge_lag_days,
                offdiag_interval_days=offdiag_interval_days,
            )
        )
    return diagnostics


def _log_compile_diagnostics(diagnostics: list[CompileDiagnostic]) -> None:
    for issue in diagnostics:
        logger.warning("%s: %s", issue.parameter, issue.issue)


def collect_first_order_approximation_warnings(
    prior_registry: PriorRegistry,
    *,
    ssm_spec: SSMSpec | None = None,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    offdiag_interval_days: dict[int, float] | None = None,
) -> list[CompileDiagnostic]:
    """Return warnings when exact matrix-log DT->CT diagnostics diverge from beta/dt."""
    if ssm_spec is None:
        return []
    parameter_layout = SSMParameterLayout.from_spec(ssm_spec)
    base_decay_site = parameter_layout.site_by_kind(SiteKind.DRIFT_BASE_DECAY)
    offdiag_site = parameter_layout.site_by_kind(SiteKind.DRIFT_OFFDIAG)
    if base_decay_site is None or offdiag_site is None:
        return []
    base_decay_prior = _prior_for_site(prior_registry, base_decay_site.name)
    offdiag_prior = _prior_for_site(prior_registry, offdiag_site.name)
    if base_decay_prior is None or offdiag_prior is None:
        return []

    base_decay_mu = _positive_prior_mean_values(base_decay_prior)
    offdiag_mu = _prior_values_1d(offdiag_prior.params.get("mu"))
    if base_decay_mu.size == 0 or offdiag_mu.size == 0:
        return []

    taylor_drift = _assemble_mean_drift_from_prior_values(
        ssm_spec,
        parameter_layout,
        base_decay_mu=base_decay_mu,
        offdiag_mu=offdiag_mu,
    )
    diag_abs = np.abs(np.diag(taylor_drift))
    ti_mask = np.zeros_like(diag_abs, dtype=bool)
    drift_component = _structural_dense_component(ssm_spec)
    if drift_component.time_invariant_mask is not None:
        candidate = np.asarray(drift_component.time_invariant_mask, dtype=bool)
        if candidate.size == diag_abs.size:
            ti_mask = candidate
    eligible_mask = (diag_abs >= NUMERICAL_EPSILON) & ~ti_mask
    if not np.any(eligible_mask):
        return []
    eligible_indices = np.where(eligible_mask)[0]
    min_diag = float(np.min(diag_abs[eligible_indices]))
    if min_diag < NUMERICAL_EPSILON:
        return []
    min_diag_latent_idx = int(eligible_indices[int(np.argmin(diag_abs[eligible_indices]))])
    min_diag_flat_idx = parameter_layout.drift_base_decay_index.get(min_diag_latent_idx)

    min_diag_name = (
        resolve_scalar_parameter_name(
            ssm_spec, parameter_layout, base_decay_site.name, min_diag_flat_idx
        )
        if min_diag_flat_idx is not None
        else None
    )
    min_diag_label = f"{min_diag_name}" if min_diag_name else f"latent[{min_diag_latent_idx}]"

    warnings: list[CompileDiagnostic] = []
    for idx, offdiag_value in enumerate(offdiag_mu):
        if idx >= len(parameter_layout.offdiag_positions):
            continue
        effect_idx, cause_idx = parameter_layout.offdiag_positions[idx]
        interval_days = _resolve_offdiag_interval_days(
            idx,
            effect_idx=effect_idx,
            cause_idx=cause_idx,
            edge_lag_days=edge_lag_days,
            offdiag_interval_days=offdiag_interval_days,
        )
        if interval_days is None:
            continue

        beta_name = resolve_scalar_parameter_name(
            ssm_spec, parameter_layout, offdiag_site.name, idx
        )
        if beta_name is not None:
            latent_names = axis_names_with_fallback(
                ssm_spec.latent_names,
                expected=ssm_spec.n_latent,
                prefix="latent",
            )
            cause_name = latent_names[cause_idx]
            effect_name = latent_names[effect_idx]
            offdiag_label = f"{beta_name} ({cause_name} -> {effect_name})"
        else:
            offdiag_label = f"drift_offdiag[{idx}]"

        try:
            exact_drift = matrix_log_diagnostic_drift(
                ssm_spec,
                taylor_drift,
                interval_days=interval_days,
            )
        except ValueError as exc:
            warnings.append(
                _compile_warning(
                    code="dt_ct_approximation_warning",
                    parameter="drift_offdiag",
                    issue=f"{offdiag_label}: exact matrix-log CT diagnostic failed: {exc}",
                    suggested_adjustment=(
                        "Shrink the DT beta prior or elicit the prior directly on a real, stable "
                        "CT drift scale."
                    ),
                    compiled_site_name=offdiag_site.name,
                    compiled_flat_index=idx,
                    failure_stage="compiled_parameters",
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dt_ct_approximation",
                        primary_score=1.0,
                    ),
                )
            )
            continue

        exact_value = float(exact_drift[effect_idx, cause_idx])
        deviation = abs(exact_value - float(offdiag_value)) / max(
            abs(exact_value), NUMERICAL_EPSILON
        )
        ratio = abs(exact_value) / min_diag
        if deviation <= _LOGM_RELATIVE_DEVIATION_WARNING_THRESHOLD and ratio <= 0.2:
            continue
        warnings.append(
            _compile_warning(
                code="dt_ct_approximation_warning",
                parameter="drift_offdiag",
                issue=(
                    f"{offdiag_label}: matrix-log mismatch; exact CT coupling at "
                    f"{_format_interval_days(interval_days)} is {abs(exact_value):.3f} 1/day "
                    f"versus the elementwise beta/dt value {abs(float(offdiag_value)):.3f} "
                    f"1/day; logm deviation is {deviation * 100:.0f}% and the exact coupling "
                    f"is {ratio * 100:.0f}% of the smallest realised CT diagonal damping "
                    f"({min_diag:.3f} 1/day, {min_diag_label})."
                ),
                suggested_adjustment=(
                    "Use the exact matrix-log CT scale when revising this edge: shorten the "
                    "reference interval, shrink the DT beta prior, or elicit the prior directly "
                    "on the CT rate."
                ),
                compiled_site_name=offdiag_site.name,
                compiled_flat_index=idx,
                failure_stage="compiled_parameters",
                pathology_certificate=PriorPathologyCertificate(
                    kind="dt_ct_approximation",
                    primary_score=deviation,
                    secondary_score=ratio,
                ),
            )
        )
    return warnings


def _prior_values_1d(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    array = np.asarray(value if isinstance(value, list | tuple) else [value], dtype=float)
    return array.reshape(-1)


def _prior_param_values(
    params: Mapping[str, Any], key: str, *, n: int, default: float
) -> np.ndarray:
    values = _prior_values_1d(params.get(key))
    if values.size == 0:
        return np.full(n, default, dtype=float)
    if values.size == 1:
        return np.full(n, float(values[0]), dtype=float)
    if values.size != n:
        raise ValueError(f"Prior field {key!r} has {values.size} values; expected {n}.")
    return values.astype(float)


def _positive_prior_mean_values(prior: PriorSpec) -> np.ndarray:
    params = prior.params
    sizes = [
        _prior_values_1d(params.get(key)).size
        for key in ("sigma", "mu", "loc", "concentration", "rate", "value")
    ]
    n = max(sizes, default=0)
    if n == 0:
        return np.asarray([], dtype=float)

    scale = _prior_param_values(params, "sigma", n=n, default=1.0)
    loc_key = "mu" if "mu" in params else "loc"
    loc = _prior_param_values(params, loc_key, n=n, default=0.0)
    concentration = _prior_param_values(params, "concentration", n=n, default=1.0)
    rate = _prior_param_values(params, "rate", n=n, default=1.0)
    value = _prior_param_values(params, "value", n=n, default=1.0)

    means = np.empty(n, dtype=float)
    for idx in range(n):
        if prior.family == PriorDistributionFamily.HALF_NORMAL:
            means[idx] = scale[idx] * math.sqrt(2.0 / math.pi)
        elif prior.family == PriorDistributionFamily.GAMMA:
            means[idx] = concentration[idx] / rate[idx]
        elif prior.family == PriorDistributionFamily.LOG_NORMAL:
            means[idx] = math.exp(loc[idx] + 0.5 * scale[idx] ** 2)
        elif prior.family == PriorDistributionFamily.EXPONENTIAL:
            means[idx] = 1.0 / rate[idx]
        elif prior.family == PriorDistributionFamily.DELTA:
            means[idx] = value[idx]
        else:
            raise ValueError(f"Unsupported positive prior family {prior.family.value!r}.")
    return means


def _assemble_mean_drift_from_prior_values(
    ssm_spec: SSMSpec,
    parameter_layout: SSMParameterLayout,
    *,
    base_decay_mu: np.ndarray,
    offdiag_mu: np.ndarray,
) -> np.ndarray:
    drift_component = _structural_dense_component(ssm_spec)
    drift = np.asarray(drift_component.drift_template, dtype=float).copy()
    for flat_idx, (effect_idx, cause_idx) in enumerate(parameter_layout.offdiag_positions):
        if flat_idx < offdiag_mu.size:
            drift[effect_idx, cause_idx] = float(offdiag_mu[flat_idx])
    offdiag = drift.copy()
    np.fill_diagonal(offdiag, 0.0)
    row_abs = np.sum(np.abs(offdiag), axis=1)
    for flat_idx, latent_idx in enumerate(parameter_layout.drift_base_decay_positions):
        if flat_idx < base_decay_mu.size:
            drift[latent_idx, latent_idx] = -(
                float(base_decay_mu[flat_idx])
                + float(row_abs[latent_idx])
                + float(drift_component.stability_margin)
            )
    if drift_component.time_invariant_mask is not None:
        ti_mask = np.asarray(drift_component.time_invariant_mask, dtype=bool)
        if ti_mask.size == drift.shape[0]:
            drift[np.diag_indices(drift.shape[0])] = np.where(
                ti_mask,
                -1e-6,
                np.diag(drift),
            )
    return drift


def _resolve_offdiag_interval_days(
    flat_idx: int,
    *,
    effect_idx: int,
    cause_idx: int,
    edge_lag_days: dict[tuple[int, int], float] | None,
    offdiag_interval_days: dict[int, float] | None,
) -> float | None:
    interval = (offdiag_interval_days or {}).get(flat_idx)
    if interval is None:
        interval = (edge_lag_days or {}).get((effect_idx, cause_idx))
    if interval is None:
        return None
    interval = float(interval)
    if interval <= 0:
        return None
    return interval


def _transition_from_elementwise_dt_terms(
    drift: np.ndarray,
    interval_days: float,
) -> np.ndarray:
    transition = np.eye(drift.shape[0], dtype=float)
    for idx in range(drift.shape[0]):
        transition[idx, idx] = math.exp(float(drift[idx, idx]) * interval_days)

    offdiag_mask = ~np.eye(drift.shape[0], dtype=bool)
    transition[offdiag_mask] = drift[offdiag_mask] * interval_days
    return transition


def matrix_log_diagnostic_drift(
    ssm_spec: SSMSpec,
    drift: np.ndarray,
    *,
    interval_days: float,
) -> np.ndarray:
    """Compute the full matrix-log CT drift used by dynamics diagnostics."""
    if interval_days <= 0:
        raise ValueError("matrix-log CT dynamics diagnostics require a positive interval.")

    transition = _transition_from_elementwise_dt_terms(drift, interval_days)
    log_transition = scipy.linalg.logm(transition)
    imaginary_scale = float(np.max(np.abs(np.imag(log_transition))))
    if imaginary_scale > _LOGM_IMAG_TOL:
        raise ValueError(
            "Matrix-log CT dynamics diagnostics require an embeddable real transition matrix; "
            f"max imaginary logm component is {imaginary_scale:.3g}."
        )
    exact_drift = np.real(log_transition) / interval_days

    drift_component = _structural_dense_component(ssm_spec)
    if drift_component.time_invariant_mask is not None:
        ti_mask = np.asarray(drift_component.time_invariant_mask, dtype=bool)
        if ti_mask.size == exact_drift.shape[0]:
            exact_drift[np.diag_indices(exact_drift.shape[0])] = np.where(
                ti_mask,
                -1e-6,
                np.diag(exact_drift),
            )
    return exact_drift


def logm_diagnostic_mean_drift(
    prior_registry: PriorRegistry,
    ssm_spec: SSMSpec,
    *,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
) -> np.ndarray | None:
    """Return the exact matrix-log mean drift for Stage 4 dynamics diagnostics."""
    parameter_layout = SSMParameterLayout.from_spec(ssm_spec)
    base_decay_site = parameter_layout.site_by_kind(SiteKind.DRIFT_BASE_DECAY)
    offdiag_site = parameter_layout.site_by_kind(SiteKind.DRIFT_OFFDIAG)
    if base_decay_site is None or offdiag_site is None:
        return None
    base_decay_prior = _prior_for_site(prior_registry, base_decay_site.name)
    offdiag_prior = _prior_for_site(prior_registry, offdiag_site.name)
    if base_decay_prior is None or offdiag_prior is None:
        return None
    base_decay_mu = _positive_prior_mean_values(base_decay_prior)
    offdiag_mu = _prior_values_1d(offdiag_prior.params.get("mu"))
    if base_decay_mu.size == 0 and offdiag_mu.size == 0:
        return None

    drift = _assemble_mean_drift_from_prior_values(
        ssm_spec,
        parameter_layout,
        base_decay_mu=base_decay_mu,
        offdiag_mu=offdiag_mu,
    )
    intervals = sorted(
        {float(interval) for interval in (edge_lag_days or {}).values() if float(interval) > 0}
    )
    if not intervals:
        if np.any(np.abs(offdiag_mu) >= NUMERICAL_EPSILON):
            raise ValueError(
                "Matrix-log CT dynamics diagnostics require edge lag metadata for "
                "off-diagonal drift priors."
            )
        return drift
    if len(intervals) > 1:
        raise ValueError(
            "Matrix-log CT dynamics diagnostics require one structural lag interval; "
            f"got {intervals}."
        )
    return matrix_log_diagnostic_drift(ssm_spec, drift, interval_days=intervals[0])


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


def _build_site_prior_payload(
    site: SiteDescriptor,
    entries: list[tuple[int, dict[str, float | int]]],
    current: dict[str, float | int],
) -> dict[str, list[float] | list[int]]:
    """Build an array-valued prior payload keyed by one unique sample site."""
    if not entries:
        raise ValueError(
            f"_build_site_prior_payload({site.name!r}) called with no entries; "
            "callers must filter out fields without any bound prior before invoking."
        )
    n_total = site_size(site.shape)

    include_mu = "mu" in current or any("mu" in normalized for _, normalized in entries)
    include_sigma = "sigma" in current or any("sigma" in normalized for _, normalized in entries)
    include_loc = "loc" in current or any("loc" in normalized for _, normalized in entries)
    include_family = "family" in current or any("family" in normalized for _, normalized in entries)
    include_lower = "lower" in current or any("lower" in normalized for _, normalized in entries)
    include_upper = "upper" in current or any("upper" in normalized for _, normalized in entries)
    include_concentration = "concentration" in current or any(
        "concentration" in normalized for _, normalized in entries
    )
    include_rate = "rate" in current or any("rate" in normalized for _, normalized in entries)
    include_value = "value" in current or any("value" in normalized for _, normalized in entries)

    lower_indices = {idx for idx, normalized in entries if "lower" in normalized}
    upper_indices = {idx for idx, normalized in entries if "upper" in normalized}
    all_indices = set(range(n_total))

    if include_lower and "lower" not in current and lower_indices != all_indices:
        raise ValueError(
            f"_build_site_prior_payload({site.name!r}): some entries specify 'lower' but no "
            "baseline was provided in the prior default. Provide a default 'lower' in current, "
            "or ensure every entry specifies one."
        )
    if include_upper and "upper" not in current and upper_indices != all_indices:
        raise ValueError(
            f"_build_site_prior_payload({site.name!r}): some entries specify 'upper' but no "
            "baseline was provided in the prior default."
        )

    mu_arr = [float(current.get("mu", 0.0))] * n_total if include_mu else None
    sigma_arr = [float(current.get("sigma", 0.5))] * n_total if include_sigma else None
    loc_arr = [float(current.get("loc", 0.0))] * n_total if include_loc else None
    family_arr = [int(current.get("family", 0))] * n_total if include_family else None
    lower_default = None
    if include_lower:
        lower_default = (
            float(current["lower"])
            if "lower" in current
            else float(
                next(normalized["lower"] for _, normalized in entries if "lower" in normalized)
            )
        )
    upper_default = None
    if include_upper:
        upper_default = (
            float(current["upper"])
            if "upper" in current
            else float(
                next(normalized["upper"] for _, normalized in entries if "upper" in normalized)
            )
        )
    lower_arr = [lower_default] * n_total if include_lower else None
    upper_arr = [upper_default] * n_total if include_upper else None
    concentration_arr = (
        [float(current.get("concentration", 1.0))] * n_total if include_concentration else None
    )
    rate_arr = [float(current.get("rate", 1.0))] * n_total if include_rate else None
    value_arr = [float(current.get("value", 1.0))] * n_total if include_value else None

    for idx, normalized in entries:
        if idx < 0 or idx >= n_total:
            raise ValueError(
                f"Prior binding for site {site.name!r} has flat index {idx}; "
                f"expected 0 <= index < {n_total}."
            )
        if "mu" in normalized and mu_arr is not None:
            mu_arr[idx] = float(normalized["mu"])
        if "sigma" in normalized and sigma_arr is not None:
            sigma_arr[idx] = float(normalized["sigma"])
        if "loc" in normalized and loc_arr is not None:
            loc_arr[idx] = float(normalized["loc"])
        if "family" in normalized and family_arr is not None:
            family_arr[idx] = int(normalized["family"])
        if "lower" in normalized and lower_arr is not None:
            lower_arr[idx] = float(normalized["lower"])
        if "upper" in normalized and upper_arr is not None:
            upper_arr[idx] = float(normalized["upper"])
        if "concentration" in normalized and concentration_arr is not None:
            concentration_arr[idx] = float(normalized["concentration"])
        if "rate" in normalized and rate_arr is not None:
            rate_arr[idx] = float(normalized["rate"])
        if "value" in normalized and value_arr is not None:
            value_arr[idx] = float(normalized["value"])

    result: dict[str, list[float] | list[int]] = {}
    if mu_arr is not None:
        result["mu"] = mu_arr
    if sigma_arr is not None:
        result["sigma"] = sigma_arr
    if loc_arr is not None:
        result["loc"] = loc_arr
    if family_arr is not None:
        result["family"] = family_arr
    if lower_arr is not None:
        result["lower"] = lower_arr
    if upper_arr is not None:
        result["upper"] = upper_arr
    if concentration_arr is not None:
        result["concentration"] = concentration_arr
    if rate_arr is not None:
        result["rate"] = rate_arr
    if value_arr is not None:
        result["value"] = value_arr
    return result


def _support_name(support: SupportClass) -> str:
    if support == SupportClass.POSITIVE:
        return "positive"
    if support == SupportClass.CORRELATION:
        return "correlation"
    return "real"


def _normalized_params_for_site_prior(support: SupportClass, prior: PriorSpec):
    normalized = prior_spec_to_normalized_params(prior)
    if support == SupportClass.POSITIVE:
        if prior.family != PriorDistributionFamily.HALF_NORMAL:
            normalized["family"] = get_positive_runtime_family_index(prior.family)
        return normalized
    if prior.family != PriorDistributionFamily.NORMAL:
        normalized["family"] = get_real_runtime_family_index(prior.family)
    elif support == SupportClass.CORRELATION:
        normalized["family"] = get_real_runtime_family_index(
            PriorDistributionFamily.TRUNCATED_NORMAL
        )
    return normalized


def _site_prior_from_normalized(
    support: SupportClass,
    normalized: dict[str, Any],
) -> PriorSpec:
    return prior_spec_from_normalized_params(normalized, support=_support_name(support))


def _prior_for_site(registry: PriorRegistry, site_name: str) -> PriorSpec | None:
    return registry.priors_by_site.get(site_name)


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
) -> tuple[PriorRegistry, SemanticBindingRegistry, list[CompileDiagnostic]]:
    """Compile prior proposals into a site-keyed prior registry with explicit index maps."""
    active_sites = build_site_registry(ssm_spec) if ssm_spec is not None else []
    prior_entries: dict[str, PriorSpec] = (
        {site.name: default_prior_for_descriptor(site) for site in active_sites}
        if active_sites
        else dict(default_prior_registry().priors_by_site)
    )
    site_by_name = {site.name: site for site in active_sites}
    role_by_name = _collect_role_lookup(model_spec)
    per_site: dict[str, list[tuple[int, dict[str, float | int]]]] = {}
    has_model_spec = model_spec is not None and (
        not isinstance(model_spec, dict) or bool(model_spec)
    )
    resolved_model_spec = model_spec if has_model_spec else None
    if resolved_model_spec is not None and ssm_spec is not None:
        bindings = build_semantic_prior_bindings(
            ssm_spec,
            resolved_model_spec,
            causal_spec=causal_spec,
        )
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
        bindings = empty_prior_bindings()
    binding_by_parameter = bindings.by_parameter
    parameter_layout = SSMParameterLayout.from_spec(ssm_spec) if ssm_spec is not None else None
    errors: list[str] = []
    offdiag_interval_days: dict[int, float] = {}

    for param_name, prior_spec in raw_priors.items():
        try:
            distribution = prior_spec.get("distribution", "Normal")
            raw_prior_params = prior_spec.get("params", {})
            degenerate_issues = _validate_nondegenerate_prior(
                param_name, distribution, raw_prior_params
            )
            if degenerate_issues:
                errors.extend(degenerate_issues)
                continue
            normalized = normalize_prior_params(distribution, raw_prior_params)
            binding = binding_by_parameter.get(param_name)
            if binding is None:
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
                continue

            if binding.transform == PriorAuthoringTransform.SITE_WIDE:
                site = site_by_name.get(binding.site_name)
                if site is None:
                    raise ValueError(
                        f"Prior {param_name!r} maps to inactive site {binding.site_name!r}."
                    )
                prior_entries[binding.site_name] = _site_prior_from_normalized(
                    site.support,
                    dict(normalized),
                )
                continue

            if binding.transform == PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY:
                construct_name = param_name.removeprefix("rho_").removeprefix("ar_")
                ref_days = prior_spec.get("reference_interval_days")
                resolved_ref_days = float(ref_days) if ref_days is not None else None
                if resolved_ref_days is not None and resolved_ref_days <= 0:
                    errors.append(
                        f"AR prior '{param_name}' reference_interval_days must be positive, "
                        f"got {resolved_ref_days:.3g}"
                    )
                    continue
                dt = (
                    resolved_ref_days
                    if resolved_ref_days is not None
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

                sigma_ar = float(normalized.get("sigma", 0.2))
                if sigma_ar <= 0.0:
                    errors.append(
                        f"AR prior '{param_name}' resolved to non-positive sigma "
                        f"{sigma_ar:.3g} during compilation."
                    )
                    continue
                base_decay = -math.log(mu_ar) / dt
                sd = sigma_ar / (mu_ar * dt)
                _append_structured_prior(
                    per_site,
                    binding.site_name,
                    binding.flat_index,
                    {
                        "family": get_positive_runtime_family_index(PriorDistributionFamily.GAMMA),
                        "concentration": (base_decay / sd) ** 2,
                        "rate": base_decay / (sd**2),
                    },
                )
                continue

            if binding.transform == PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE:
                if parameter_layout is None or ssm_spec is None:
                    raise ValueError(
                        "Dynamics effect prior compilation requires a translated SSMSpec runtime."
                    )
                if binding.site_kind == SiteKind.INPUT_EFFECT:
                    ref_days = prior_spec.get("reference_interval_days")
                    resolved_ref_days = float(ref_days) if ref_days is not None else None
                    if resolved_ref_days is not None and resolved_ref_days <= 0:
                        errors.append(
                            f"Known-input effect prior '{param_name}' reference_interval_days "
                            f"must be positive, got {resolved_ref_days:.3g}"
                        )
                        continue
                    dt = (
                        resolved_ref_days
                        if resolved_ref_days is not None
                        else get_construct_dt_days(causal_spec)
                    )
                elif binding.effect_idx is not None and binding.cause_idx is not None:
                    dt = _resolve_cross_lag_interval_days(
                        param_name=param_name,
                        prior_spec=prior_spec,
                        flat_index=binding.flat_index,
                        parameter_layout=parameter_layout,
                        ssm_spec=ssm_spec,
                        edge_lag_days=edge_lag_days,
                        causal_spec=causal_spec,
                        effect_idx=binding.effect_idx,
                        cause_idx=binding.cause_idx,
                    )
                    offdiag_interval_days[binding.flat_index] = dt
                else:
                    raise ValueError(
                        f"Dynamics effect prior {param_name!r} is missing effect/cause metadata."
                    )
                _append_structured_prior(
                    per_site,
                    binding.site_name,
                    binding.flat_index,
                    {
                        "mu": normalized.get("mu", 0.0) / dt,
                        "sigma": normalized.get("sigma", 0.5) / dt,
                    },
                )
                continue

            if binding.transform == PriorAuthoringTransform.INITIAL_STATE_CORRELATION:
                _append_structured_prior(
                    per_site,
                    binding.site_name,
                    binding.flat_index,
                    _coerce_initial_state_correlation_prior(normalized),
                )
                continue

            _append_structured_prior(
                per_site,
                binding.site_name,
                binding.flat_index,
                normalized,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue

    if errors:
        raise PriorCompilationError(errors)

    for site_name, entries in per_site.items():
        site = site_by_name.get(site_name)
        if site is None:
            raise ValueError(f"Prior site {site_name!r} maps to no active sample site.")
        current_prior = prior_entries[site.name]
        current = _normalized_params_for_site_prior(site.support, current_prior)
        result = _build_site_prior_payload(site, entries, current)
        prior_entries[site.name] = _site_prior_from_normalized(site.support, result)

    prior_registry = PriorRegistry(prior_entries)

    diagnostics: list[CompileDiagnostic] = []
    if ssm_spec is not None:
        diagnostics = collect_compile_diagnostics(
            ssm_spec,
            edge_lag_days=edge_lag_days,
            raw_priors=raw_priors,
            prior_registry=prior_registry,
            offdiag_interval_days=offdiag_interval_days,
        )
        _log_compile_diagnostics(diagnostics)

    return prior_registry, bindings, diagnostics


def bind_parameters(
    bindings: SemanticBindingRegistry,
    ssm_spec: SSMSpec,  # noqa: ARG001 - retained so call sites document spec provenance
) -> list[dict[str, Any]]:
    """Map semantic parameter names to NumPyro sample sites."""
    return [
        {
            "parameter": binding.parameter_name,
            "site_name": binding.site_name,
            "prior_field": binding.prior_field,
            "flat_index": binding.flat_index,
            "site_kind": binding.site_kind.value,
            "transform": binding.transform.value,
            "construct_names": list(binding.construct_names),
            "indicator_names": list(binding.indicator_names),
            "component_index": binding.component_index,
            "effect_idx": binding.effect_idx,
            "cause_idx": binding.cause_idx,
        }
        for binding in sorted(bindings.bindings, key=lambda item: item.parameter_name)
    ]
