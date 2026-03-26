"""Pure prior-compilation and binding stages for SSM compilation."""

from __future__ import annotations

import math
from typing import Any

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.compilation_errors import AggregatedCompileError
from causal_ssm_agent.models.likelihoods.base import NUMERICAL_EPSILON
from causal_ssm_agent.models.ssm.model import SSMPriors, SSMSpec
from causal_ssm_agent.models.ssm_compilation_common import (
    SAMPLE_SITE_FOR_PRIOR_FIELD,
    PriorIndexMaps,
    build_array_prior_payload,
    normalize_prior_params,
    split_compound_name,
)
from causal_ssm_agent.models.ssm_prior_indexing import build_prior_index_maps
from causal_ssm_agent.models.ssm_spec_translation import get_construct_dt_days
from causal_ssm_agent.orchestrator.schemas_model import ModelSpec, ParameterRole

logger = get_prefect_logger("causal_ssm_agent.models.ssm_compilation")


class PriorCompilationError(AggregatedCompileError):
    """Aggregate independent prior-compilation failures into one exception."""

    header = "Prior compilation failed"


def _iter_offdiag_positions(ssm_spec: SSMSpec) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    if ssm_spec.drift_mask is None:
        return positions

    for effect_idx in range(ssm_spec.n_latent):
        for cause_idx in range(ssm_spec.n_latent):
            if effect_idx != cause_idx and ssm_spec.drift_mask[effect_idx, cause_idx]:
                positions.append((effect_idx, cause_idx))
    return positions


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


def collect_lagged_drift_prior_issues(
    ssm_priors: SSMPriors,
    ssm_spec: SSMSpec,
    *,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    raw_priors: dict[str, dict] | None = None,
) -> list[dict[str, str]]:
    """Collect lagged-edge DT/CT diagnostics that should stop Stage 4 retries early."""
    edge_lags = edge_lag_days or {}
    if not edge_lags:
        return []

    offdiag_prior = ssm_priors.drift_offdiag
    if offdiag_prior is None:
        return []

    offdiag_mu = offdiag_prior.get("mu")
    if offdiag_mu is None:
        return []

    if isinstance(offdiag_mu, (int, float)):
        offdiag_mu = [offdiag_mu]
    if not offdiag_mu:
        return []

    positions = _iter_offdiag_positions(ssm_spec)
    issues_by_parameter: dict[str, list[str]] = {}

    for flat_idx, (effect_idx, cause_idx) in enumerate(positions):
        if flat_idx >= len(offdiag_mu) or (effect_idx, cause_idx) not in edge_lags:
            continue

        parameter_name, cause_name, effect_name = _drift_parameter_name(
            ssm_spec,
            effect_idx,
            cause_idx,
        )
        offdiag_abs = abs(float(offdiag_mu[flat_idx]))

        if offdiag_abs < NUMERICAL_EPSILON:
            continue

        expected_lag_days = edge_lags[(effect_idx, cause_idx)]
        implied_timescale_days = 1.0 / offdiag_abs
        ratio = max(implied_timescale_days, expected_lag_days) / max(
            min(implied_timescale_days, expected_lag_days),
            NUMERICAL_EPSILON,
        )
        if ratio > 5.0:
            prior_spec = (raw_priors or {}).get(parameter_name) or {}
            ref_days = prior_spec.get("reference_interval_days")
            authored_interval_days = (
                float(ref_days)
                if ref_days is not None and float(ref_days) > 0
                else expected_lag_days
            )
            interval_source = (
                f"`reference_interval_days={authored_interval_days:.1f}`"
                if ref_days is not None and float(ref_days) > 0
                else (
                    "`reference_interval_days` omitted, so the prior is being interpreted on "
                    f"the default model interval ({expected_lag_days:.1f}d)"
                )
            )
            strength_note = (
                "too weak/slow" if implied_timescale_days > expected_lag_days else "too strong/fast"
            )
            issues_by_parameter.setdefault(parameter_name, []).append(
                f"The authored prior is being treated as an effect over {authored_interval_days:.1f}d "
                f"({interval_source}). After interval normalization, it implies a characteristic "
                f"timescale of {implied_timescale_days:.1f}d for {cause_name}->{effect_name}, "
                f"which is {strength_note} for a {expected_lag_days:.1f}d lagged edge "
                f"({ratio:.0f}x mismatch)."
            )

    return [
        {
            "parameter": parameter,
            "issue": " ".join(messages),
            "suggested_adjustment": (
                "Keep `params` on the interval you actually want to author. If the evidence "
                "comes from a longer study interval, set `reference_interval_days` to that "
                "interval. Otherwise change the prior mean/sigma on the current authored "
                "interval scale so the normalized one-step effect matches this lagged edge."
            ),
        }
        for parameter, messages in issues_by_parameter.items()
    ]


def collect_compile_diagnostics(
    ssm_priors: SSMPriors,
    ssm_spec: SSMSpec,
    *,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    raw_priors: dict[str, dict] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Collect structured compiler diagnostics for downstream consumers."""
    return {
        "lagged_drift_prior_issues": collect_lagged_drift_prior_issues(
            ssm_priors,
            ssm_spec,
            edge_lag_days=edge_lag_days,
            raw_priors=raw_priors,
        )
    }


def _log_compile_diagnostics(diagnostics: dict[str, list[dict[str, str]]]) -> None:
    for issue in diagnostics.get("lagged_drift_prior_issues") or []:
        logger.warning("%s: %s", issue["parameter"], issue["issue"])


def warn_first_order_approximation(ssm_priors: SSMPriors) -> None:
    """Warn when the first-order DT->CT approximation is likely inaccurate."""
    diag_prior = ssm_priors.drift_diag
    offdiag_prior = ssm_priors.drift_offdiag
    if diag_prior is None or offdiag_prior is None:
        return

    diag_mu = diag_prior.get("mu")
    offdiag_mu = offdiag_prior.get("mu")
    if diag_mu is None or offdiag_mu is None:
        return

    if isinstance(diag_mu, (int, float)):
        diag_mu = [diag_mu]
    if isinstance(offdiag_mu, (int, float)):
        offdiag_mu = [offdiag_mu]
    if not diag_mu or not offdiag_mu:
        return

    min_diag = min(abs(float(value)) for value in diag_mu)
    if min_diag < NUMERICAL_EPSILON:
        return

    for idx, offdiag_value in enumerate(offdiag_mu):
        ratio = abs(float(offdiag_value)) / min_diag
        if ratio <= 0.2:
            continue
        logger.warning(
            "First-order DT->CT approximation may be inaccurate: "
            "off-diagonal drift[%d] magnitude (%.3f) is %.0f%% of "
            "minimum diagonal magnitude (%.3f). Consider a shorter "
            "reference interval or eliciting priors directly on CT rates.",
            idx,
            abs(float(offdiag_value)),
            ratio * 100,
            min_diag,
        )
        break


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
) -> tuple[SSMPriors, PriorIndexMaps, dict[str, list[dict[str, str]]]]:
    """Compile prior proposals into ``SSMPriors`` with explicit index maps."""
    ssm_priors = SSMPriors()
    role_by_name = _collect_role_lookup(model_spec)
    per_element: dict[str, list[tuple[int, dict[str, float | int]]]] = {}
    index_maps = build_prior_index_maps(ssm_spec, model_spec, causal_spec=causal_spec)
    (
        offdiag_param_index,
        lambda_param_index,
        diag_param_index,
        diffusion_diag_param_index,
        diffusion_offdiag_param_index,
        t0_offdiag_param_index,
    ) = index_maps
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
                ref_days = prior_spec.get("reference_interval_days")
                if ref_days is not None and ref_days > 0:
                    dt = float(ref_days)
                else:
                    dt = 1.0
                    if ssm_spec is not None and ssm_spec.latent_names:
                        latent_set = set(ssm_spec.latent_names)
                        compound = param_name.removeprefix("beta_")
                        split = split_compound_name(compound, latent_set, latent_set)
                        if split is not None:
                            _cause, effect = split
                            dt = get_construct_dt_days(causal_spec, effect)
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

            if param_name in diffusion_diag_param_index:
                attr, idx = diffusion_diag_param_index[param_name]
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

    diagnostics: dict[str, list[dict[str, str]]] = {"lagged_drift_prior_issues": []}
    warn_first_order_approximation(ssm_priors)
    if ssm_spec is not None:
        diagnostics = collect_compile_diagnostics(
            ssm_priors,
            ssm_spec,
            edge_lag_days=edge_lag_days,
            raw_priors=raw_priors,
        )
        _log_compile_diagnostics(diagnostics)

    return ssm_priors, index_maps, diagnostics


def bind_parameters(
    model_spec: ModelSpec | dict | None,
    ssm_spec: SSMSpec,
    index_maps: PriorIndexMaps | None = None,
    *,
    causal_spec: dict | None = None,
) -> list[dict[str, Any]]:
    """Map semantic parameter names to NumPyro sample sites."""
    if model_spec is None:
        return []
    if index_maps is None:
        index_maps = build_prior_index_maps(ssm_spec, model_spec, causal_spec=causal_spec)

    (
        offdiag_index,
        lambda_index,
        diag_index,
        diffusion_diag_index,
        diffusion_offdiag_index,
        t0_offdiag_index,
    ) = index_maps

    bindings: list[dict[str, Any]] = []
    ordered_maps = (
        diag_index,
        offdiag_index,
        diffusion_diag_index,
        diffusion_offdiag_index,
        t0_offdiag_index,
        lambda_index,
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
