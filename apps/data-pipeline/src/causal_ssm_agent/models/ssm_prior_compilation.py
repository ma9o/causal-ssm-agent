"""Pure prior-compilation and binding stages for SSM compilation."""

from __future__ import annotations

import math
from typing import Any

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.likelihoods.base import NUMERICAL_EPSILON
from causal_ssm_agent.models.ssm.model import SSMPriors, SSMSpec
from causal_ssm_agent.models.ssm_compilation_common import (
    KEYWORD_RULES,
    ROLE_TO_SSM,
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


def check_drift_lag_consistency(
    ssm_priors: SSMPriors,
    ssm_spec: SSMSpec,
    *,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
) -> None:
    """Check CT drift rates against expected lag metadata from the causal structure."""
    edge_lags = edge_lag_days or {}
    if not edge_lags:
        return

    offdiag_prior = ssm_priors.drift_offdiag
    if offdiag_prior is None or "mu" not in offdiag_prior:
        return

    mu_arr = offdiag_prior["mu"]
    if not isinstance(mu_arr, list):
        return

    offdiag_positions: list[tuple[int, int]] = []
    if ssm_spec.drift_mask is not None:
        for effect_idx in range(ssm_spec.n_latent):
            for cause_idx in range(ssm_spec.n_latent):
                if effect_idx != cause_idx and ssm_spec.drift_mask[effect_idx, cause_idx]:
                    offdiag_positions.append((effect_idx, cause_idx))

    for flat_idx, (effect_idx, cause_idx) in enumerate(offdiag_positions):
        if flat_idx >= len(mu_arr) or (effect_idx, cause_idx) not in edge_lags:
            continue

        mu_ct = abs(float(mu_arr[flat_idx]))
        if mu_ct < NUMERICAL_EPSILON:
            continue

        expected_lag_days = edge_lags[(effect_idx, cause_idx)]
        implied_timescale_days = 1.0 / mu_ct
        ratio = max(implied_timescale_days, expected_lag_days) / max(
            min(implied_timescale_days, expected_lag_days),
            NUMERICAL_EPSILON,
        )
        if ratio <= 5.0:
            continue

        cause_name = (
            ssm_spec.latent_names[cause_idx] if ssm_spec.latent_names else f"latent_{cause_idx}"
        )
        effect_name = (
            ssm_spec.latent_names[effect_idx] if ssm_spec.latent_names else f"latent_{effect_idx}"
        )
        logger.warning(
            "Drift rate for %s->%s implies timescale %.1f days, but edge lag suggests %.1f days "
            "(%.0fx mismatch). The literature prior may be calibrated to a different observation "
            "interval than the causal model expects.",
            cause_name,
            effect_name,
            implied_timescale_days,
            expected_lag_days,
            ratio,
        )


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
    per_element: dict[str, list[tuple[int, dict[str, float]]]],
    attr: str,
    idx: int,
    normalized: dict[str, float],
) -> None:
    per_element.setdefault(attr, []).append((idx, normalized))


def compile_priors(
    raw_priors: dict[str, dict],
    model_spec: ModelSpec | dict | None,
    ssm_spec: SSMSpec | None,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    causal_spec: dict | None = None,
) -> tuple[SSMPriors, PriorIndexMaps]:
    """Compile prior proposals into ``SSMPriors`` with explicit index maps."""
    ssm_priors = SSMPriors()
    role_by_name = _collect_role_lookup(model_spec)
    per_element: dict[str, list[tuple[int, dict[str, float]]]] = {}
    index_maps = build_prior_index_maps(ssm_spec, model_spec, causal_spec=causal_spec)
    (
        offdiag_param_index,
        lambda_param_index,
        diag_param_index,
        diffusion_diag_param_index,
        diffusion_offdiag_param_index,
    ) = index_maps

    for param_name, prior_spec in raw_priors.items():
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
            lower = normalized.get("lower")
            upper = normalized.get("upper")
            if lower is not None and float(lower) < 0.0:
                raise ValueError(
                    f"AR prior '{param_name}' must be on the DT persistence scale in [0, 1], "
                    f"but lower bound is {float(lower):.3g}"
                )
            if upper is not None and float(upper) > 1.0:
                raise ValueError(
                    f"AR prior '{param_name}' must be on the DT persistence scale in [0, 1], "
                    f"but upper bound is {float(upper):.3g}"
                )

            mu_ar = float(normalized.get("mu", 0.5))
            if not 0.0 < mu_ar < 1.0:
                raise ValueError(
                    f"AR prior '{param_name}' must have DT persistence mean in (0, 1), got {mu_ar:.3g}"
                )
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

        role = role_by_name.get(param_name)
        if role and role in ROLE_TO_SSM:
            attr, defaults = ROLE_TO_SSM[role]
            merged = {key: normalized.get(key, value) for key, value in defaults.items()}
            setattr(ssm_priors, attr, merged)
            continue

        name_lower = param_name.lower()
        matched = False
        for keywords, attr, defaults in KEYWORD_RULES:
            matching_kw = [kw for kw in keywords if kw in name_lower]
            if not matching_kw:
                continue
            logger.debug(
                "Prior '%s': keyword fallback matched '%s' -> %s",
                param_name,
                matching_kw[0],
                attr,
            )
            merged = {key: normalized.get(key, value) for key, value in defaults.items()}
            setattr(ssm_priors, attr, merged)
            matched = True
            break
        if not matched:
            logger.debug("Prior '%s': no role or keyword match found, skipping", param_name)

    for attr, entries in per_element.items():
        current = getattr(ssm_priors, attr)
        result = build_array_prior_payload(attr, entries, current, ssm_spec)
        setattr(ssm_priors, attr, result)

    warn_first_order_approximation(ssm_priors)
    if ssm_spec is not None:
        check_drift_lag_consistency(ssm_priors, ssm_spec, edge_lag_days=edge_lag_days)

    return ssm_priors, index_maps


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
    ) = index_maps

    bindings: list[dict[str, Any]] = []
    ordered_maps = (
        diag_index,
        offdiag_index,
        diffusion_diag_index,
        diffusion_offdiag_index,
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
