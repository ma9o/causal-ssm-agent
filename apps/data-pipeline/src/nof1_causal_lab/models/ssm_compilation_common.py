"""Shared helpers and constants for the pure SSM compilation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nof1_causal_lab.distributions import (
    PriorDistributionFamily,
    get_positive_runtime_family_index,
    get_prior_family_spec,
    get_real_runtime_family_index,
)
from nof1_causal_lab.models.ssm.parameter_names import (
    INITIAL_STATE_CORRELATION_KEYWORDS,
)
from nof1_causal_lab.models.ssm.structure_runtime import SSMStructureRuntime

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec

PriorIndexMaps = tuple[
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
]


def empty_prior_index_maps() -> PriorIndexMaps:
    """Return an empty prior-index payload for spec-only code paths."""
    return ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})


SAMPLE_SITE_FOR_PRIOR_FIELD: dict[str, str] = {
    "drift_base_decay": "drift_base_decay_free",
    "drift_offdiag": "drift_offdiag_free",
    "input_effect": "input_effect_free",
    "diffusion_diag": "diffusion_diag_free",
    "diffusion_offdiag": "diffusion_lower_free",
    "cint": "cint_free",
    "static_state_sd": "static_state_sd_free",
    "lambda_free": "lambda_free",
    "manifest_means": "manifest_means_free",
    "manifest_var_diag": "manifest_var_diag_free",
    "t0_means": "t0_means_free",
    "t0_var_diag": "t0_var_diag_free",
    "t0_var_offdiag": "t0_var_lower_free",
    "obs_df": "obs_df",
    "obs_shape": "obs_shape",
    "obs_r": "obs_r",
    "obs_concentration": "obs_concentration",
    "obs_ordered_base": "obs_ordered_base",
    "obs_ordered_gaps": "obs_ordered_gaps",
    "obs_cat_intercepts": "obs_cat_intercepts",
    "obs_cat_slopes": "obs_cat_slopes",
}

SITE_TO_KEYWORDS: dict[str, list[str]] = {
    "drift_base_decay": ["rho", "ar"],
    "drift_base_decay_free": ["rho", "ar"],
    "drift_offdiag": ["beta"],
    "drift_offdiag_free": ["beta"],
    "input_effect": ["beta"],
    "input_effect_free": ["beta"],
    "diffusion_diag": ["sigma", "sd"],
    "diffusion_diag_free": ["sigma", "sd"],
    "diffusion_lower_free": ["cor"],
    "cint": ["cint"],
    "cint_free": ["cint"],
    "static_state_sd": ["tau", "baseline_factor"],
    "static_state_sd_free": ["tau", "baseline_factor"],
    "lambda_free": ["lambda", "loading"],
    "manifest_means_free": ["manifest_mean"],
    "manifest_var_diag": ["obs_sd", "measurement_error"],
    "manifest_var_diag_free": ["obs_sd", "measurement_error"],
    "t0_means": ["t0_mean"],
    "t0_means_free": ["t0_mean"],
    "t0_var_diag": ["t0_sd"],
    "t0_var_diag_free": ["t0_sd"],
    "t0_var_offdiag": list(INITIAL_STATE_CORRELATION_KEYWORDS),
    "t0_var_lower_free": list(INITIAL_STATE_CORRELATION_KEYWORDS),
    "diffusion_offdiag": ["cor"],
    "obs_df": ["obs_df"],
    "obs_shape": ["obs_shape"],
    "obs_r": ["obs_r"],
    "obs_concentration": ["obs_concentration"],
    "obs_ordered_base": ["obs_ordered_base"],
    "obs_ordered_gaps": ["obs_ordered_gaps"],
    "obs_cat_intercepts": ["obs_cat_intercepts"],
    "obs_cat_slopes": ["obs_cat_slopes"],
}
# dynamics_stability is a synthetic validation site (not a prior field) that
# covers both drift and diffusion parameters.
SITE_TO_KEYWORDS["dynamics_stability"] = ["rho", "ar", "sigma", "sd"]

# SSM parameters with fixed default priors that are not in ModelSpec and
# cannot be re-elicited.  Used to filter validation failures before mapping
# them back to user-facing parameter names.
NUISANCE_SITES: frozenset[str] = frozenset({"t0_means", "t0_cov"})

# Validation failure parameters that are global (affect all ModelSpec params).
GLOBAL_FAILURE_SITES: frozenset[str] = frozenset(
    {"prior_predictive", "dynamics_stability", "model_build", "prior_sampling"}
)


def axis_names_with_fallback(
    names: list[str] | None,
    *,
    expected: int,
    prefix: str,
) -> list[str]:
    """Return axis names with deterministic fallbacks when metadata is incomplete."""
    resolved = [str(name) for name in (names or []) if name]
    if len(resolved) >= expected:
        return resolved[:expected]
    return resolved + [f"{prefix}_{idx}" for idx in range(len(resolved), expected)]


def resolve_scalar_parameter_name(
    spec: SSMSpec,
    structure_runtime: SSMStructureRuntime,
    site_name: str,
    flat_index: int,
) -> str | None:
    """Resolve the canonical semantic name for one compiled sample-site scalar.

    Returns None when (site_name, flat_index) does not identify a known structural
    entry; callers choose their own fallback representation (e.g. ``f"{site}[{idx}]"``).
    """
    latent_names = axis_names_with_fallback(
        spec.latent_names, expected=spec.n_latent, prefix="latent"
    )
    manifest_names = axis_names_with_fallback(
        spec.manifest_names, expected=spec.n_manifest, prefix="manifest"
    )

    if site_name == "drift_base_decay_free" and flat_index < structure_runtime.n_drift_base_decay:
        latent_idx = structure_runtime.drift_base_decay_positions[flat_index]
        return f"rho_{latent_names[latent_idx]}"
    if site_name == "drift_offdiag_free" and flat_index < structure_runtime.n_drift_offdiag:
        effect_idx, cause_idx = structure_runtime.offdiag_positions[flat_index]
        return f"beta_{latent_names[cause_idx]}_{latent_names[effect_idx]}"
    if site_name == "input_effect_free" and flat_index < structure_runtime.n_input_effect:
        effect_idx, input_idx = structure_runtime.input_effect_positions[flat_index]
        input_names = axis_names_with_fallback(
            spec.input_names,
            expected=len(spec.input_names or []),
            prefix="input",
        )
        return f"beta_{input_names[input_idx]}_{latent_names[effect_idx]}"
    if site_name == "diffusion_diag_free" and flat_index < structure_runtime.n_diffusion_diag:
        latent_idx = structure_runtime.diffusion_diag_positions[flat_index]
        return f"sigma_{latent_names[latent_idx]}"
    if site_name == "diffusion_lower_free" and flat_index < structure_runtime.n_diffusion_lower:
        row, col = structure_runtime.diffusion_lower_positions[flat_index]
        return f"cor_{latent_names[col]}_{latent_names[row]}"
    if site_name == "cint_free" and flat_index < structure_runtime.n_cint:
        latent_idx = structure_runtime.cint_free_positions[flat_index]
        return f"cint_{latent_names[latent_idx]}"
    if site_name == "lambda_free" and flat_index < structure_runtime.n_lambda_free:
        manifest_idx, latent_idx = structure_runtime.lambda_free_positions[flat_index]
        return f"lambda_{manifest_names[manifest_idx]}_{latent_names[latent_idx]}"
    if site_name == "manifest_means_free" and flat_index < structure_runtime.n_manifest_means:
        manifest_idx = structure_runtime.manifest_means_free_positions[flat_index]
        return f"manifest_mean_{manifest_names[manifest_idx]}"
    if site_name == "manifest_var_diag_free" and flat_index < structure_runtime.n_manifest_var_diag:
        manifest_idx = structure_runtime.manifest_var_free_positions[flat_index]
        return f"obs_sd_{manifest_names[manifest_idx]}"
    if site_name == "t0_means_free" and flat_index < structure_runtime.n_t0_means:
        latent_idx = structure_runtime.t0_means_free_positions[flat_index]
        return f"t0_mean_{latent_names[latent_idx]}"
    if site_name == "t0_var_diag_free" and flat_index < structure_runtime.n_t0_diag:
        latent_idx = structure_runtime.t0_diag_free_positions[flat_index]
        return f"t0_sd_{latent_names[latent_idx]}"
    if site_name == "t0_var_lower_free" and flat_index < structure_runtime.n_t0_correlation:
        row, col = structure_runtime.t0_correlation_positions[flat_index]
        return f"cor0_{latent_names[col]}_{latent_names[row]}"
    return None


def normalize_prior_params(
    distribution: PriorDistributionFamily | str,
    params: dict,
) -> dict[str, float | int]:
    """Convert a typed prior distribution into the SSMPriors parameter shape."""
    try:
        spec = get_prior_family_spec(distribution)
    except ValueError as exc:
        raise ValueError(f"Unsupported prior distribution family: {distribution!r}") from exc

    family = spec.family

    if family == PriorDistributionFamily.NORMAL:
        return {"mu": params.get("mu", 0.0), "sigma": params.get("sigma", 1.0)}

    if family == PriorDistributionFamily.TRUNCATED_NORMAL:
        return {
            "family": get_real_runtime_family_index(family),
            "mu": params.get("mu", 0.0),
            "sigma": params.get("sigma", 1.0),
            "lower": params.get("lower", -1.0),
            "upper": params.get("upper", 1.0),
        }

    if family == PriorDistributionFamily.HALF_NORMAL:
        return {"sigma": params.get("sigma", 1.0)}

    if family == PriorDistributionFamily.BETA:
        alpha = params.get("alpha", 2.0)
        beta = params.get("beta", 2.0)
        mu = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return {"mu": mu, "sigma": var**0.5}

    if family == PriorDistributionFamily.UNIFORM:
        lower = params.get("lower", -1.0)
        upper = params.get("upper", 1.0)
        mu = (lower + upper) / 2
        sigma = (upper - lower) / 4
        return {
            "family": get_real_runtime_family_index(family),
            "mu": mu,
            "sigma": sigma,
            "lower": lower,
            "upper": upper,
        }

    if family == PriorDistributionFamily.GAMMA:
        return {
            "family": get_positive_runtime_family_index(family),
            "concentration": params.get("concentration", 2.0),
            "rate": params.get("rate", 1.0),
        }

    if family == PriorDistributionFamily.LOG_NORMAL:
        return {
            "family": get_positive_runtime_family_index(family),
            "loc": params.get("mu", 0.0),
            "sigma": params.get("sigma", 1.0),
        }

    if family == PriorDistributionFamily.EXPONENTIAL:
        return {
            "family": get_positive_runtime_family_index(family),
            "rate": params.get("rate", 1.0),
        }

    if family == PriorDistributionFamily.DELTA:
        return {
            "family": get_positive_runtime_family_index(family),
            "value": params.get("value", 1.0),
        }

    raise ValueError(f"Unsupported prior distribution family: {distribution!r}")


def dump_prior_payloads(priors: dict[str, Any] | None) -> dict[str, dict]:
    """Normalize prior proposals into plain ``dict`` payloads."""
    return {
        name: value.model_dump() if hasattr(value, "model_dump") else dict(value)
        for name, value in (priors or {}).items()
    }


def expected_prior_size(attr: str, ssm_spec: SSMSpec | None) -> int | None:
    """Return the structural size for an array-valued prior field."""
    if ssm_spec is None:
        return None

    structure_runtime = SSMStructureRuntime(ssm_spec)

    if attr == "drift_base_decay":
        return structure_runtime.n_drift_base_decay

    if attr == "diffusion_diag":
        return structure_runtime.n_diffusion_diag

    if attr == "drift_offdiag":
        return structure_runtime.n_drift_offdiag

    if attr == "input_effect":
        return structure_runtime.n_input_effect

    if attr == "cint":
        return structure_runtime.n_cint

    if attr == "static_state_sd":
        return structure_runtime.n_static_state_sd

    if attr == "manifest_means":
        return structure_runtime.n_manifest_means

    if attr == "lambda_free":
        return structure_runtime.n_lambda_free

    if attr == "manifest_var_diag":
        return structure_runtime.n_manifest_var_diag

    if attr == "diffusion_offdiag":
        return structure_runtime.n_diffusion_lower

    if attr == "t0_var_diag":
        return structure_runtime.n_t0_diag

    if attr == "t0_var_offdiag":
        return structure_runtime.n_t0_correlation

    return None


def build_array_prior_payload(
    attr: str,
    entries: list[tuple[int, dict[str, float | int]]],
    current: dict[str, float | int],
    ssm_spec: SSMSpec | None,
) -> dict[str, list[float] | list[int]]:
    """Build the array-valued SSMPriors payload for a structured parameter family."""
    if not entries:
        raise ValueError(
            f"build_array_prior_payload({attr!r}) called with no entries; "
            "callers must filter out fields without any bound prior before invoking."
        )
    expected_size = expected_prior_size(attr, ssm_spec)
    n_total = max(idx for idx, _ in entries) + 1
    if expected_size is not None:
        n_total = max(n_total, expected_size)

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
            f"build_array_prior_payload({attr!r}): some entries specify 'lower' but no "
            "baseline was provided in the SSMPriors default — refusing to silently fill "
            "with ±1e6. Provide a default 'lower' in current, or ensure every entry "
            "specifies one."
        )
    if include_upper and "upper" not in current and upper_indices != all_indices:
        raise ValueError(
            f"build_array_prior_payload({attr!r}): some entries specify 'upper' but no "
            "baseline was provided in the SSMPriors default — refusing to silently fill "
            "with ±1e6."
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
