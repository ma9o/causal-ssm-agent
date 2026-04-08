"""Shared helpers and constants for the pure SSM compilation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from causal_ssm_agent.distributions import (
    PriorDistributionFamily,
    PriorRuntimeKind,
    get_positive_runtime_family_index,
    get_prior_family_spec,
    get_real_runtime_family_index,
)
from causal_ssm_agent.models.ssm.parameter_names import (
    INITIAL_STATE_CORRELATION_KEYWORDS,
)
from causal_ssm_agent.models.ssm.parameter_names import (
    split_compound_name as _split_compound_name,
)
from causal_ssm_agent.models.ssm.structure_runtime import SSMStructureRuntime

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec

split_compound_name = _split_compound_name

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
]

SAMPLE_SITE_FOR_PRIOR_FIELD: dict[str, str] = {
    "drift_diag": "drift_diag_pop",
    "drift_offdiag": "drift_offdiag_pop",
    "diffusion_diag": "diffusion_diag_pop",
    "diffusion_offdiag": "diffusion_lower",
    "lambda_free": "lambda_free",
    "manifest_var_diag": "manifest_var_diag",
    "t0_means": "t0_means_pop",
    "t0_var_diag": "t0_var_diag",
    "t0_var_offdiag": "t0_var_lower",
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
    "drift_diag": ["rho", "ar"],
    "drift_offdiag": ["beta"],
    "diffusion_diag": ["sigma", "sd"],
    "lambda_free": ["lambda", "loading"],
    "manifest_var_diag": ["obs_sd", "measurement_error"],
    "t0_means": ["t0_mean"],
    "t0_means_pop": ["t0_mean"],
    "t0_var_diag": ["t0_sd"],
    "t0_var_offdiag": list(INITIAL_STATE_CORRELATION_KEYWORDS),
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
NUISANCE_SITES: frozenset[str] = frozenset({"cint_pop", "cint", "t0_means", "t0_cov"})

# Validation failure parameters that are global (affect all ModelSpec params).
GLOBAL_FAILURE_SITES: frozenset[str] = frozenset(
    {"prior_predictive", "dynamics_stability", "model_build", "prior_sampling"}
)


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
    runtime_kind = spec.runtime_kind

    if runtime_kind == PriorRuntimeKind.NORMAL:
        return {"mu": params.get("mu", 0.0), "sigma": params.get("sigma", 1.0)}

    if runtime_kind == PriorRuntimeKind.TRUNCATED_NORMAL:
        return {
            "family": get_real_runtime_family_index(runtime_kind),
            "mu": params.get("mu", 0.0),
            "sigma": params.get("sigma", 1.0),
            "lower": params.get("lower", -1.0),
            "upper": params.get("upper", 1.0),
        }

    if runtime_kind == PriorRuntimeKind.HALF_NORMAL:
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
            "family": get_real_runtime_family_index(runtime_kind),
            "mu": mu,
            "sigma": sigma,
            "lower": lower,
            "upper": upper,
        }

    if runtime_kind == PriorRuntimeKind.GAMMA:
        return {
            "family": get_positive_runtime_family_index(runtime_kind),
            "concentration": params.get("concentration", 2.0),
            "rate": params.get("rate", 1.0),
        }

    if runtime_kind == PriorRuntimeKind.LOG_NORMAL:
        return {
            "family": get_positive_runtime_family_index(runtime_kind),
            "loc": params.get("mu", 0.0),
            "sigma": params.get("sigma", 1.0),
        }

    if runtime_kind == PriorRuntimeKind.EXPONENTIAL:
        return {
            "family": get_positive_runtime_family_index(runtime_kind),
            "rate": params.get("rate", 1.0),
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

    if attr == "drift_diag":
        return structure_runtime.n_drift_diag

    if attr == "diffusion_diag":
        return structure_runtime.n_diffusion_diag

    if attr == "drift_offdiag":
        return structure_runtime.n_drift_offdiag

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

    mu_arr = [float(current.get("mu", 0.0))] * n_total if include_mu else None
    sigma_arr = [float(current.get("sigma", 0.5))] * n_total if include_sigma else None
    loc_arr = [float(current.get("loc", 0.0))] * n_total if include_loc else None
    family_arr = [int(current.get("family", 0))] * n_total if include_family else None
    lower_arr = [float(current.get("lower", -1e6))] * n_total if include_lower else None
    upper_arr = [float(current.get("upper", 1e6))] * n_total if include_upper else None
    concentration_arr = (
        [float(current.get("concentration", 1.0))] * n_total if include_concentration else None
    )
    rate_arr = [float(current.get("rate", 1.0))] * n_total if include_rate else None

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

    return result
