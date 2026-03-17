"""Shared helpers and constants for the pure SSM compilation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from causal_ssm_agent.orchestrator.schemas_model import ParameterRole

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec

PriorIndexMaps = tuple[
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
]

ROLE_TO_SSM: dict[ParameterRole, tuple[str, dict[str, float]]] = {
    ParameterRole.AR_COEFFICIENT: ("drift_diag", {"mu": -0.5, "sigma": 1.0}),
    ParameterRole.FIXED_EFFECT: ("drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
    ParameterRole.RESIDUAL_SD: ("diffusion_diag", {"sigma": 1.0}),
    ParameterRole.LOADING: ("lambda_free", {"mu": 0.5, "sigma": 0.5}),
    ParameterRole.CORRELATION: ("diffusion_offdiag", {"mu": 0.0, "sigma": 0.5}),
}

KEYWORD_RULES: list[tuple[list[str], str, dict[str, float]]] = [
    (["rho", "ar"], "drift_diag", {"mu": -0.5, "sigma": 1.0}),
    (["beta"], "drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
    (["sigma", "sd"], "diffusion_diag", {"sigma": 1.0}),
    (["lambda", "loading"], "lambda_free", {"mu": 0.5, "sigma": 0.5}),
    (["cor"], "diffusion_offdiag", {"mu": 0.0, "sigma": 0.5}),
]

SAMPLE_SITE_FOR_PRIOR_FIELD: dict[str, str] = {
    "drift_diag": "drift_diag_pop",
    "drift_offdiag": "drift_offdiag_pop",
    "diffusion_diag": "diffusion_diag_pop",
    "diffusion_offdiag": "diffusion_lower",
    "lambda_free": "lambda_free",
}

# Inverted view of KEYWORD_RULES: SSM prior field → parameter keywords.
# Used by prior_predictive.get_failed_parameters() to map SSM-level validation
# failures back to user-facing ModelSpec parameter names.
SITE_TO_KEYWORDS: dict[str, list[str]] = {field: keywords for keywords, field, _ in KEYWORD_RULES}
# dynamics_stability is a synthetic validation site (not a prior field) that
# covers both drift and diffusion parameters.
SITE_TO_KEYWORDS["dynamics_stability"] = ["rho", "ar", "sigma", "sd"]

# SSM parameters with fixed default priors that are not in ModelSpec and
# cannot be re-elicited.  Used to filter validation failures before mapping
# them back to user-facing parameter names.
NUISANCE_SITES: frozenset[str] = frozenset(
    {"cint_pop", "cint", "t0_means_pop", "t0_means", "t0_var_diag", "t0_cov"}
)

# Validation failure parameters that are global (affect all ModelSpec params).
GLOBAL_FAILURE_SITES: frozenset[str] = frozenset(
    {"prior_predictive", "dynamics_stability", "model_build", "prior_sampling"}
)


def normalize_prior_params(distribution: str, params: dict) -> dict[str, float]:
    """Convert distribution-specific params to the mu/sigma shape used by SSMPriors."""
    dist_lower = distribution.lower()

    if dist_lower in {"normal", "truncatednormal"}:
        return {"mu": params.get("mu", 0.0), "sigma": params.get("sigma", 1.0)}

    if dist_lower == "halfnormal":
        return {"sigma": params.get("sigma", 1.0)}

    if dist_lower == "beta":
        alpha = params.get("alpha", 2.0)
        beta = params.get("beta", 2.0)
        mu = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return {"mu": mu, "sigma": var**0.5}

    if dist_lower == "uniform":
        lower = params.get("lower", -1.0)
        upper = params.get("upper", 1.0)
        mu = (lower + upper) / 2
        sigma = (upper - lower) / 4
        return {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper}

    return {"mu": params.get("mu", 0.0), "sigma": params.get("sigma", 1.0)}


def split_compound_name(
    compound: str,
    valid_first: set[str],
    valid_second: set[str],
) -> tuple[str, str] | None:
    """Split an underscore-joined name into two known names."""
    parts = compound.split("_")
    for idx in range(1, len(parts)):
        first = "_".join(parts[:idx])
        second = "_".join(parts[idx:])
        if first in valid_first and second in valid_second:
            return first, second
    return None


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

    if attr in {"drift_diag", "diffusion_diag"}:
        return ssm_spec.n_latent

    if attr == "drift_offdiag":
        if ssm_spec.drift_mask is None:
            return ssm_spec.n_latent * (ssm_spec.n_latent - 1)
        count = 0
        for effect_idx in range(ssm_spec.n_latent):
            for cause_idx in range(ssm_spec.n_latent):
                if effect_idx != cause_idx and ssm_spec.drift_mask[effect_idx, cause_idx]:
                    count += 1
        return count

    if attr == "lambda_free":
        if ssm_spec.lambda_mask is None:
            return None
        return int(np.asarray(ssm_spec.lambda_mask).sum())

    if attr == "diffusion_offdiag":
        if ssm_spec.diffusion != "free":
            return 0
        return ssm_spec.n_latent * (ssm_spec.n_latent - 1) // 2

    return None


def build_array_prior_payload(
    attr: str,
    entries: list[tuple[int, dict[str, float]]],
    current: dict[str, float],
    ssm_spec: SSMSpec | None,
) -> dict[str, list[float]]:
    """Build the array-valued SSMPriors payload for a structured parameter family."""
    expected_size = expected_prior_size(attr, ssm_spec)
    n_total = max(idx for idx, _ in entries) + 1
    if expected_size is not None:
        n_total = max(n_total, expected_size)

    include_mu = "mu" in current or any("mu" in normalized for _, normalized in entries)
    include_sigma = "sigma" in current or any("sigma" in normalized for _, normalized in entries)

    mu_arr = [float(current.get("mu", 0.0))] * n_total if include_mu else None
    sigma_arr = [float(current.get("sigma", 0.5))] * n_total if include_sigma else None

    for idx, normalized in entries:
        if "mu" in normalized and mu_arr is not None:
            mu_arr[idx] = float(normalized["mu"])
        if "sigma" in normalized and sigma_arr is not None:
            sigma_arr[idx] = float(normalized["sigma"])

    result: dict[str, list[float]] = {}
    if mu_arr is not None:
        result["mu"] = mu_arr
    if sigma_arr is not None:
        result["sigma"] = sigma_arr

    if any("lower" in normalized for _, normalized in entries):
        lower_arr = [-1e6] * n_total
        upper_arr = [1e6] * n_total
        for idx, normalized in entries:
            lower_arr[idx] = float(normalized.get("lower", -1e6))
            upper_arr[idx] = float(normalized.get("upper", 1e6))
        result["lower"] = lower_arr
        result["upper"] = upper_arr

    return result
