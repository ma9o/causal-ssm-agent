"""Shared helpers and constants for the pure SSM compilation pipeline."""

from __future__ import annotations

from typing import Any

from nof1_causal_lab.distributions import (
    PriorDistributionFamily,
    get_positive_runtime_family_index,
    get_prior_family_spec,
    get_real_runtime_family_index,
)
from nof1_causal_lab.models.ssm.parameter_names import (
    INITIAL_STATE_CORRELATION_KEYWORDS,
)

# Substring keys mapping a diagnostic's `parameter` field to the user-facing
# model-parameter keyword set it implicates. The matcher in
# ``prior_predictive.get_failed_parameters`` does ``if site_prefix in
# result_param:`` so each key must cover one variant the diagnostic can use:
# prior_field forms and block-runtime site names.
SITE_TO_KEYWORDS: dict[str, list[str]] = {
    "dynamics_decay": ["rho", "ar", "decay"],
    "decay": ["rho", "ar", "decay"],
    "linear_edge_weight": ["beta", "linear_weight"],
    "dynamics_weight": ["beta", "linear_weight", "multiplicative_weight"],
    "weight": ["beta", "linear_weight", "multiplicative_weight"],
    "dynamics_cint": ["cint"],
    "input_effect": ["beta"],
    "diffusion_diag": ["sigma", "sd"],
    "diffusion_offdiag": ["cor"],
    "diffusion_lower_free": ["cor"],
    "static_state_sd": ["tau", "baseline_factor"],
    "lambda_free": ["lambda", "loading"],
    "manifest_means": ["manifest_mean"],
    "manifest_var_diag": ["obs_sd", "measurement_error"],
    "t0_means": ["t0_mean"],
    "t0_var_diag": ["t0_sd"],
    "t0_var_offdiag": list(INITIAL_STATE_CORRELATION_KEYWORDS),
    "t0_var_lower_free": list(INITIAL_STATE_CORRELATION_KEYWORDS),
    "obs_df": ["obs_df"],
    "obs_shape": ["obs_shape"],
    "obs_r": ["obs_r"],
    "obs_concentration": ["obs_concentration"],
    "obs_ordered_base": ["obs_ordered_base"],
    "obs_ordered_gaps": ["obs_ordered_gaps"],
    "obs_cat_intercepts": ["obs_cat_intercepts"],
    "obs_cat_slopes": ["obs_cat_slopes"],
    "hill_emax": ["hill_emax"],
    "hill_ec50": ["hill_ec50"],
    "hill_n": ["hill_n"],
    "multiplicative_weight": ["multiplicative_weight"],
    # dynamics_stability is a synthetic validation parameter (not a prior
    # field) that covers dynamics and diffusion parameters.
    "dynamics_stability": ["rho", "ar", "beta", "sigma", "sd"],
}

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


def normalize_prior_params(
    distribution: PriorDistributionFamily | str,
    params: dict,
) -> dict[str, float | int]:
    """Convert a typed prior distribution into compiler-normalized parameter params."""
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
