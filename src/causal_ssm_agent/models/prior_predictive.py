"""Prior Predictive Validation for Stage 4.

Validates proposed priors by sampling from the prior predictive distribution
and checking for domain violations (NaN/Inf, constraint violations, extreme
values, scale plausibility).
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl

from causal_ssm_agent.orchestrator.schemas_model import ModelSpec
from causal_ssm_agent.workers.schemas_prior import PriorProposal, PriorValidationResult

logger = logging.getLogger(__name__)

# Map ModelSpec parameter keywords to SSM sample site names
# Same pattern as _PRIOR_RULES in ssm_builder.py
_PARAM_TO_SITE: list[tuple[list[str], list[str], str]] = [
    (["rho", "ar"], ["drift_diag_pop"], "positive"),
    (["sigma", "sd"], ["diffusion_diag_pop", "manifest_var_diag"], "positive"),
    (["beta"], ["drift_offdiag_pop"], "none"),
]


def _pivot_raw_data(raw_data: pl.DataFrame) -> pd.DataFrame:
    """Pivot long-format raw data to wide format for model building.

    Same long-to-wide pivot used in build_model_task.
    """
    time_col = "time_bucket" if "time_bucket" in raw_data.columns else "timestamp"
    wide_data = (
        raw_data.with_columns(pl.col("value").cast(pl.Float64, strict=False))
        .pivot(on="indicator", index=time_col, values="value")
        .sort(time_col)
    )
    X = wide_data.to_pandas()
    if time_col in X.columns:
        X = X.rename(columns={time_col: "time"})
    return X


def _compute_data_stats(raw_data: pl.DataFrame) -> dict[str, dict]:
    """Compute per-indicator mean, std, min, max from raw data."""
    stats = {}
    for row in (
        raw_data.group_by("indicator")
        .agg(
            [
                pl.col("value").cast(pl.Float64, strict=False).mean().alias("mean"),
                pl.col("value").cast(pl.Float64, strict=False).std().alias("std"),
                pl.col("value").cast(pl.Float64, strict=False).min().alias("min"),
                pl.col("value").cast(pl.Float64, strict=False).max().alias("max"),
            ]
        )
        .iter_rows(named=True)
    ):
        stats[row["indicator"]] = {
            "mean": row["mean"],
            "std": row["std"],
            "min": row["min"],
            "max": row["max"],
        }
    return stats


def _check_nan_inf(samples: dict[str, jnp.ndarray]) -> PriorValidationResult | None:
    """Check for NaN or Inf in any sample site."""
    bad_sites = []
    for name, values in samples.items():
        arr = np.asarray(values)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            bad_sites.append(name)

    if bad_sites:
        return PriorValidationResult(
            parameter="prior_predictive",
            is_valid=False,
            issue=f"NaN/Inf detected in sample sites: {', '.join(bad_sites)}",
            suggested_adjustment="Check for degenerate priors or numerical overflow",
        )
    return None


def _check_constraint_violations(
    samples: dict[str, jnp.ndarray],
    threshold: float = 0.01,
) -> list[PriorValidationResult]:
    """Check for constraint violations in sampled parameters.

    Positive-constrained sites (HalfNormal-sampled): diffusion_diag_pop, manifest_var_diag
    should not have negative values.
    """
    results = []
    positive_sites = ["diffusion_diag_pop", "manifest_var_diag", "t0_var_diag"]

    for site_name in positive_sites:
        if site_name not in samples:
            continue
        arr = np.asarray(samples[site_name])
        n_total = arr.size
        if n_total == 0:
            continue
        n_violations = int(np.sum(arr < 0))
        violation_rate = n_violations / n_total
        if violation_rate > threshold:
            results.append(
                PriorValidationResult(
                    parameter=site_name,
                    is_valid=False,
                    issue=(
                        f"Constraint violation: {violation_rate:.1%} of {site_name} samples "
                        f"are negative (should be positive)"
                    ),
                    suggested_adjustment="Use HalfNormal or other positive-constrained prior",
                )
            )

    return results


def _check_extreme_values(
    samples: dict[str, jnp.ndarray],
    threshold: float = 0.10,
    extreme_cutoff: float = 1e6,
) -> list[PriorValidationResult]:
    """Check for extreme parameter values indicating priors too wide."""
    results = []
    # Check parameter sites (not deterministic outputs like drift, diffusion)
    param_sites = [
        k
        for k in samples
        if k.endswith("_pop") or k.endswith("_diag") or k == "cint_pop" or k == "lambda_free"
    ]
    for site_name in param_sites:
        arr = np.asarray(samples[site_name])
        n_total = arr.size
        if n_total == 0:
            continue
        n_extreme = int(np.sum(np.abs(arr) > extreme_cutoff))
        extreme_rate = n_extreme / n_total
        if extreme_rate > threshold:
            results.append(
                PriorValidationResult(
                    parameter=site_name,
                    is_valid=False,
                    issue=(
                        f"Extreme values: {extreme_rate:.1%} of {site_name} samples "
                        f"have |value| > {extreme_cutoff:.0e}"
                    ),
                    suggested_adjustment="Tighten the prior (reduce sigma)",
                )
            )

    return results


def _check_scale_plausibility(
    samples: dict[str, jnp.ndarray],
    data_stats: dict[str, dict],
    manifest_names: list[str],
    n_subsample: int = 50,
    ratio_threshold: float = 100.0,
) -> list[PriorValidationResult]:
    """Check implied observation scale vs data scale.

    For a subsample of draws, compute stationary covariance analytically:
      solve_lyapunov(drift, diffusion @ diffusion.T) -> Sigma_inf
      implied_obs_cov = lambda @ Sigma_inf @ lambda.T + manifest_cov

    Compare sqrt(diag(implied_obs_cov)) to data std per indicator.
    """
    from causal_ssm_agent.models.ssm.discretization import solve_lyapunov

    results = []

    if "drift" not in samples or "diffusion" not in samples:
        return results

    drift_samples = np.asarray(samples["drift"])
    diffusion_samples = np.asarray(samples["diffusion"])

    # Get lambda and manifest_cov if available
    lambda_samples = np.asarray(samples.get("lambda")) if "lambda" in samples else None
    manifest_cov_samples = (
        np.asarray(samples.get("manifest_cov")) if "manifest_cov" in samples else None
    )

    n_total = drift_samples.shape[0]
    idx = np.random.default_rng(42).choice(n_total, size=min(n_subsample, n_total), replace=False)

    implied_stds_list = []
    n_unstable = 0

    for i in idx:
        drift_i = jnp.array(drift_samples[i])
        diff_i = jnp.array(diffusion_samples[i])
        diff_cov_i = diff_i @ diff_i.T

        try:
            sigma_inf = solve_lyapunov(drift_i, diff_cov_i)
            sigma_inf_np = np.asarray(sigma_inf)

            # Check stability: Sigma_inf should be positive semi-definite
            if np.any(np.isnan(sigma_inf_np)) or np.any(np.diag(sigma_inf_np) < 0):
                n_unstable += 1
                continue

            # Compute implied observation covariance
            if lambda_samples is not None:
                lam = jnp.array(lambda_samples[i] if lambda_samples.ndim == 3 else lambda_samples)
            else:
                lam = jnp.eye(drift_i.shape[0])

            implied_obs = np.asarray(lam @ sigma_inf @ lam.T)
            if manifest_cov_samples is not None:
                mcov = (
                    manifest_cov_samples[i]
                    if manifest_cov_samples.ndim == 3
                    else manifest_cov_samples
                )
                implied_obs = implied_obs + np.asarray(mcov)

            implied_std = np.sqrt(np.maximum(np.diag(implied_obs), 0))
            implied_stds_list.append(implied_std)

        except Exception:
            n_unstable += 1
            continue

    if n_unstable > len(idx) * 0.5:
        results.append(
            PriorValidationResult(
                parameter="dynamics_stability",
                is_valid=False,
                issue=(
                    f"Unstable dynamics: {n_unstable}/{len(idx)} prior draws have "
                    f"unstable drift (Lyapunov solver failed)"
                ),
                suggested_adjustment="Tighten drift_diag prior toward more negative values",
            )
        )

    if not implied_stds_list:
        return results

    median_implied = np.median(implied_stds_list, axis=0)

    for j, name in enumerate(manifest_names):
        if j >= len(median_implied):
            break
        if name not in data_stats or data_stats[name]["std"] is None:
            continue

        data_std = data_stats[name]["std"]
        if data_std == 0 or data_std is None:
            continue

        ratio = float(median_implied[j]) / data_std
        if ratio > ratio_threshold or ratio < 1.0 / ratio_threshold:
            results.append(
                PriorValidationResult(
                    parameter=f"scale_{name}",
                    is_valid=False,
                    issue=(
                        f"Scale mismatch for {name}: implied std "
                        f"({median_implied[j]:.2g}) vs data std ({data_std:.2g}), "
                        f"ratio={ratio:.1g}"
                    ),
                    suggested_adjustment=("Adjust diffusion/drift priors to match data scale"),
                )
            )

    return results


def validate_prior_predictive(
    model_spec: ModelSpec | dict,
    priors: dict[str, PriorProposal] | dict[str, dict],
    raw_data: pl.DataFrame | None = None,
    n_samples: int = 500,
) -> tuple[bool, list[PriorValidationResult]]:
    """Validate priors via prior predictive sampling.

    Checks for:
    1. Model builds successfully
    2. No NaN/Inf in samples
    3. Constraint violations (positive params < 0, etc.)
    4. Extreme values (|param| > 1e6)
    5. Scale plausibility vs data (if raw_data provided)

    Args:
        model_spec: Model specification
        priors: Prior proposals for each parameter
        raw_data: Raw timestamped data (optional, for scale plausibility check)
        n_samples: Number of prior predictive samples

    Returns:
        Tuple of (is_valid, list of validation results)
    """
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

    priors_dict = {}
    for name, prior in priors.items():
        if isinstance(prior, PriorProposal):
            priors_dict[name] = prior.model_dump()
        else:
            priors_dict[name] = prior

    # Parse model_spec for manifest names
    if isinstance(model_spec, dict):
        spec_obj = ModelSpec.model_validate(model_spec)
    else:
        spec_obj = model_spec

    manifest_names = [lik.variable for lik in spec_obj.likelihoods]

    # 1. Build model
    try:
        builder = SSMModelBuilder(model_spec=model_spec, priors=priors_dict)

        if raw_data is not None and not raw_data.is_empty():
            X_wide = _pivot_raw_data(raw_data)
        else:
            # Create minimal dummy data for building
            cols = {name: np.zeros(10) for name in manifest_names}
            cols["time"] = np.arange(10, dtype=float)
            X_wide = pd.DataFrame(cols)

        builder.build_model(X_wide)
    except Exception as e:
        return False, [
            PriorValidationResult(
                parameter="model_build",
                is_valid=False,
                issue=f"Model build failed: {e}",
                suggested_adjustment="Fix model_spec or priors to enable model construction",
            )
        ]

    # 2. Sample prior predictive
    try:
        samples = builder.sample_prior_predictive(samples=n_samples)
    except Exception as e:
        return False, [
            PriorValidationResult(
                parameter="prior_sampling",
                is_valid=False,
                issue=f"Prior predictive sampling failed: {e}",
                suggested_adjustment="Check priors for numerical issues",
            )
        ]

    # 3. Run checks
    results: list[PriorValidationResult] = []

    # Check 1: NaN/Inf
    nan_result = _check_nan_inf(samples)
    if nan_result is not None:
        results.append(nan_result)

    # Check 2: Constraint violations
    results.extend(_check_constraint_violations(samples))

    # Check 3: Extreme values
    results.extend(_check_extreme_values(samples))

    # Check 4: Scale plausibility (only if raw_data provided)
    if raw_data is not None and not raw_data.is_empty():
        data_stats = _compute_data_stats(raw_data)
        results.extend(_check_scale_plausibility(samples, data_stats, manifest_names))

    is_valid = all(r.is_valid for r in results)

    # If no issues found, add passing results per parameter
    if not results:
        for param_name in priors_dict:
            results.append(
                PriorValidationResult(
                    parameter=param_name,
                    is_valid=True,
                    issue=None,
                    suggested_adjustment=None,
                )
            )
        is_valid = True

    return is_valid, results


def format_validation_report(
    is_valid: bool,
    results: list[PriorValidationResult],
) -> str:
    """Format validation results as a human-readable report."""
    lines = []

    if is_valid:
        lines.append("Prior predictive validation PASSED")
    else:
        lines.append("Prior predictive validation FAILED")

    failed = [r for r in results if not r.is_valid]
    if failed:
        lines.append("")
        for r in failed:
            lines.append(f"- {r.parameter}: {r.issue}")

    return "\n".join(lines)


def format_parameter_feedback(
    parameter_name: str,
    results: list[PriorValidationResult],
    prior: dict | None = None,
    data_stats: dict[str, dict] | None = None,
) -> str:
    """Format per-parameter validation feedback for LLM re-elicitation.

    Creates a structured message that tells the LLM what went wrong with
    a specific parameter's prior and provides context for revision.

    Args:
        parameter_name: Name of the parameter to generate feedback for
        results: All validation results from the prior predictive check
        prior: The previous prior proposal dict (for showing what was tried)
        data_stats: Per-indicator data statistics (for scale context)

    Returns:
        Formatted feedback string for inclusion in re-elicitation prompt
    """
    # Find results relevant to this parameter
    # Global failures (affect all parameters) are always included
    _GLOBAL_FAILURES = {"prior_predictive", "dynamics_stability", "model_build", "prior_sampling"}
    param_lower = parameter_name.lower()
    relevant = [
        r
        for r in results
        if not r.is_valid
        and (
            r.parameter == parameter_name
            or param_lower in r.parameter.lower()
            or r.parameter.lower().startswith("scale_")  # scale mismatch affects all
            or r.parameter in _GLOBAL_FAILURES
        )
    ]

    if not relevant:
        return ""

    lines = []

    # Show what was previously proposed
    if prior:
        dist = prior.get("distribution", "Unknown")
        params = prior.get("params", {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"Your previous prior for {parameter_name} was {dist}({params_str}).")

    lines.append("Prior predictive validation FAILED:")
    for r in relevant:
        lines.append(f"- {r.issue}")
        if r.suggested_adjustment:
            lines.append(f"  Suggested: {r.suggested_adjustment}")

    # Add data scale context if available
    if data_stats:
        scale_lines = []
        for indicator, stats in data_stats.items():
            std = stats.get("std")
            mean = stats.get("mean")
            if std is not None and mean is not None:
                scale_lines.append(f"  {indicator}: mean={mean:.2g}, std={std:.2g}")
        if scale_lines:
            lines.append("")
            lines.append("Data scale reference:")
            lines.extend(scale_lines)

    lines.append("")
    lines.append("Please revise your prior to be consistent with the data scale.")

    return "\n".join(lines)


def get_failed_parameters(
    results: list[PriorValidationResult],
    parameter_names: list[str],
) -> list[str]:
    """Extract parameter names that contributed to validation failure.

    Maps validation result parameter names (which may be SSM site names like
    'drift_diag_pop' or 'scale_mood') back to ModelSpec parameter names.

    Args:
        results: Validation results from prior predictive check
        parameter_names: All ModelSpec parameter names

    Returns:
        List of ModelSpec parameter names that need re-elicitation
    """
    failed_results = [r for r in results if not r.is_valid]
    if not failed_results:
        return []

    # Check for global failures that affect all parameters
    global_failures = {"prior_predictive", "model_build", "prior_sampling"}
    if any(r.parameter in global_failures for r in failed_results):
        return list(parameter_names)

    # Map SSM site names back to parameter names using keyword matching
    # Same keyword patterns as _PRIOR_RULES in ssm_builder.py
    _SITE_TO_KEYWORDS: dict[str, list[str]] = {
        "drift_diag": ["rho", "ar"],
        "drift_offdiag": ["beta"],
        "diffusion_diag": ["sigma", "sd"],
        "dynamics_stability": ["rho", "ar", "sigma", "sd"],  # drift + diffusion
    }

    failed_params = set()
    for r in failed_results:
        result_param = r.parameter.lower()

        # Direct match
        for param_name in parameter_names:
            if param_name.lower() in result_param or result_param in param_name.lower():
                failed_params.add(param_name)
                continue

        # Keyword-based match via SSM site names
        for site_prefix, keywords in _SITE_TO_KEYWORDS.items():
            if site_prefix in result_param:
                for param_name in parameter_names:
                    if any(kw in param_name.lower() for kw in keywords):
                        failed_params.add(param_name)

        # Scale mismatch (scale_<indicator>) -> all parameters affect scale
        if result_param.startswith("scale_"):
            failed_params.update(parameter_names)

    return list(failed_params) if failed_params else list(parameter_names)
