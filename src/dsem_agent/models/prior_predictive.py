"""Prior Predictive Validation for Stage 4.

Validates proposed priors by sampling from the prior predictive distribution
and checking for domain violations (NaN/Inf, wrong sign for constrained params).

NOTE: CT-SEM implementation pending merge from numpyro-ctsem.
Currently provides stub validation that passes through.
"""

import numpy as np
import polars as pl

from dsem_agent.orchestrator.schemas_glmm import GLMMSpec
from dsem_agent.workers.schemas_prior import PriorProposal, PriorValidationResult


def validate_prior_predictive(
    glmm_spec: GLMMSpec | dict,
    priors: dict[str, PriorProposal] | dict[str, dict],
    raw_data: pl.DataFrame | None = None,
    n_samples: int = 500,
) -> tuple[bool, list[PriorValidationResult]]:
    """Validate priors via prior predictive sampling.

    Checks for:
    1. Model builds successfully
    2. No NaN/Inf in samples
    3. Domain violations (negative for positive-constrained, outside [0,1] for unit_interval)

    Args:
        glmm_spec: GLMM specification
        priors: Prior proposals for each parameter
        raw_data: Raw timestamped data (optional)
        n_samples: Number of prior predictive samples

    Returns:
        Tuple of (is_valid, list of validation results)

    NOTE: Full implementation will be merged from numpyro-ctsem.
    Currently returns valid for all priors.
    """
    # TODO: Implement with CTSEMModelBuilder once merged
    # For now, return valid to allow pipeline to proceed

    # Convert inputs to dicts if needed
    if isinstance(glmm_spec, GLMMSpec):
        glmm_dict = glmm_spec.model_dump()
    else:
        glmm_dict = glmm_spec

    priors_dict = {}
    for name, prior in priors.items():
        if isinstance(prior, PriorProposal):
            priors_dict[name] = prior.model_dump()
        else:
            priors_dict[name] = prior

    # Return valid results for all parameters
    results: list[PriorValidationResult] = []
    for param_name in priors_dict.keys():
        results.append(PriorValidationResult(
            parameter=param_name,
            is_valid=True,
            issue=None,
            suggested_adjustment=None,
        ))

    return True, results


def _validate_parameter(
    param_name: str,
    prior_spec: dict,
    idata: "InferenceData",
) -> PriorValidationResult:
    """Validate a single parameter's prior samples."""
    if param_name not in idata.prior:
        return PriorValidationResult(
            parameter=param_name,
            is_valid=True,
            issue=None,
            suggested_adjustment=None,
        )

    samples = idata.prior[param_name].values.flatten()

    # Check for NaN/Inf
    n_invalid = np.sum(~np.isfinite(samples))
    if n_invalid > 0:
        pct = 100 * n_invalid / len(samples)
        return PriorValidationResult(
            parameter=param_name,
            is_valid=False,
            issue=f"{pct:.1f}% of samples are NaN/Inf",
            suggested_adjustment=None,
        )

    # Check domain violations based on distribution
    dist_name = prior_spec.get("distribution", "Normal")
    constraint = _get_constraint_from_distribution(dist_name)

    if constraint == "positive":
        n_negative = np.sum(samples < 0)
        if n_negative > 0:
            pct = 100 * n_negative / len(samples)
            return PriorValidationResult(
                parameter=param_name,
                is_valid=False,
                issue=f"{pct:.1f}% of samples are negative (should be positive)",
                suggested_adjustment=None,
            )

    elif constraint == "unit_interval":
        n_outside = np.sum((samples < 0) | (samples > 1))
        if n_outside > 0:
            pct = 100 * n_outside / len(samples)
            return PriorValidationResult(
                parameter=param_name,
                is_valid=False,
                issue=f"{pct:.1f}% of samples outside [0, 1]",
                suggested_adjustment=None,
            )

    return PriorValidationResult(
        parameter=param_name,
        is_valid=True,
        issue=None,
        suggested_adjustment=None,
    )


def _validate_prior_predictive_samples(
    var_name: str,
    samples: np.ndarray,
    distribution: str,
) -> PriorValidationResult:
    """Validate prior predictive samples for an observed variable."""
    samples = samples.flatten()

    # Check for NaN/Inf
    n_invalid = np.sum(~np.isfinite(samples))
    if n_invalid > 0:
        pct = 100 * n_invalid / len(samples)
        return PriorValidationResult(
            parameter=f"prior_pred_{var_name}",
            is_valid=False,
            issue=f"{pct:.1f}% of prior predictive samples are NaN/Inf",
            suggested_adjustment=None,
        )

    # Check domain violations based on distribution
    valid = samples[np.isfinite(samples)]
    if len(valid) == 0:
        return PriorValidationResult(
            parameter=f"prior_pred_{var_name}",
            is_valid=False,
            issue="No valid samples",
            suggested_adjustment=None,
        )

    if distribution in ("Poisson", "NegativeBinomial", "Gamma"):
        n_negative = np.sum(valid < 0)
        if n_negative > 0:
            pct = 100 * n_negative / len(valid)
            return PriorValidationResult(
                parameter=f"prior_pred_{var_name}",
                is_valid=False,
                issue=f"{pct:.1f}% of samples are negative (should be positive)",
                suggested_adjustment=None,
            )

    elif distribution in ("Bernoulli", "Beta"):
        n_outside = np.sum((valid < 0) | (valid > 1))
        if n_outside > 0:
            pct = 100 * n_outside / len(valid)
            return PriorValidationResult(
                parameter=f"prior_pred_{var_name}",
                is_valid=False,
                issue=f"{pct:.1f}% of samples outside [0, 1]",
                suggested_adjustment=None,
            )

    return PriorValidationResult(
        parameter=f"prior_pred_{var_name}",
        is_valid=True,
        issue=None,
        suggested_adjustment=None,
    )


def _get_constraint_from_distribution(dist_name: str) -> str:
    """Get the implicit constraint from distribution name."""
    positive_dists = {"HalfNormal", "Gamma", "InverseGamma", "Exponential", "HalfCauchy"}
    unit_interval_dists = {"Beta"}

    if dist_name in positive_dists:
        return "positive"
    elif dist_name in unit_interval_dists:
        return "unit_interval"
    else:
        return "none"


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
