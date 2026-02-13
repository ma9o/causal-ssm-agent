"""Prior Predictive Validation for Stage 4.

Validates proposed priors by sampling from the prior predictive distribution
and checking for domain violations (NaN/Inf, wrong sign for constrained params).

NOTE: Currently provides stub validation that passes through.
Full implementation with SSMModelBuilder is pending.
"""

import polars as pl

from causal_ssm_agent.orchestrator.schemas_model import ModelSpec
from causal_ssm_agent.workers.schemas_prior import PriorProposal, PriorValidationResult


def validate_prior_predictive(
    model_spec: ModelSpec | dict,  # noqa: ARG001
    priors: dict[str, PriorProposal] | dict[str, dict],
    raw_data: pl.DataFrame | None = None,  # noqa: ARG001
    n_samples: int = 500,  # noqa: ARG001
) -> tuple[bool, list[PriorValidationResult]]:
    """Validate priors via prior predictive sampling.

    Checks for:
    1. Model builds successfully
    2. No NaN/Inf in samples
    3. Domain violations (negative for positive-constrained, outside [0,1] for unit_interval)

    Args:
        model_spec: Model specification
        priors: Prior proposals for each parameter
        raw_data: Raw timestamped data (optional)
        n_samples: Number of prior predictive samples

    Returns:
        Tuple of (is_valid, list of validation results)

    NOTE: Stub implementation. Returns valid for all priors.
    """

    priors_dict = {}
    for name, prior in priors.items():
        if isinstance(prior, PriorProposal):
            priors_dict[name] = prior.model_dump()
        else:
            priors_dict[name] = prior

    # Return valid results for all parameters
    results: list[PriorValidationResult] = []
    for param_name in priors_dict:
        results.append(
            PriorValidationResult(
                parameter=param_name,
                is_valid=True,
                issue=None,
                suggested_adjustment=None,
            )
        )

    return True, results


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
