"""Stage 4: Prior Elicitation.

Core logic for LLM-assisted prior elicitation, decoupled from Prefect/Inspect.
Uses dependency injection for the LLM generate function.

See docs/modeling/functional_spec.md for design rationale.
"""

import json
from dataclasses import dataclass, field

from .prompts import (
    PRIOR_ELICITATION_SYSTEM,
    PRIOR_ELICITATION_USER,
)
from dsem_agent.utils.llm import OrchestratorGenerateFn, parse_json_response


@dataclass
class ElicitedPrior:
    """A single elicited prior distribution."""

    parameter: str
    mean: float
    std: float
    reasoning: str
    source: str = "llm"  # "llm" or "default"


@dataclass
class Stage4Result:
    """Result of Stage 4: elicited priors for all parameters."""

    priors: dict[str, ElicitedPrior]
    model_spec: dict
    elicitation_responses: list[dict] = field(default_factory=list)
    n_paraphrases: int = 1

    def to_prior_dict(self) -> dict:
        """Convert to dict format for PyMC construction."""
        result = {}
        for param_name, prior in self.priors.items():
            result[param_name] = {
                "mean": prior.mean,
                "std": prior.std,
                "reasoning": prior.reasoning,
                "source": prior.source,
            }
        return result


def _format_model_structure(model_spec: dict) -> str:
    """Format model structure for the prompt."""
    lines = []

    # Constructs
    lines.append("### Constructs")
    for name, spec in model_spec.get("constructs", {}).items():
        role = spec.get("role", "unknown")
        temporal = spec.get("temporal_status", "unknown")
        gran = spec.get("granularity", "N/A")
        outcome = " (OUTCOME)" if spec.get("is_outcome") else ""
        lines.append(f"- **{name}**: {role}, {temporal}, granularity={gran}{outcome}")

    # Edges
    lines.append("\n### Causal Relationships")
    for param_name, spec in model_spec.get("edges", {}).items():
        cause = spec["cause"]
        effect = spec["effect"]
        timing = "lagged" if spec.get("lagged") else "contemporaneous"
        lines.append(f"- {cause} → {effect} ({timing})")

    # Measurement
    lines.append("\n### Indicators")
    for name, spec in model_spec.get("measurement", {}).items():
        construct = spec["construct"]
        dtype = spec["dtype"]
        lines.append(f"- {name}: measures {construct} ({dtype})")

    return "\n".join(lines)


def _format_parameters(model_spec: dict) -> str:
    """Format parameters requiring priors."""
    lines = []

    # AR coefficients
    lines.append("### AR(1) Coefficients (temporal persistence)")
    for name, spec in model_spec.get("constructs", {}).items():
        if spec.get("ar_prior"):
            lines.append(f"- **rho_{name}**: Autocorrelation of {name} (must be in [0, 1])")

    # Cross-lag coefficients
    lines.append("\n### Causal Effect Coefficients")
    for param_name, spec in model_spec.get("edges", {}).items():
        cause = spec["cause"]
        effect = spec["effect"]
        lines.append(f"- **{param_name}**: Effect of {cause} on {effect}")

    # Residual variances
    lines.append("\n### Residual Standard Deviations")
    for name, spec in model_spec.get("constructs", {}).items():
        if spec.get("sigma_prior"):
            lines.append(f"- **sigma_{name}**: Unexplained variation in {name} (must be positive)")

    return "\n".join(lines)


def _parse_elicited_priors(
    response: str,
    model_spec: dict,
    default_priors: dict,
) -> dict[str, ElicitedPrior]:
    """Parse LLM response into ElicitedPrior objects."""
    priors = {}

    try:
        elicited = parse_json_response(response)
    except ValueError:
        # Fall back to defaults if parsing fails
        elicited = {}

    # Build expected parameters
    expected_params = set()

    for name, spec in model_spec.get("constructs", {}).items():
        if spec.get("ar_prior"):
            expected_params.add(f"rho_{name}")
        if spec.get("sigma_prior"):
            expected_params.add(f"sigma_{name}")

    for param_name in model_spec.get("edges", {}).keys():
        expected_params.add(param_name)

    # Process each parameter
    for param in expected_params:
        if param in elicited and isinstance(elicited[param], dict):
            prior_data = elicited[param]
            priors[param] = ElicitedPrior(
                parameter=param,
                mean=float(prior_data.get("mean", 0.0)),
                std=float(prior_data.get("std", 1.0)),
                reasoning=prior_data.get("reasoning", ""),
                source="llm",
            )
        else:
            # Use default
            if param.startswith("rho_"):
                default = default_priors.get("ar", {})
                # For AR, use midpoint of uniform as "mean"
                priors[param] = ElicitedPrior(
                    parameter=param,
                    mean=0.5,
                    std=0.25,
                    reasoning="Default: moderate persistence expected",
                    source="default",
                )
            elif param.startswith("sigma_"):
                priors[param] = ElicitedPrior(
                    parameter=param,
                    mean=1.0,
                    std=0.5,
                    reasoning="Default: unit-scale residual variance",
                    source="default",
                )
            elif param.startswith("beta_"):
                default = default_priors.get("beta", {})
                priors[param] = ElicitedPrior(
                    parameter=param,
                    mean=default.get("mean", 0.0),
                    std=default.get("std", 0.5),
                    reasoning="Default: weakly informative prior centered at zero",
                    source="default",
                )

    return priors


def _aggregate_priors(
    responses: list[dict[str, ElicitedPrior]],
) -> dict[str, ElicitedPrior]:
    """Aggregate multiple elicitation responses (AutoElicit-style).

    Uses pooled mean and inflated std to capture uncertainty across paraphrases.
    """
    if not responses:
        return {}

    if len(responses) == 1:
        return responses[0]

    # Collect all parameter names
    all_params = set()
    for resp in responses:
        all_params.update(resp.keys())

    aggregated = {}
    for param in all_params:
        values = [r[param] for r in responses if param in r]

        if not values:
            continue

        means = [v.mean for v in values]
        stds = [v.std for v in values]
        reasonings = [v.reasoning for v in values if v.reasoning]

        # Pooled mean
        pooled_mean = sum(means) / len(means)

        # Pooled std: sqrt(mean(std²) + var(means))
        # This inflates uncertainty to account for disagreement across paraphrases
        mean_var = sum(s**2 for s in stds) / len(stds)
        variance_of_means = sum((m - pooled_mean)**2 for m in means) / len(means)
        pooled_std = (mean_var + variance_of_means) ** 0.5

        # Combine reasonings
        combined_reasoning = reasonings[0] if reasonings else ""
        if len(reasonings) > 1:
            combined_reasoning += f" (aggregated from {len(responses)} elicitations)"

        aggregated[param] = ElicitedPrior(
            parameter=param,
            mean=pooled_mean,
            std=pooled_std,
            reasoning=combined_reasoning,
            source="llm_aggregated" if len(responses) > 1 else "llm",
        )

    return aggregated


async def run_stage4(
    model_spec: dict,
    question: str,
    generate: OrchestratorGenerateFn,
    default_priors: dict,
    n_paraphrases: int = 1,
) -> Stage4Result:
    """
    Run prior elicitation for all model parameters.

    Args:
        model_spec: Model specification from specify_model()
        question: The research question for context
        generate: Async function (messages, tools, follow_ups) -> completion
        default_priors: Fallback priors if elicitation fails
        n_paraphrases: Number of paraphrased elicitations (1 = no paraphrasing)

    Returns:
        Stage4Result with elicited priors
    """
    model_structure = _format_model_structure(model_spec)
    parameters = _format_parameters(model_spec)

    # Build elicitation messages
    messages = [
        {"role": "system", "content": PRIOR_ELICITATION_SYSTEM},
        {"role": "user", "content": PRIOR_ELICITATION_USER.format(
            question=question,
            model_structure=model_structure,
            parameters=parameters,
        )},
    ]

    # Single elicitation (no paraphrasing for now)
    # TODO: Implement paraphrased elicitation for n_paraphrases > 1
    completion = await generate(messages, None, None)

    elicitation_responses = [completion]

    # Parse response
    priors = _parse_elicited_priors(completion, model_spec, default_priors)

    return Stage4Result(
        priors=priors,
        model_spec=model_spec,
        elicitation_responses=elicitation_responses,
        n_paraphrases=1,
    )
