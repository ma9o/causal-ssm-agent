"""Stage 4 Orchestrator: Model Specification Proposal.

The orchestrator proposes a complete model specification based on the CausalSpec,
enumerating all parameters needing priors with search context for literature.

Deterministic parts (parameter enumeration, unambiguous distributions/links,
role-based constraints) are pre-computed from the CausalSpec. The LLM only
provides genuine decisions: distribution choices for ambiguous dtypes, loading
constraints, search_context strings, and reasoning.
"""

from causal_ssm_agent.orchestrator.prompts.model_proposal import (
    SYSTEM as MODEL_PROPOSAL_SYSTEM,
)
from causal_ssm_agent.orchestrator.prompts.model_proposal import (
    USER as MODEL_PROPOSAL_USER,
)
from causal_ssm_agent.orchestrator.prompts.model_proposal import (
    format_ambiguous_indicators,
    format_loading_params,
    format_parameters,
    format_resolved_likelihoods,
)
from causal_ssm_agent.orchestrator.schemas_model import (
    VALID_LIKELIHOODS_FOR_DTYPE,
    VALID_LINKS_FOR_DISTRIBUTION,
    Stage4OrchestratorResult,
)
from causal_ssm_agent.utils.causal_spec import get_constructs, get_edges, get_indicators
from causal_ssm_agent.utils.llm import (
    OrchestratorGenerateFn,
    make_validate_model_spec_tool,
)


def derive_deterministic_spec(
    causal_spec: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Pre-compute all deterministic parts of the model specification.

    Derives everything that can be determined from the CausalSpec without
    LLM judgment: parameter enumeration, unambiguous distribution/link choices,
    and role-based constraints.

    Args:
        causal_spec: The full CausalSpec dict

    Returns:
        Tuple of:
        - resolved_likelihoods: [{variable, distribution, link, reasoning}]
          for indicators whose distribution AND link are fully determined by dtype
        - ambiguous_indicators: [{variable, dtype, valid_distributions, valid_links}]
          for indicators that need LLM distribution/link choices
        - parameters: [{name, role, constraint, description}]
          all parameters pre-enumerated with roles and deterministic constraints
        - loading_params: [{name, role, constraint, description, indicator, construct}]
          loading parameters that need LLM constraint decision (positive or none)
    """
    constructs = get_constructs(causal_spec)
    edges = get_edges(causal_spec)
    indicators = get_indicators(causal_spec)

    # --- Likelihoods ---
    resolved_likelihoods: list[dict] = []
    ambiguous_indicators: list[dict] = []

    for ind in indicators:
        name = ind["name"]
        dtype = ind.get("measurement_dtype", "continuous")
        valid_dists = VALID_LIKELIHOODS_FOR_DTYPE.get(dtype, set())

        if len(valid_dists) == 1:
            dist = next(iter(valid_dists))
            valid_links = VALID_LINKS_FOR_DISTRIBUTION[dist]
            if len(valid_links) == 1:
                # Fully deterministic: single distribution, single link
                link = next(iter(valid_links))
                resolved_likelihoods.append(
                    {
                        "variable": name,
                        "distribution": dist.value,
                        "link": link.value,
                        "reasoning": f"{dtype} dtype → {dist.value} / {link.value}",
                    }
                )
            else:
                # Distribution is forced, but link has multiple options
                ambiguous_indicators.append(
                    {
                        "variable": name,
                        "dtype": dtype,
                        "fixed_distribution": dist.value,
                        "valid_links": sorted(lf.value for lf in valid_links),
                    }
                )
        else:
            # Multiple valid distributions — build a map of link options per dist
            link_options: dict[str, list[str]] = {}
            for d in sorted(valid_dists, key=lambda x: x.value):
                links = VALID_LINKS_FOR_DISTRIBUTION[d]
                link_options[d.value] = sorted(lf.value for lf in links)
            ambiguous_indicators.append(
                {
                    "variable": name,
                    "dtype": dtype,
                    "valid_distributions": sorted(d.value for d in valid_dists),
                    "link_options": link_options,
                }
            )

    # --- Parameters ---
    parameters: list[dict] = []
    loading_params: list[dict] = []

    # Count indicators per construct for loading detection
    indicators_per_construct: dict[str, list[str]] = {}
    for ind in indicators:
        cn = ind.get("construct_name")
        if cn:
            indicators_per_construct.setdefault(cn, []).append(ind["name"])

    # AR coefficients for time-varying endogenous constructs
    for c in constructs:
        if c.get("temporal_status") == "time_varying" and c.get("role") == "endogenous":
            parameters.append(
                {
                    "name": f"rho_{c['name']}",
                    "role": "ar_coefficient",
                    "constraint": "correlation",
                    "description": f"AR(1) persistence for {c['name']}",
                }
            )

    # Fixed effects for each causal edge
    for edge in edges:
        cause = edge["cause"]
        effect = edge["effect"]
        parameters.append(
            {
                "name": f"beta_{cause}_{effect}",
                "role": "fixed_effect",
                "constraint": "none",
                "description": f"Effect of {cause} on {effect}",
            }
        )

    # Residual SDs for each construct
    for c in constructs:
        parameters.append(
            {
                "name": f"sigma_{c['name']}",
                "role": "residual_sd",
                "constraint": "positive",
                "description": f"Residual/innovation SD for {c['name']}",
            }
        )

    # Loadings for multi-indicator constructs (non-reference indicators only)
    # Convention: first indicator per construct is the reference (fixed at 1.0)
    reference_set: set[str] = set()
    for ind in indicators:
        cn = ind.get("construct_name")
        if cn and cn in indicators_per_construct and len(indicators_per_construct[cn]) > 1:
            if cn not in reference_set:
                reference_set.add(cn)  # First indicator = reference, no param
            else:
                loading_params.append(
                    {
                        "name": f"lambda_{ind['name']}_{cn}",
                        "role": "loading",
                        "constraint": "positive",  # default; LLM can override to "none"
                        "description": f"Factor loading for {ind['name']} on {cn}",
                        "indicator": ind["name"],
                        "construct": cn,
                    }
                )

    return resolved_likelihoods, ambiguous_indicators, parameters, loading_params


async def propose_model_spec(
    causal_spec: dict,
    data_summary: str,
    question: str,
    generate: OrchestratorGenerateFn,
) -> Stage4OrchestratorResult:
    """Orchestrator proposes complete model specification.

    Pre-computes deterministic parts from the CausalSpec, then asks the LLM
    only for genuine decisions (ambiguous distributions, loading constraints,
    search contexts, and reasoning).

    Args:
        causal_spec: The full CausalSpec dict (latent + measurement)
        data_summary: Summary of the data (time points, subjects, etc.)
        question: The research question for context
        generate: Async generate function (messages, tools, follow_ups) -> str

    Returns:
        Stage4OrchestratorResult with ModelSpec
    """
    # Pre-compute everything deterministic
    resolved_likelihoods, ambiguous_indicators, parameters, loading_params = (
        derive_deterministic_spec(causal_spec)
    )

    # All parameters (including loadings) for the prompt
    all_params = parameters + [
        {k: v for k, v in lp.items() if k not in ("indicator", "construct")}
        for lp in loading_params
    ]

    # Format for the prompt
    resolved_str = format_resolved_likelihoods(resolved_likelihoods)
    ambiguous_str = format_ambiguous_indicators(ambiguous_indicators)
    params_str = format_parameters(all_params)
    loading_str = format_loading_params(loading_params)

    # Build messages
    messages = [
        {"role": "system", "content": MODEL_PROPOSAL_SYSTEM},
        {
            "role": "user",
            "content": MODEL_PROPOSAL_USER.format(
                question=question,
                resolved_likelihoods=resolved_str,
                ambiguous_indicators=ambiguous_str,
                parameters=params_str,
                loading_params=loading_str,
                data_summary=data_summary,
            ),
        },
    ]

    # Generate with validation feedback loop
    tool, capture = make_validate_model_spec_tool(
        causal_spec,
        resolved_likelihoods=resolved_likelihoods,
        ambiguous_indicators=ambiguous_indicators,
        parameters=all_params,
        loading_params=loading_params,
    )
    completion = await generate(messages, [tool], None)

    # Use the captured spec from the validation tool
    model_spec = capture.get("spec")
    if model_spec is None:
        raise ValueError("Model spec validation never passed. Raw response:\n" + completion[:500])

    return Stage4OrchestratorResult(
        model_spec=model_spec,
        raw_response=completion,
    )


def build_data_summary(measurements_data: dict) -> str:
    """Build a summary of the measurement data for the orchestrator.

    Args:
        measurements_data: Dict of granularity -> polars DataFrame

    Returns:
        Human-readable summary string
    """
    lines = ["### Data Overview\n"]

    for granularity, df in measurements_data.items():
        if granularity == "time_invariant":
            n_indicators = len([c for c in df.columns if c != "time_bucket"])
            lines.append(f"- **Time-invariant**: {n_indicators} indicators")
        else:
            n_rows = df.height
            n_indicators = len([c for c in df.columns if c != "time_bucket"])
            lines.append(f"- **{granularity}**: {n_rows} time points × {n_indicators} indicators")

            # Add basic stats for numeric columns
            if n_rows > 0:
                sample_cols = [c for c in df.columns if c != "time_bucket"][:3]
                for col in sample_cols:
                    try:
                        mean = df[col].mean()
                        std = df[col].std()
                        if mean is not None and std is not None:
                            lines.append(f"    {col}: mean={mean:.2f}, std={std:.2f}")
                    except Exception:
                        pass

    return "\n".join(lines)
