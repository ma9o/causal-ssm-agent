"""Stage 4 Orchestrator: Deterministic Model Specification Helpers.

Pre-computes everything that can be determined from the CausalSpec without
LLM judgment: parameter enumeration, unambiguous distribution/link choices,
and role-based constraints.

The agentic LLM logic now lives in ``orchestrator/stage4.py``.
"""

from causal_ssm_agent.orchestrator.schemas_model import (
    VALID_LIKELIHOODS_FOR_DTYPE,
    VALID_LINKS_FOR_DISTRIBUTION,
)
from causal_ssm_agent.utils.causal_spec import get_constructs, get_edges, get_indicators


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
                    "constraint": "unit_interval",
                    "description": f"AR(1) discrete-time persistence for {c['name']}",
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
