"""Accessor helpers for CausalSpec dicts.

Replaces the repeated pattern of causal_spec.get("latent", {}).get("constructs", [])
with clear, typed accessor functions.
"""


def get_constructs(causal_spec: dict) -> list[dict]:
    """Get constructs from a CausalSpec dict."""
    return causal_spec.get("latent", {}).get("constructs", [])


def get_edges(causal_spec: dict) -> list[dict]:
    """Get causal edges from a CausalSpec dict."""
    return causal_spec.get("latent", {}).get("edges", [])


def get_indicators(causal_spec: dict) -> list[dict]:
    """Get indicators from a CausalSpec dict."""
    return causal_spec.get("measurement", {}).get("indicators", [])


def get_indicator_info(causal_spec: dict) -> dict[str, dict]:
    """Extract indicator info from a CausalSpec dict.

    Returns:
        Dict mapping indicator name to {dtype, construct_name}
    """
    return {
        ind["name"]: {
            "dtype": ind.get("measurement_dtype"),
            "construct_name": ind.get("construct_name"),
        }
        for ind in get_indicators(causal_spec)
    }


def get_indicator_dtypes(causal_spec: dict) -> dict[str, str]:
    """Extract indicator name -> measurement_dtype mapping.

    Returns:
        Dict mapping indicator name to dtype string (e.g. "continuous", "binary")
    """
    return {
        ind["name"]: ind.get("measurement_dtype", "continuous")
        for ind in get_indicators(causal_spec)
    }


_WORKER_INDICATOR_KEYS = (
    "name",
    "measurement_dtype",
    "how_to_measure",
    "source_columns",
    "aggregation",
)


def make_extraction_context(causal_spec: dict) -> dict:
    """Build minimal context needed by Stage 2 extraction workers.

    Workers need:
    - indicators: name, measurement_dtype, how_to_measure, source_columns
    - outcome: name, description (for prompt context)

    Does not include: construct_name, aggregation, ordinal_levels,
    latent edges, or non-outcome constructs.
    """
    slim_indicators = [
        {k: ind[k] for k in _WORKER_INDICATOR_KEYS if k in ind}
        for ind in get_indicators(causal_spec)
    ]
    outcome = get_outcome_construct(causal_spec)
    slim_outcome = (
        {"name": outcome["name"], "description": outcome.get("description", "")}
        if outcome
        else None
    )
    return {
        "measurement": {"indicators": slim_indicators},
        "latent": {"constructs": [slim_outcome] if slim_outcome else []},
    }


def get_outcome_construct(causal_spec_or_latent: dict) -> dict | None:
    """Get the outcome construct dict from a CausalSpec or latent model dict.

    Handles both full CausalSpec dicts and bare latent model dicts.

    Returns:
        The outcome construct dict, or None if not found
    """
    # Handle both CausalSpec (has "latent" key) and bare latent model
    if "latent" in causal_spec_or_latent:
        constructs = get_constructs(causal_spec_or_latent)
    else:
        constructs = causal_spec_or_latent.get("constructs", [])

    for c in constructs:
        if c.get("is_outcome"):
            return c
    return None


def get_outcome_name(causal_spec_or_latent: dict) -> str | None:
    """Get the outcome construct name from a CausalSpec or latent model dict.

    Convenience wrapper around get_outcome_construct() that returns just the name.

    Args:
        causal_spec_or_latent: Either a full CausalSpec dict or a bare latent model dict.

    Returns:
        Name of the outcome construct, or None if not found.
    """
    outcome = get_outcome_construct(causal_spec_or_latent)
    return outcome["name"] if outcome else None
