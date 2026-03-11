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


def make_extraction_context(causal_spec: dict) -> dict:
    """Build minimal context needed by Stage 2 extraction workers.

    Extracts only indicators and the outcome construct from the full CausalSpec,
    avoiding passing edges and all latent constructs to each worker.
    """
    outcome = get_outcome_construct(causal_spec)
    return {
        "measurement": {"indicators": get_indicators(causal_spec)},
        "latent": {"constructs": [outcome] if outcome else []},
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
