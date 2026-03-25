"""Accessor helpers for CausalSpec dicts.

The contract now has two structural layers:
- ``latent``: user-facing theoretical DAG
- ``estimation``: retained executable state-space projection

Use the explicit latent/estimation accessors in new code. The historical
``get_constructs()`` / ``get_edges()`` helpers still refer to the latent DAG.
"""

import networkx as nx

from causal_ssm_agent.utils.observation_semantics import (
    get_anchor_policy,
    get_summary_operator,
    get_support_kind,
)


def get_constructs(causal_spec: dict) -> list[dict]:
    """Get latent constructs from a CausalSpec dict."""
    return causal_spec.get("latent", {}).get("constructs", [])


def get_edges(causal_spec: dict) -> list[dict]:
    """Get latent DAG edges from a CausalSpec dict."""
    return causal_spec.get("latent", {}).get("edges", [])


def get_latent_constructs(causal_spec: dict) -> list[dict]:
    """Get user-facing latent constructs from a CausalSpec dict."""
    return get_constructs(causal_spec)


def get_latent_edges(causal_spec: dict) -> list[dict]:
    """Get user-facing latent DAG edges from a CausalSpec dict."""
    return get_edges(causal_spec)


def get_indicators(causal_spec: dict) -> list[dict]:
    """Get indicators from a CausalSpec dict."""
    return causal_spec.get("measurement", {}).get("indicators", [])


def get_estimation_spec(causal_spec: dict) -> dict:
    """Get the estimation projection, failing loudly when it is missing."""
    estimation = causal_spec.get("estimation")
    if not isinstance(estimation, dict):
        raise ValueError("causal_spec.estimation is required for estimation-sensitive access")
    return estimation


def get_estimation_state_order(causal_spec: dict) -> list[str]:
    """Get retained latent states in canonical estimation order."""
    return list(get_estimation_spec(causal_spec).get("state_order") or [])


def get_estimation_edges(causal_spec: dict) -> list[dict]:
    """Get retained directed edges in the estimation projection."""
    return list(get_estimation_spec(causal_spec).get("edges") or [])


def get_estimation_constructs(causal_spec: dict) -> list[dict]:
    """Get retained latent construct payloads in estimation-state order."""
    latent_lookup = {
        construct["name"]: construct for construct in get_latent_constructs(causal_spec) if construct.get("name")
    }
    return [
        latent_lookup[name]
        for name in get_estimation_state_order(causal_spec)
        if name in latent_lookup
    ]


def get_induced_dependencies(causal_spec: dict) -> list[dict]:
    """Get induced dependencies created by marginalizing latent roots."""
    return list(get_estimation_spec(causal_spec).get("induced_dependencies") or [])


def get_effective_observation_window(indicator: dict, model_clock: str | None) -> str | None:
    """Return the effective support window for an indicator."""
    return indicator.get("observation_window") or model_clock


def get_indicator_info(causal_spec: dict) -> dict[str, dict]:
    """Extract indicator info from a CausalSpec dict.

    Returns:
        Dict mapping indicator name to semantic extraction/measurement metadata.
    """
    return {
        ind["name"]: {
            "dtype": ind.get("measurement_dtype"),
            "construct_name": ind.get("construct_name"),
            "ordinal_levels": ind.get("ordinal_levels"),
            "support_kind": get_support_kind(ind),
            "summary_operator": get_summary_operator(ind),
            "anchor_policy": get_anchor_policy(ind),
            "observation_window": ind.get("observation_window"),
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
    "observation_window",
    "ordinal_levels",
)


def make_extraction_context(causal_spec: dict) -> dict:
    """Build minimal context needed by Stage 2 extraction workers.

    Workers need:
    - indicators: name, measurement_dtype, how_to_measure, source_columns,
      aggregation, support_kind, summary_operator, anchor_policy, observation_window
    - outcome: name, description (for prompt context)

    Does not include: construct_name, latent edges, or non-outcome constructs.
    Includes ordinal_levels only for ordinal indicators so workers can use a
    stable numeric codebook.
    """
    model_clock = causal_spec.get("measurement", {}).get("model_clock")
    slim_indicators = [
        {
            **{k: ind[k] for k in _WORKER_INDICATOR_KEYS if k in ind},
            "support_kind": get_support_kind(ind),
            "summary_operator": get_summary_operator(ind),
            "anchor_policy": get_anchor_policy(ind),
            **(
                {
                    "observation_window": get_effective_observation_window(ind, model_clock),
                }
                if get_effective_observation_window(ind, model_clock)
                else {}
            ),
        }
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


# ---------------------------------------------------------------------------
# Graph utilities (merged from effects.py)
# ---------------------------------------------------------------------------


def build_digraph(latent_model: dict) -> nx.DiGraph:
    """Build a simple DiGraph from a latent model's edge list.

    Args:
        latent_model: Dict with 'edges' list of {cause, effect} dicts

    Returns:
        nx.DiGraph with one node per referenced construct
    """
    return build_digraph_from_edges(latent_model.get("edges", []))


def build_digraph_from_edges(edges: list[dict]) -> nx.DiGraph:
    """Build a simple DiGraph from an edge list."""
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge["cause"], edge["effect"])
    return G


def _get_treatments_from_graph(
    *,
    node_names: list[str],
    edges: list[dict],
    outcome: str | None,
) -> list[str]:
    """Return nodes with a directed path to the outcome within the given graph."""
    if not outcome:
        return []

    G = build_digraph_from_edges(edges)
    G.add_nodes_from(node_names)
    if outcome not in G:
        return []

    return sorted(
        node
        for node in node_names
        if node != outcome and G.has_node(node) and nx.has_path(G, node, outcome)
    )


def get_all_treatments(latent_model: dict) -> list[str]:
    """Get all potential treatments from latent model.

    A treatment is any construct that has a causal path to the outcome.

    Args:
        latent_model: Dict with 'constructs' and 'edges'

    Returns:
        Sorted list of treatment construct names
    """
    return _get_treatments_from_graph(
        node_names=[
            construct["name"]
            for construct in latent_model.get("constructs", [])
            if construct.get("name")
        ],
        edges=list(latent_model.get("edges", []) or []),
        outcome=get_outcome_name(latent_model),
    )


def get_estimable_treatments(causal_spec: dict) -> list[str]:
    """Get intervention targets that remain in the retained estimation graph."""
    state_order = get_estimation_state_order(causal_spec)
    return _get_treatments_from_graph(
        node_names=state_order,
        edges=get_estimation_edges(causal_spec),
        outcome=get_outcome_name(causal_spec),
    )
