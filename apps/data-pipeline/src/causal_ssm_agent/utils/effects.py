"""Utilities for deriving treatments from latent models."""

import networkx as nx


def build_digraph(latent_model: dict) -> nx.DiGraph:
    """Build a simple DiGraph from a latent model's edge list.

    This is the single source of truth for creating a bare graph from
    latent_model edges. Use this instead of inlining the 3-line pattern.

    Args:
        latent_model: Dict with 'edges' list of {cause, effect} dicts

    Returns:
        nx.DiGraph with one node per referenced construct
    """
    G = nx.DiGraph()
    for edge in latent_model.get("edges", []):
        G.add_edge(edge["cause"], edge["effect"])
    return G


def get_outcome_from_latent_model(latent_model: dict) -> str | None:
    """Get the outcome variable from a latent model.

    Args:
        latent_model: Dict with 'constructs' list

    Returns:
        Name of the outcome construct, or None if not found
    """
    for construct in latent_model.get("constructs", []):
        if construct.get("is_outcome", False):
            return construct["name"]
    return None


def get_all_treatments(latent_model: dict) -> list[str]:
    """Get all potential treatments from latent model.

    A treatment is any construct that has a causal path to the outcome.
    We want to rank ALL of these by their effect size on the outcome.

    Args:
        latent_model: Dict with 'constructs' and 'edges'

    Returns:
        Sorted list of treatment construct names
    """
    outcome = get_outcome_from_latent_model(latent_model)
    if not outcome:
        return []

    G = build_digraph(latent_model)

    # Find all nodes with causal paths to outcome
    treatments = [node for node in G.nodes() if node != outcome and nx.has_path(G, node, outcome)]

    return sorted(treatments)
