"""Semantic helpers for scientific CausalDesign and measurement artifacts."""

import networkx as nx

from nof1_causal_lab.json_types import UncheckedJsonObject
from nof1_causal_lab.utils.observation_semantics import (
    get_observation_semantics,
)


def get_constructs(causal_design: UncheckedJsonObject) -> list[UncheckedJsonObject]:
    """Get latent constructs from a CausalDesign dict."""
    return causal_design.get("latent", {}).get("constructs", [])


def get_indicators(causal_design: UncheckedJsonObject) -> list[UncheckedJsonObject]:
    """Get indicators from a CausalDesign dict."""
    return causal_design.get("measurement", {}).get("indicators", [])


def get_indicator_polarity(indicator: UncheckedJsonObject) -> str:
    """Return the declared indicator polarity, failing loudly when absent."""
    polarity = indicator.get("construct_polarity")
    if polarity not in {"positive", "negative"}:
        raise ValueError(
            f"Indicator {indicator.get('name')!r} is missing a valid construct_polarity"
        )
    return str(polarity)


# How strongly a pinned reference loading anchors the latent scale, by dtype:
# continuous channels standardize (data units pin scale and location), ordinal
# channels pin scale through the fixed logistic link, binary/count pin scale
# through their link but carry less information, and a pinned categorical
# loading anchors nothing because the free class slopes absorb it.
_REFERENCE_DTYPE_TIERS = {
    "continuous": 0,
    "ordinal": 1,
    "binary": 2,
    "count": 2,
    "categorical": 3,
}


def choose_reference_indicator(
    indicators: list[UncheckedJsonObject],
) -> UncheckedJsonObject | None:
    """Choose a deterministic marker indicator for one construct.

    Prefer the dtype whose fixed loading anchors the latent scale most strongly
    (see ``_REFERENCE_DTYPE_TIERS``); within a tier prefer positive polarity so
    the latent orientation matches the construct name, then declaration order.
    """
    if not indicators:
        return None

    def _rank(item: tuple[int, UncheckedJsonObject]) -> tuple[int, int, int]:
        declaration_index, indicator = item
        dtype = str(indicator.get("measurement_dtype") or "")
        tier = _REFERENCE_DTYPE_TIERS.get(dtype, 2)
        polarity_rank = 0 if get_indicator_polarity(indicator) == "positive" else 1
        return (tier, polarity_rank, declaration_index)

    return min(enumerate(indicators), key=_rank)[1]


def build_reference_indicator_lookup(indicators: list[UncheckedJsonObject]) -> dict[str, str]:
    """Return construct -> chosen reference indicator name."""
    grouped: dict[str, list[UncheckedJsonObject]] = {}
    for indicator in indicators:
        construct_name = indicator.get("construct_name")
        if isinstance(construct_name, str):
            grouped.setdefault(construct_name, []).append(indicator)

    lookup: dict[str, str] = {}
    for construct_name, construct_indicators in grouped.items():
        reference = choose_reference_indicator(construct_indicators)
        if reference is not None:
            lookup[construct_name] = str(reference["name"])
    return lookup


def get_effective_observation_window(
    indicator: UncheckedJsonObject,
    model_clock: str | None,
) -> str | None:
    """Return the effective support window for an indicator."""
    return indicator.get("observation_window") or model_clock


def get_measurement_indicator_info(
    measurement_structure: UncheckedJsonObject,
) -> dict[str, UncheckedJsonObject]:
    """Extract indicator info from a MeasurementStructure dict."""
    result: dict[str, UncheckedJsonObject] = {}
    model_clock = measurement_structure.get("model_clock")
    for ind in measurement_structure.get("indicators", []):
        sem = get_observation_semantics(ind)
        result[ind["name"]] = {
            "dtype": ind.get("measurement_dtype"),
            "construct_name": ind.get("construct_name"),
            "ordinal_levels": ind.get("ordinal_levels"),
            "support_kind": sem.support_kind.value,
            "summary_operator": sem.summary_operator.value,
            "anchor_policy": sem.anchor_policy.value,
            "observation_window": get_effective_observation_window(ind, model_clock),
        }
    return result


_WORKER_INDICATOR_KEYS = (
    "name",
    "measurement_dtype",
    "how_to_measure",
    "source_columns",
    "aggregation",
    "observation_window",
    "ordinal_levels",
)


def make_measurement_extraction_context(
    measurement_structure: UncheckedJsonObject,
) -> UncheckedJsonObject:
    """Build minimal context needed by extraction extraction workers.

    Workers need:
    - indicators: name, measurement_dtype, how_to_measure, source_columns,
      aggregation, support_kind, summary_operator, anchor_policy, observation_window

    Does not include: construct_name, latent edges, or non-outcome constructs.
    Includes ordinal_levels only for ordinal indicators so workers can use a
    stable numeric codebook.
    """
    model_clock = measurement_structure.get("model_clock")
    slim_indicators = []
    for ind in measurement_structure.get("indicators", []):
        sem = get_observation_semantics(ind)
        entry = {
            **{k: ind[k] for k in _WORKER_INDICATOR_KEYS if k in ind},
            "support_kind": sem.support_kind.value,
            "summary_operator": sem.summary_operator.value,
            "anchor_policy": sem.anchor_policy.value,
        }
        effective_window = get_effective_observation_window(ind, model_clock)
        if effective_window:
            entry["observation_window"] = effective_window
        slim_indicators.append(entry)
    return {
        "model_clock": model_clock,
        "indicators": slim_indicators,
    }


def get_outcome_construct(
    causal_design_or_latent: UncheckedJsonObject,
) -> UncheckedJsonObject | None:
    """Get the outcome construct dict from a CausalDesign or latent structure dict.

    Handles both full CausalDesign dicts and bare latent structure dicts.

    Returns:
        The outcome construct dict, or None if not found
    """
    # Handle both CausalDesign (has "latent" key) and bare latent structure
    if "latent" in causal_design_or_latent:
        constructs = get_constructs(causal_design_or_latent)
    else:
        constructs = causal_design_or_latent.get("constructs", [])

    for c in constructs:
        if c.get("is_outcome"):
            return c
    return None


def get_outcome_name(causal_design_or_latent: UncheckedJsonObject) -> str | None:
    """Get the outcome construct name from a CausalDesign or latent structure dict.

    Convenience wrapper around get_outcome_construct() that returns just the name.

    Args:
        causal_design_or_latent: Either a full CausalDesign dict or a bare latent structure dict.

    Returns:
        Name of the outcome construct, or None if not found.
    """
    outcome = get_outcome_construct(causal_design_or_latent)
    return outcome["name"] if outcome else None


# ---------------------------------------------------------------------------
# Graph utilities (merged from effects.py)
# ---------------------------------------------------------------------------


def build_digraph(latent_structure: UncheckedJsonObject) -> nx.DiGraph:
    """Build a simple DiGraph from a latent structure's edge list.

    Args:
        latent_structure: Dict with 'edges' list of {cause, effect} dicts

    Returns:
        nx.DiGraph with one node per referenced construct
    """
    return build_digraph_from_edges(latent_structure.get("edges", []))


def build_digraph_from_edges(edges: list[UncheckedJsonObject]) -> nx.DiGraph:
    """Build a simple DiGraph from an edge list."""
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge["cause"], edge["effect"])
    return G


def _get_treatments_from_graph(
    *,
    node_names: list[str],
    edges: list[UncheckedJsonObject],
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


def get_all_treatments(latent_structure: UncheckedJsonObject) -> list[str]:
    """Get all potential treatments from latent structure.

    A treatment is any construct that has a causal path to the outcome.

    Args:
        latent_structure: Dict with 'constructs' and 'edges'

    Returns:
        Sorted list of treatment construct names
    """
    return _get_treatments_from_graph(
        node_names=[
            construct["name"]
            for construct in latent_structure.get("constructs", [])
            if construct.get("name")
        ],
        edges=list(latent_structure.get("edges", []) or []),
        outcome=get_outcome_name(latent_structure),
    )
