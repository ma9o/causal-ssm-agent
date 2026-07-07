"""Accessor helpers for CausalDesign dicts.

The contract now has two structural layers:
- ``latent``: user-facing theoretical DAG
- ``estimation``: retained executable state-space projection

Use the explicit latent/estimation accessors in new code. The historical
``get_constructs()`` / ``get_edges()`` helpers still refer to the latent DAG.
"""

from collections import defaultdict
from typing import Any, cast

import networkx as nx

from nof1_causal_lab.utils.observation_semantics import (
    get_observation_semantics,
)


def get_constructs(causal_design: dict) -> list[dict]:
    """Get latent constructs from a CausalDesign dict."""
    return causal_design.get("latent", {}).get("constructs", [])


def get_indicators(causal_design: dict) -> list[dict]:
    """Get indicators from a CausalDesign dict."""
    return causal_design.get("measurement", {}).get("indicators", [])


def get_indicator_polarity(indicator: dict) -> str:
    """Return the declared indicator polarity, failing loudly when absent."""
    polarity = indicator.get("construct_polarity")
    if polarity not in {"positive", "negative"}:
        raise ValueError(
            f"Indicator {indicator.get('name')!r} is missing a valid construct_polarity"
        )
    return str(polarity)


def choose_reference_indicator(indicators: list[dict]) -> dict | None:
    """Choose a deterministic marker indicator for one construct.

    Prefer the first positive-polarity indicator so the latent orientation matches
    the construct name whenever the measurement structure provides one. If none are
    positive, fall back to the first declared indicator.
    """
    if not indicators:
        return None

    for indicator in indicators:
        if get_indicator_polarity(indicator) == "positive":
            return indicator
    return indicators[0]


def build_reference_indicator_lookup(indicators: list[dict]) -> dict[str, str]:
    """Return construct -> chosen reference indicator name."""
    grouped: dict[str, list[dict]] = {}
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


def get_estimation_spec(causal_design: dict) -> dict:
    """Get the estimation projection, failing loudly when it is missing."""
    estimation = causal_design.get("estimation")
    if not isinstance(estimation, dict):
        raise ValueError("causal_design.estimation is required for estimation-sensitive access")
    return estimation


def get_estimation_state_order(causal_design: dict) -> list[str]:
    """Get retained latent states in canonical estimation order."""
    return list(get_estimation_spec(causal_design).get("state_order") or [])


def get_estimation_edges(causal_design: dict) -> list[dict]:
    """Get retained directed edges in the estimation projection."""
    return list(get_estimation_spec(causal_design).get("edges") or [])


def get_known_inputs(causal_design: dict) -> list[dict]:
    """Get known input declarations from the estimation projection."""
    return list(get_estimation_spec(causal_design).get("known_inputs") or [])


def get_known_input_source_indicators(causal_design: dict) -> set[str]:
    """Return indicator names consumed as deterministic transition inputs."""
    return {
        str(known_input["source_indicator"])
        for known_input in get_known_inputs(causal_design)
        if known_input.get("source_indicator")
    }


def get_manifest_indicators(causal_design: dict) -> list[dict]:
    """Get indicators that remain in the manifest likelihood."""
    input_sources = get_known_input_source_indicators(causal_design)
    return [
        indicator
        for indicator in get_indicators(causal_design)
        if indicator.get("name") not in input_sources
    ]


def get_estimation_constructs(causal_design: dict) -> list[dict]:
    """Get retained latent construct payloads in estimation-state order."""
    latent_lookup = {
        construct["name"]: construct
        for construct in get_constructs(causal_design)
        if construct.get("name")
    }
    return [
        latent_lookup[name]
        for name in get_estimation_state_order(causal_design)
        if name in latent_lookup
    ]


def get_induced_dependencies(causal_design: dict) -> list[dict[str, Any]]:
    """Get induced dependencies created by marginalizing latent roots."""
    return list(get_estimation_spec(causal_design).get("induced_dependencies") or [])


def _canonical_scale_name(sources: list[str]) -> str:
    """Canonical identifier for a confounder equivalence class."""
    return "tau_" + "__".join(sorted(sources))


def get_marginalized_scales(causal_design: dict) -> list[dict]:
    """Return the scale-indexed view of marginalized confounders.

    Confounders sharing the same *footprint* — the union of retained state
    pairs they induce a dependency between — form one identifiable equivalence
    class. The likelihood depends only on the class's sum-of-squares
    contribution, so each class corresponds to exactly one static-factor scale
    parameter, regardless of how many source confounders it aggregates.

    Each returned entry is:
        - ``parameter``: canonical name ``tau_<sorted_sources_joined>``.
        - ``kind``: ``initial_state_correlation`` | ``innovation_correlation``.
        - ``sources``: the source confounders aggregated into this scale.
        - ``affected_states``: states in the scale's loading column.
        - ``directions``: state pairs this scale contributes to.

    Raises:
        ValueError: a confounder's ``kind`` is inconsistent across the deps it
        appears in (structurally impossible under the projection contract; if
        it happens, the estimation spec is malformed).
    """
    deps = get_induced_dependencies(causal_design)

    footprint_by_confounder: dict[str, set[str]] = defaultdict(set)
    kind_by_confounder: dict[str, str] = {}
    directions_by_confounder: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for dep in deps:
        # Induced dependencies always carry a non-None ``kind`` string by the
        # estimation-projection contract (see estimation_projection.py).
        kind = cast("str", dep.get("kind"))
        between = tuple(dep.get("between") or ())
        if len(between) != 2:
            continue
        for confounder in dep.get("source_confounders") or ():
            footprint_by_confounder[confounder].update(between)
            directions_by_confounder[confounder].append(between)
            prior_kind = kind_by_confounder.setdefault(confounder, kind)
            if prior_kind != kind:
                raise ValueError(
                    f"Confounder {confounder!r} participates in dependencies with "
                    f"inconsistent kinds ({prior_kind!r} and {kind!r}); a confounder "
                    "must project to exactly one covariance block."
                )

    members_by_footprint: dict[tuple[str, frozenset[str]], list[str]] = defaultdict(list)
    for confounder, footprint in footprint_by_confounder.items():
        key = (kind_by_confounder[confounder], frozenset(footprint))
        members_by_footprint[key].append(confounder)

    scales: list[dict] = []
    for (kind, footprint), members in members_by_footprint.items():
        sources = sorted(members)
        directions: set[tuple[str, str]] = set()
        for confounder in sources:
            directions.update(directions_by_confounder[confounder])
        scales.append(
            {
                "parameter": _canonical_scale_name(sources),
                "kind": kind,
                "sources": sources,
                "affected_states": sorted(footprint),
                "directions": sorted(directions),
            }
        )
    scales.sort(key=lambda scale: scale["parameter"])
    return scales


def get_effective_observation_window(indicator: dict, model_clock: str | None) -> str | None:
    """Return the effective support window for an indicator."""
    return indicator.get("observation_window") or model_clock


def get_indicator_info(causal_design: dict) -> dict[str, dict]:
    """Extract indicator info from a CausalDesign dict.

    Returns:
        Dict mapping indicator name to semantic extraction/measurement metadata.
    """
    result: dict[str, dict] = {}
    for ind in get_indicators(causal_design):
        sem = get_observation_semantics(ind)
        result[ind["name"]] = {
            "dtype": ind.get("measurement_dtype"),
            "construct_name": ind.get("construct_name"),
            "ordinal_levels": ind.get("ordinal_levels"),
            "support_kind": sem.support_kind.value,
            "summary_operator": sem.summary_operator.value,
            "anchor_policy": sem.anchor_policy.value,
            "observation_window": ind.get("observation_window"),
        }
    return result


def get_indicator_dtypes(causal_design: dict) -> dict[str, str]:
    """Extract indicator name -> measurement_dtype mapping.

    Returns:
        Dict mapping indicator name to dtype string (e.g. "continuous", "binary")
    """
    return {
        ind["name"]: ind.get("measurement_dtype", "continuous")
        for ind in get_indicators(causal_design)
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


def make_extraction_context(causal_design: dict) -> dict:
    """Build minimal context needed by Stage 2 extraction workers.

    Workers need:
    - indicators: name, measurement_dtype, how_to_measure, source_columns,
      aggregation, support_kind, summary_operator, anchor_policy, observation_window
    - outcome: name, description (for prompt context)

    Does not include: construct_name, latent edges, or non-outcome constructs.
    Includes ordinal_levels only for ordinal indicators so workers can use a
    stable numeric codebook.
    """
    model_clock = causal_design.get("measurement", {}).get("model_clock")
    slim_indicators = []
    for ind in get_indicators(causal_design):
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
    outcome = get_outcome_construct(causal_design)
    slim_outcome = (
        {"name": outcome["name"], "description": outcome.get("description", "")}
        if outcome
        else None
    )
    return {
        "measurement": {"indicators": slim_indicators},
        "latent": {"constructs": [slim_outcome] if slim_outcome else []},
    }


def get_outcome_construct(causal_design_or_latent: dict) -> dict | None:
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


def get_outcome_name(causal_design_or_latent: dict) -> str | None:
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


def build_digraph(latent_structure: dict) -> nx.DiGraph:
    """Build a simple DiGraph from a latent structure's edge list.

    Args:
        latent_structure: Dict with 'edges' list of {cause, effect} dicts

    Returns:
        nx.DiGraph with one node per referenced construct
    """
    return build_digraph_from_edges(latent_structure.get("edges", []))


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


def get_all_treatments(latent_structure: dict) -> list[str]:
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


def get_estimable_treatments(causal_design: dict) -> list[str]:
    """Get intervention targets that remain in the retained estimation graph."""
    state_order = get_estimation_state_order(causal_design)
    known_input_names = [item["construct"] for item in get_known_inputs(causal_design)]
    return _get_treatments_from_graph(
        node_names=[*state_order, *known_input_names],
        edges=get_estimation_edges(causal_design),
        outcome=get_outcome_name(causal_design),
    )
