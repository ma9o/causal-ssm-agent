"""Identifiability checking using y0's ID algorithm.

Uses Pearl's do-calculus via y0 to check if causal effects are identifiable
given observed/unobserved constructs. This properly handles:
- Backdoor criterion
- Front-door criterion
- Instrumental variables (under linearity assumption)
- Other identification strategies from Shpitser & Pearl

Design principle: Users specify DAGs with explicit latent confounders. We convert
to ADMG internally using y0's from_latent_variable_dag() for identification.

Note on IV: y0's nonparametric do-calculus cannot identify effects via IV alone.
However, our framework uses linear SEMs where IV identification is valid. We detect IV
structures separately and mark them as identifiable (with linearity assumption).
"""

import logging
from typing import Any

import networkx as nx
from y0.algorithm.identify import identify_outcomes
from y0.dsl import Variable
from y0.graph import NxMixedGraph

from causal_ssm_agent.utils.effects import (
    build_digraph,
    get_all_treatments,
    get_outcome_from_latent_model,
)

logger = logging.getLogger(__name__)


def check_identifiability(
    latent_model: dict,
    measurement_model: dict,
) -> dict[str, Any]:
    """Check which treatment effects are identifiable using y0's ID algorithm.

    Uses 2-timestep unrolling (per A3a and arXiv:2504.20172) to correctly
    handle lagged confounding. Checks identifiability of X_t → Y_t for each
    potential treatment X.

    Args:
        latent_model: Dict with 'constructs' and 'edges'
        measurement_model: Dict with 'indicators' mapping constructs to measures

    Returns:
        Dict with:
            - identifiable_treatments: Map of treatment -> identification details
                * method: 'do_calculus' or 'instrumental_variable'
                * estimand: Closed-form estimand or IV placeholder
                * marginalized_confounders: Unobserved constructs the estimand integrates out
                * instruments: Optional list of IVs when applicable
            - non_identifiable_treatments: Map of treatment -> confounder context
                * confounders: Unobserved constructs blocking identification
                * notes: Optional explanation when confounders cannot be enumerated
            - graph_info: Debug info about the graph structure
    """
    outcome = get_outcome_from_latent_model(latent_model)
    if not outcome:
        raise ValueError("No outcome found in latent model (missing is_outcome=true)")

    # Determine which constructs have measurements (observed)
    observed_constructs = get_observed_constructs(measurement_model)

    # Get all potential treatments (observed constructs with paths to outcome)
    # Only observed constructs can be treatments - you can't do(X) on unobserved X
    all_treatments = [t for t in get_all_treatments(latent_model) if t in observed_constructs]

    # Determine if outcome is time-varying or time-invariant
    outcome_is_time_varying = _is_time_varying(latent_model, outcome)

    # Check each treatment
    identifiable_treatments: dict[str, dict[str, Any]] = {}
    non_identifiable_treatments: dict[str, dict[str, Any]] = {}

    # If outcome itself is unobserved, no effects are identifiable
    if outcome not in observed_constructs:
        for treatment in all_treatments:
            non_identifiable_treatments[treatment] = {
                "confounders": [outcome],
                "notes": "outcome is unobserved",
            }
        return {
            "identifiable_treatments": identifiable_treatments,
            "non_identifiable_treatments": non_identifiable_treatments,
            "graph_info": {
                "observed_constructs": list(observed_constructs),
                "total_constructs": len(latent_model["constructs"]),
                "unobserved_confounders": [],
                "n_directed_edges": 0,
            },
        }

    # Convert DAG to ADMG via 2-timestep unrolling
    admg, unobserved_confounders = dag_to_admg(latent_model, observed_constructs)

    for treatment in all_treatments:
        # Build timestamped variable names for y0 query
        treatment_is_time_varying = _is_time_varying(latent_model, treatment)

        if treatment_is_time_varying:
            treatment_node = _node_name(treatment, "t")
        else:
            treatment_node = treatment

        if outcome_is_time_varying:
            outcome_node = _node_name(outcome, "t")
        else:
            outcome_node = outcome

        treatment_var = Variable(treatment_node)
        outcome_var = Variable(outcome_node)

        try:
            estimand = identify_outcomes(
                admg,
                treatments={treatment_var},
                outcomes={outcome_var},
            )

            if estimand is not None:
                # Map estimand back to original names for readability
                estimand_str = str(estimand)
                identifiable_treatments[treatment] = {
                    "method": "do_calculus",
                    "estimand": estimand_str,
                    "marginalized_confounders": sorted(unobserved_confounders),
                }
            else:
                # y0's nonparametric check failed - try IV identification
                # IV works under linearity (which our framework assumes)
                instruments = find_instruments(
                    latent_model, observed_constructs, treatment, outcome
                )
                if instruments:
                    # IV identification available under linearity
                    iv_list = ", ".join(instruments)
                    identifiable_treatments[treatment] = {
                        "method": "instrumental_variable",
                        "estimand": f"IV({iv_list}) [requires linearity]",
                        "marginalized_confounders": sorted(unobserved_confounders),
                        "instruments": instruments,
                    }
                else:
                    blockers = find_blocking_confounders(
                        latent_model, observed_constructs, treatment, outcome
                    )
                    non_identifiable_treatments[treatment] = {
                        "confounders": blockers,
                    }
        except (ValueError, KeyError, nx.NetworkXError) as e:
            logger.warning("Identifiability check for treatment '%s' failed: %s", treatment, e)
            non_identifiable_treatments[treatment] = {
                "confounders": ["unknown (graph error)"],
                "notes": f"graph projection failed: {e}",
            }

    return {
        "identifiable_treatments": identifiable_treatments,
        "non_identifiable_treatments": non_identifiable_treatments,
        "graph_info": {
            "observed_constructs": list(observed_constructs),
            "total_constructs": len(latent_model["constructs"]),
            "unobserved_confounders": list(unobserved_confounders),
            "n_directed_edges": len(list(admg.directed.edges())),
        },
    }


def _is_time_varying(latent_model: dict, construct_name: str) -> bool:
    """Check if a construct is time-varying (vs time-invariant)."""
    for construct in latent_model["constructs"]:
        if construct["name"] == construct_name:
            return construct.get("temporal_status", "time_varying") != "time_invariant"
    return True  # Default to time-varying if not found


def get_observed_constructs(measurement_model: dict) -> set[str]:
    """Get set of constructs that have at least one measurement indicator."""
    observed = set()
    for indicator in measurement_model.get("indicators", []):
        construct = indicator.get("construct_name")
        if not construct:
            continue
        observed.add(construct)
    return observed


def _node_name(construct: str, timestep: str) -> str:
    """Create timestamped node name like 'X_t' or 'X_{t-1}'."""
    return f"{construct}_{timestep}"


def unroll_temporal_dag(
    latent_model: dict,
    observed_constructs: set[str],
) -> nx.DiGraph:
    """Unroll a temporal causal graph to a 2-timestep DAG for identification.

    Under AR(1) (A3) and bounded latent reach (A3a), a 2-timestep unrolling
    suffices to decide identifiability (per arXiv:2504.20172).

    Node creation:
    - Time-varying constructs → C_t, C_{t-1}
    - Time-invariant constructs → C (single node, no timestep suffix)

    Edge creation:
    - Contemporaneous edges (lagged=False): cause_t → effect_t
    - Lagged edges (lagged=True): cause_{t-1} → effect_t
    - AR(1) for endogenous time-varying: C_{t-1} → C_t
    - Time-invariant to time-varying: C → effect_t (for each timestep)

    Hidden labels:
    - Observed constructs: all timesteps have hidden=False
    - Unobserved constructs: all timesteps have hidden=True

    Args:
        latent_model: Dict with 'constructs' and 'edges'
        observed_constructs: Set of construct names that have measurements

    Returns:
        nx.DiGraph with timestamped nodes and hidden labels for y0
    """
    dag = nx.DiGraph()

    # Categorize constructs by temporal status
    time_varying = set()
    time_invariant = set()

    for construct in latent_model["constructs"]:
        name = construct["name"]
        temporal_status = construct.get("temporal_status", "time_varying")

        if temporal_status == "time_invariant":
            time_invariant.add(name)
        else:
            time_varying.add(name)

    # Add nodes for time-varying constructs (both timesteps)
    for name in time_varying:
        is_hidden = name not in observed_constructs
        dag.add_node(_node_name(name, "t"), hidden=is_hidden, construct=name, timestep="t")
        dag.add_node(_node_name(name, "{t-1}"), hidden=is_hidden, construct=name, timestep="{t-1}")

    # Add nodes for time-invariant constructs (single node)
    for name in time_invariant:
        is_hidden = name not in observed_constructs
        dag.add_node(name, hidden=is_hidden, construct=name, timestep=None)

    # Add AR(1) edges for OBSERVED time-varying constructs only
    # For identification purposes, AR(1) on hidden nodes doesn't add confounding
    # information - it just models U's internal dynamics. What matters is the
    # confounding edges from U to observed nodes.
    #
    # Including AR(1) for hidden nodes causes y0's projection to incorrectly
    # include hidden nodes in the ADMG (because hidden U_{t-1} would have hidden
    # U_t as a successor, and y0 adds edges between ALL pairs of successors).
    for name in time_varying:
        if name in observed_constructs:
            dag.add_edge(_node_name(name, "{t-1}"), _node_name(name, "t"))

    # Add edges from the latent model
    for edge in latent_model.get("edges", []):
        cause = edge["cause"]
        effect = edge["effect"]
        lagged = edge.get("lagged", False)

        cause_is_time_invariant = cause in time_invariant
        effect_is_time_invariant = effect in time_invariant

        if cause_is_time_invariant and effect_is_time_invariant:
            # Both time-invariant: single edge
            dag.add_edge(cause, effect)
        elif cause_is_time_invariant:
            # Time-invariant cause affects time-varying effect at current time
            # (Time-invariant constructs represent stable traits that affect all timepoints)
            dag.add_edge(cause, _node_name(effect, "t"))
            # Also affects t-1 if we're modeling the full 2-timestep window
            dag.add_edge(cause, _node_name(effect, "{t-1}"))
        elif effect_is_time_invariant:
            # Time-varying cause cannot affect time-invariant effect
            # (This would violate the definition of time-invariant)
            # Skip this edge - should be caught by schema validation
            continue
        elif lagged:
            # Lagged edge: cause_{t-1} → effect_t
            dag.add_edge(_node_name(cause, "{t-1}"), _node_name(effect, "t"))
        else:
            # Contemporaneous edge: cause_t → effect_t
            dag.add_edge(_node_name(cause, "t"), _node_name(effect, "t"))
            # Mirror the relationship in the earlier timestep to avoid clamping
            dag.add_edge(_node_name(cause, "{t-1}"), _node_name(effect, "{t-1}"))

    return dag


def _validate_max_lag_one(latent_model: dict) -> None:
    """Validate that all edges have lag ≤ 1 (assumption A3a).

    Under assumption A3a (latent confounders have bounded temporal reach),
    we require all edges to have lag ≤ 1. This allows identification to be
    decided using a 2-timestep graph segment (per arXiv:2504.20172).

    The schema enforces this via `lagged: bool`, but we assert here to make
    the assumption explicit and catch any violations.

    Raises:
        AssertionError: If any edge has a lag value other than 0 or 1
    """
    for edge in latent_model.get("edges", []):
        lagged = edge.get("lagged", False)
        assert isinstance(lagged, bool), (
            f"Edge {edge.get('cause')} -> {edge.get('effect')} has non-boolean 'lagged' value: {lagged}. "
            f"Assumption A3a requires all edges to have lag ≤ 1 (lagged: true/false). "
            f"See arXiv:2504.20172 for why this is required for finite identification."
        )


def dag_to_admg(
    latent_model: dict,
    observed_constructs: set[str],
) -> tuple[NxMixedGraph, set[str]]:
    """Convert a temporal DAG to ADMG via 2-timestep unrolling.

    Uses time-unrolling (per arXiv:2504.20172) to correctly handle lagged
    confounding, then projects to ADMG using y0's from_latent_variable_dag().

    Args:
        latent_model: Dict with 'constructs' and 'edges'
        observed_constructs: Set of construct names that have measurements

    Returns:
        Tuple of (NxMixedGraph, set of unobserved confounder names)

    Raises:
        AssertionError: If any edge violates assumption A3a (lag > 1)
    """
    # Validate assumption A3a: all edges have lag ≤ 1
    _validate_max_lag_one(latent_model)

    # Build 2-timestep unrolled DAG
    dag = unroll_temporal_dag(latent_model, observed_constructs)

    # Find unobserved constructs that will create confounding
    # An unobserved node with 2+ observed children creates bidirected edges
    all_constructs = {c["name"] for c in latent_model["constructs"]}
    unobserved = all_constructs - observed_constructs

    unobserved_confounders = set()
    for node in dag.nodes():
        if not dag.nodes[node].get("hidden", False):
            continue

        # Get the original construct name
        construct = dag.nodes[node].get("construct", node)
        if construct not in unobserved:
            continue

        # Count observed children (children with hidden=False)
        observed_children = [
            child for child in dag.successors(node) if not dag.nodes[child].get("hidden", False)
        ]
        if len(observed_children) >= 2:
            unobserved_confounders.add(construct)

    # Convert to ADMG using y0's built-in projection
    admg = NxMixedGraph.from_latent_variable_dag(dag)

    return admg, unobserved_confounders


def find_blocking_confounders(
    latent_model: dict,
    observed_constructs: set[str],
    treatment: str,
    outcome: str,
) -> list[str]:
    """Find unobserved constructs that confound the treatment-outcome relationship.

    A confounder U creates a backdoor path from treatment to outcome. For U to
    be a *blocking* confounder (one that needs a proxy):
    - U must be an ancestor of treatment
    - U must reach outcome through a path that does NOT go through treatment
    - U must have at least one direct child that is observed

    The last condition ensures U is a "proximal" confounder. If U only affects
    observed nodes through other unobserved nodes (U1 → U2 → X), then U2 is the
    proximal confounder that needs a proxy, not U1. Observing U1 wouldn't help
    if U2 remains unobserved.

    Note: This may over-report if identification strategies like front-door
    handle some confounding. The actual identification decision is made by
    y0's identify_outcomes() algorithm.
    """
    G = build_digraph(latent_model)

    all_constructs = {c["name"] for c in latent_model["constructs"]}
    unobserved = all_constructs - observed_constructs

    # Create graph with treatment removed to check backdoor paths
    G_sans_treatment = G.copy()
    if treatment in G_sans_treatment:
        G_sans_treatment.remove_node(treatment)

    blocking = []
    for u in unobserved:
        if u not in G:
            continue
        if u in (treatment, outcome):
            continue

        # Check if U has any direct observed children
        # If not, U's confounding effect is mediated through other unobserved nodes
        direct_children = list(G.successors(u))
        has_observed_child = any(c in observed_constructs for c in direct_children)
        if not has_observed_child:
            continue

        # U is a blocking confounder if:
        # 1. U is an ancestor of treatment
        # 2. U reaches outcome WITHOUT going through treatment (backdoor path)
        is_ancestor_of_treatment = nx.has_path(G, u, treatment)
        reaches_outcome_via_backdoor = (
            u in G_sans_treatment
            and outcome in G_sans_treatment
            and nx.has_path(G_sans_treatment, u, outcome)
        )

        if is_ancestor_of_treatment and reaches_outcome_via_backdoor:
            blocking.append(u)

    return blocking


def find_instruments(
    latent_model: dict,
    observed_constructs: set[str],
    treatment: str,
    outcome: str,
) -> list[str]:
    """Find valid instrumental variables for the treatment-outcome relationship.

    Based on DoWhy's graph-theoretic approach (py-why/dowhy), adapted to handle
    explicit unobserved confounders. A valid instrument Z for X → Y requires:

    1. Relevance: Z is a direct parent of X (Z → X edge exists)
    2. Exclusion: Z is not an ancestor of Y when X's incoming edges are removed
       (Z affects Y only through X)
    3. As-if-random (Exogeneity): Z is not a descendant of any node that causes Y
       (Z is not affected by confounders of the X-Y relationship)

    This enables IV identification under linear SEM assumptions, even when
    y0's nonparametric do-calculus says the effect is not identifiable.

    Reference: https://github.com/py-why/dowhy/blob/main/dowhy/graph.py

    Args:
        latent_model: Dict with 'constructs' and 'edges'
        observed_constructs: Set of observed construct names
        treatment: The treatment variable name
        outcome: The outcome variable name

    Returns:
        List of valid instrument names (observed constructs that satisfy IV conditions)
    """
    G = build_digraph(latent_model)

    if treatment not in G or outcome not in G:
        return []

    # Get direct parents of treatment (potential instruments must be parents)
    parents_treatment = set(G.predecessors(treatment))

    # Do surgery: remove incoming edges to treatment
    G_surgered = G.copy()
    incoming_to_treatment = list(G_surgered.in_edges(treatment))
    G_surgered.remove_edges_from(incoming_to_treatment)

    # Get ancestors of outcome in the surgered graph
    ancestors_outcome = nx.ancestors(G_surgered, outcome) if outcome in G_surgered else set()

    # Condition 1 & 2 (Relevance + Exclusion):
    # Instruments must be parents of treatment AND not ancestors of outcome
    candidate_instruments = parents_treatment - ancestors_outcome

    # Condition 3 (As-if-random/Exogeneity):
    # Instruments must not be descendants of any ancestor of outcome
    # This ensures Z is not affected by confounders
    descendants_of_ancestors = set()
    for ancestor in ancestors_outcome:
        descendants_of_ancestors.update(nx.descendants(G_surgered, ancestor))

    valid_instruments = candidate_instruments - descendants_of_ancestors

    # Filter to only observed instruments
    observed_instruments = [z for z in valid_instruments if z in observed_constructs]

    return observed_instruments


def analyze_unobserved_constructs(
    latent_model: dict,
    measurement_model: dict,
    identifiability_result: dict,
) -> dict[str, Any]:
    """Analyze which unobserved constructs can be marginalized in model specification.

    When y0 identifies an effect despite unobserved confounding, it means the
    identification strategy (front-door, IV, etc.) handles that confounding without
    needing explicit modeling of the confounder. Such confounders can be "marginalized"
    - their effects get absorbed into error terms.

    Confounders that BLOCK identification cannot be marginalized - they need proxies
    or must be explicitly modeled as latent variables.

    Note: This analysis only considers whether U blocks OBSERVED treatments. An
    unobserved treatment U is inherently non-identifiable (we can't intervene on it),
    but that's a separate concern from whether U creates confounding for other effects.

    Args:
        latent_model: Dict with 'constructs' and 'edges'
        measurement_model: Dict with 'indicators'
        identifiability_result: Output from check_identifiability()

    Returns:
        Dict with:
            - can_marginalize: Set of unobserved constructs safe to ignore in model spec
            - marginalize_reason: Dict explaining why each can be marginalized
            - blocking_details: Map of blocking confounder -> treatments they obstruct
    """
    observed = get_observed_constructs(measurement_model)
    all_constructs = {c["name"] for c in latent_model["constructs"]}
    unobserved = all_constructs - observed

    # Collect confounders that block identification of OBSERVED treatments
    # (Ignore unobserved treatments - they're inherently non-identifiable)
    blocking_any = set()
    blocking_details: dict[str, list[str]] = {}  # confounder -> list of treatments it blocks

    for treatment, info in identifiability_result.get("non_identifiable_treatments", {}).items():
        # Skip if the treatment itself is unobserved (it's blocking itself)
        if treatment in unobserved:
            continue

        blockers = info.get("confounders", []) if isinstance(info, dict) else []
        for blocker in blockers:
            if blocker in unobserved and blocker != treatment:
                blocking_any.add(blocker)
                blocking_details.setdefault(blocker, []).append(treatment)

    # Sort treatment lists for deterministic output
    for blocker in blocking_details:
        blocking_details[blocker] = sorted(blocking_details[blocker])

    # Classify unobserved constructs
    can_marginalize = unobserved - blocking_any

    # Build explanations
    marginalize_reason = {}
    for u in can_marginalize:
        # Check if U creates any confounding at all (is in unobserved_confounders)
        if u in identifiability_result["graph_info"].get("unobserved_confounders", []):
            marginalize_reason[u] = (
                "confounding handled by identification strategy (front-door or similar)"
            )
        else:
            marginalize_reason[u] = (
                "does not create confounding (single child or no observed children)"
            )

    return {
        "can_marginalize": can_marginalize,
        "marginalize_reason": marginalize_reason,
        "blocking_details": blocking_details,
    }


def get_correlation_pairs_from_marginalization(
    latent_model: dict,
    measurement_model: dict,
    identifiability_result: dict,
) -> list[tuple[str, str, str]]:
    """Return (state1, state2, confounder) pairs needing correlation parameters.

    When an unobserved confounder with 2+ observed children is marginalized,
    its children's innovations become correlated. This function identifies those
    pairs so the model spec can add correlation parameters.

    Only confounders that are actual confounders (listed in
    graph_info["unobserved_confounders"], meaning 2+ observed children) AND
    are in can_marginalize get pairs. Single-child unobserved constructs don't
    create confounding and need no correlation.

    Args:
        latent_model: Dict with 'constructs' and 'edges'
        measurement_model: Dict with 'indicators'
        identifiability_result: Output from check_identifiability()

    Returns:
        List of (state1, state2, confounder_name) tuples, sorted for determinism.
        state1 < state2 lexicographically within each tuple.
    """
    from itertools import combinations

    analysis = analyze_unobserved_constructs(
        latent_model, measurement_model, identifiability_result
    )
    can_marginalize = analysis["can_marginalize"]
    # Only confounders with 2+ observed children (listed in unobserved_confounders)
    actual_confounders = set(
        identifiability_result.get("graph_info", {}).get("unobserved_confounders", [])
    )
    marginalizable_confounders = can_marginalize & actual_confounders

    if not marginalizable_confounders:
        return []

    # Build confounder -> observed children mapping from edges
    observed = get_observed_constructs(measurement_model)
    pairs: list[tuple[str, str, str]] = []

    for confounder in sorted(marginalizable_confounders):
        children = sorted(
            {
                edge["effect"]
                for edge in latent_model.get("edges", [])
                if edge["cause"] == confounder and edge["effect"] in observed
            }
        )
        for s1, s2 in combinations(children, 2):
            pairs.append((s1, s2, confounder))

    return pairs


def format_identifiability_report(result: dict, outcome: str) -> str:
    """Format identifiability check results for logging."""
    lines = []

    identifiable = result.get("identifiable_treatments", {})
    non_identifiable = result.get("non_identifiable_treatments", {})
    n_identifiable = len(identifiable)
    n_non_identifiable = len(non_identifiable)
    total = n_identifiable + n_non_identifiable

    if not non_identifiable:
        lines.append(f"✓ All {total} treatment effects on {outcome} are identifiable!")
    else:
        lines.append(
            f"✗ {n_non_identifiable}/{total} treatments have non-identifiable effects on {outcome}:"
        )
        for treatment in sorted(non_identifiable.keys()):
            details = non_identifiable[treatment]
            blockers = details.get("confounders", []) if isinstance(details, dict) else []
            notes = details.get("notes") if isinstance(details, dict) else None
            if blockers:
                lines.append(f"  - {treatment} (blocked by: {', '.join(blockers)})")
            elif notes:
                lines.append(f"  - {treatment} ({notes})")
            else:
                lines.append(f"  - {treatment} (structural non-identifiability)")

    if identifiable:
        lines.append(f"\n✓ {n_identifiable} treatments have identifiable effects:")
        for treatment in sorted(identifiable.keys())[:5]:
            method = identifiable[treatment].get("method", "unknown")
            lines.append(f"  - {treatment} via {method}")
        if n_identifiable > 5:
            lines.append(f"  ... and {n_identifiable - 5} more")

    info = result["graph_info"]
    lines.append(
        f"\nGraph: {len(info['observed_constructs'])}/{info['total_constructs']} constructs observed, "
        f"{info['n_directed_edges']} directed edges"
    )
    if info["unobserved_confounders"]:
        lines.append(f"Unobserved confounders: {', '.join(info['unobserved_confounders'])}")

    return "\n".join(lines)


def format_marginalization_report(analysis: dict) -> str:
    """Format the marginalization analysis for logging.

    Args:
        analysis: Output from analyze_unobserved_constructs()

    Returns:
        Formatted string report
    """
    lines = []

    can_marginalize = analysis["can_marginalize"]
    blocking_details = analysis.get("blocking_details", {})
    needs_modeling = set(blocking_details.keys())

    lines.append("=" * 60)
    lines.append("UNOBSERVED CONSTRUCT ANALYSIS FOR MODEL SPECIFICATION")
    lines.append("=" * 60)

    if can_marginalize:
        lines.append(f"\n✓ CAN MARGINALIZE ({len(can_marginalize)} constructs):")
        lines.append("  These can be omitted from model spec - effects absorbed into error terms")
        for u in sorted(can_marginalize):
            reason = analysis["marginalize_reason"].get(u, "")
            lines.append(f"  - {u}: {reason}")

    if needs_modeling:
        lines.append(f"\n✗ NEEDS MODELING ({len(needs_modeling)} constructs):")
        lines.append("  These block identification - need proxies or explicit latent variables")
        for u in sorted(needs_modeling):
            treatments = ", ".join(blocking_details.get(u, []))
            reason = f"blocks identification of: {treatments}" if treatments else ""
            lines.append(f"  - {u}: {reason}")

    if not can_marginalize and not needs_modeling:
        lines.append("\n✓ All constructs are observed - no marginalization analysis needed")

    return "\n".join(lines)


def inject_marginalized_correlations(model_spec: dict, causal_spec: dict) -> None:
    """Add cor_* parameters to model_spec for marginalized confounders.

    When an unobserved confounder is marginalized, its observed children have
    correlated innovations in the SSM. This adds correlation parameters so the
    noise covariance is correctly specified.

    Mutates model_spec["parameters"] in place.

    Args:
        model_spec: Model specification dict (must have "parameters" key)
        causal_spec: CausalSpec dict with "latent", "measurement", "identifiability"
    """
    id_status = causal_spec.get("identifiability", {})
    if not id_status:
        return

    cor_pairs = get_correlation_pairs_from_marginalization(
        causal_spec.get("latent", {}),
        causal_spec.get("measurement", {}),
        id_status,
    )

    parameter_specs = model_spec.get("parameters", [])
    existing_names = {p.get("name") for p in parameter_specs}

    for s1, s2, confounder in cor_pairs:
        name = f"cor_{s1}_{s2}"
        if name not in existing_names:
            parameter_specs.append(
                {
                    "name": name,
                    "role": "correlation",
                    "constraint": "correlation",
                    "description": (
                        f"Residual correlation between {s1} and {s2} "
                        f"(marginalized confounder: {confounder})"
                    ),
                    "search_context": "",
                }
            )
            existing_names.add(name)

    model_spec["parameters"] = parameter_specs
