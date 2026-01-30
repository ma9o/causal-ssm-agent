"""Identifiability checking using y0's ID algorithm.

Uses Pearl's do-calculus via y0 to check if causal effects are identifiable
given observed/unobserved constructs. This properly handles:
- Backdoor criterion
- Front-door criterion
- Other identification strategies from Shpitser & Pearl

Design principle: Users specify DAGs with explicit latent confounders. We convert
to ADMG internally using y0's from_latent_variable_dag() for identification.
"""

from typing import Any

import networkx as nx
from y0.algorithm.identify import identify_outcomes
from y0.dsl import Variable
from y0.graph import NxMixedGraph

from dsem_agent.utils.effects import get_outcome_from_latent_model


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
            - outcome: The outcome construct name
            - identifiable_treatments: Dict mapping treatment name to estimand string
            - non_identifiable_treatments: Set of treatment names that aren't identifiable
            - blocking_confounders: Dict mapping treatment to list of blocking confounders
            - graph_info: Debug info about the graph structure
    """
    outcome = get_outcome_from_latent_model(latent_model)
    if not outcome:
        raise ValueError("No outcome found in latent model (missing is_outcome=true)")

    # Determine which constructs have measurements (observed)
    observed_constructs = get_observed_constructs(measurement_model)

    # Convert DAG to ADMG via 2-timestep unrolling
    admg, unobserved_confounders = dag_to_admg(latent_model, observed_constructs)

    # Get all potential treatments (constructs with paths to outcome)
    all_treatments = _get_treatments_from_graph(latent_model, outcome)

    # Determine if outcome is time-varying or time-invariant
    outcome_is_time_varying = _is_time_varying(latent_model, outcome)

    # Check each treatment
    identifiable_treatments = {}
    non_identifiable_treatments = set()
    blocking_confounders = {}

    for treatment in all_treatments:
        # Skip if treatment or outcome not observed
        if treatment not in observed_constructs:
            non_identifiable_treatments.add(treatment)
            blocking_confounders[treatment] = [treatment]  # Treatment itself is unobserved
            continue

        if outcome not in observed_constructs:
            non_identifiable_treatments.add(treatment)
            blocking_confounders[treatment] = [outcome]
            continue

        # Build timestamped variable names for y0 query
        treatment_is_time_varying = _is_time_varying(latent_model, treatment)

        if treatment_is_time_varying:
            treatment_node = _node_name(treatment, 't')
        else:
            treatment_node = treatment

        if outcome_is_time_varying:
            outcome_node = _node_name(outcome, 't')
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
                identifiable_treatments[treatment] = estimand_str
            else:
                non_identifiable_treatments.add(treatment)
                blockers = find_blocking_confounders(
                    latent_model, observed_constructs, treatment, outcome
                )
                blocking_confounders[treatment] = blockers
        except Exception:
            non_identifiable_treatments.add(treatment)
            blocking_confounders[treatment] = ['unknown (graph error)']

    return {
        'outcome': outcome,
        'identifiable_treatments': identifiable_treatments,
        'non_identifiable_treatments': non_identifiable_treatments,
        'blocking_confounders': blocking_confounders,
        'graph_info': {
            'observed_constructs': list(observed_constructs),
            'total_constructs': len(latent_model['constructs']),
            'unobserved_confounders': list(unobserved_confounders),
            'n_directed_edges': len(list(admg.directed.edges())),
            'n_bidirected_edges': len(list(admg.undirected.edges())),
        },
    }


def _is_time_varying(latent_model: dict, construct_name: str) -> bool:
    """Check if a construct is time-varying (vs time-invariant)."""
    for construct in latent_model['constructs']:
        if construct['name'] == construct_name:
            return construct.get('temporal_status', 'time_varying') != 'time_invariant'
    return True  # Default to time-varying if not found


def _get_treatments_from_graph(latent_model: dict, outcome: str) -> list[str]:
    """Get all constructs with causal paths to outcome."""
    G = nx.DiGraph()
    for edge in latent_model.get('edges', []):
        G.add_edge(edge['cause'], edge['effect'])

    return [
        node for node in G.nodes()
        if node != outcome and nx.has_path(G, node, outcome)
    ]


def get_observed_constructs(measurement_model: dict) -> set[str]:
    """Get set of constructs that have at least one measurement indicator."""
    observed = set()
    for indicator in measurement_model.get('indicators', []):
        observed.add(indicator['construct'])
    return observed


def _node_name(construct: str, timestep: str) -> str:
    """Create timestamped node name like 'X_t' or 'X_{t-1}'."""
    return f"{construct}_{timestep}"


def _parse_node_name(node: str) -> tuple[str, str | None]:
    """Parse node name back to (construct, timestep).

    Returns (construct, timestep) where timestep is 't', '{t-1}', or None for time-invariant.
    """
    if node.endswith('_t'):
        return node[:-2], 't'
    elif node.endswith('_{t-1}'):
        return node[:-6], '{t-1}'
    else:
        return node, None


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
    endogenous_time_varying = set()

    for construct in latent_model['constructs']:
        name = construct['name']
        temporal_status = construct.get('temporal_status', 'time_varying')
        role = construct.get('role', 'endogenous')

        if temporal_status == 'time_invariant':
            time_invariant.add(name)
        else:
            time_varying.add(name)
            if role == 'endogenous':
                endogenous_time_varying.add(name)

    # Add nodes for time-varying constructs (both timesteps)
    for name in time_varying:
        is_hidden = name not in observed_constructs
        dag.add_node(_node_name(name, 't'), hidden=is_hidden, construct=name, timestep='t')
        dag.add_node(_node_name(name, '{t-1}'), hidden=is_hidden, construct=name, timestep='{t-1}')

    # Add nodes for time-invariant constructs (single node)
    for name in time_invariant:
        is_hidden = name not in observed_constructs
        dag.add_node(name, hidden=is_hidden, construct=name, timestep=None)

    # Add AR(1) edges for endogenous time-varying constructs
    for name in endogenous_time_varying:
        dag.add_edge(_node_name(name, '{t-1}'), _node_name(name, 't'))

    # Add edges from the latent model
    for edge in latent_model.get('edges', []):
        cause = edge['cause']
        effect = edge['effect']
        lagged = edge.get('lagged', False)

        cause_is_time_invariant = cause in time_invariant
        effect_is_time_invariant = effect in time_invariant

        if cause_is_time_invariant and effect_is_time_invariant:
            # Both time-invariant: single edge
            dag.add_edge(cause, effect)
        elif cause_is_time_invariant:
            # Time-invariant cause affects time-varying effect at current time
            # (Time-invariant constructs represent stable traits that affect all timepoints)
            dag.add_edge(cause, _node_name(effect, 't'))
            # Also affects t-1 if we're modeling the full 2-timestep window
            dag.add_edge(cause, _node_name(effect, '{t-1}'))
        elif effect_is_time_invariant:
            # Time-varying cause cannot affect time-invariant effect
            # (This would violate the definition of time-invariant)
            # Skip this edge - should be caught by schema validation
            continue
        elif lagged:
            # Lagged edge: cause_{t-1} → effect_t
            dag.add_edge(_node_name(cause, '{t-1}'), _node_name(effect, 't'))
        else:
            # Contemporaneous edge: cause_t → effect_t
            dag.add_edge(_node_name(cause, 't'), _node_name(effect, 't'))

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
    for edge in latent_model.get('edges', []):
        lagged = edge.get('lagged', False)
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
    all_constructs = {c['name'] for c in latent_model['constructs']}
    unobserved = all_constructs - observed_constructs

    unobserved_confounders = set()
    for node in dag.nodes():
        if not dag.nodes[node].get('hidden', False):
            continue

        # Get the original construct name
        construct = dag.nodes[node].get('construct', node)
        if construct not in unobserved:
            continue

        # Count observed children (children with hidden=False)
        observed_children = [
            child for child in dag.successors(node)
            if not dag.nodes[child].get('hidden', False)
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
    """Find unobserved constructs that confound the treatment-outcome relationship."""
    G = nx.DiGraph()
    for edge in latent_model.get('edges', []):
        G.add_edge(edge['cause'], edge['effect'])

    all_constructs = {c['name'] for c in latent_model['constructs']}
    unobserved = all_constructs - observed_constructs

    blocking = []
    for u in unobserved:
        if u not in G:
            continue
        has_path_to_treatment = nx.has_path(G, u, treatment) if u != treatment else False
        has_path_to_outcome = nx.has_path(G, u, outcome) if u != outcome else False
        if has_path_to_treatment and has_path_to_outcome:
            blocking.append(u)

    return blocking


def analyze_unobserved_constructs(
    latent_model: dict,
    measurement_model: dict,
    identifiability_result: dict,
) -> dict[str, Any]:
    """Analyze which unobserved constructs can be marginalized in DSEM specification.

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
            - can_marginalize: Set of unobserved constructs safe to ignore in DSEM spec
            - needs_modeling: Set of unobserved constructs that block identification
            - marginalize_reason: Dict explaining why each can be marginalized
            - modeling_reason: Dict explaining why each needs modeling
    """
    observed = get_observed_constructs(measurement_model)
    all_constructs = {c['name'] for c in latent_model['constructs']}
    unobserved = all_constructs - observed

    # Collect confounders that block identification of OBSERVED treatments
    # (Ignore unobserved treatments - they're inherently non-identifiable)
    blocking_any = set()
    blocking_details: dict[str, list[str]] = {}  # confounder -> list of treatments it blocks

    for treatment, blockers in identifiability_result.get('blocking_confounders', {}).items():
        # Skip if the treatment itself is unobserved (it's blocking itself)
        if treatment in unobserved:
            continue

        for blocker in blockers:
            # Only count confounders that are different from the treatment
            if blocker in unobserved and blocker != treatment:
                blocking_any.add(blocker)
                if blocker not in blocking_details:
                    blocking_details[blocker] = []
                blocking_details[blocker].append(treatment)

    # Classify unobserved constructs
    can_marginalize = unobserved - blocking_any
    needs_modeling = unobserved & blocking_any

    # Build explanations
    marginalize_reason = {}
    for u in can_marginalize:
        # Check if U creates any confounding at all (is in unobserved_confounders)
        if u in identifiability_result['graph_info'].get('unobserved_confounders', []):
            marginalize_reason[u] = "confounding handled by identification strategy (front-door or similar)"
        else:
            marginalize_reason[u] = "does not create confounding (single child or no observed children)"

    modeling_reason = {
        u: f"blocks identification of: {', '.join(sorted(blocking_details[u]))}"
        for u in needs_modeling
    }

    return {
        'can_marginalize': can_marginalize,
        'needs_modeling': needs_modeling,
        'marginalize_reason': marginalize_reason,
        'modeling_reason': modeling_reason,
    }


def format_identifiability_report(result: dict) -> str:
    """Format identifiability check results for logging."""
    lines = []

    outcome = result['outcome']
    n_identifiable = len(result['identifiable_treatments'])
    n_non_identifiable = len(result['non_identifiable_treatments'])
    total = n_identifiable + n_non_identifiable

    if not result['non_identifiable_treatments']:
        lines.append(f"✓ All {total} treatment effects on {outcome} are identifiable!")
    else:
        lines.append(f"✗ {n_non_identifiable}/{total} treatments have non-identifiable effects on {outcome}:")
        for treatment in sorted(result['non_identifiable_treatments']):
            blockers = result['blocking_confounders'].get(treatment, [])
            if blockers:
                lines.append(f"  - {treatment} (blocked by: {', '.join(blockers)})")
            else:
                lines.append(f"  - {treatment} (structural non-identifiability)")

    if result['identifiable_treatments']:
        lines.append(f"\n✓ {n_identifiable} treatments have identifiable effects:")
        for treatment in sorted(result['identifiable_treatments'].keys())[:5]:
            lines.append(f"  - {treatment}")
        if n_identifiable > 5:
            lines.append(f"  ... and {n_identifiable - 5} more")

    info = result['graph_info']
    lines.append(
        f"\nGraph: {len(info['observed_constructs'])}/{info['total_constructs']} constructs observed, "
        f"{info['n_directed_edges']} directed edges, {info['n_bidirected_edges']} bidirected edges"
    )
    if info['unobserved_confounders']:
        lines.append(f"Unobserved confounders: {', '.join(info['unobserved_confounders'])}")

    return '\n'.join(lines)


def format_marginalization_report(analysis: dict) -> str:
    """Format the marginalization analysis for logging.

    Args:
        analysis: Output from analyze_unobserved_constructs()

    Returns:
        Formatted string report
    """
    lines = []

    can_marginalize = analysis['can_marginalize']
    needs_modeling = analysis['needs_modeling']

    lines.append("=" * 60)
    lines.append("UNOBSERVED CONSTRUCT ANALYSIS FOR DSEM SPECIFICATION")
    lines.append("=" * 60)

    if can_marginalize:
        lines.append(f"\n✓ CAN MARGINALIZE ({len(can_marginalize)} constructs):")
        lines.append("  These can be omitted from DSEM spec - effects absorbed into error terms")
        for u in sorted(can_marginalize):
            reason = analysis['marginalize_reason'].get(u, '')
            lines.append(f"  - {u}: {reason}")

    if needs_modeling:
        lines.append(f"\n✗ NEEDS MODELING ({len(needs_modeling)} constructs):")
        lines.append("  These block identification - need proxies or explicit latent variables")
        for u in sorted(needs_modeling):
            reason = analysis['modeling_reason'].get(u, '')
            lines.append(f"  - {u}: {reason}")

    if not can_marginalize and not needs_modeling:
        lines.append("\n✓ All constructs are observed - no marginalization analysis needed")

    return '\n'.join(lines)
