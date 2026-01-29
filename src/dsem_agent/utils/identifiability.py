"""Identifiability checking using y0's ID algorithm.

Uses Pearl's do-calculus via y0 to check if causal effects are identifiable
given observed/unobserved constructs. This properly handles:
- Backdoor criterion
- Front-door criterion
- Other identification strategies from Shpitser & Pearl
"""

from itertools import combinations
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

    Since there's exactly one outcome in the model, we check identifiability
    for each treatment → outcome effect and return which treatments are
    identifiable vs not.

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

    # Build projected ADMG
    admg, unobserved_confounders = build_projected_admg(latent_model, observed_constructs)

    # Get all potential treatments (constructs with paths to outcome)
    all_treatments = _get_treatments_from_graph(latent_model, outcome)

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

        # Check identifiability using y0
        treatment_var = Variable(treatment)
        outcome_var = Variable(outcome)

        try:
            estimand = identify_outcomes(
                admg,
                treatments={treatment_var},
                outcomes={outcome_var},
            )

            if estimand is not None:
                identifiable_treatments[treatment] = str(estimand)
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


def build_projected_admg(
    latent_model: dict,
    observed_constructs: set[str],
) -> tuple[NxMixedGraph, set[str]]:
    """Build a projected ADMG for identifiability analysis.

    In the projected ADMG:
    - Only observed constructs are nodes
    - Directed edges between observed constructs are preserved
    - Unobserved constructs with multiple children create bidirected edges
      (representing confounding)

    Args:
        latent_model: Dict with 'constructs' and 'edges'
        observed_constructs: Set of construct names that have measurements

    Returns:
        Tuple of (NxMixedGraph, set of unobserved confounder names)
    """
    # Create Variable objects for observed constructs
    var_map = {name: Variable(name) for name in observed_constructs}

    # Collect directed edges (only between observed constructs)
    directed_edges = []
    for edge in latent_model.get('edges', []):
        cause, effect = edge['cause'], edge['effect']
        if cause in observed_constructs and effect in observed_constructs:
            directed_edges.append((var_map[cause], var_map[effect]))

    # Find unobserved constructs and their children
    all_constructs = {c['name'] for c in latent_model['constructs']}
    unobserved = all_constructs - observed_constructs

    # Build child map for unobserved constructs
    unobserved_children: dict[str, set[str]] = {u: set() for u in unobserved}
    for edge in latent_model.get('edges', []):
        cause, effect = edge['cause'], edge['effect']
        if cause in unobserved and effect in observed_constructs:
            unobserved_children[cause].add(effect)

    # Create bidirected edges for unobserved confounders
    # An unobserved node U with children A, B, C creates edges A↔B, A↔C, B↔C
    bidirected_edges = []
    unobserved_confounders = set()

    for u, children in unobserved_children.items():
        if len(children) >= 2:
            unobserved_confounders.add(u)
            for c1, c2 in combinations(children, 2):
                bidirected_edges.append((var_map[c1], var_map[c2]))

    # Build the ADMG
    admg = NxMixedGraph.from_edges(
        directed=directed_edges,
        undirected=bidirected_edges,
    )

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
