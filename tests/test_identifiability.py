"""Test identifiability checking using y0."""

import pytest

from dsem_agent.utils.identifiability import (
    build_projected_admg,
    check_identifiability,
    format_identifiability_report,
    get_observed_constructs,
)


def test_get_observed_constructs():
    """Test extraction of observed constructs from measurement model."""
    measurement_model = {
        'indicators': [
            {'name': 'ind1', 'construct': 'A', 'how_to_measure': 'test'},
            {'name': 'ind2', 'construct': 'B', 'how_to_measure': 'test'},
            {'name': 'ind3', 'construct': 'A', 'how_to_measure': 'test'},  # Duplicate
        ]
    }

    observed = get_observed_constructs(measurement_model)
    assert observed == {'A', 'B'}


def test_identifiability_simple_chain():
    """Test identifiability on simple chain: A -> B -> C (all observed)."""
    latent_model = {
        'constructs': [
            {'name': 'A', 'role': 'exogenous'},
            {'name': 'B', 'role': 'endogenous'},
            {'name': 'C', 'role': 'endogenous', 'is_outcome': True},
        ],
        'edges': [
            {'cause': 'A', 'effect': 'B', 'description': 'A causes B'},
            {'cause': 'B', 'effect': 'C', 'description': 'B causes C'},
        ],
    }

    measurement_model = {
        'indicators': [
            {'name': 'a_ind', 'construct': 'A', 'how_to_measure': 'test'},
            {'name': 'b_ind', 'construct': 'B', 'how_to_measure': 'test'},
            {'name': 'c_ind', 'construct': 'C', 'how_to_measure': 'test'},
        ]
    }

    result = check_identifiability(latent_model, measurement_model)

    # All effects should be identifiable (no unobserved confounders)
    assert result['outcome'] == 'C'
    assert len(result['non_identifiable_treatments']) == 0
    assert 'A' in result['identifiable_treatments']
    assert 'B' in result['identifiable_treatments']
    assert result['graph_info']['n_bidirected_edges'] == 0


def test_identifiability_with_unobserved_confounder():
    """Test non-identifiability with unobserved confounder.

    Graph: A -> B, U -> A, U -> B (U is unobserved)
    This creates confounding A <-> B in the projected ADMG.
    """
    latent_model = {
        'constructs': [
            {'name': 'A', 'role': 'endogenous'},
            {'name': 'B', 'role': 'endogenous', 'is_outcome': True},
            {'name': 'U', 'role': 'exogenous'},  # Unobserved confounder
        ],
        'edges': [
            {'cause': 'A', 'effect': 'B', 'description': 'A causes B'},
            {'cause': 'U', 'effect': 'A', 'description': 'U causes A'},
            {'cause': 'U', 'effect': 'B', 'description': 'U causes B'},
        ],
    }

    # Only A and B are observed (U has no indicators)
    measurement_model = {
        'indicators': [
            {'name': 'a_ind', 'construct': 'A', 'how_to_measure': 'test'},
            {'name': 'b_ind', 'construct': 'B', 'how_to_measure': 'test'},
        ]
    }

    result = check_identifiability(latent_model, measurement_model)

    # A should NOT be identifiable due to unobserved confounder U
    assert result['outcome'] == 'B'
    assert 'A' in result['non_identifiable_treatments']
    assert 'U' in result['blocking_confounders']['A']
    assert result['graph_info']['n_bidirected_edges'] == 1  # A <-> B


def test_identifiability_front_door():
    """Test front-door identifiability.

    Graph: X -> M -> Y, with unobserved U -> X, U -> Y
    The effect X -> Y is identifiable via front-door criterion through M.
    """
    latent_model = {
        'constructs': [
            {'name': 'X', 'role': 'endogenous'},
            {'name': 'M', 'role': 'endogenous'},  # Mediator
            {'name': 'Y', 'role': 'endogenous', 'is_outcome': True},
            {'name': 'U', 'role': 'exogenous'},  # Unobserved confounder
        ],
        'edges': [
            {'cause': 'X', 'effect': 'M', 'description': 'X causes M'},
            {'cause': 'M', 'effect': 'Y', 'description': 'M causes Y'},
            {'cause': 'U', 'effect': 'X', 'description': 'U causes X'},
            {'cause': 'U', 'effect': 'Y', 'description': 'U causes Y'},
        ],
    }

    # X, M, Y observed; U unobserved
    measurement_model = {
        'indicators': [
            {'name': 'x_ind', 'construct': 'X', 'how_to_measure': 'test'},
            {'name': 'm_ind', 'construct': 'M', 'how_to_measure': 'test'},
            {'name': 'y_ind', 'construct': 'Y', 'how_to_measure': 'test'},
        ]
    }

    result = check_identifiability(latent_model, measurement_model)

    # X should be identifiable via front-door through M
    assert result['outcome'] == 'Y'
    assert 'X' in result['identifiable_treatments']
    # The estimand should involve M (front-door formula)
    assert 'M' in result['identifiable_treatments']['X']


def test_identifiability_unobserved_treatment():
    """Test when the treatment itself is unobserved."""
    latent_model = {
        'constructs': [
            {'name': 'X', 'role': 'exogenous'},
            {'name': 'Y', 'role': 'endogenous', 'is_outcome': True},
        ],
        'edges': [
            {'cause': 'X', 'effect': 'Y', 'description': 'X causes Y'},
        ],
    }

    # Only Y is observed
    measurement_model = {
        'indicators': [
            {'name': 'y_ind', 'construct': 'Y', 'how_to_measure': 'test'},
        ]
    }

    result = check_identifiability(latent_model, measurement_model)

    # X not identifiable because X is not observed
    assert 'X' in result['non_identifiable_treatments']
    assert 'X' in result['blocking_confounders']['X']  # Treatment itself is the blocker


def test_build_projected_admg():
    """Test ADMG construction with bidirected edges for confounders."""
    latent_model = {
        'constructs': [
            {'name': 'A'},
            {'name': 'B'},
            {'name': 'C'},
            {'name': 'U'},  # Will be unobserved, causes A and B
        ],
        'edges': [
            {'cause': 'A', 'effect': 'C', 'description': 'test'},
            {'cause': 'B', 'effect': 'C', 'description': 'test'},
            {'cause': 'U', 'effect': 'A', 'description': 'test'},
            {'cause': 'U', 'effect': 'B', 'description': 'test'},
        ],
    }

    observed = {'A', 'B', 'C'}
    admg, confounders = build_projected_admg(latent_model, observed)

    # Should have A <-> B bidirected edge from U
    assert 'U' in confounders
    assert len(list(admg.undirected.edges())) == 1


def test_format_identifiability_report():
    """Test formatting of identifiability report."""
    result = {
        'outcome': 'Y',
        'identifiable_treatments': {'X': 'P(Y|X)'},
        'non_identifiable_treatments': {'A'},
        'blocking_confounders': {'A': ['U', 'V']},
        'graph_info': {
            'observed_constructs': ['X', 'Y', 'A'],
            'total_constructs': 5,
            'unobserved_confounders': ['U', 'V'],
            'n_directed_edges': 3,
            'n_bidirected_edges': 1,
        },
    }

    report = format_identifiability_report(result)

    assert '1/2 treatments have non-identifiable effects on Y' in report
    assert 'A (blocked by: U, V)' in report
    assert '1 treatments have identifiable effects' in report
    assert 'X' in report
    assert '3/5 constructs observed' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
