"""Tests for generic Ito raw-moment recurrence generation."""

from __future__ import annotations

import sympy as sp

from causal_ssm_agent.models.sde_moments import (
    coefficient_of_moment,
    ito_moment_ode_system,
    moment_symbol,
    multi_indices,
)


def test_two_state_ou_recurrences_match_paper_equations() -> None:
    """Ito generator recovers the paper's two-state additive OU moment ODEs."""
    x, y = sp.symbols("x y")
    a, b, c, d, e, f, p, r, s = sp.symbols("a b c d e f p r s")
    drift = sp.Matrix(
        [
            a * e + b * f - a * x - b * y,
            c * e + d * f - c * x - d * y,
        ]
    )
    diffusion = sp.Matrix(
        [
            [p, 0],
            [r, s],
        ]
    )

    recurrences = ito_moment_ode_system(
        (x, y),
        drift,
        diffusion=diffusion,
        alphas=[(1, 0), (0, 1), (2, 0), (0, 2), (1, 1)],
    )
    m00 = moment_symbol((0, 0))

    expected = {
        (1, 0): a * e + b * f - a * moment_symbol((1, 0)) - b * moment_symbol((0, 1)),
        (0, 1): c * e + d * f - c * moment_symbol((1, 0)) - d * moment_symbol((0, 1)),
        (2, 0): (
            2 * (a * e + b * f) * moment_symbol((1, 0))
            - 2 * a * moment_symbol((2, 0))
            - 2 * b * moment_symbol((1, 1))
            + p**2
        ),
        (0, 2): (
            2 * (c * e + d * f) * moment_symbol((0, 1))
            - 2 * c * moment_symbol((1, 1))
            - 2 * d * moment_symbol((0, 2))
            + r**2
            + s**2
        ),
        (1, 1): (
            (c * e + d * f) * moment_symbol((1, 0))
            + (a * e + b * f) * moment_symbol((0, 1))
            - c * moment_symbol((2, 0))
            - b * moment_symbol((0, 2))
            - (a + d) * moment_symbol((1, 1))
            + p * r
        ),
    }
    for alpha, rhs in expected.items():
        got = sp.expand(recurrences[alpha].subs(m00, 1))
        assert sp.simplify(got - rhs) == 0


def test_linear_additive_recurrence_matches_matrix_formula() -> None:
    """Generic Ito recurrences reduce to mu'=Fmu and M'=FM+MF' + Q."""
    z0, z1, z2 = sp.symbols("z0 z1 z2")
    state = (z0, z1, z2)
    drift_matrix = sp.Matrix(
        [
            [sp.Rational(-5, 2), sp.Rational(1, 3), 0],
            [sp.Rational(1, 5), sp.Rational(-7, 3), sp.Rational(1, 4)],
            [0, sp.Rational(-1, 6), sp.Rational(-3, 2)],
        ]
    )
    diffusion_cov = sp.diag(sp.Rational(1, 4), sp.Rational(1, 9), sp.Rational(1, 16))
    drift = drift_matrix * sp.Matrix(state)

    recurrences = ito_moment_ode_system(
        state,
        drift,
        diffusion_cov=diffusion_cov,
        max_order=2,
    )
    zero = (0, 0, 0)
    basis = [zero, *multi_indices(3, 2)]

    expected: dict[tuple[int, ...], dict[tuple[int, ...], sp.Expr]] = {}
    for i in range(3):
        alpha = tuple(1 if idx == i else 0 for idx in range(3))
        expected[alpha] = {}
        for k in range(3):
            beta = tuple(1 if idx == k else 0 for idx in range(3))
            expected[alpha][beta] = expected[alpha].get(beta, 0) + drift_matrix[i, k]

    for i in range(3):
        for j in range(i, 3):
            alpha_list = [0, 0, 0]
            alpha_list[i] += 1
            alpha_list[j] += 1
            alpha = tuple(alpha_list)
            expected[alpha] = {zero: diffusion_cov[i, j]}
            for k in range(3):
                beta_list = [0, 0, 0]
                beta_list[k] += 1
                beta_list[j] += 1
                beta = tuple(beta_list)
                expected[alpha][beta] = expected[alpha].get(beta, 0) + drift_matrix[i, k]

                beta_list = [0, 0, 0]
                beta_list[i] += 1
                beta_list[k] += 1
                beta = tuple(beta_list)
                expected[alpha][beta] = expected[alpha].get(beta, 0) + drift_matrix[j, k]

    for alpha, rhs in recurrences.items():
        for beta in basis:
            assert sp.simplify(coefficient_of_moment(rhs, beta) - expected[alpha].get(beta, 0)) == 0
