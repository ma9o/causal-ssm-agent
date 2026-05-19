"""Generic Ito moment recurrences for polynomial SDEs."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

MomentAlpha = tuple[int, ...]


def moment_symbol(alpha: Sequence[int], *, prefix: str = "m") -> sp.Symbol:
    """Return the symbolic raw-moment name for a multi-index."""
    suffix = "_".join(str(int(power)) for power in alpha)
    return sp.Symbol(f"{prefix}_{suffix}")


def state_monomial(
    alpha: Sequence[int],
    state_symbols: Sequence[sp.Symbol],
) -> sp.Expr:
    """Return ``prod_i x_i ** alpha_i`` for a raw moment multi-index."""
    if len(alpha) != len(state_symbols):
        raise ValueError(
            f"Moment index has length {len(alpha)}, but state has length {len(state_symbols)}."
        )

    term = sp.Integer(1)
    for symbol, power in zip(state_symbols, alpha, strict=True):
        term *= symbol ** int(power)
    return term


def multi_indices(
    n_state: int,
    max_order: int,
    *,
    min_order: int = 1,
) -> list[MomentAlpha]:
    """Enumerate raw-moment multi-indices by total order."""
    if n_state <= 0:
        raise ValueError(f"n_state must be positive, got {n_state}.")
    if max_order < 0:
        raise ValueError(f"max_order must be non-negative, got {max_order}.")
    if min_order < 0:
        raise ValueError(f"min_order must be non-negative, got {min_order}.")
    if min_order > max_order:
        return []

    indices: list[MomentAlpha] = []
    for alpha in itertools.product(range(max_order + 1), repeat=n_state):
        order = sum(alpha)
        if min_order <= order <= max_order:
            indices.append(tuple(int(power) for power in alpha))
    return sorted(indices, key=lambda item: (sum(item), item))


def moment_expression_from_polynomial(
    expr: sp.Expr,
    state_symbols: Sequence[sp.Symbol],
    *,
    prefix: str = "m",
) -> sp.Expr:
    """Replace each state monomial in ``expr`` with its raw-moment symbol."""
    expanded = sp.expand(expr)
    poly = sp.Poly(expanded, *state_symbols)
    result = sp.Integer(0)
    for powers, coefficient in poly.terms():
        result += coefficient * moment_symbol(powers, prefix=prefix)
    return sp.factor(result)


def ito_generator_for_monomial(
    alpha: Sequence[int],
    state_symbols: Sequence[sp.Symbol],
    drift: sp.MatrixBase,
    *,
    diffusion: sp.MatrixBase | None = None,
    diffusion_cov: sp.MatrixBase | None = None,
) -> sp.Expr:
    """Apply the Ito generator to one state monomial.

    For ``dX = f(X) dt + G(X) dW`` and ``h_alpha(X) = prod_i X_i ** alpha_i``,
    this returns ``L h_alpha`` where
    ``L h = grad(h).f + 1/2 trace((G G.T) Hessian(h))``.
    """
    n_state = len(state_symbols)
    if len(alpha) != n_state:
        raise ValueError(
            f"Moment index has length {len(alpha)}, but state has length {n_state}."
        )
    drift_matrix = sp.Matrix(drift)
    if drift_matrix.shape != (n_state, 1):
        raise ValueError(f"drift must have shape ({n_state}, 1), got {drift_matrix.shape}.")
    if diffusion is None and diffusion_cov is None:
        raise ValueError("Provide either diffusion or diffusion_cov.")
    if diffusion_cov is None:
        diffusion_matrix = sp.Matrix(diffusion)
        if diffusion_matrix.shape[0] != n_state:
            raise ValueError(
                f"diffusion must have {n_state} rows, got {diffusion_matrix.shape[0]}."
            )
        diffusion_cov_matrix = diffusion_matrix * diffusion_matrix.T
    else:
        diffusion_cov_matrix = sp.Matrix(diffusion_cov)
    if diffusion_cov_matrix.shape != (n_state, n_state):
        raise ValueError(
            f"diffusion_cov must have shape ({n_state}, {n_state}), "
            f"got {diffusion_cov_matrix.shape}."
        )

    h = state_monomial(alpha, state_symbols)
    generator = sp.Integer(0)
    for i, x_i in enumerate(state_symbols):
        generator += sp.diff(h, x_i) * drift_matrix[i]
    for i, x_i in enumerate(state_symbols):
        for j, x_j in enumerate(state_symbols):
            generator += (
                sp.Rational(1, 2)
                * diffusion_cov_matrix[i, j]
                * sp.diff(h, x_i, x_j)
            )
    return sp.expand(generator)


def ito_moment_recurrence(
    alpha: Sequence[int],
    state_symbols: Sequence[sp.Symbol],
    drift: sp.MatrixBase,
    *,
    diffusion: sp.MatrixBase | None = None,
    diffusion_cov: sp.MatrixBase | None = None,
    prefix: str = "m",
) -> sp.Expr:
    """Return the deterministic ODE right-hand side for one raw SDE moment."""
    generator = ito_generator_for_monomial(
        alpha,
        state_symbols,
        drift,
        diffusion=diffusion,
        diffusion_cov=diffusion_cov,
    )
    return moment_expression_from_polynomial(generator, state_symbols, prefix=prefix)


def ito_moment_ode_system(
    state_symbols: Sequence[sp.Symbol],
    drift: sp.MatrixBase,
    *,
    diffusion: sp.MatrixBase | None = None,
    diffusion_cov: sp.MatrixBase | None = None,
    max_order: int | None = None,
    alphas: Sequence[Sequence[int]] | None = None,
    prefix: str = "m",
) -> dict[MomentAlpha, sp.Expr]:
    """Return Ito-derived raw-moment ODEs for requested multi-indices."""
    if alphas is None:
        if max_order is None:
            raise ValueError("Provide max_order or explicit alphas.")
        alphas = multi_indices(len(state_symbols), max_order)
    return {
        tuple(int(power) for power in alpha): ito_moment_recurrence(
            tuple(int(power) for power in alpha),
            state_symbols,
            drift,
            diffusion=diffusion,
            diffusion_cov=diffusion_cov,
            prefix=prefix,
        )
        for alpha in alphas
    }


def coefficient_of_moment(
    expr: sp.Expr,
    alpha: Sequence[int],
    *,
    prefix: str = "m",
) -> sp.Expr:
    """Return the coefficient multiplying a raw-moment symbol in ``expr``."""
    return sp.expand(expr).coeff(moment_symbol(alpha, prefix=prefix))
