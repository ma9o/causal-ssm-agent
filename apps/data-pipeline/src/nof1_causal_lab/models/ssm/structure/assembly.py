"""Single source of truth for the linear-path structural assembly algorithms.

The stability-by-construction drift assembler and the sparse-element
cint assembler historically existed in two places — ``SSMParameterLayout``
and the ``StructuralDenseLinearSpec`` / ``StructuralInterceptSpec``
component specs in this package. The duplication carried a silent-
divergence risk; both impls had to evolve together to stay numerically
equivalent.

This module owns the canonical implementations. Both
``SSMParameterLayout`` and the spec components delegate here. The
functions are pure (no class state) so they can be JIT'd freely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np


def drift_base_decay_positions(
    drift_diag_mask: np.ndarray | jnp.ndarray, n_latent: int
) -> list[int]:
    """Latent indices marked as free in the drift diagonal mask."""
    return [idx for idx in range(n_latent) if bool(drift_diag_mask[idx])]


def drift_offdiag_positions(
    drift_offdiag_mask: np.ndarray | jnp.ndarray, n_latent: int
) -> list[tuple[int, int]]:
    """``(i, j)`` positions marked as free in the drift off-diagonal mask."""
    positions: list[tuple[int, int]] = []
    for i in range(n_latent):
        for j in range(n_latent):
            if i != j and bool(drift_offdiag_mask[i, j]):
                positions.append((i, j))
    return positions


def cint_free_positions(
    cint_mask: np.ndarray | jnp.ndarray, n_latent: int
) -> list[int]:
    """Latent indices marked as free in the cint mask."""
    return [idx for idx in range(n_latent) if bool(cint_mask[idx])]


# Generic helpers used by the non-drift blocks and by SSMParameterLayout.


def dense_vector_positions(mask: np.ndarray | jnp.ndarray, n: int) -> list[int]:
    """``[idx]`` positions marked free on a 1-D length-``n`` mask."""
    return [idx for idx in range(n) if bool(mask[idx])]


def rect_matrix_positions(
    mask: np.ndarray | jnp.ndarray, n_rows: int, n_cols: int
) -> list[tuple[int, int]]:
    """``(row, col)`` positions marked free on a rectangular mask."""
    return [(i, j) for i in range(n_rows) for j in range(n_cols) if bool(mask[i, j])]


def chol_diag_positions(
    mask: np.ndarray | jnp.ndarray, n: int
) -> list[int]:
    """Diagonal positions marked free on a square Cholesky-shape mask."""
    return [idx for idx in range(n) if bool(mask[idx, idx])]


def strict_lower_positions(
    mask: np.ndarray | jnp.ndarray, n: int
) -> list[tuple[int, int]]:
    """Strict-lower-triangle ``(row, col)`` positions marked free."""
    return [(row, col) for row in range(n) for col in range(row) if bool(mask[row, col])]


def assemble_dense_linear_drift(
    drift_template: jnp.ndarray,
    base_decay_positions: list[int],
    offdiag_positions_list: list[tuple[int, int]],
    base_decay_free: jnp.ndarray | None,
    offdiag_free: jnp.ndarray | None,
    stability_margin: float,
    time_invariant_mask: np.ndarray | jnp.ndarray | None,
) -> jnp.ndarray:
    """Stability-by-construction dense-linear drift assembler.

    Builds ``A`` from a template by:

    1. Inserting free off-diagonal entries at their masked positions.
    2. Setting diagonal entries from base-decay free values, subtracting
       the sum of row absolute values + ``stability_margin`` so the
       resulting ``A`` has eigenvalues in the stable region.
    3. Forcing diagonal entries of time-invariant latents toward zero so
       their drift is near-stationary.

    This is the single canonical implementation; both
    :class:`StructuralDenseLinearSpec` and :class:`SSMParameterLayout`
    delegate here.
    """
    drift = jnp.asarray(drift_template)
    if base_decay_free is not None:
        base_decay_free = jnp.asarray(base_decay_free, dtype=drift.dtype)
    if offdiag_free is not None:
        offdiag_free = jnp.asarray(offdiag_free, dtype=drift.dtype)

    if offdiag_free is not None:
        for idx, (i, j) in enumerate(offdiag_positions_list):
            drift = drift.at[i, j].set(offdiag_free[idx])
    if base_decay_free is not None:
        offdiag_drift = drift - jnp.diag(jnp.diag(drift))
        row_abs = jnp.sum(jnp.abs(offdiag_drift), axis=1)
        margin = jnp.asarray(stability_margin, dtype=drift.dtype)
        for idx, latent_idx in enumerate(base_decay_positions):
            base_decay = base_decay_free[idx]
            drift = drift.at[latent_idx, latent_idx].set(
                -(base_decay + row_abs[latent_idx] + margin)
            )
    if time_invariant_mask is not None:
        ti_mask = jnp.asarray(time_invariant_mask, dtype=bool)
        diag_vals = jnp.diag(drift)
        new_diag = jnp.where(ti_mask, jnp.asarray(-1e-6, dtype=drift.dtype), diag_vals)
        drift = drift - jnp.diag(diag_vals) + jnp.diag(new_diag)
    return drift


def assemble_intercept_cint(
    cint_template: jnp.ndarray,
    free_positions: list[int],
    cint_free: jnp.ndarray | None,
) -> jnp.ndarray:
    """Sparse-element-substitution cint assembler.

    Builds ``c`` from a template by inserting free values at their
    masked positions. Single canonical implementation.
    """
    cint = jnp.asarray(cint_template)
    if cint_free is not None:
        cint_free = jnp.asarray(cint_free, dtype=cint.dtype)
        for idx, latent_idx in enumerate(free_positions):
            cint = cint.at[latent_idx].set(cint_free[idx])
    return cint


# ---------------------------------------------------------------------------
# Generic sparse-substitution helpers (used by the BlockSpec abstraction
# in ``blocks.py`` and by ``SSMParameterLayout``)
# ---------------------------------------------------------------------------


def assemble_sparse_vector(
    template: jnp.ndarray,
    free_positions: list[int],
    free: jnp.ndarray | None,
) -> jnp.ndarray:
    """Insert free values into a 1-D template at marked positions."""
    out = jnp.asarray(template)
    if free is not None:
        free = jnp.asarray(free, dtype=out.dtype)
        for idx, latent_idx in enumerate(free_positions):
            out = out.at[latent_idx].set(free[idx])
    return out


def assemble_sparse_matrix(
    template: jnp.ndarray,
    free_positions: list[tuple[int, int]],
    free: jnp.ndarray | None,
) -> jnp.ndarray:
    """Insert free values into a 2-D template at marked ``(i, j)`` positions."""
    out = jnp.asarray(template)
    if free is not None and len(free_positions) > 0:
        free = jnp.asarray(free, dtype=out.dtype)
        for idx, (i, j) in enumerate(free_positions):
            out = out.at[i, j].set(free[idx])
    return out


def assemble_diffusion_chol(
    diffusion_chol_template: jnp.ndarray,
    diag_positions: list[int],
    lower_positions: list[tuple[int, int]],
    diag_free: jnp.ndarray | None,
    lower_free: jnp.ndarray | None,
    time_invariant_mask: np.ndarray | jnp.ndarray | None,
) -> jnp.ndarray:
    """Process-noise Cholesky assembler shared by ``DiffusionBlockSpec`` and ``SSMSpec``."""
    diffusion = jnp.asarray(diffusion_chol_template)
    if diag_free is not None:
        diag_free = jnp.asarray(diag_free, dtype=diffusion.dtype)
        for idx, latent_idx in enumerate(diag_positions):
            diffusion = diffusion.at[latent_idx, latent_idx].set(diag_free[idx])
    if lower_free is not None:
        lower_free = jnp.asarray(lower_free, dtype=diffusion.dtype)
        for idx, (row, col) in enumerate(lower_positions):
            diffusion = diffusion.at[row, col].set(lower_free[idx])
    if time_invariant_mask is not None:
        ti = jnp.asarray(time_invariant_mask, dtype=bool)
        diag_vals = jnp.diag(diffusion)
        new_diag = jnp.where(ti, jnp.asarray(1e-6, dtype=diffusion.dtype), diag_vals)
        diffusion = diffusion - jnp.diag(diag_vals) + jnp.diag(new_diag)
    return diffusion


def assemble_manifest_chol(
    template: jnp.ndarray,
    free_positions: list[int],
    free: jnp.ndarray | None,
) -> jnp.ndarray:
    """Manifest-noise diagonal-Cholesky assembler. Inserts free diagonal
    values at marked positions; off-diagonal is fixed at template."""
    chol = jnp.asarray(template)
    if free is not None and len(free_positions) > 0:
        free = jnp.asarray(free, dtype=chol.dtype)
        for idx, manifest_idx in enumerate(free_positions):
            chol = chol.at[manifest_idx, manifest_idx].set(free[idx])
    return chol
