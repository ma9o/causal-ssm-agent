"""Shared covariance repair helpers for stable Cholesky factorization.

Generic over the square dimension: helpers operate on any covariance, latent
(``D``) or manifest (``M``), so they annotate it with the neutral axis ``N``
(see :mod:`nof1_causal_lab.models.ssm.shapes`). Instrumented for runtime shape
checks, hence no ``from __future__ import annotations`` (eager annotations keep
the jaxtyping imports resolvable for beartype).
"""

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.shapes import Array, Float, FloatScalar, Shaped

# Defined locally so the covariance core does not depend on execution contracts.
CHOL_JITTER = 1e-8

INITIAL_STATE_COV_MIN_EIGENVALUE = 1e-6


def symmetrize(M: Float[Array, "*batch N N"]) -> Float[Array, "*batch N N"]:
    """Symmetrize a square matrix or stack of square matrices."""
    return 0.5 * (M + jnp.swapaxes(M, -1, -2))


def symmetrize_with_jitter(
    M: Float[Array, "*batch N N"], *, jitter: float = CHOL_JITTER
) -> Float[Array, "*batch N N"]:
    """Symmetrize a square matrix or stack of square matrices and add diagonal jitter."""
    eye = jnp.eye(M.shape[-1], dtype=M.dtype)
    return symmetrize(M) + eye * jitter


def inflate_missing_variance(
    cov: Float[Array, "N N"], mask_float: Shaped[Array, " N"]
) -> Float[Array, "N N"]:
    """Inflate covariance diagonal for unobserved channels.

    Args:
        cov: Covariance matrix to inflate.
        mask_float: Observation mask (observed = 1.0/True, missing = 0.0/False);
            consumed as ``1.0 - mask_float``, so bool or float is accepted.
    """
    from nof1_causal_lab.models.ssm.execution.contracts import MISSING_DATA_LARGE_VAR

    return cov + jnp.diag((1.0 - mask_float) * MISSING_DATA_LARGE_VAR)


def stabilize_covariance_for_cholesky(
    cov: Float[Array, "N N"],
    *,
    min_eigenvalue: float = CHOL_JITTER,
) -> tuple[Float[Array, "N N"], FloatScalar]:
    """Symmetrize and diagonally repair a covariance matrix for stable Cholesky."""
    cov_sym = 0.5 * (cov + cov.T)
    min_eig = jnp.min(jnp.linalg.eigvalsh(cov_sym))
    jitter = jnp.maximum(min_eigenvalue - min_eig, 0.0) + min_eigenvalue
    eye = jnp.eye(cov.shape[0], dtype=cov.dtype)
    return cov_sym + jitter * eye, min_eig


def stable_cholesky(
    cov: Float[Array, "N N"],
    *,
    min_eigenvalue: float = CHOL_JITTER,
) -> Float[Array, "N N"]:
    """Return a stable Cholesky factor from a possibly near-PSD covariance."""
    stabilized_cov, _min_eig = stabilize_covariance_for_cholesky(
        cov,
        min_eigenvalue=min_eigenvalue,
    )
    return jnp.linalg.cholesky(stabilized_cov)


def logdet_from_cholesky(cholesky: Float[Array, "*batch N N"]) -> Float[Array, "*batch"]:
    """Log-determinant of ``A = L Lᵀ`` from its lower-triangular Cholesky factor ``L``.

    Computes ``2 · Σ log diag(L)``, batched over any leading dimensions. Expects a
    valid Cholesky factor (strictly positive diagonal); callers guarantee positive
    definiteness upstream (e.g. via :func:`symmetrize_with_jitter` /
    :func:`stable_cholesky`), so no diagonal clipping is needed.
    """
    diagonal = jnp.diagonal(cholesky, axis1=-2, axis2=-1)
    return 2.0 * jnp.sum(jnp.log(diagonal), axis=-1)
