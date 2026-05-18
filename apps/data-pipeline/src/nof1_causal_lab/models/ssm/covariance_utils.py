"""Shared covariance repair helpers for stable Cholesky factorization."""

from __future__ import annotations

import jax.numpy as jnp

# Defined locally to avoid circular import with inference.targets.base.
CHOL_JITTER = 1e-8

INITIAL_STATE_COV_MIN_EIGENVALUE = 1e-6


def symmetrize(M: jnp.ndarray) -> jnp.ndarray:
    """Symmetrize a square matrix or stack of square matrices."""
    return 0.5 * (M + jnp.swapaxes(M, -1, -2))


def symmetrize_with_jitter(M: jnp.ndarray, *, jitter: float = CHOL_JITTER) -> jnp.ndarray:
    """Symmetrize a square matrix or stack of square matrices and add diagonal jitter."""
    eye = jnp.eye(M.shape[-1], dtype=M.dtype)
    return symmetrize(M) + eye * jitter


def inflate_missing_variance(cov: jnp.ndarray, mask_float: jnp.ndarray) -> jnp.ndarray:
    """Inflate covariance diagonal for unobserved channels.

    Args:
        cov: Covariance matrix to inflate.
        mask_float: Float observation mask (1.0 = observed, 0.0 = missing).
    """
    from nof1_causal_lab.models.ssm.inference.targets.base import MISSING_DATA_LARGE_VAR

    return cov + jnp.diag((1.0 - mask_float) * MISSING_DATA_LARGE_VAR)


def stabilize_covariance_for_cholesky(
    cov: jnp.ndarray,
    *,
    min_eigenvalue: float = CHOL_JITTER,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Symmetrize and diagonally repair a covariance matrix for stable Cholesky."""
    cov_sym = 0.5 * (cov + cov.T)
    min_eig = jnp.min(jnp.linalg.eigvalsh(cov_sym))
    jitter = jnp.maximum(min_eigenvalue - min_eig, 0.0) + min_eigenvalue
    eye = jnp.eye(cov.shape[0], dtype=cov.dtype)
    return cov_sym + jitter * eye, min_eig


def stable_cholesky(
    cov: jnp.ndarray,
    *,
    min_eigenvalue: float = CHOL_JITTER,
) -> jnp.ndarray:
    """Return a stable Cholesky factor from a possibly near-PSD covariance."""
    stabilized_cov, _min_eig = stabilize_covariance_for_cholesky(
        cov,
        min_eigenvalue=min_eigenvalue,
    )
    return jnp.linalg.cholesky(stabilized_cov)
