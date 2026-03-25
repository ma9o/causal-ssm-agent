"""Shared covariance repair helpers for stable Cholesky factorization."""

from __future__ import annotations

import jax.numpy as jnp

from causal_ssm_agent.models.likelihoods.base import CHOL_JITTER

INITIAL_STATE_COV_MIN_EIGENVALUE = 1e-6


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
