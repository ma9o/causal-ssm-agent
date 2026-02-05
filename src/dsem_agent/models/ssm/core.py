"""Core utility functions for state-space models.

Contains helper functions for parameter transformations and matrix operations
that are not specific to discretization or inference.
"""

import jax.numpy as jnp


def cholesky_of_diffusion(diffusion_raw: jnp.ndarray) -> jnp.ndarray:
    """Convert raw lower triangular diffusion to covariance.

    DIFFUSION is specified as a lower-triangular Cholesky factor.
    The covariance is G*G'.

    Args:
        diffusion_raw: n x n lower triangular matrix G

    Returns:
        G*G': n x n covariance matrix
    """
    # Ensure lower triangular
    G = jnp.tril(diffusion_raw)
    return G @ G.T


def ensure_stability(drift: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Ensure drift matrix eigenvalues have negative real parts.

    For a stable state-space model, all eigenvalues of the drift matrix must have
    negative real parts. This function projects to the nearest stable matrix.

    Args:
        drift: n x n drift matrix
        eps: small positive value for stability margin

    Returns:
        Stabilized drift matrix
    """
    # Compute eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eig(drift)

    # Project eigenvalues with positive real parts
    real_parts = jnp.real(eigenvalues)
    stable_real = jnp.where(real_parts > -eps, -eps, real_parts)
    stable_eigenvalues = stable_real + 1j * jnp.imag(eigenvalues)

    # Reconstruct matrix
    stable_drift = eigenvectors @ jnp.diag(stable_eigenvalues) @ jnp.linalg.inv(
        eigenvectors
    )

    return jnp.real(stable_drift)
