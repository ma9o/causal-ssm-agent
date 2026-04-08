"""Test helpers for constructing explicit SSMSpec instances."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.models.ssm.model import (
    SSMSpec,
    full_cholesky_mask,
    full_diagonal_mask,
    full_drift_offdiag_mask,
    full_vector_mask,
    strict_lower_triangle_mask,
    zero_diagonal_mask,
    zero_loading_mask,
    zero_square_mask,
    zero_vector_mask,
)


def full_drift_mask(n_latent: int) -> np.ndarray:
    """Return the fully free combined drift support mask used by tests."""
    return np.eye(n_latent, dtype=bool) | full_drift_offdiag_mask(n_latent)


def split_drift_mask(drift_mask: np.ndarray, n_latent: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a combined drift support matrix into diagonal and off-diagonal masks."""
    mask = np.asarray(drift_mask, dtype=bool)
    if mask.shape != (n_latent, n_latent):
        raise ValueError(f"drift_mask must have shape ({n_latent}, {n_latent}), got {mask.shape}")
    drift_diag_mask = np.diag(mask).copy()
    drift_offdiag_mask = mask.copy()
    np.fill_diagonal(drift_offdiag_mask, False)
    return drift_diag_mask, drift_offdiag_mask


def combined_drift_mask(spec: SSMSpec) -> np.ndarray:
    """Recover the combined drift support matrix from a compiled spec."""
    mask = np.asarray(spec.drift_offdiag_mask, dtype=bool).copy()
    np.fill_diagonal(mask, np.asarray(spec.drift_diag_mask, dtype=bool))
    return mask


def make_ssm_spec(**kwargs: Any) -> SSMSpec:
    """Build an SSMSpec with explicit default structural masks."""
    kwargs = dict(kwargs)
    n_latent = kwargs["n_latent"]
    n_manifest = kwargs["n_manifest"]
    drift_mask_present = "drift_mask" in kwargs
    drift_mask = kwargs.pop("drift_mask", None)
    if drift_mask_present and drift_mask is None:
        kwargs.setdefault("drift_diag_mask", None)
        kwargs.setdefault("drift_offdiag_mask", None)
    elif drift_mask is not None:
        drift_diag_mask, drift_offdiag_mask = split_drift_mask(drift_mask, n_latent)
        kwargs.setdefault("drift_diag_mask", drift_diag_mask)
        kwargs.setdefault("drift_offdiag_mask", drift_offdiag_mask)
    elif "drift_diag_mask" not in kwargs and "drift_offdiag_mask" not in kwargs:
        if "drift" in kwargs:
            kwargs.setdefault("drift_diag_mask", zero_diagonal_mask(n_latent))
            kwargs.setdefault("drift_offdiag_mask", zero_square_mask(n_latent))
        else:
            kwargs.setdefault("drift_diag_mask", full_diagonal_mask(n_latent))
            kwargs.setdefault("drift_offdiag_mask", full_drift_offdiag_mask(n_latent))
    if "drift" not in kwargs:
        kwargs.setdefault("drift", jnp.zeros((n_latent, n_latent)))
    kwargs.setdefault("cint_mask", zero_vector_mask(n_latent))
    kwargs.setdefault("cint", jnp.zeros(n_latent))
    kwargs.setdefault("lambda_mask", zero_loading_mask(n_manifest, n_latent))
    kwargs.setdefault("lambda_mat", jnp.eye(n_manifest, n_latent))
    if "diffusion_mask" in kwargs:
        diffusion_mask = kwargs.pop("diffusion_mask")
        kwargs.setdefault("diffusion_chol_mask", diffusion_mask)
    kwargs.setdefault("diffusion_chol_mask", full_cholesky_mask(n_latent))
    if "diffusion" in kwargs:
        diffusion = kwargs.pop("diffusion")
        kwargs.setdefault("diffusion_chol", diffusion)
    kwargs.setdefault("diffusion_chol", jnp.eye(n_latent))
    kwargs.setdefault("manifest_means_mask", zero_vector_mask(n_manifest))
    kwargs.setdefault("manifest_means", jnp.zeros(n_manifest))
    if "manifest_var_mask" in kwargs:
        manifest_var_mask = kwargs.pop("manifest_var_mask")
        kwargs.setdefault("manifest_chol_diag_mask", manifest_var_mask)
    kwargs.setdefault("manifest_chol_diag_mask", full_diagonal_mask(n_manifest))
    if "manifest_var" in kwargs:
        manifest_var = kwargs.pop("manifest_var")
        kwargs.setdefault("manifest_chol", manifest_var)
    kwargs.setdefault("manifest_chol", jnp.zeros((n_manifest, n_manifest)))
    kwargs.setdefault("t0_means_mask", full_vector_mask(n_latent))
    kwargs.setdefault("t0_means", jnp.zeros(n_latent))
    if "t0_var_diag_mask" in kwargs:
        t0_var_diag_mask = kwargs.pop("t0_var_diag_mask")
        kwargs.setdefault("t0_chol_diag_mask", t0_var_diag_mask)
    kwargs.setdefault("t0_chol_diag_mask", full_diagonal_mask(n_latent))
    kwargs.setdefault("t0_correlation_mask", strict_lower_triangle_mask(n_latent))
    if "t0_var" in kwargs:
        t0_var = kwargs.pop("t0_var")
        kwargs.setdefault("t0_chol", t0_var)
    kwargs.setdefault("t0_chol", jnp.eye(n_latent))
    return SSMSpec(**kwargs)


def diagonal_diffusion_kwargs(n_latent: int) -> dict[str, Any]:
    return {
        "diffusion_chol": jnp.eye(n_latent),
        "diffusion_chol_mask": np.diag(full_diagonal_mask(n_latent)),
    }


def full_diffusion_kwargs(n_latent: int) -> dict[str, Any]:
    return {
        "diffusion_chol": jnp.eye(n_latent),
        "diffusion_chol_mask": full_cholesky_mask(n_latent),
    }


def diagonal_manifest_var_kwargs(n_manifest: int) -> dict[str, Any]:
    return {
        "manifest_chol": jnp.zeros((n_manifest, n_manifest)),
        "manifest_chol_diag_mask": full_diagonal_mask(n_manifest),
    }


def fixed_manifest_var_kwargs(chol: jnp.ndarray) -> dict[str, Any]:
    return {
        "manifest_chol": chol,
        "manifest_chol_diag_mask": zero_diagonal_mask(int(chol.shape[0])),
    }


def diagonal_t0_var_kwargs(n_latent: int) -> dict[str, Any]:
    return {
        "t0_chol": jnp.eye(n_latent),
        "t0_chol_diag_mask": full_diagonal_mask(n_latent),
        "t0_correlation_mask": zero_square_mask(n_latent),
    }


def full_t0_var_kwargs(n_latent: int) -> dict[str, Any]:
    return {
        "t0_chol": jnp.eye(n_latent),
        "t0_chol_diag_mask": full_diagonal_mask(n_latent),
        "t0_correlation_mask": strict_lower_triangle_mask(n_latent),
    }
