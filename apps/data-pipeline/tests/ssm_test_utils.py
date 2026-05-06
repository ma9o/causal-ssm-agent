"""Test helpers for constructing explicit SSMSpec instances."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np

from causal_ssm_agent.models.ssm import discretize_system
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
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime


def make_lgss_data(
    *,
    T: int = 100,
    dt: float = 1.0,
    drift_diag: float = -0.3,
    diff_sd: float = 0.3,
    obs_sd: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Build 1D linear-Gaussian SSM data plus a free-parameter SSMSpec.

    Returns a dict with ``observations``, ``times``, ``spec``, the true
    parameter values (``true_drift_diag``, ``true_diff_diag``, ``true_obs_sd``),
    and ``n_latent`` for convenience. Used by recovery checks that fit the
    same canonical 1D model with different inference methods.
    """
    n_latent, n_manifest = 1, 1

    true_drift = jnp.array([[drift_diag]])
    true_diff_cov = jnp.array([[diff_sd**2]])
    true_obs_var = jnp.array([[obs_sd**2]])

    Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
    Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
    R_chol = jla.cholesky(true_obs_var, lower=True)

    key = random.PRNGKey(seed)
    states = [jnp.zeros(n_latent)]
    for _ in range(T - 1):
        key, nk = random.split(key)
        states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
    latent = jnp.stack(states)

    key, obs_key = random.split(key)
    observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
    times = jnp.arange(T, dtype=float) * dt

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift_diag_mask=full_diagonal_mask(n_latent),
        drift_offdiag_mask=full_drift_offdiag_mask(n_latent),
        drift=jnp.zeros((n_latent, n_latent)),
        cint_mask=zero_vector_mask(n_latent),
        cint=jnp.zeros(n_latent),
        lambda_mask=zero_loading_mask(n_manifest, n_latent),
        lambda_mat=jnp.eye(n_manifest, n_latent),
        diffusion_chol_mask=np.diag(full_diagonal_mask(n_latent)),
        diffusion_chol=jnp.eye(n_latent),
        manifest_means_mask=zero_vector_mask(n_manifest),
        manifest_means=jnp.zeros(n_manifest),
        manifest_chol_diag_mask=full_diagonal_mask(n_manifest),
        manifest_chol=jnp.zeros((n_manifest, n_manifest)),
        t0_means_mask=zero_vector_mask(n_latent),
        t0_means=jnp.zeros(n_latent),
        t0_chol_diag_mask=zero_diagonal_mask(n_latent),
        t0_correlation_mask=zero_square_mask(n_latent),
        t0_chol=jnp.eye(n_latent),
    )

    return {
        "observations": observations,
        "times": times,
        "spec": spec,
        "true_drift_diag": drift_diag,
        "true_diff_diag": diff_sd,
        "true_obs_sd": obs_sd,
        "n_latent": n_latent,
    }


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


def make_observation_support_runtime(**kwargs: Any) -> ObservationSupportRuntime:
    """Build ObservationSupportRuntime while accepting 2D interval coefficient inputs."""
    support_kinds = kwargs["support_kinds"]
    kwargs.setdefault(
        "summary_operators",
        ["mean" if kind == "interval" else "last" for kind in support_kinds],
    )
    kwargs.setdefault(
        "anchor_policies",
        [
            "support_start" if operator == "first" else "support_end"
            for operator in kwargs["summary_operators"]
        ],
    )
    prev = np.asarray(kwargs["interval_prev_coeffs"], dtype=np.float64)
    curr = np.asarray(kwargs["interval_curr_coeffs"], dtype=np.float64)
    weights = np.asarray(kwargs["interval_weights"], dtype=np.float64)
    if prev.ndim == 2:
        prev = prev[..., None]
        curr = curr[..., None]
        weights = weights[..., None]
    kwargs["interval_prev_coeffs"] = prev
    kwargs["interval_curr_coeffs"] = curr
    kwargs["interval_weights"] = weights
    emission_slots = kwargs.get("emission_slot_indices")
    if emission_slots is None:
        support_end = np.asarray(kwargs["support_end_times"])
        emission_slots = np.where(np.isfinite(support_end), 0, -1).astype(np.int64)
    kwargs["emission_slot_indices"] = emission_slots
    return ObservationSupportRuntime(**kwargs)


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
