"""Pure-JAX matrix assembly for SSM parameters.

Single source of truth for building SSM matrices (drift, diffusion, lambda)
from raw parameter arrays. Used by both:
- SSMModel._sample_* (numpyro model, single sample)
- _assemble_deterministics (vmap over particles)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np

    from causal_ssm_agent.models.ssm.model import SSMSpec


def lower_triangle_positions(
    n_latent: int,
    mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Enumerate lower-triangle positions, optionally filtered by a boolean mask."""
    positions: list[tuple[int, int]] = []
    for row in range(n_latent):
        for col in range(row):
            if mask is None or bool(mask[row, col]):
                positions.append((row, col))
    return positions


class SSMAssembler:
    """Pure-JAX functions to build SSM matrices from raw parameter arrays.

    Pre-computes position lists and masks from SSMSpec so that assembly
    functions are self-contained closures suitable for jax.vmap.
    """

    def __init__(self, spec: SSMSpec) -> None:
        self.n_latent = spec.n_latent
        self.n_manifest = spec.n_manifest

        # Drift: pre-compute off-diagonal positions from mask
        self.offdiag_positions: list[tuple[int, int]] = []
        for i in range(spec.n_latent):
            for j in range(spec.n_latent):
                if i != j and spec.drift_mask[i, j]:
                    self.offdiag_positions.append((i, j))

        self.ti_mask: jnp.ndarray | None = (
            jnp.array(spec.time_invariant_mask) if spec.time_invariant_mask is not None else None
        )
        self.t0_correlation_positions = lower_triangle_positions(
            spec.n_latent,
            spec.t0_correlation_mask if spec.t0_var != "diag" else None,
        )

        # Lambda: pre-compute mode, template, and free positions
        if isinstance(spec.lambda_mat, jnp.ndarray):
            self.lambda_mode = "template"
            self.lambda_template = jnp.array(spec.lambda_mat)
            self.lambda_free_positions: list[tuple[int, int]] = [
                (i, j)
                for i in range(spec.n_manifest)
                for j in range(spec.n_latent)
                if spec.lambda_mask[i, j]
            ]
        else:
            self.lambda_mode = "legacy"
            self.lambda_template = jnp.eye(spec.n_manifest, spec.n_latent)
            self.lambda_free_positions = [
                (i, j) for i in range(spec.n_latent, spec.n_manifest) for j in range(spec.n_latent)
            ]

        # Manifest variance: diagonal Cholesky template with optional sparse free positions
        if isinstance(spec.manifest_var, jnp.ndarray) and spec.manifest_var_mask is not None:
            self.manifest_var_mode = "template"
            self.manifest_var_template = jnp.array(spec.manifest_var)
            self.manifest_var_free_positions: list[int] = [
                idx for idx in range(spec.n_manifest) if bool(spec.manifest_var_mask[idx])
            ]
        elif isinstance(spec.manifest_var, jnp.ndarray):
            self.manifest_var_mode = "fixed"
            self.manifest_var_template = jnp.array(spec.manifest_var)
            self.manifest_var_free_positions = []
        else:
            self.manifest_var_mode = "diag"
            self.manifest_var_template = jnp.zeros((spec.n_manifest, spec.n_manifest))
            self.manifest_var_free_positions = list(range(spec.n_manifest))

    def assemble_drift(
        self, drift_diag_pop: jnp.ndarray, drift_offdiag_pop: jnp.ndarray
    ) -> jnp.ndarray:
        """Build drift matrix from diagonal and off-diagonal parameter values."""
        drift_diag = -jnp.abs(drift_diag_pop)
        if self.ti_mask is not None:
            drift_diag = jnp.where(self.ti_mask, -1e-6, drift_diag)
        drift = jnp.diag(drift_diag)
        for idx, (i, j) in enumerate(self.offdiag_positions):
            drift = drift.at[i, j].set(drift_offdiag_pop[idx])
        return drift

    def assemble_diffusion(
        self, diff_diag: jnp.ndarray, diff_lower: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Build diffusion Cholesky from diagonal (and optional lower triangle)."""
        diffusion = jnp.diag(diff_diag)
        if diff_lower is not None:
            lower_idx = 0
            for i in range(self.n_latent):
                for j in range(i):
                    diffusion = diffusion.at[i, j].set(diff_lower[lower_idx])
                    lower_idx += 1
        if self.ti_mask is not None:
            diag_vals = jnp.diag(diffusion)
            new_diag = jnp.where(self.ti_mask, 1e-6, diag_vals)
            diffusion = diffusion - jnp.diag(diag_vals) + jnp.diag(new_diag)
        return diffusion

    def assemble_t0_cov(
        self,
        t0_diag: jnp.ndarray,
        t0_correlation: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build initial-state covariance from standard deviations and correlations."""
        std = jnp.asarray(t0_diag)
        corr = jnp.eye(self.n_latent, dtype=std.dtype)
        if t0_correlation is not None:
            for idx, (row, col) in enumerate(self.t0_correlation_positions):
                corr = corr.at[row, col].set(t0_correlation[idx])
                corr = corr.at[col, row].set(t0_correlation[idx])
        cov = corr * (std[:, None] * std[None, :])
        return 0.5 * (cov + cov.T)

    def assemble_lambda(self, free_loadings: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build lambda (factor loading) matrix.

        Handles all three modes (template+mask, fixed, legacy) via the
        pre-computed template and free positions from __init__.
        """
        lam = self.lambda_template
        if free_loadings is not None and len(self.lambda_free_positions) > 0:
            for idx, (i, j) in enumerate(self.lambda_free_positions):
                lam = lam.at[i, j].set(free_loadings[idx])
        return lam

    def assemble_manifest_chol(self, free_diag: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build manifest-noise Cholesky from a template and sparse free diagonal."""
        manifest_chol = self.manifest_var_template
        if free_diag is not None and len(self.manifest_var_free_positions) > 0:
            for idx, manifest_idx in enumerate(self.manifest_var_free_positions):
                manifest_chol = manifest_chol.at[manifest_idx, manifest_idx].set(free_diag[idx])
        return manifest_chol
