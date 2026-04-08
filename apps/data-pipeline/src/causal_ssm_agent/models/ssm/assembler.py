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

        self.drift_template = jnp.array(spec.drift)
        self.drift_diag_positions: list[int] = [
            idx for idx in range(spec.n_latent) if bool(spec.drift_diag_mask[idx])
        ]

        # Drift: pre-compute off-diagonal positions from mask
        self.offdiag_positions: list[tuple[int, int]] = []
        for i in range(spec.n_latent):
            for j in range(spec.n_latent):
                if i != j and spec.drift_offdiag_mask[i, j]:
                    self.offdiag_positions.append((i, j))

        self.ti_mask: jnp.ndarray | None = (
            jnp.array(spec.time_invariant_mask) if spec.time_invariant_mask is not None else None
        )
        self.cint_template = jnp.array(spec.cint)
        self.cint_free_positions: list[int] = [
            idx for idx in range(spec.n_latent) if bool(spec.cint_mask[idx])
        ]
        self.diffusion_template = jnp.array(spec.diffusion_chol)
        self.diffusion_diag_positions: list[int] = [
            idx for idx in range(spec.n_latent) if bool(spec.diffusion_chol_mask[idx, idx])
        ]
        self.diffusion_lower_positions: list[tuple[int, int]] = [
            (row, col)
            for row in range(spec.n_latent)
            for col in range(row)
            if bool(spec.diffusion_chol_mask[row, col])
        ]

        # Lambda: explicit template plus free-loading mask
        if isinstance(spec.lambda_mat, str):
            raise ValueError("SSMAssembler requires a canonical loading template.")
        self.lambda_template = jnp.array(spec.lambda_mat)
        self.lambda_free_positions: list[tuple[int, int]] = [
            (i, j)
            for i in range(spec.n_manifest)
            for j in range(spec.n_latent)
            if spec.lambda_mask[i, j]
        ]
        self.manifest_means_template = jnp.array(spec.manifest_means)
        self.manifest_means_free_positions: list[int] = [
            idx for idx in range(spec.n_manifest) if bool(spec.manifest_means_mask[idx])
        ]

        # Manifest variance: explicit template plus sparse free diagonal positions.
        self.manifest_var_template = jnp.array(spec.manifest_chol)
        self.manifest_var_free_positions: list[int] = [
            idx for idx in range(spec.n_manifest) if bool(spec.manifest_chol_diag_mask[idx])
        ]

        self.t0_means_template = jnp.array(spec.t0_means)
        self.t0_means_free_positions: list[int] = [
            idx for idx in range(spec.n_latent) if bool(spec.t0_means_mask[idx])
        ]

        # Initial-state covariance: explicit template plus free diagonal/correlation masks.
        self.t0_var_template = jnp.array(spec.t0_chol)
        self.t0_diag_free_positions: list[int] = [
            idx for idx in range(spec.n_latent) if bool(spec.t0_chol_diag_mask[idx])
        ]
        self.t0_correlation_positions = lower_triangle_positions(
            spec.n_latent,
            spec.t0_correlation_mask,
        )
        t0_cov_template = self.t0_var_template @ self.t0_var_template.T
        self.t0_base_std = jnp.sqrt(jnp.clip(jnp.diag(t0_cov_template), a_min=0.0))
        denom = self.t0_base_std[:, None] * self.t0_base_std[None, :]
        self.t0_base_corr = jnp.where(
            denom > 0,
            t0_cov_template / denom,
            jnp.eye(spec.n_latent, dtype=t0_cov_template.dtype),
        )
        self.t0_base_corr = 0.5 * (self.t0_base_corr + self.t0_base_corr.T)
        self.t0_base_corr = self.t0_base_corr.at[jnp.diag_indices(spec.n_latent)].set(1.0)

    def assemble_drift(
        self,
        drift_diag_pop: jnp.ndarray | None = None,
        drift_offdiag_pop: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build drift matrix from diagonal and off-diagonal parameter values."""
        drift = self.drift_template
        if drift_diag_pop is not None:
            for idx, latent_idx in enumerate(self.drift_diag_positions):
                drift = drift.at[latent_idx, latent_idx].set(-jnp.abs(drift_diag_pop[idx]))
        if drift_offdiag_pop is not None:
            for idx, (i, j) in enumerate(self.offdiag_positions):
                drift = drift.at[i, j].set(drift_offdiag_pop[idx])
        if self.ti_mask is not None:
            diag_vals = jnp.diag(drift)
            new_diag = jnp.where(self.ti_mask, -1e-6, diag_vals)
            drift = drift - jnp.diag(diag_vals) + jnp.diag(new_diag)
        return drift

    def assemble_diffusion(
        self,
        diff_diag: jnp.ndarray | None = None,
        diff_lower: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build diffusion Cholesky from a template and sparse free entries."""
        diffusion = self.diffusion_template
        if diff_diag is not None:
            for idx, latent_idx in enumerate(self.diffusion_diag_positions):
                diffusion = diffusion.at[latent_idx, latent_idx].set(diff_diag[idx])
        if diff_lower is not None:
            for idx, (row, col) in enumerate(self.diffusion_lower_positions):
                diffusion = diffusion.at[row, col].set(diff_lower[idx])
        if self.ti_mask is not None:
            diag_vals = jnp.diag(diffusion)
            new_diag = jnp.where(self.ti_mask, 1e-6, diag_vals)
            diffusion = diffusion - jnp.diag(diag_vals) + jnp.diag(new_diag)
        return diffusion

    def assemble_cint(self, free_cint: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build continuous intercept from a template and sparse free entries."""
        cint = self.cint_template
        if free_cint is not None:
            for idx, latent_idx in enumerate(self.cint_free_positions):
                cint = cint.at[latent_idx].set(free_cint[idx])
        return cint

    def assemble_t0_cov(
        self,
        t0_diag: jnp.ndarray | None = None,
        t0_correlation: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build initial-state covariance from a template and sparse free entries."""
        std = self.t0_base_std
        if t0_diag is not None:
            for idx, latent_idx in enumerate(self.t0_diag_free_positions):
                std = std.at[latent_idx].set(t0_diag[idx])
        corr = self.t0_base_corr
        if t0_correlation is not None:
            for idx, (row, col) in enumerate(self.t0_correlation_positions):
                corr = corr.at[row, col].set(t0_correlation[idx])
                corr = corr.at[col, row].set(t0_correlation[idx])
        cov = corr * (std[:, None] * std[None, :])
        return 0.5 * (cov + cov.T)

    def assemble_lambda(self, free_loadings: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build lambda (factor loading) matrix from template + explicit mask."""
        lam = self.lambda_template
        if free_loadings is not None and len(self.lambda_free_positions) > 0:
            for idx, (i, j) in enumerate(self.lambda_free_positions):
                lam = lam.at[i, j].set(free_loadings[idx])
        return lam

    def assemble_manifest_means(self, free_means: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build manifest means from a template and sparse free entries."""
        manifest_means = self.manifest_means_template
        if free_means is not None:
            for idx, manifest_idx in enumerate(self.manifest_means_free_positions):
                manifest_means = manifest_means.at[manifest_idx].set(free_means[idx])
        return manifest_means

    def assemble_manifest_chol(self, free_diag: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build manifest-noise Cholesky from a template and sparse free diagonal."""
        manifest_chol = self.manifest_var_template
        if free_diag is not None and len(self.manifest_var_free_positions) > 0:
            for idx, manifest_idx in enumerate(self.manifest_var_free_positions):
                manifest_chol = manifest_chol.at[manifest_idx, manifest_idx].set(free_diag[idx])
        return manifest_chol

    def assemble_t0_means(self, free_means: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build initial-state means from a template and sparse free entries."""
        t0_means = self.t0_means_template
        if free_means is not None:
            for idx, latent_idx in enumerate(self.t0_means_free_positions):
                t0_means = t0_means.at[latent_idx].set(free_means[idx])
        return t0_means
