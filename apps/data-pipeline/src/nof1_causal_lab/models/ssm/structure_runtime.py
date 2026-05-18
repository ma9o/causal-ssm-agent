"""Pure-JAX structural runtime for assembled SSM parameters.

Single source of truth for compiled matrix structure after ``SSMSpec``
translation. It owns:
- Fixed numeric templates
- Canonical free-entry positions and flat-index lookups
- Matrix/vector assembly from sampled free values

Used by both:
- ``SSMModel._sample_*`` for single-sample NumPyro execution
- deterministic reconstruction utilities that ``vmap`` over draws
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from causal_ssm_agent.models.ssm.covariance_utils import symmetrize

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


def _cast_like(values: jnp.ndarray | None, template: jnp.ndarray) -> jnp.ndarray | None:
    """Cast free parameter vectors to the assembled template dtype."""
    if values is None:
        return None
    return jnp.asarray(values, dtype=template.dtype)


class SSMStructureRuntime:
    """Canonical runtime structure derived once from ``SSMSpec``.

    Downstream runtime consumers should read templates, free-entry positions,
    and flat-index mappings from this object rather than re-deriving them from
    raw mask/template fields on ``SSMSpec``.
    """

    def __init__(self, spec: SSMSpec) -> None:
        self.n_latent = spec.n_latent
        self.n_manifest = spec.n_manifest
        self.stability_margin = float(spec.stability_margin)

        self.drift_template = jnp.array(spec.drift)
        self.drift_base_decay_positions: list[int] = [
            idx for idx in range(spec.n_latent) if bool(spec.drift_diag_mask[idx])
        ]
        self.drift_base_decay_index = {
            latent_idx: flat_idx
            for flat_idx, latent_idx in enumerate(self.drift_base_decay_positions)
        }

        # Drift: pre-compute off-diagonal positions from mask
        self.offdiag_positions: list[tuple[int, int]] = []
        for i in range(spec.n_latent):
            for j in range(spec.n_latent):
                if i != j and spec.drift_offdiag_mask[i, j]:
                    self.offdiag_positions.append((i, j))
        self.offdiag_index = {
            position: flat_idx for flat_idx, position in enumerate(self.offdiag_positions)
        }

        self.ti_mask: jnp.ndarray | None = (
            jnp.array(spec.time_invariant_mask) if spec.time_invariant_mask is not None else None
        )
        self.cint_template = jnp.array(spec.cint)
        self.cint_free_positions: list[int] = [
            idx for idx in range(spec.n_latent) if bool(spec.cint_mask[idx])
        ]
        self.cint_free_index = {
            latent_idx: flat_idx for flat_idx, latent_idx in enumerate(self.cint_free_positions)
        }
        self.static_factor_loadings = jnp.array(spec.static_factor_loadings)
        self.static_factor_names = list(spec.static_factor_names or [])
        self.static_state_sd_free_positions: list[int] = [
            idx
            for idx in range(self.static_factor_loadings.shape[1])
            if bool(spec.static_state_sd_mask[idx])
        ]
        self.static_state_sd_free_index = {
            factor_idx: flat_idx
            for flat_idx, factor_idx in enumerate(self.static_state_sd_free_positions)
        }
        self.static_factor_name_index = {
            name: idx for idx, name in enumerate(self.static_factor_names)
        }
        self.static_state_sd_template = jnp.array(spec.static_state_sds)
        self.input_effect_template = jnp.array(spec.input_effect)
        self.input_effect_positions: list[tuple[int, int]] = [
            (state_idx, input_idx)
            for state_idx in range(spec.n_latent)
            for input_idx in range(len(spec.input_names or []))
            if bool(spec.input_effect_mask[state_idx, input_idx])
        ]
        self.input_effect_index = {
            position: flat_idx for flat_idx, position in enumerate(self.input_effect_positions)
        }
        self.diffusion_chol_template = jnp.array(spec.diffusion_chol)
        self.diffusion_diag_positions: list[int] = [
            idx for idx in range(spec.n_latent) if bool(spec.diffusion_chol_mask[idx, idx])
        ]
        self.diffusion_diag_index = {
            latent_idx: flat_idx
            for flat_idx, latent_idx in enumerate(self.diffusion_diag_positions)
        }
        self.diffusion_lower_positions: list[tuple[int, int]] = [
            (row, col)
            for row in range(spec.n_latent)
            for col in range(row)
            if bool(spec.diffusion_chol_mask[row, col])
        ]
        self.diffusion_lower_index = {
            position: flat_idx for flat_idx, position in enumerate(self.diffusion_lower_positions)
        }

        # Lambda: explicit template plus free-loading mask
        if isinstance(spec.lambda_mat, str):
            raise ValueError("SSMStructureRuntime requires a canonical loading template.")
        self.lambda_template = jnp.array(spec.lambda_mat)
        self.lambda_free_positions: list[tuple[int, int]] = [
            (i, j)
            for i in range(spec.n_manifest)
            for j in range(spec.n_latent)
            if spec.lambda_mask[i, j]
        ]
        self.lambda_free_index = {
            position: flat_idx for flat_idx, position in enumerate(self.lambda_free_positions)
        }
        self.manifest_means_template = jnp.array(spec.manifest_means)
        self.manifest_means_free_positions: list[int] = [
            idx for idx in range(spec.n_manifest) if bool(spec.manifest_means_mask[idx])
        ]
        self.manifest_means_free_index = {
            manifest_idx: flat_idx
            for flat_idx, manifest_idx in enumerate(self.manifest_means_free_positions)
        }

        # Manifest variance: explicit template plus sparse free diagonal positions.
        self.manifest_chol_template = jnp.array(spec.manifest_chol)
        self.manifest_var_free_positions: list[int] = [
            idx for idx in range(spec.n_manifest) if bool(spec.manifest_chol_diag_mask[idx])
        ]
        self.manifest_var_free_index = {
            manifest_idx: flat_idx
            for flat_idx, manifest_idx in enumerate(self.manifest_var_free_positions)
        }

        self.t0_means_template = jnp.array(spec.t0_means)
        self.t0_means_free_positions: list[int] = [
            idx for idx in range(spec.n_latent) if bool(spec.t0_means_mask[idx])
        ]
        self.t0_means_free_index = {
            latent_idx: flat_idx for flat_idx, latent_idx in enumerate(self.t0_means_free_positions)
        }

        # Initial-state covariance: explicit template plus free diagonal/correlation masks.
        self.t0_chol_template = jnp.array(spec.t0_chol)
        self.t0_diag_free_positions: list[int] = [
            idx for idx in range(spec.n_latent) if bool(spec.t0_chol_diag_mask[idx])
        ]
        self.t0_diag_free_index = {
            latent_idx: flat_idx for flat_idx, latent_idx in enumerate(self.t0_diag_free_positions)
        }
        self.t0_correlation_positions = lower_triangle_positions(
            spec.n_latent,
            spec.t0_correlation_mask,
        )
        self.t0_correlation_index = {
            position: flat_idx for flat_idx, position in enumerate(self.t0_correlation_positions)
        }
        self.n_drift_base_decay = len(self.drift_base_decay_positions)
        self.n_drift_offdiag = len(self.offdiag_positions)
        self.n_cint = len(self.cint_free_positions)
        self.n_static_state_sd = len(self.static_state_sd_free_positions)
        self.n_input_effect = len(self.input_effect_positions)
        self.n_diffusion_diag = len(self.diffusion_diag_positions)
        self.n_diffusion_lower = len(self.diffusion_lower_positions)
        self.n_lambda_free = len(self.lambda_free_positions)
        self.n_manifest_means = len(self.manifest_means_free_positions)
        self.n_manifest_var_diag = len(self.manifest_var_free_positions)
        self.n_t0_means = len(self.t0_means_free_positions)
        self.n_t0_diag = len(self.t0_diag_free_positions)
        self.n_t0_correlation = len(self.t0_correlation_positions)

        self.manifest_cov_template = self.manifest_chol_template @ self.manifest_chol_template.T
        self.t0_cov_template = self.t0_chol_template @ self.t0_chol_template.T
        self.t0_base_std = jnp.sqrt(jnp.clip(jnp.diag(self.t0_cov_template), min=0.0))
        denom = self.t0_base_std[:, None] * self.t0_base_std[None, :]
        self.t0_base_corr = jnp.where(
            denom > 0,
            self.t0_cov_template / denom,
            jnp.eye(spec.n_latent, dtype=self.t0_cov_template.dtype),
        )
        self.t0_base_corr = symmetrize(self.t0_base_corr)
        self.t0_base_corr = self.t0_base_corr.at[jnp.diag_indices(spec.n_latent)].set(1.0)

    def drift_support_mask(self) -> jnp.ndarray:
        """Return potential nonzero drift support from fixed template and free entries."""
        fixed_nonzero = jnp.abs(self.drift_template) > 0
        free = jnp.zeros_like(fixed_nonzero, dtype=bool)
        for latent_idx in self.drift_base_decay_positions:
            free = free.at[latent_idx, latent_idx].set(True)
        for row, col in self.offdiag_positions:
            free = free.at[row, col].set(True)
        return fixed_nonzero | free

    def loading_support_mask(self) -> jnp.ndarray:
        """Return potential nonzero loading support from fixed template and free entries."""
        fixed_nonzero = jnp.abs(self.lambda_template) > 0
        free = jnp.zeros_like(fixed_nonzero, dtype=bool)
        for manifest_idx, latent_idx in self.lambda_free_positions:
            free = free.at[manifest_idx, latent_idx].set(True)
        return fixed_nonzero | free

    def assemble_drift(
        self,
        drift_base_decay_free: jnp.ndarray | None = None,
        drift_offdiag_free: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build drift matrix from base decay and off-diagonal parameter values."""
        drift = self.drift_template
        drift_base_decay_free = _cast_like(drift_base_decay_free, drift)
        drift_offdiag_free = _cast_like(drift_offdiag_free, drift)
        if drift_offdiag_free is not None:
            for idx, (i, j) in enumerate(self.offdiag_positions):
                drift = drift.at[i, j].set(drift_offdiag_free[idx])
        if drift_base_decay_free is not None:
            offdiag_drift = drift - jnp.diag(jnp.diag(drift))
            row_abs = jnp.sum(jnp.abs(offdiag_drift), axis=1)
            margin = jnp.asarray(self.stability_margin, dtype=drift.dtype)
            for idx, latent_idx in enumerate(self.drift_base_decay_positions):
                base_decay = drift_base_decay_free[idx]
                drift = drift.at[latent_idx, latent_idx].set(
                    -(base_decay + row_abs[latent_idx] + margin)
                )
        if self.ti_mask is not None:
            diag_vals = jnp.diag(drift)
            new_diag = jnp.where(self.ti_mask, jnp.asarray(-1e-6, dtype=drift.dtype), diag_vals)
            drift = drift - jnp.diag(diag_vals) + jnp.diag(new_diag)
        return drift

    def assemble_diffusion(
        self,
        diff_diag: jnp.ndarray | None = None,
        diff_lower: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build diffusion Cholesky from a template and sparse free entries."""
        diffusion = self.diffusion_chol_template
        diff_diag = _cast_like(diff_diag, diffusion)
        diff_lower = _cast_like(diff_lower, diffusion)
        if diff_diag is not None:
            for idx, latent_idx in enumerate(self.diffusion_diag_positions):
                diffusion = diffusion.at[latent_idx, latent_idx].set(diff_diag[idx])
        if diff_lower is not None:
            for idx, (row, col) in enumerate(self.diffusion_lower_positions):
                diffusion = diffusion.at[row, col].set(diff_lower[idx])
        if self.ti_mask is not None:
            diag_vals = jnp.diag(diffusion)
            new_diag = jnp.where(self.ti_mask, jnp.asarray(1e-6, dtype=diffusion.dtype), diag_vals)
            diffusion = diffusion - jnp.diag(diag_vals) + jnp.diag(new_diag)
        return diffusion

    def assemble_cint(self, free_cint: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build continuous intercept from a template and sparse free entries."""
        cint = self.cint_template
        free_cint = _cast_like(free_cint, cint)
        if free_cint is not None:
            for idx, latent_idx in enumerate(self.cint_free_positions):
                cint = cint.at[latent_idx].set(free_cint[idx])
        return cint

    def assemble_input_effect(
        self,
        free_input_effect: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build known-input transition effects from a template and sparse free entries."""
        input_effect = self.input_effect_template
        free_input_effect = _cast_like(free_input_effect, input_effect)
        if free_input_effect is not None:
            for idx, (state_idx, input_idx) in enumerate(self.input_effect_positions):
                input_effect = input_effect.at[state_idx, input_idx].set(free_input_effect[idx])
        return input_effect

    def assemble_t0_cov(
        self,
        t0_diag: jnp.ndarray | None = None,
        t0_correlation: jnp.ndarray | None = None,
        static_state_sds: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build initial-state covariance from a template and sparse free entries."""
        std = self.t0_base_std
        t0_diag = _cast_like(t0_diag, std)
        if t0_diag is not None:
            for idx, latent_idx in enumerate(self.t0_diag_free_positions):
                std = std.at[latent_idx].set(t0_diag[idx])
        corr = self.t0_base_corr
        t0_correlation = _cast_like(t0_correlation, corr)
        if t0_correlation is not None:
            for idx, (row, col) in enumerate(self.t0_correlation_positions):
                corr = corr.at[row, col].set(t0_correlation[idx])
                corr = corr.at[col, row].set(t0_correlation[idx])
        cov = corr * (std[:, None] * std[None, :])
        factor_sds = self.static_state_sd_template
        static_state_sds = _cast_like(static_state_sds, factor_sds)
        if static_state_sds is not None:
            for idx, factor_idx in enumerate(self.static_state_sd_free_positions):
                factor_sds = factor_sds.at[factor_idx].set(static_state_sds[idx])
        if factor_sds.size:
            factor_cov = (
                self.static_factor_loadings
                @ jnp.diag(factor_sds**2)
                @ self.static_factor_loadings.T
            )
            cov = cov + factor_cov
        return symmetrize(cov)

    def assemble_static_state_sds(
        self,
        free_sds: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Build compiled baseline-factor scales from a sparse free vector."""
        static_sds = self.static_state_sd_template
        free_sds = _cast_like(free_sds, static_sds)
        if free_sds is not None:
            for idx, factor_idx in enumerate(self.static_state_sd_free_positions):
                static_sds = static_sds.at[factor_idx].set(free_sds[idx])
        return static_sds

    def assemble_lambda(self, free_loadings: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build lambda (factor loading) matrix from template + explicit mask."""
        lam = self.lambda_template
        free_loadings = _cast_like(free_loadings, lam)
        if free_loadings is not None and len(self.lambda_free_positions) > 0:
            for idx, (i, j) in enumerate(self.lambda_free_positions):
                lam = lam.at[i, j].set(free_loadings[idx])
        return lam

    def assemble_manifest_means(self, free_means: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build manifest means from a template and sparse free entries."""
        manifest_means = self.manifest_means_template
        free_means = _cast_like(free_means, manifest_means)
        if free_means is not None:
            for idx, manifest_idx in enumerate(self.manifest_means_free_positions):
                manifest_means = manifest_means.at[manifest_idx].set(free_means[idx])
        return manifest_means

    def assemble_manifest_chol(self, free_diag: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build manifest-noise Cholesky from a template and sparse free diagonal."""
        manifest_chol = self.manifest_chol_template
        free_diag = _cast_like(free_diag, manifest_chol)
        if free_diag is not None and len(self.manifest_var_free_positions) > 0:
            for idx, manifest_idx in enumerate(self.manifest_var_free_positions):
                manifest_chol = manifest_chol.at[manifest_idx, manifest_idx].set(free_diag[idx])
        return manifest_chol

    def assemble_t0_means(self, free_means: jnp.ndarray | None = None) -> jnp.ndarray:
        """Build initial-state means from a template and sparse free entries."""
        t0_means = self.t0_means_template
        free_means = _cast_like(free_means, t0_means)
        if free_means is not None:
            for idx, latent_idx in enumerate(self.t0_means_free_positions):
                t0_means = t0_means.at[latent_idx].set(free_means[idx])
        return t0_means
