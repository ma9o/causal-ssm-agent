"""Parameter layout for block-based SSM specs.

``SSMParameterLayout`` is intentionally narrow: it derives sample-site
sizes and flat-index lookup maps from the block specs on ``SSMSpec``.
It does not own templates and it does not assemble matrices. Assembly
belongs to the block that owns the parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.dynamics.composite import (
    StructuralDenseLinearSpec,
    StructuralInterceptSpec,
)
from nof1_causal_lab.models.ssm.structure.assembly import (
    chol_diag_positions,
    dense_vector_positions,
    drift_base_decay_positions,
    drift_offdiag_positions,
    rect_matrix_positions,
    strict_lower_positions,
)

if TYPE_CHECKING:
    import numpy as np

    from nof1_causal_lab.models.ssm.model import SSMSpec


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


@dataclass(frozen=True)
class SSMParameterLayout:
    """Derived structural layout for sample sites.

    The dense-linear drift sites are present only when ``spec.drift_spec``
    starts with the canonical structural drift/intercept components.
    Nonlinear drift specs sample their component parameters through the
    composite path, so the dense-linear drift positions are empty there.
    """

    spec: SSMSpec
    drift_base_decay_positions: list[int]
    offdiag_positions: list[tuple[int, int]]
    drift_base_decay_index: dict[int, int]
    offdiag_index: dict[tuple[int, int], int]
    cint_free_positions: list[int]
    cint_free_index: dict[int, int]
    static_state_sd_free_positions: list[int]
    static_state_sd_free_index: dict[int, int]
    static_factor_name_index: dict[str, int]
    input_effect_positions: list[tuple[int, int]]
    input_effect_index: dict[tuple[int, int], int]
    diffusion_diag_positions: list[int]
    diffusion_diag_index: dict[int, int]
    diffusion_lower_positions: list[tuple[int, int]]
    diffusion_lower_index: dict[tuple[int, int], int]
    lambda_free_positions: list[tuple[int, int]]
    lambda_free_index: dict[tuple[int, int], int]
    manifest_means_free_positions: list[int]
    manifest_means_free_index: dict[int, int]
    manifest_var_free_positions: list[int]
    manifest_var_free_index: dict[int, int]
    t0_means_free_positions: list[int]
    t0_means_free_index: dict[int, int]
    t0_diag_free_positions: list[int]
    t0_diag_free_index: dict[int, int]
    t0_correlation_positions: list[tuple[int, int]]
    t0_correlation_index: dict[tuple[int, int], int]

    @classmethod
    def from_spec(cls, spec: SSMSpec) -> SSMParameterLayout:
        components = spec.drift_spec.components
        drift_component = (
            components[0]
            if components and isinstance(components[0], StructuralDenseLinearSpec)
            else None
        )
        cint_component = (
            components[1]
            if len(components) >= 2 and isinstance(components[1], StructuralInterceptSpec)
            else None
        )

        if drift_component is None:
            drift_base_positions: list[int] = []
            offdiag_positions_list: list[tuple[int, int]] = []
        else:
            drift_base_positions = drift_base_decay_positions(
                drift_component.drift_diag_mask,
                spec.n_latent,
            )
            offdiag_positions_list = drift_offdiag_positions(
                drift_component.drift_offdiag_mask,
                spec.n_latent,
            )

        cint_positions = (
            dense_vector_positions(cint_component.cint_mask, spec.n_latent)
            if cint_component is not None
            else []
        )
        n_static_factor = int(jnp.asarray(spec.static_factor_loadings).shape[1])
        static_positions = [
            idx
            for idx in range(n_static_factor)
            if bool(spec.static_state_sd_block.mask[idx])
        ]
        input_positions = rect_matrix_positions(
            spec.input_effect_block.mask,
            spec.n_latent,
            spec.input_effect_block.n_cols,
        )
        diffusion_diag = chol_diag_positions(
            spec.diffusion_block.diffusion_chol_mask,
            spec.n_latent,
        )
        diffusion_lower = strict_lower_positions(
            spec.diffusion_block.diffusion_chol_mask,
            spec.n_latent,
        )
        lambda_positions = rect_matrix_positions(
            spec.lambda_block.mask,
            spec.n_manifest,
            spec.n_latent,
        )
        manifest_mean_positions = dense_vector_positions(
            spec.manifest_means_block.mask,
            spec.n_manifest,
        )
        manifest_var_positions = dense_vector_positions(
            spec.manifest_chol_block.diag_mask,
            spec.n_manifest,
        )
        t0_mean_positions = dense_vector_positions(
            spec.t0_means_block.mask,
            spec.n_latent,
        )
        t0_diag_positions_list = dense_vector_positions(
            spec.t0_chol_block.diag_mask,
            spec.n_latent,
        )
        t0_corr_positions = lower_triangle_positions(
            spec.n_latent,
            spec.t0_chol_block.correlation_mask,
        )

        return cls(
            spec=spec,
            drift_base_decay_positions=drift_base_positions,
            offdiag_positions=offdiag_positions_list,
            drift_base_decay_index={
                latent_idx: flat_idx
                for flat_idx, latent_idx in enumerate(drift_base_positions)
            },
            offdiag_index={
                position: flat_idx
                for flat_idx, position in enumerate(offdiag_positions_list)
            },
            cint_free_positions=cint_positions,
            cint_free_index={
                latent_idx: flat_idx for flat_idx, latent_idx in enumerate(cint_positions)
            },
            static_state_sd_free_positions=static_positions,
            static_state_sd_free_index={
                factor_idx: flat_idx
                for flat_idx, factor_idx in enumerate(static_positions)
            },
            static_factor_name_index={
                name: idx for idx, name in enumerate(spec.static_factor_names or [])
            },
            input_effect_positions=input_positions,
            input_effect_index={
                position: flat_idx for flat_idx, position in enumerate(input_positions)
            },
            diffusion_diag_positions=diffusion_diag,
            diffusion_diag_index={
                latent_idx: flat_idx for flat_idx, latent_idx in enumerate(diffusion_diag)
            },
            diffusion_lower_positions=diffusion_lower,
            diffusion_lower_index={
                position: flat_idx for flat_idx, position in enumerate(diffusion_lower)
            },
            lambda_free_positions=lambda_positions,
            lambda_free_index={
                position: flat_idx for flat_idx, position in enumerate(lambda_positions)
            },
            manifest_means_free_positions=manifest_mean_positions,
            manifest_means_free_index={
                manifest_idx: flat_idx
                for flat_idx, manifest_idx in enumerate(manifest_mean_positions)
            },
            manifest_var_free_positions=manifest_var_positions,
            manifest_var_free_index={
                manifest_idx: flat_idx
                for flat_idx, manifest_idx in enumerate(manifest_var_positions)
            },
            t0_means_free_positions=t0_mean_positions,
            t0_means_free_index={
                latent_idx: flat_idx
                for flat_idx, latent_idx in enumerate(t0_mean_positions)
            },
            t0_diag_free_positions=t0_diag_positions_list,
            t0_diag_free_index={
                latent_idx: flat_idx
                for flat_idx, latent_idx in enumerate(t0_diag_positions_list)
            },
            t0_correlation_positions=t0_corr_positions,
            t0_correlation_index={
                position: flat_idx for flat_idx, position in enumerate(t0_corr_positions)
            },
        )

    @property
    def n_drift_base_decay(self) -> int:
        return len(self.drift_base_decay_positions)

    @property
    def n_drift_offdiag(self) -> int:
        return len(self.offdiag_positions)

    @property
    def n_cint(self) -> int:
        return len(self.cint_free_positions)

    @property
    def n_static_state_sd(self) -> int:
        return len(self.static_state_sd_free_positions)

    @property
    def n_input_effect(self) -> int:
        return len(self.input_effect_positions)

    @property
    def n_diffusion_diag(self) -> int:
        return len(self.diffusion_diag_positions)

    @property
    def n_diffusion_lower(self) -> int:
        return len(self.diffusion_lower_positions)

    @property
    def n_lambda_free(self) -> int:
        return len(self.lambda_free_positions)

    @property
    def n_manifest_means(self) -> int:
        return len(self.manifest_means_free_positions)

    @property
    def n_manifest_var_diag(self) -> int:
        return len(self.manifest_var_free_positions)

    @property
    def n_t0_means(self) -> int:
        return len(self.t0_means_free_positions)

    @property
    def n_t0_diag(self) -> int:
        return len(self.t0_diag_free_positions)

    @property
    def n_t0_correlation(self) -> int:
        return len(self.t0_correlation_positions)
