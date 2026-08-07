"""Hydration of serialized SSM specifications into executable core values."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.compile.contracts import SerializedSSMSpec
    from nof1_causal_lab.models.ssm.model import SSMSpec


def deserialize_ssm_spec(payload: SerializedSSMSpec) -> SSMSpec:
    """Restore an SSMSpec from a serialized compiled artifact."""
    from nof1_causal_lab.models.ssm.dynamics.serialization import dynamics_spec_from_dict
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.structure import (
        DiffusionBlockSpec,
        ManifestCholBlockSpec,
        SparseMatrixBlockSpec,
        SparseVectorBlockSpec,
        T0CholBlockSpec,
    )

    def _bool_array(block: UncheckedJsonObject, key: str) -> np.ndarray:
        return np.asarray(block[key], dtype=bool)

    def _float_array(block: UncheckedJsonObject, key: str) -> jnp.ndarray:
        return jnp.asarray(block[key], dtype=jnp.float32)

    def _optional_bool_array(block: UncheckedJsonObject, key: str) -> np.ndarray | None:
        value = block.get(key)
        return None if value is None else np.asarray(value, dtype=bool)

    def _diffusion_block(block: UncheckedJsonObject) -> DiffusionBlockSpec:
        return DiffusionBlockSpec(
            n_latent=int(block["n_latent"]),
            diffusion_chol_support=_bool_array(block, "diffusion_chol_support"),
            diffusion_chol_template=_float_array(block, "diffusion_chol_template"),
            time_invariant_mask=_optional_bool_array(block, "time_invariant_mask"),
        )

    def _sparse_matrix_block(block: UncheckedJsonObject) -> SparseMatrixBlockSpec:
        return SparseMatrixBlockSpec(
            n_rows=int(block["n_rows"]),
            n_cols=int(block["n_cols"]),
            free_support=_bool_array(block, "free_support"),
            template=_float_array(block, "template"),
            free_site_name=str(block["free_site_name"]),
            det_site_name=str(block["det_site_name"]),
            support=SupportClass(block["support"]),
            site_kind=SiteKind(block["site_kind"]),
            assembly_group=str(block["assembly_group"]),
            fixed_spec_field=str(block["fixed_spec_field"]),
            priors_field=str(block["priors_field"]),
        )

    def _sparse_vector_block(block: UncheckedJsonObject) -> SparseVectorBlockSpec:
        return SparseVectorBlockSpec(
            n=int(block["n"]),
            free_support=_bool_array(block, "free_support"),
            template=_float_array(block, "template"),
            free_site_name=str(block["free_site_name"]),
            det_site_name=str(block["det_site_name"]),
            support=SupportClass(block["support"]),
            site_kind=SiteKind(block["site_kind"]),
            assembly_group=str(block["assembly_group"]),
            fixed_spec_field=str(block["fixed_spec_field"]),
            priors_field=str(block["priors_field"]),
        )

    def _manifest_chol_block(block: UncheckedJsonObject) -> ManifestCholBlockSpec:
        return ManifestCholBlockSpec(
            n_manifest=int(block["n_manifest"]),
            diag_support=_bool_array(block, "diag_support"),
            template=_float_array(block, "template"),
        )

    def _t0_chol_block(block: UncheckedJsonObject) -> T0CholBlockSpec:
        return T0CholBlockSpec(
            n_latent=int(block["n_latent"]),
            diag_support=_bool_array(block, "diag_support"),
            correlation_support=_bool_array(block, "correlation_support"),
            template=_float_array(block, "template"),
        )

    return SSMSpec(
        n_latent=payload.n_latent,
        n_manifest=payload.n_manifest,
        dynamics_spec=dynamics_spec_from_dict(payload.dynamics_spec),
        diffusion_block=_diffusion_block(payload.diffusion_block),
        lambda_block=_sparse_matrix_block(payload.lambda_block),
        manifest_means_block=_sparse_vector_block(payload.manifest_means_block),
        manifest_chol_block=_manifest_chol_block(payload.manifest_chol_block),
        t0_means_block=_sparse_vector_block(payload.t0_means_block),
        t0_chol_block=_t0_chol_block(payload.t0_chol_block),
        input_effect_block=_sparse_matrix_block(payload.input_effect_block),
        static_state_sd_block=_sparse_vector_block(payload.static_state_sd_block),
        static_factor_loadings=jnp.asarray(payload.static_factor_loadings, dtype=jnp.float32),
        diffusion_dists=list(payload.diffusion_dists),
        manifest_dists=list(payload.manifest_dists),
        manifest_level_counts=payload.manifest_level_counts,
        manifest_links=payload.manifest_links,
        manifest_standardized=payload.manifest_standardized,
        manifest_cat_anchor=payload.manifest_cat_anchor,
        latent_names=payload.latent_names,
        manifest_names=payload.manifest_names,
        input_names=payload.input_names,
        input_source_indicators=payload.input_source_indicators,
        input_scales=payload.input_scales,
        input_missing_policies=(
            [str(policy) for policy in payload.input_missing_policies]
            if payload.input_missing_policies is not None
            else None
        ),
        input_lagged=payload.input_lagged,
        static_factor_names=payload.static_factor_names,
    )


__all__ = ["deserialize_ssm_spec"]
