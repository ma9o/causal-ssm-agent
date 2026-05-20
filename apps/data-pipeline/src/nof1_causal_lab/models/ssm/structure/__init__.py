"""Block-level SSM structure specs and assembly helpers."""

from nof1_causal_lab.models.ssm.structure.blocks import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
    default_diffusion_block,
    default_input_effect_block,
    default_lambda_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
)

__all__ = [
    "DiffusionBlockSpec",
    "ManifestCholBlockSpec",
    "SparseMatrixBlockSpec",
    "SparseVectorBlockSpec",
    "T0CholBlockSpec",
    "default_diffusion_block",
    "default_input_effect_block",
    "default_lambda_block",
    "default_manifest_chol_block",
    "default_manifest_means_block",
    "default_static_state_sd_block",
    "default_t0_chol_block",
    "default_t0_means_block",
]
