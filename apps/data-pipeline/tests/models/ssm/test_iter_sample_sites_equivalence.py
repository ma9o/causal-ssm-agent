"""Equivalence gate: spec.iter_sample_sites() must match build_site_registry.

This test is the verification gate for step 1 of the block-spec refactor.
Once green and stable, build_site_registry's hardcoded ``core_site_specs``
table can be deleted (step 3); until then both encodings must agree.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.models.ssm import SSMSpec
from nof1_causal_lab.models.ssm.parameterization import build_site_registry
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from tests.ssm_test_utils import (
    default_diffusion_block,
    default_input_effect_block,
    default_lambda_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
    full_structural_dense_drift_spec,
    structural_dense_drift_spec,
)


def _full_default_spec(n_latent: int = 2, n_manifest: int = 2) -> SSMSpec:
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=full_structural_dense_drift_spec(n_latent),
        diffusion_block=default_diffusion_block(n_latent),
        lambda_block=default_lambda_block(n_manifest, n_latent),
        manifest_means_block=default_manifest_means_block(n_manifest),
        manifest_chol_block=default_manifest_chol_block(n_manifest),
        t0_means_block=default_t0_means_block(n_latent),
        t0_chol_block=default_t0_chol_block(n_latent),
        input_effect_block=default_input_effect_block(n_latent),
        static_state_sd_block=default_static_state_sd_block(),
    )


def _sparse_spec_with_inputs_and_static(n_latent: int = 3) -> SSMSpec:
    n_manifest = 4
    n_input = 2
    n_static = 1
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=structural_dense_drift_spec(
            n_latent=n_latent,
            drift_diag_mask=np.ones(n_latent, dtype=bool),
            drift_offdiag_mask=np.eye(n_latent, dtype=bool)
            ^ np.ones((n_latent, n_latent), dtype=bool),
            drift_template=jnp.zeros((n_latent, n_latent)),
            cint_mask=np.ones(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent),
        ),
        diffusion_block=default_diffusion_block(n_latent),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            mask=np.ones((n_manifest, n_latent), dtype=bool),
            template=jnp.zeros((n_manifest, n_latent)),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=SparseVectorBlockSpec(
            n=n_manifest,
            mask=np.ones(n_manifest, dtype=bool),
            template=jnp.zeros(n_manifest),
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=default_manifest_chol_block(n_manifest),
        t0_means_block=default_t0_means_block(n_latent),
        t0_chol_block=default_t0_chol_block(n_latent),
        input_effect_block=SparseMatrixBlockSpec(
            n_rows=n_latent,
            n_cols=n_input,
            mask=np.ones((n_latent, n_input), dtype=bool),
            template=jnp.zeros((n_latent, n_input)),
            free_site_name="input_effect_free",
            det_site_name="input_effect",
            support=SupportClass.REAL,
            site_kind=SiteKind.INPUT_EFFECT,
            assembly_group="input_effect",
            fixed_spec_field="input_effect",
            priors_field="input_effect",
        ),
        static_state_sd_block=SparseVectorBlockSpec(
            n=n_static,
            mask=np.ones(n_static, dtype=bool),
            template=jnp.zeros(n_static),
            free_site_name="static_state_sd_free",
            det_site_name="static_state_sds",
            support=SupportClass.POSITIVE,
            site_kind=SiteKind.STATIC_STATE_SD,
            assembly_group="t0",
            fixed_spec_field="static_state_sds",
            priors_field="static_state_sd",
        ),
        static_factor_loadings=jnp.zeros((n_latent, n_static)),
        input_names=[f"input_{i}" for i in range(n_input)],
        input_source_indicators=[f"input_{i}" for i in range(n_input)],
        input_scales=[1.0] * n_input,
        input_missing_policies=["zero"] * n_input,
        static_factor_names=[f"factor_{i}" for i in range(n_static)],
    )


def _all_fixed_spec(n_latent: int = 2) -> SSMSpec:
    n_manifest = 2
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=structural_dense_drift_spec(
            n_latent=n_latent,
            drift_diag_mask=np.zeros(n_latent, dtype=bool),
            drift_offdiag_mask=np.zeros((n_latent, n_latent), dtype=bool),
            drift_template=jnp.zeros((n_latent, n_latent)),
            cint_mask=np.zeros(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent),
        ),
        diffusion_block=DiffusionBlockSpec(
            n_latent=n_latent,
            diffusion_chol_mask=np.zeros((n_latent, n_latent), dtype=bool),
            diffusion_chol_template=jnp.eye(n_latent),
        ),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            mask=np.zeros((n_manifest, n_latent), dtype=bool),
            template=jnp.eye(n_manifest, n_latent),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=SparseVectorBlockSpec(
            n=n_manifest,
            mask=np.zeros(n_manifest, dtype=bool),
            template=jnp.zeros(n_manifest),
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=n_manifest,
            diag_mask=np.zeros(n_manifest, dtype=bool),
            template=jnp.eye(n_manifest),
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            mask=np.zeros(n_latent, dtype=bool),
            template=jnp.zeros(n_latent),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        ),
        t0_chol_block=T0CholBlockSpec(
            n_latent=n_latent,
            diag_mask=np.zeros(n_latent, dtype=bool),
            correlation_mask=np.zeros((n_latent, n_latent), dtype=bool),
            template=jnp.eye(n_latent),
        ),
        input_effect_block=default_input_effect_block(n_latent),
        static_state_sd_block=default_static_state_sd_block(),
    )


def _descriptor_fields(s):
    return (
        s.name,
        tuple(s.shape),
        s.support,
        s.assembly_group,
        s.site_kind,
        s.transform_kind,
        s.deterministic_name,
        s.fixed_spec_field,
        s.priors_field,
        s.runtime_prior_key,
        s.is_runtime_prior_controlled,
    )


def _compare_block_owned_with_registry(spec: SSMSpec) -> None:
    # registry has likelihood extras; restrict to the dense-linear core sites.
    legacy = [s for s in build_site_registry(spec) if s.assembly_group != "likelihood"]
    block_owned = sorted(spec.iter_sample_sites(), key=lambda s: s.name)

    legacy_by_name = {s.name: s for s in legacy}
    new_by_name = {s.name: s for s in block_owned}
    assert set(legacy_by_name) == set(new_by_name), (
        f"site-name mismatch: legacy={sorted(legacy_by_name)}, new={sorted(new_by_name)}"
    )
    for name in legacy_by_name:
        assert _descriptor_fields(legacy_by_name[name]) == _descriptor_fields(new_by_name[name]), (
            f"descriptor field mismatch for site {name!r}"
        )


def test_full_default_spec_equivalence():
    _compare_block_owned_with_registry(_full_default_spec())


def test_full_default_3x3_equivalence():
    _compare_block_owned_with_registry(_full_default_spec(n_latent=3, n_manifest=4))


def test_sparse_spec_with_inputs_and_static_equivalence():
    _compare_block_owned_with_registry(_sparse_spec_with_inputs_and_static())


def test_all_fixed_spec_yields_no_sites():
    spec = _all_fixed_spec()
    assert list(spec.iter_sample_sites()) == []
    _compare_block_owned_with_registry(spec)
