"""Tests for vector-field linearisation and structural drift specs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist

from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeSpec,
    DenseLinearSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    LinearEdgeSpec,
    compile_composite,
    infer_linearisation,
)
from nof1_causal_lab.models.ssm.priors import PriorRegistry, PriorSpec
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from tests.ssm_test_utils import (
    default_diffusion_block,
    default_input_effect_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
    structural_dense_drift_spec,
)


def _decay_prior_registry(values) -> PriorRegistry:
    return PriorRegistry(
        {
            "vf_0_decay": PriorSpec(
                PriorDistributionFamily.DELTA,
                {"value": values},
            )
        }
    )


class TestInferLinearisation:
    def test_dense_linear_only_is_constant(self):
        spec = CompositeSpec(
            n_latent=2,
            components=(
                DenseLinearSpec(
                    drift_prior=ndist.Normal(jnp.zeros((2, 2)), 1.0),
                    cint_prior=ndist.Normal(jnp.zeros(2), 1.0),
                ),
            ),
        )
        compiled = compile_composite(spec)
        assert infer_linearisation(compiled.vector_field) == "constant"

    def test_diagonal_decay_plus_linear_edge_is_constant(self):
        spec = CompositeSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.5)),
                LinearEdgeSpec(source=0, target=1, weight_prior=ndist.Normal(0.0, 1.0)),
            ),
        )
        compiled = compile_composite(spec)
        assert infer_linearisation(compiled.vector_field) == "constant"

    def test_hill_makes_it_trajectory(self):
        spec = CompositeSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.5)),
                HillEdgeSpec(
                    source=0,
                    target=1,
                    emax_prior=ndist.LogNormal(0.0, 0.5),
                    ec50_prior=ndist.LogNormal(0.0, 0.5),
                    n_prior=ndist.TruncatedNormal(loc=2.0, scale=0.5, low=1.0, high=4.0),
                ),
            ),
        )
        compiled = compile_composite(spec)
        assert infer_linearisation(compiled.vector_field) == "trajectory"


class TestSSMModelCompositeDispatch:
    """The NumPyro model samples nonlinear drift and delegates at backend boundary."""

    def test_nonlinear_drift_uses_vector_field_backend_method(self):
        import numpyro
        from numpyro import handlers

        from nof1_causal_lab.models.ssm import SSMModel, SSMSpec
        from nof1_causal_lab.models.ssm.inference.targets.base import RuntimeDynamics
        from nof1_causal_lab.models.ssm.structure import (
            DiffusionBlockSpec,
            ManifestCholBlockSpec,
            SparseMatrixBlockSpec,
            SparseVectorBlockSpec,
            T0CholBlockSpec,
        )

        class CompositeAwareBackend:
            checkpoint_loglik = False

            def compute_log_likelihood(
                self,
                dynamics,
                _measurement_params,
                _initial_state,
                _observations,
                time_intervals,
                **_kwargs,
            ):
                assert isinstance(dynamics, RuntimeDynamics)
                numpyro.deterministic(
                    "backend_n_vf_components",
                    jnp.asarray(len(dynamics.vf_params)),
                )
                numpyro.deterministic("backend_decay", dynamics.vf_params[0]["decay"])
                return jnp.zeros_like(time_intervals)

        spec = SSMSpec(
            n_latent=2,
            n_manifest=1,
            drift_spec=CompositeSpec(
                n_latent=2,
                components=(DiagonalDecaySpec(decay_prior=ndist.Delta(jnp.array([0.3, 0.5]))),),
            ),
            diffusion_block=DiffusionBlockSpec(
                n_latent=2,
                diffusion_chol_mask=np.zeros((2, 2), dtype=bool),
                diffusion_chol_template=jnp.eye(2) * 0.1,
            ),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=2,
                mask=np.zeros((1, 2), dtype=bool),
                template=jnp.array([[1.0, 0.0]]),
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
            manifest_means_block=default_manifest_means_block(1),
            manifest_chol_block=ManifestCholBlockSpec(
                n_manifest=1,
                diag_mask=np.zeros(1, dtype=bool),
                template=jnp.array([[0.2]]),
            ),
            t0_means_block=SparseVectorBlockSpec(
                n=2,
                mask=np.zeros(2, dtype=bool),
                template=jnp.zeros(2),
                free_site_name="t0_means_free",
                det_site_name="t0_means",
                support=SupportClass.REAL,
                site_kind=SiteKind.T0_MEANS,
                assembly_group="t0",
                fixed_spec_field="t0_means",
                priors_field="t0_means",
            ),
            t0_chol_block=T0CholBlockSpec(
                n_latent=2,
                diag_mask=np.zeros(2, dtype=bool),
                correlation_mask=np.zeros((2, 2), dtype=bool),
                template=jnp.eye(2) * 0.3,
            ),
            input_effect_block=default_input_effect_block(2),
            static_state_sd_block=default_static_state_sd_block(),
        )
        model = SSMModel(spec, priors=_decay_prior_registry([0.3, 0.5]))
        tr = handlers.trace(handlers.seed(model.model, rng_seed=0)).get_trace(
            observations=jnp.zeros((4, 1)),
            times=jnp.arange(4, dtype=jnp.float64),
            likelihood_backend=CompositeAwareBackend(),
        )

        assert "vf_0_decay" in tr
        assert int(tr["backend_n_vf_components"]["value"]) == 1
        np.testing.assert_allclose(
            tr["backend_decay"]["value"],
            np.array([0.3, 0.5]),
        )


class TestStructuralLinearSpecs:
    """Numerical-equivalence pins for the unification artifacts.

    ``StructuralDenseLinearSpec`` and ``StructuralInterceptSpec`` produce
    the same assembled drift and cint as the spec-level block assembly
    paths used by ``SSMModel._sample_drift`` / ``_sample_cint``.

    These two specs together express the linear SSM as a two-component
    composite, with equivalent numerics.
    """

    def test_dense_linear_drift_matches_spec_assembly(self):
        from numpyro.handlers import condition, seed, trace

        from nof1_causal_lab.models.ssm import SSMSpec
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            StructuralDenseLinearSpec,
            compile_composite,
        )
        from nof1_causal_lab.models.ssm.structure import (
            SparseMatrixBlockSpec,
        )

        n_latent = 3
        drift_offdiag_mask = np.array(
            [
                [False, True, False],
                [False, False, True],
                [True, False, False],
            ],
            dtype=bool,
        )
        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=1,
            drift_spec=structural_dense_drift_spec(
                n_latent=n_latent,
                drift_diag_mask=np.ones(n_latent, dtype=bool),
                drift_offdiag_mask=drift_offdiag_mask,
                drift_template=jnp.zeros((n_latent, n_latent)),
                cint_mask=np.zeros(n_latent, dtype=bool),
                cint_template=jnp.zeros(n_latent),
            ),
            diffusion_block=default_diffusion_block(n_latent),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=n_latent,
                mask=np.zeros((1, n_latent), dtype=bool),
                template=jnp.array([[0.0, 1.0, 0.0]]),
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
            manifest_means_block=default_manifest_means_block(1),
            manifest_chol_block=default_manifest_chol_block(1),
            t0_means_block=default_t0_means_block(n_latent),
            t0_chol_block=default_t0_chol_block(n_latent),
            input_effect_block=default_input_effect_block(n_latent),
            static_state_sd_block=default_static_state_sd_block(),
        )

        base_decay_values = jnp.array([0.3, 0.5, 0.7])
        offdiag_values = jnp.array([0.2, -0.1, 0.4])

        drift_component = spec.drift_spec.components[0]
        expected_drift = drift_component.assemble_drift(base_decay_values, offdiag_values)

        component = StructuralDenseLinearSpec(
            n_latent=n_latent,
            drift_diag_mask=drift_component.drift_diag_mask,
            drift_offdiag_mask=drift_component.drift_offdiag_mask,
            drift_template=jnp.asarray(drift_component.drift_template),
            stability_margin=float(drift_component.stability_margin),
            time_invariant_mask=drift_component.time_invariant_mask,
            base_decay_prior=ndist.LogNormal(0.0, 1.0),
            offdiag_prior=ndist.Normal(jnp.zeros(3), 1.0),
        )
        assert component.n_drift_base_decay == drift_component.n_drift_base_decay
        assert component.n_drift_offdiag == drift_component.n_drift_offdiag

        composite = CompositeSpec(n_latent=n_latent, components=(component,))
        compiled = compile_composite(composite)

        with (
            seed(rng_seed=0),
            condition(
                data={
                    "vf_0_base_decay": base_decay_values,
                    "vf_0_offdiag": offdiag_values,
                }
            ),
            trace() as tr,
        ):
            params = compiled.sample_params()

        composite_drift = params[0]["drift"]
        np.testing.assert_allclose(composite_drift, expected_drift, atol=1e-12)
        assert "vf_0_drift" in tr
        np.testing.assert_allclose(tr["vf_0_drift"]["value"], expected_drift, atol=1e-12)
        assert "cint" not in params[0]

    def test_no_free_drift_returns_template(self):
        from numpyro.handlers import seed

        from nof1_causal_lab.models.ssm import SSMSpec
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            StructuralDenseLinearSpec,
            compile_composite,
        )
        from nof1_causal_lab.models.ssm.structure import (
            SparseMatrixBlockSpec,
        )
        from tests.ssm_test_utils import zero_diagonal_mask, zero_square_mask

        spec = SSMSpec(
            n_latent=2,
            n_manifest=1,
            drift_spec=structural_dense_drift_spec(
                n_latent=2,
                drift_diag_mask=zero_diagonal_mask(2),
                drift_offdiag_mask=zero_square_mask(2),
                drift_template=jnp.eye(2) * -0.4,
                cint_mask=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
            diffusion_block=default_diffusion_block(2),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=2,
                mask=np.zeros((1, 2), dtype=bool),
                template=jnp.array([[0.0, 1.0]]),
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
            manifest_means_block=default_manifest_means_block(1),
            manifest_chol_block=default_manifest_chol_block(1),
            t0_means_block=default_t0_means_block(2),
            t0_chol_block=default_t0_chol_block(2),
            input_effect_block=default_input_effect_block(2),
            static_state_sd_block=default_static_state_sd_block(),
        )
        drift_component = spec.drift_spec.components[0]

        component = StructuralDenseLinearSpec(
            n_latent=2,
            drift_diag_mask=drift_component.drift_diag_mask,
            drift_offdiag_mask=drift_component.drift_offdiag_mask,
            drift_template=jnp.asarray(drift_component.drift_template),
        )
        composite = CompositeSpec(n_latent=2, components=(component,))
        compiled = compile_composite(composite)
        with seed(rng_seed=0):
            params = compiled.sample_params()
        np.testing.assert_allclose(params[0]["drift"], np.eye(2) * -0.4)

    def test_intercept_matches_spec_assembly(self):
        from numpyro.handlers import condition, seed, trace

        from nof1_causal_lab.models.ssm import SSMSpec
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            StructuralInterceptSpec,
            compile_composite,
        )
        from nof1_causal_lab.models.ssm.structure import (
            SparseMatrixBlockSpec,
        )
        from tests.ssm_test_utils import full_vector_mask

        n_latent = 3
        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=1,
            drift_spec=structural_dense_drift_spec(
                n_latent=n_latent,
                drift_diag_mask=np.ones(n_latent, dtype=bool),
                drift_offdiag_mask=np.ones((n_latent, n_latent), dtype=bool)
                & ~np.eye(n_latent, dtype=bool),
                drift_template=jnp.zeros((n_latent, n_latent)),
                cint_mask=full_vector_mask(n_latent),
                cint_template=jnp.zeros(n_latent),
            ),
            diffusion_block=default_diffusion_block(n_latent),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=n_latent,
                mask=np.zeros((1, n_latent), dtype=bool),
                template=jnp.array([[0.0, 1.0, 0.0]]),
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
            manifest_means_block=default_manifest_means_block(1),
            manifest_chol_block=default_manifest_chol_block(1),
            t0_means_block=default_t0_means_block(n_latent),
            t0_chol_block=default_t0_chol_block(n_latent),
            input_effect_block=default_input_effect_block(n_latent),
            static_state_sd_block=default_static_state_sd_block(),
        )
        cint_values = jnp.array([0.1, -0.2, 0.3])
        cint_component = spec.drift_spec.components[1]
        expected_cint = cint_component.assemble_cint(cint_values)

        component = StructuralInterceptSpec(
            n_latent=n_latent,
            cint_mask=cint_component.cint_mask,
            cint_template=jnp.asarray(cint_component.cint_template),
            cint_prior=ndist.Normal(jnp.zeros(n_latent), 1.0),
        )
        composite = CompositeSpec(n_latent=n_latent, components=(component,))
        compiled = compile_composite(composite)
        with (
            seed(rng_seed=0),
            condition(data={"vf_0_cint": cint_values}),
            trace() as tr,
        ):
            params = compiled.sample_params()
        np.testing.assert_allclose(params[0]["cint"], expected_cint, atol=1e-12)
        assert "vf_0_cint_full" in tr
        np.testing.assert_allclose(tr["vf_0_cint_full"]["value"], expected_cint, atol=1e-12)
