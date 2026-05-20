"""Tests for the canonical-model unification layer.

Phase A of the unification proposed in
``docs/reference/unification-rfc.md`` introduces ``RuntimeSSM`` plus
two adapters. This test file pins:

- The ``linearisation`` classifier returns ``"constant"`` iff every
  component has a state-independent Jacobian.
- ``runtime_from_composite`` round-trips a compiled spec into a
  canonical envelope.
- ``runtime_from_dense_linear`` builds a single-component canonical
  envelope from raw ``(drift, cint)`` matrices.
- Both adapters produce canonical models whose vector fields evaluate
  identically to a hand-built equivalent.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeSpec,
    DenseLinearSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    Intervention,
    LinearEdgeSpec,
    RuntimeSSM,
    VectorFieldArgs,
    compile_composite,
    infer_linearisation,
    runtime_from_composite,
    runtime_from_dense_linear,
    runtime_from_ssm_model,
)
from nof1_causal_lab.models.ssm.inference.targets.kernels import (
    build_observation_kernel,
)


def _gaussian_kernel(R):
    return build_observation_kernel(
        DistributionFamily.GAUSSIAN,
        LinkFunction.IDENTITY,
        manifest_cov=np.asarray(R),
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
                LinearEdgeSpec(
                    source=0, target=1, weight_prior=ndist.Normal(0.0, 1.0)
                ),
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
                    n_prior=ndist.TruncatedNormal(
                        loc=2.0, scale=0.5, low=1.0, high=4.0
                    ),
                ),
            ),
        )
        compiled = compile_composite(spec)
        assert infer_linearisation(compiled.vector_field) == "trajectory"


class TestCanonicalFromComposite:
    def test_propagates_fields(self):
        spec = CompositeSpec(
            n_latent=2,
            components=(DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.5)),),
        )
        compiled = compile_composite(spec)
        init_mean = jnp.array([1.0, 0.5])
        init_cov = jnp.eye(2) * 0.1
        diffusion_cov = jnp.eye(2) * 0.05
        H = jnp.array([[1.0, 0.0]])
        d_meas = jnp.array([0.0])
        R = jnp.array([[0.01]])
        kernel = _gaussian_kernel(R)

        canonical = runtime_from_composite(
            compiled,
            init_mean=init_mean,
            init_cov=init_cov,
            diffusion_cov=diffusion_cov,
            H=H,
            d_meas=d_meas,
            R=R,
            obs_kernel=kernel,
        )
        assert isinstance(canonical, RuntimeSSM)
        assert canonical.vector_field is compiled.vector_field
        assert canonical.sample_params is compiled.sample_params
        assert jnp.allclose(canonical.init_mean, init_mean)
        assert canonical.linearisation == "constant"  # DiagonalDecay only
        assert canonical.obs_kernel is kernel

    def test_marks_hill_as_trajectory(self):
        spec = CompositeSpec(
            n_latent=2,
            components=(
                HillEdgeSpec(
                    source=0,
                    target=1,
                    emax_prior=ndist.LogNormal(0.0, 0.5),
                    ec50_prior=ndist.LogNormal(0.0, 0.5),
                    n_prior=ndist.TruncatedNormal(
                        loc=2.0, scale=0.5, low=1.0, high=4.0
                    ),
                ),
            ),
        )
        compiled = compile_composite(spec)
        canonical = runtime_from_composite(
            compiled,
            init_mean=jnp.zeros(2),
            init_cov=jnp.eye(2),
            diffusion_cov=jnp.eye(2),
            H=jnp.eye(2),
            d_meas=jnp.zeros(2),
            R=jnp.eye(2),
            obs_kernel=_gaussian_kernel(jnp.eye(2)),
        )
        assert canonical.linearisation == "trajectory"


class TestCanonicalFromDenseLinear:
    def test_single_component_envelope(self):
        drift = jnp.array([[-1.0, 0.3], [0.0, -0.5]])
        cint = jnp.array([0.1, -0.2])
        init_mean = jnp.zeros(2)
        kernel = _gaussian_kernel(jnp.eye(1) * 0.1)
        canonical = runtime_from_dense_linear(
            drift,
            cint,
            init_mean=init_mean,
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.05,
            H=jnp.array([[1.0, 0.0]]),
            d_meas=jnp.zeros(1),
            R=jnp.eye(1) * 0.1,
            obs_kernel=kernel,
        )
        assert canonical.linearisation == "constant"
        # The sample_params returns the Delta-distributed pair
        params = canonical.sample_params()
        assert len(params) == 1
        assert jnp.allclose(params[0]["drift"], drift)
        assert jnp.allclose(params[0]["cint"], cint)

    def test_drift_matches_explicit_evaluation(self):
        """A canonical built from (A, c) must evaluate ``f(η) = A·η + c``
        at any state."""
        drift = jnp.array([[-1.5, 0.4], [0.2, -0.8]])
        cint = jnp.array([0.3, -0.1])
        canonical = runtime_from_dense_linear(
            drift,
            cint,
            init_mean=jnp.zeros(2),
            init_cov=jnp.eye(2),
            diffusion_cov=jnp.eye(2),
            H=jnp.eye(2),
            d_meas=jnp.zeros(2),
            R=jnp.eye(2),
            obs_kernel=_gaussian_kernel(jnp.eye(2)),
        )
        params = canonical.sample_params()
        eta = jnp.array([1.0, 2.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        drift_out = canonical.vector_field(jnp.asarray(0.0), eta, args)
        assert jnp.allclose(drift_out, drift @ eta + cint, atol=1e-6)


class TestRuntimeFromSSMModel:
    """Bridge from ``SSMModel`` (declarative spec + numpyro wrapper) into
    :class:`RuntimeSSM`. This is the Phase-4 unification seam: composite
    consumers will eventually take ``SSMModel`` directly and call this
    factory internally."""

    def _gaussian_kernel_2(self):
        return build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=np.eye(2),
        )

    def test_composite_spec_pulls_drift_from_drift_spec(self):
        from nof1_causal_lab.models.ssm import SSMModel, SSMSpec
        from nof1_causal_lab.models.ssm.structure import (
            DiffusionBlockSpec,
            ManifestCholBlockSpec,
            SparseMatrixBlockSpec,
            SparseVectorBlockSpec,
            T0CholBlockSpec,
            default_input_effect_block,
            default_manifest_means_block,
            default_static_state_sd_block,
        )

        drift_spec = CompositeSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(decay_prior=ndist.Delta(jnp.array([0.3, 0.5]))),
            ),
        )
        spec = SSMSpec(
            n_latent=2,
            n_manifest=1,
            drift_spec=drift_spec,
            diffusion_block=DiffusionBlockSpec(
                n_latent=2,
                diffusion_chol_mask=np.zeros((2, 2), dtype=bool),
                diffusion_chol_template=jnp.eye(2) * 0.07,
            ),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1, n_cols=2,
                mask=np.zeros((1, 2), dtype=bool),
                template=jnp.array([[0.0, 1.0]]),
                free_site_name="lambda_free", det_site_name="lambda",
            ),
            manifest_means_block=default_manifest_means_block(1),
            manifest_chol_block=ManifestCholBlockSpec(
                n_manifest=1,
                diag_mask=np.zeros(1, dtype=bool),
                template=jnp.array([[0.14]]),
            ),
            t0_means_block=SparseVectorBlockSpec(
                n=2,
                mask=np.zeros(2, dtype=bool),
                template=jnp.array([1.5, 0.0]),
                free_site_name="t0_means_free", det_site_name="t0_means",
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
        model = SSMModel(spec)

        runtime = runtime_from_ssm_model(
            model,
            obs_kernel=build_observation_kernel(
                DistributionFamily.GAUSSIAN,
                LinkFunction.IDENTITY,
                manifest_cov=np.array([[0.02]]),
            ),
        )

        assert runtime.linearisation == "constant"
        params = runtime.sample_params()
        assert len(params) == 1
        np.testing.assert_allclose(params[0]["decay"], np.array([0.3, 0.5]))

        eta = jnp.array([2.0, 1.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        drift_out = runtime.vector_field(jnp.asarray(0.0), eta, args)
        # DiagonalDecay contributes -decay * eta
        np.testing.assert_allclose(drift_out, np.array([-0.6, -0.5]), atol=1e-6)

        np.testing.assert_allclose(runtime.init_mean, np.array([1.5, 0.0]))
        np.testing.assert_allclose(runtime.H, np.array([[0.0, 1.0]]))


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
            default_diffusion_block,
            default_input_effect_block,
            default_manifest_chol_block,
            default_manifest_means_block,
            default_static_state_sd_block,
            default_t0_chol_block,
            default_t0_means_block,
        )
        from tests.ssm_test_utils import linear_drift_spec_from_combined_mask

        n_latent = 3
        drift_mask = np.array(
            [
                [True, True, False],
                [False, True, True],
                [True, False, True],
            ],
            dtype=bool,
        )
        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=1,
            drift_spec=linear_drift_spec_from_combined_mask(
                n_latent, drift_mask=drift_mask
            ),
            diffusion_block=default_diffusion_block(n_latent),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1, n_cols=n_latent,
                mask=np.zeros((1, n_latent), dtype=bool),
                template=jnp.array([[0.0, 1.0, 0.0]]),
                free_site_name="lambda_free", det_site_name="lambda",
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

        expected_drift = spec.assemble_drift(
            base_decay_values, offdiag_values
        )
        drift_component, _ = spec.structural_drift_components()

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
                    "drift_base_decay_free": base_decay_values,
                    "drift_offdiag_free": offdiag_values,
                }
            ),
            trace() as tr,
        ):
            params = compiled.sample_params()

        composite_drift = params[0]["drift"]
        np.testing.assert_allclose(composite_drift, expected_drift, atol=1e-12)
        assert "drift" in tr
        np.testing.assert_allclose(tr["drift"]["value"], expected_drift, atol=1e-12)
        assert "cint" not in params[0]

    def test_no_free_drift_returns_template(self):
        from numpyro.handlers import seed

        from nof1_causal_lab.models.ssm import SSMSpec
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            StructuralDenseLinearSpec,
            compile_composite,
            linear_drift_spec,
        )
        from nof1_causal_lab.models.ssm.model import (
            zero_diagonal_mask,
            zero_square_mask,
        )
        from nof1_causal_lab.models.ssm.structure import (
            SparseMatrixBlockSpec,
            default_diffusion_block,
            default_input_effect_block,
            default_manifest_chol_block,
            default_manifest_means_block,
            default_static_state_sd_block,
            default_t0_chol_block,
            default_t0_means_block,
        )

        spec = SSMSpec(
            n_latent=2,
            n_manifest=1,
            drift_spec=linear_drift_spec(
                n_latent=2,
                drift_diag_mask=zero_diagonal_mask(2),
                drift_offdiag_mask=zero_square_mask(2),
                drift_template=jnp.eye(2) * -0.4,
                cint_mask=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
            diffusion_block=default_diffusion_block(2),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1, n_cols=2,
                mask=np.zeros((1, 2), dtype=bool),
                template=jnp.array([[0.0, 1.0]]),
                free_site_name="lambda_free", det_site_name="lambda",
            ),
            manifest_means_block=default_manifest_means_block(1),
            manifest_chol_block=default_manifest_chol_block(1),
            t0_means_block=default_t0_means_block(2),
            t0_chol_block=default_t0_chol_block(2),
            input_effect_block=default_input_effect_block(2),
            static_state_sd_block=default_static_state_sd_block(),
        )
        drift_component, _ = spec.structural_drift_components()

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
            linear_drift_spec,
        )
        from nof1_causal_lab.models.ssm.model import full_vector_mask
        from nof1_causal_lab.models.ssm.structure import (
            SparseMatrixBlockSpec,
            default_diffusion_block,
            default_input_effect_block,
            default_manifest_chol_block,
            default_manifest_means_block,
            default_static_state_sd_block,
            default_t0_chol_block,
            default_t0_means_block,
        )

        n_latent = 3
        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=1,
            drift_spec=linear_drift_spec(
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
                n_rows=1, n_cols=n_latent,
                mask=np.zeros((1, n_latent), dtype=bool),
                template=jnp.array([[0.0, 1.0, 0.0]]),
                free_site_name="lambda_free", det_site_name="lambda",
            ),
            manifest_means_block=default_manifest_means_block(1),
            manifest_chol_block=default_manifest_chol_block(1),
            t0_means_block=default_t0_means_block(n_latent),
            t0_chol_block=default_t0_chol_block(n_latent),
            input_effect_block=default_input_effect_block(n_latent),
            static_state_sd_block=default_static_state_sd_block(),
        )
        cint_values = jnp.array([0.1, -0.2, 0.3])
        expected_cint = spec.assemble_cint(cint_values)
        _, cint_component = spec.structural_drift_components()

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
            condition(data={"cint_free": cint_values}),
            trace() as tr,
        ):
            params = compiled.sample_params()
        np.testing.assert_allclose(params[0]["cint"], expected_cint, atol=1e-12)
        assert "cint" in tr
        np.testing.assert_allclose(tr["cint"]["value"], expected_cint, atol=1e-12)
