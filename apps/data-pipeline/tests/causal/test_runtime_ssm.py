"""Tests for vector-field linearisation and component-native dynamics specs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist

from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    Intervention,
    LinearEdgeSpec,
    StateDecaySpec,
    StateInterceptSpec,
    VectorFieldArgs,
    compile_composite,
    infer_linearisation,
)
from nof1_causal_lab.models.ssm.priors import PriorRegistry, PriorSpec
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from tests.ssm_test_utils import (
    default_input_effect_block,
    default_manifest_means_block,
    default_static_state_sd_block,
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
    def test_state_decay_only_is_constant(self):
        spec = CompositeSpec(
            n_latent=2,
            components=(
                StateDecaySpec(target=0, decay_prior=ndist.LogNormal(0.0, 0.5)),
                StateDecaySpec(target=1, decay_prior=ndist.LogNormal(0.0, 0.5)),
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
    """The NumPyro model samples nonlinear dynamics and delegates at backend boundary."""

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
            dynamics_spec=CompositeSpec(
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


class TestComponentNativeLinearDynamics:
    """Numerical pins for linear dynamics expressed as first-class components."""

    def test_state_decay_and_edges_match_expected_vector_field(self):
        from numpyro.handlers import condition, seed

        spec = CompositeSpec(
            n_latent=3,
            components=(
                StateDecaySpec(target=0, decay_prior=ndist.LogNormal(0.0, 1.0)),
                StateDecaySpec(target=1, decay_prior=ndist.LogNormal(0.0, 1.0)),
                StateDecaySpec(target=2, decay_prior=ndist.LogNormal(0.0, 1.0)),
                LinearEdgeSpec(source=1, target=0, weight_prior=ndist.Normal(0.0, 1.0)),
                LinearEdgeSpec(source=2, target=1, weight_prior=ndist.Normal(0.0, 1.0)),
                LinearEdgeSpec(source=0, target=2, weight_prior=ndist.Normal(0.0, 1.0)),
            ),
        )
        compiled = compile_composite(spec)
        with (
            seed(rng_seed=0),
            condition(
                data={
                    "vf_0_decay": jnp.asarray(0.3),
                    "vf_1_decay": jnp.asarray(0.5),
                    "vf_2_decay": jnp.asarray(0.7),
                    "vf_3_weight": jnp.asarray(0.2),
                    "vf_4_weight": jnp.asarray(-0.1),
                    "vf_5_weight": jnp.asarray(0.4),
                }
            ),
        ):
            params = compiled.sample_params()

        eta = jnp.array([1.0, 2.0, 3.0])
        actual = compiled.vector_field(
            jnp.asarray(0.0),
            eta,
            VectorFieldArgs(params=params, intervention=Intervention.none()),
        )
        expected = jnp.array(
            [
                -0.3 * 1.0 + 0.2 * 2.0,
                -0.5 * 2.0 - 0.1 * 3.0,
                -0.7 * 3.0 + 0.4 * 1.0,
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_delta_state_decay_samples_component_param(self):
        from numpyro.handlers import seed

        spec = CompositeSpec(
            n_latent=2,
            components=(StateDecaySpec(target=1, decay_prior=ndist.Delta(jnp.asarray(0.4))),),
        )
        compiled = compile_composite(spec)
        with seed(rng_seed=0):
            params = compiled.sample_params()

        assert params[0]["decay"].shape == ()
        np.testing.assert_allclose(params[0]["decay"], 0.4, atol=1e-12)

    def test_state_intercepts_add_to_selected_targets(self):
        from numpyro.handlers import condition, seed

        spec = CompositeSpec(
            n_latent=3,
            components=(
                StateInterceptSpec(target=0, cint_prior=ndist.Normal(0.0, 1.0)),
                StateInterceptSpec(target=2, cint_prior=ndist.Normal(0.0, 1.0)),
            ),
        )
        compiled = compile_composite(spec)
        with (
            seed(rng_seed=0),
            condition(
                data={
                    "vf_0_cint": jnp.asarray(0.1),
                    "vf_1_cint": jnp.asarray(-0.2),
                }
            ),
        ):
            params = compiled.sample_params()

        actual = compiled.vector_field(
            jnp.asarray(0.0),
            jnp.zeros(3),
            VectorFieldArgs(params=params, intervention=Intervention.none()),
        )
        np.testing.assert_allclose(actual, jnp.array([0.1, 0.0, -0.2]), atol=1e-12)
