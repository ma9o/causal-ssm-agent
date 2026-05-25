"""Round-trip tests for the dict-config bridge to ``DynamicsSpec``.

The bridge in ``models.ssm.dynamics.serialization`` materialises a
structure-only runtime ``DynamicsSpec``. Priors are bound separately through
the canonical site-prior registry. These tests pin the bridge:

- Every supported family materialises into the right ``ndist`` class.
- Every component kind (``StateDecay``, ``DiagonalDecay``, ``StateIntercept``,
  ``Intercept``, ``LinearEdge``, ``HillEdge``, ``MultiplicativeEdge``) compiles
  end-to-end from a dict and produces working vector-field params when supplied
  a site-prior resolver.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
from numpyro.handlers import seed

from nof1_causal_lab.models.ssm.dynamics import (
    DynamicsSpec,
    Intervention,
    StateDecay,
    StateIntercept,
    VectorFieldArgs,
    compile_dynamics,
    dynamics_spec_from_dict,
)
from nof1_causal_lab.models.ssm.priors import materialize_prior_distribution
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from tests.ssm_test_utils import (
    default_diffusion_block,
    default_input_effect_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
)


def _delta_prior_fn(compiled, values: dict[str, object] | None = None):
    values = values or {}
    shapes = {site.name: site.shape for site in compiled.site_registry}

    def _prior(site_name: str):
        value = values.get(site_name, 1.0)
        return ndist.Delta(jnp.broadcast_to(jnp.asarray(value), shapes[site_name]))

    return _prior


class TestMaterializePrior:
    def test_normal(self):
        d = materialize_prior_distribution(
            {"family": "Normal", "params": {"mu": 0.3, "sigma": 0.5}}
        )
        assert isinstance(d, ndist.Normal)
        assert float(d.loc) == 0.3
        assert float(d.scale) == 0.5

    def test_log_normal(self):
        d = materialize_prior_distribution(
            {"family": "LogNormal", "params": {"mu": 0.0, "sigma": 0.7}}
        )
        assert isinstance(d, ndist.LogNormal)

    def test_gamma(self):
        d = materialize_prior_distribution(
            {"family": "Gamma", "params": {"concentration": 2.0, "rate": 4.0}}
        )
        assert isinstance(d, ndist.Gamma)
        assert float(d.concentration) == 2.0
        assert float(d.rate) == 4.0

    def test_truncated_normal_with_bounds(self):
        d = materialize_prior_distribution(
            {
                "family": "TruncatedNormal",
                "params": {"mu": 2.0, "sigma": 0.5, "lower": 1.0, "upper": 4.0},
            }
        )
        # NumPyro's TruncatedNormal is a factory; verify behaviour via samples
        import jax.random as jr

        samples = d.sample(jr.PRNGKey(0), (1000,))
        assert float(jnp.min(samples)) >= 1.0
        assert float(jnp.max(samples)) <= 4.0

    def test_shape_broadcasts(self):
        """A matrix-shape Normal prior must broadcast its scalar mu/sigma."""
        d = materialize_prior_distribution(
            {
                "family": "Normal",
                "params": {"mu": 0.0, "sigma": 1.0},
                "shape": [3, 3],
            }
        )
        assert d.batch_shape + d.event_shape == (3, 3)


class TestDynamicsSpecFromDict:
    def test_scalar_background_components_compile(self):
        config = {
            "n_latent": 2,
            "components": [
                {"kind": "StateDecay", "target": 0},
                {"kind": "StateIntercept", "target": 1},
            ],
        }
        spec = dynamics_spec_from_dict(config)
        assert isinstance(spec, DynamicsSpec)
        assert spec.n_latent == 2

        compiled = compile_dynamics(dynamics_spec_from_dict(config))
        assert isinstance(compiled.vector_field.components[0], StateDecay)
        assert isinstance(compiled.vector_field.components[1], StateIntercept)
        with seed(rng_seed=0):
            params = compiled.sample_params(_delta_prior_fn(compiled))
        assert params[0]["decay"].shape == ()
        assert params[1]["cint"].shape == ()

    def test_ssri_chain_round_trip(self):
        """The Hill/Multiplicative/LinearEdge chain emits the right
        component types in the right order from a serialised dict."""
        DOSE, ADHERENCE, C_P, C_E, AFFECTIVE = 0, 1, 2, 3, 4
        config = {
            "n_latent": 5,
            "components": [
                {"kind": "DiagonalDecay"},
                {"kind": "Intercept"},
                {
                    "kind": "MultiplicativeEdge",
                    "source_a": DOSE,
                    "source_b": ADHERENCE,
                    "target": C_P,
                },
                {"kind": "LinearEdge", "source": C_P, "target": C_E},
                {"kind": "HillEdge", "source": C_E, "target": AFFECTIVE},
            ],
        }
        compiled = compile_dynamics(dynamics_spec_from_dict(config))
        kinds = [type(c).__name__ for c in compiled.vector_field.components]
        assert kinds == [
            "DiagonalDecay",
            "Intercept",
            "MultiplicativeEdge",
            "LinearEdge",
            "HillEdge",
        ]
        # Indices preserved
        mult = compiled.vector_field.components[2]
        lin = compiled.vector_field.components[3]
        hill = compiled.vector_field.components[4]
        assert (mult.source_a, mult.source_b, mult.target) == (DOSE, ADHERENCE, C_P)
        assert (lin.source, lin.target) == (C_P, C_E)
        assert (hill.source, hill.target) == (C_E, AFFECTIVE)


class TestEndToEndDynamicsFromDict:
    def test_dict_compiled_vector_field_matches_hand_built(self):
        """Sample params from a dict-compiled spec, plug into the vector
        field, and verify the output is numerically sensible."""
        config = {
            "n_latent": 2,
            "components": [
                {"kind": "DiagonalDecay"},
                {"kind": "LinearEdge", "source": 0, "target": 1},
            ],
        }
        compiled = compile_dynamics(dynamics_spec_from_dict(config))
        params = compiled.sample_params(
            _delta_prior_fn(compiled, {"vf_0_decay": jnp.array([1.0, 1.0]), "vf_1_weight": 0.3})
        )
        eta = jnp.array([2.0, 1.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        dynamics = compiled.vector_field(jnp.asarray(0.0), eta, args)
        # decay·η = [-2, -1]; LinearEdge adds 0.3·2 = 0.6 to target=1
        assert jnp.allclose(dynamics, jnp.array([-2.0, -0.4]), atol=1e-6)


class TestErrorPaths:
    def test_unknown_kind_raises(self):
        import pytest

        config = {"n_latent": 1, "components": [{"kind": "Bogus"}]}
        with pytest.raises(ValueError, match="Bogus"):
            dynamics_spec_from_dict(config)


class TestBlockSpecEquivalence:
    """Numerical-equivalence pins for the non-dynamics ``BlockSpec`` family.

    Each ``*BlockSpec`` in ``dynamics/blocks.py`` delegates to the shared
    structural assembly helpers. The spec-level assembly methods and the
    block sample path must produce identical output for the same free values.
    """

    def test_diffusion_block_matches_spec_assembly(self):
        from numpyro.handlers import condition, seed

        from nof1_causal_lab.models.ssm import SSMSpec
        from nof1_causal_lab.models.ssm.dynamics import DynamicsSpec
        from nof1_causal_lab.models.ssm.structure import (
            DiffusionBlockSpec,
            SparseMatrixBlockSpec,
        )
        from tests.ssm_test_utils import full_diagonal_support

        spec = SSMSpec(
            n_latent=2,
            n_manifest=1,
            dynamics_spec=DynamicsSpec(n_latent=2),
            diffusion_block=DiffusionBlockSpec(
                n_latent=2,
                diffusion_chol_support=np.diag(full_diagonal_support(2)),
                diffusion_chol_template=jnp.eye(2),
            ),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=2,
                free_support=np.zeros((1, 2), dtype=bool),
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
        diag_vals = jnp.array([0.4, 0.6])
        expected = spec.diffusion_block.assemble(diag_vals, None)

        block = DiffusionBlockSpec(
            n_latent=2,
            diffusion_chol_support=spec.diffusion_block.diffusion_chol_support,
            diffusion_chol_template=jnp.asarray(spec.diffusion_block.diffusion_chol_template),
            time_invariant_mask=spec.diffusion_block.time_invariant_mask,
        )

        with seed(rng_seed=0), condition(data={"diffusion_diag_free": diag_vals}):
            block_assembled = block.sample_params(lambda _: ndist.LogNormal(0.0, 1.0))["diffusion"]

        np.testing.assert_allclose(block_assembled, expected, atol=1e-12)

    def test_sparse_vector_block_matches_spec_assembly(self):
        from numpyro.handlers import condition, seed

        from nof1_causal_lab.models.ssm import SSMSpec
        from nof1_causal_lab.models.ssm.structure import (
            SparseMatrixBlockSpec,
            SparseVectorBlockSpec,
        )
        from tests.ssm_test_utils import full_vector_support

        spec = SSMSpec(
            n_latent=3,
            n_manifest=1,
            dynamics_spec=DynamicsSpec(n_latent=3),
            diffusion_block=default_diffusion_block(3),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=3,
                free_support=np.zeros((1, 3), dtype=bool),
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
            t0_means_block=SparseVectorBlockSpec(
                n=3,
                free_support=full_vector_support(3),
                template=jnp.zeros(3),
                free_site_name="t0_means_free",
                det_site_name="t0_means",
                support=SupportClass.REAL,
                site_kind=SiteKind.T0_MEANS,
                assembly_group="t0",
                fixed_spec_field="t0_means",
                priors_field="t0_means",
            ),
            t0_chol_block=default_t0_chol_block(3),
            input_effect_block=default_input_effect_block(3),
            static_state_sd_block=default_static_state_sd_block(),
        )
        free = jnp.array([0.1, -0.2, 0.3])
        expected = spec.t0_means_block.assemble(free)

        block = SparseVectorBlockSpec(
            n=3,
            free_support=spec.t0_means_block.free_support,
            template=jnp.asarray(spec.t0_means_block.template),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        )
        with seed(rng_seed=0), condition(data={"t0_means_free": free}):
            assembled = block.sample_params(lambda _: ndist.Normal(jnp.zeros(3), 1.0))["t0_means"]
        np.testing.assert_allclose(assembled, expected, atol=1e-12)


class TestDynamicsSpecRoundTrip:
    """``dynamics_spec_to_dict`` is the inverse of
    ``dynamics_spec_from_dict``. Round-trip must preserve the structural
    info so persistence works end-to-end.
    """

    def test_ssri_chain_round_trips(self):
        from nof1_causal_lab.models.ssm.dynamics import dynamics_spec_to_dict

        DOSE, ADHERENCE, C_P, C_E, AFFECTIVE = 0, 1, 2, 3, 4
        config = {
            "n_latent": 5,
            "components": [
                {"kind": "DiagonalDecay"},
                {"kind": "Intercept"},
                {
                    "kind": "MultiplicativeEdge",
                    "source_a": DOSE,
                    "source_b": ADHERENCE,
                    "target": C_P,
                },
                {"kind": "LinearEdge", "source": C_P, "target": C_E},
                {"kind": "HillEdge", "source": C_E, "target": AFFECTIVE},
            ],
        }
        spec = dynamics_spec_from_dict(config)
        round_tripped = dynamics_spec_to_dict(spec)
        assert round_tripped == config

    def test_component_dynamics_round_trips(self):
        from nof1_causal_lab.models.ssm.dynamics import (
            DynamicsSpec,
            LinearEdgeSpec,
            StateDecaySpec,
            StateInterceptSpec,
            dynamics_spec_from_dict,
            dynamics_spec_to_dict,
        )

        spec = DynamicsSpec(
            n_latent=2,
            components=(
                StateDecaySpec(target=0),
                LinearEdgeSpec(source=0, target=1),
                StateInterceptSpec(target=1),
            ),
        )
        payload = dynamics_spec_to_dict(spec)
        restored = dynamics_spec_from_dict(payload)

        c0 = restored.components[0]
        assert c0.target == 0
        c1 = restored.components[1]
        assert (c1.source, c1.target) == (0, 1)
        c2 = restored.components[2]
        assert c2.target == 1

        assert dynamics_spec_to_dict(restored) == payload

    def test_fixed_hill_shape_round_trips_and_removes_sites(self):
        from nof1_causal_lab.models.ssm.dynamics import (
            DynamicsSpec,
            HillEdgeSpec,
            compile_dynamics,
            dynamics_spec_from_dict,
            dynamics_spec_to_dict,
        )

        spec = DynamicsSpec(
            n_latent=2,
            components=(
                HillEdgeSpec(
                    source=0,
                    target=1,
                    fixed_ec50=1.2,
                    fixed_n=2.0,
                ),
            ),
        )
        payload = dynamics_spec_to_dict(spec)
        assert payload["components"][0] == {
            "kind": "HillEdge",
            "source": 0,
            "target": 1,
            "fixed_ec50": 1.2,
            "fixed_n": 2.0,
        }

        restored = dynamics_spec_from_dict(payload)
        compiled = compile_dynamics(restored)
        assert [site.name for site in compiled.site_registry] == ["vf_0_Emax"]

        with seed(rng_seed=0):
            params = compiled.sample_params(
                _delta_prior_fn(compiled, {"vf_0_Emax": jnp.asarray(0.8)})
            )

        np.testing.assert_allclose(float(params[0]["Emax"]), 0.8, rtol=1e-6)
        np.testing.assert_allclose(float(params[0]["EC50"]), 1.2, rtol=1e-6)
        np.testing.assert_allclose(float(params[0]["n"]), 2.0, rtol=1e-6)

    def test_ssm_compiler_serializes_dynamics_spec(self):
        """``serialize_ssm_spec`` / ``deserialize_ssm_spec`` round-trip a
        populated ``dynamics_spec`` via the dict-config layer."""
        import jax.numpy as jnp

        from nof1_causal_lab.models.ssm import SSMSpec
        from nof1_causal_lab.models.ssm.compile.artifact import (
            deserialize_ssm_spec,
            serialize_ssm_spec,
        )
        from nof1_causal_lab.models.ssm.dynamics import (
            DiagonalDecaySpec,
            DynamicsSpec,
        )
        from nof1_causal_lab.models.ssm.structure import (
            SparseMatrixBlockSpec,
        )

        dynamics_spec = DynamicsSpec(
            n_latent=2,
            components=(DiagonalDecaySpec(),),
        )
        spec = SSMSpec(
            n_latent=2,
            n_manifest=1,
            dynamics_spec=dynamics_spec,
            diffusion_block=default_diffusion_block(2),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=2,
                free_support=np.zeros((1, 2), dtype=bool),
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
        payload = serialize_ssm_spec(spec)
        assert isinstance(payload["dynamics_spec"], dict)
        assert payload["dynamics_spec"]["n_latent"] == 2
        assert payload["dynamics_spec"]["components"][0]["kind"] == "DiagonalDecay"

        restored = deserialize_ssm_spec(payload)
        assert restored.dynamics_spec is not None
        assert restored.dynamics_spec.n_latent == 2
        from nof1_causal_lab.models.ssm.dynamics import DiagonalDecaySpec as _DD

        assert isinstance(restored.dynamics_spec.components[0], _DD)
