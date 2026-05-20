"""Round-trip tests for the dict-config bridge to ``CompositeSpec``.

Stage 4's LLM tools emit prior dict-configs (``{"family": "LogNormal",
"params": {...}}``). The bridge in ``models.ssm.dynamics.serialization`` lets
the same dict-config materialise into a runtime ``CompositeSpec`` ready
for inference. These tests pin the bridge:

- Every supported family materialises into the right ``ndist`` class.
- Every component kind (``DenseLinear``, ``DiagonalDecay``, ``Intercept``,
  ``LinearEdge``, ``HillEdge``, ``MultiplicativeEdge``) compiles end-to-end
  from a dict and produces working ``sample_params`` + vector field.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
from numpyro.handlers import seed

from nof1_causal_lab.models.ssm.dynamics import (
    CompositeSpec,
    DenseLinear,
    Intervention,
    VectorFieldArgs,
    compile_composite_from_dict,
    composite_spec_from_dict,
    materialize_prior,
)


class TestMaterializePrior:
    def test_normal(self):
        d = materialize_prior({"family": "Normal", "params": {"mu": 0.3, "sigma": 0.5}})
        assert isinstance(d, ndist.Normal)
        assert float(d.loc) == 0.3
        assert float(d.scale) == 0.5

    def test_log_normal(self):
        d = materialize_prior({"family": "LogNormal", "params": {"mu": 0.0, "sigma": 0.7}})
        assert isinstance(d, ndist.LogNormal)

    def test_gamma(self):
        d = materialize_prior(
            {"family": "Gamma", "params": {"concentration": 2.0, "rate": 4.0}}
        )
        assert isinstance(d, ndist.Gamma)
        assert float(d.concentration) == 2.0
        assert float(d.rate) == 4.0

    def test_truncated_normal_with_bounds(self):
        d = materialize_prior({
            "family": "TruncatedNormal",
            "params": {"mu": 2.0, "sigma": 0.5, "lower": 1.0, "upper": 4.0},
        })
        # NumPyro's TruncatedNormal is a factory; verify behaviour via samples
        import jax.random as jr

        samples = d.sample(jr.PRNGKey(0), (1000,))
        assert float(jnp.min(samples)) >= 1.0
        assert float(jnp.max(samples)) <= 4.0

    def test_shape_broadcasts(self):
        """A matrix-shape Normal prior must broadcast its scalar mu/sigma."""
        d = materialize_prior({
            "family": "Normal",
            "params": {"mu": 0.0, "sigma": 1.0},
            "shape": [3, 3],
        })
        assert d.batch_shape + d.event_shape == (3, 3)


class TestCompositeSpecFromDict:
    def test_dense_linear_compiles(self):
        config = {
            "n_latent": 2,
            "components": [
                {
                    "kind": "DenseLinear",
                    "priors": {
                        "drift": {"family": "Normal", "params": {"mu": 0.0, "sigma": 1.0},
                                  "shape": [2, 2]},
                        "cint": {"family": "Normal", "params": {"mu": 0.0, "sigma": 0.5},
                                 "shape": [2]},
                    },
                },
            ],
        }
        spec = composite_spec_from_dict(config)
        assert isinstance(spec, CompositeSpec)
        assert spec.n_latent == 2

        compiled = compile_composite_from_dict(config)
        assert isinstance(compiled.vector_field.components[0], DenseLinear)
        with seed(rng_seed=0):
            params = compiled.sample_params()
        assert params[0]["drift"].shape == (2, 2)
        assert params[0]["cint"].shape == (2,)

    def test_ssri_chain_round_trip(self):
        """The Hill/Multiplicative/LinearEdge chain emits the right
        component types in the right order from a serialised dict."""
        DOSE, ADHERENCE, C_P, C_E, AFFECTIVE = 0, 1, 2, 3, 4
        config = {
            "n_latent": 5,
            "components": [
                {
                    "kind": "DiagonalDecay",
                    "priors": {
                        "decay": {"family": "Gamma",
                                  "params": {"concentration": 2.0, "rate": 4.0},
                                  "shape": [5]},
                    },
                },
                {
                    "kind": "Intercept",
                    "priors": {"cint": {"family": "Normal",
                                        "params": {"mu": 0.0, "sigma": 1.0},
                                        "shape": [5]}},
                },
                {
                    "kind": "MultiplicativeEdge",
                    "source_a": DOSE, "source_b": ADHERENCE, "target": C_P,
                    "priors": {"weight": {"family": "Normal",
                                          "params": {"mu": 0.0, "sigma": 1.0}}},
                },
                {
                    "kind": "LinearEdge",
                    "source": C_P, "target": C_E,
                    "priors": {"weight": {"family": "LogNormal",
                                          "params": {"mu": 0.0, "sigma": 0.5}}},
                },
                {
                    "kind": "HillEdge",
                    "source": C_E, "target": AFFECTIVE,
                    "priors": {
                        "Emax": {"family": "LogNormal",
                                 "params": {"mu": 0.0, "sigma": 0.5}},
                        "EC50": {"family": "LogNormal",
                                 "params": {"mu": 0.0, "sigma": 0.5}},
                        "n":    {"family": "TruncatedNormal",
                                 "params": {"mu": 2.0, "sigma": 0.5,
                                            "lower": 1.0, "upper": 4.0}},
                    },
                },
            ],
        }
        compiled = compile_composite_from_dict(config)
        kinds = [type(c).__name__ for c in compiled.vector_field.components]
        assert kinds == [
            "DiagonalDecay", "Intercept", "MultiplicativeEdge",
            "LinearEdge", "HillEdge",
        ]
        # Indices preserved
        mult = compiled.vector_field.components[2]
        lin = compiled.vector_field.components[3]
        hill = compiled.vector_field.components[4]
        assert (mult.source_a, mult.source_b, mult.target) == (DOSE, ADHERENCE, C_P)
        assert (lin.source, lin.target) == (C_P, C_E)
        assert (hill.source, hill.target) == (C_E, AFFECTIVE)


class TestEndToEndDriftFromDict:
    def test_dict_compiled_drift_matches_hand_built(self):
        """Sample params from a dict-compiled spec, plug into the vector
        field, and verify the drift output is numerically sensible."""
        config = {
            "n_latent": 2,
            "components": [
                {"kind": "DiagonalDecay",
                 "priors": {"decay": {"family": "Delta",
                                      "params": {"value": 1.0},
                                      "shape": [2]}}},
                {"kind": "LinearEdge",
                 "source": 0, "target": 1,
                 "priors": {"weight": {"family": "Delta",
                                       "params": {"value": 0.3}}}},
            ],
        }
        compiled = compile_composite_from_dict(config)
        params = compiled.sample_params()
        eta = jnp.array([2.0, 1.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        drift = compiled.vector_field(jnp.asarray(0.0), eta, args)
        # decay·η = [-2, -1]; LinearEdge adds 0.3·2 = 0.6 to target=1
        assert jnp.allclose(drift, jnp.array([-2.0, -0.4]), atol=1e-6)


class TestErrorPaths:
    def test_unknown_kind_raises(self):
        import pytest

        config = {"n_latent": 1, "components": [{"kind": "Bogus"}]}
        with pytest.raises(ValueError, match="Bogus"):
            composite_spec_from_dict(config)

    def test_missing_required_prior_raises(self):
        import pytest

        config = {
            "n_latent": 2,
            "components": [
                {"kind": "HillEdge",
                 "source": 0, "target": 1,
                 "priors": {"Emax": {"family": "LogNormal",
                                     "params": {"mu": 0.0, "sigma": 0.5}}}},
            ],
        }
        with pytest.raises(ValueError, match=r"EC50|missing required prior 'n'"):
            composite_spec_from_dict(config)


class TestBlockSpecEquivalence:
    """Numerical-equivalence pins for the non-drift ``BlockSpec`` family.

    Each ``*BlockSpec`` in ``dynamics/blocks.py`` delegates to the shared
    structural assembly helpers. The spec-level assembly methods and the
    block sample path must produce identical output for the same free values.
    """

    def test_diffusion_block_matches_spec_assembly(self):
        from numpyro.handlers import condition, seed

        from nof1_causal_lab.models.ssm import SSMSpec
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            StructuralDenseLinearSpec,
            StructuralInterceptSpec,
            compile_composite,
            linear_drift_spec,
        )
        from nof1_causal_lab.models.ssm.model import (
            full_diagonal_mask,
            zero_diagonal_mask,
            zero_square_mask,
            zero_vector_mask,
        )
        from nof1_causal_lab.models.ssm.structure import (
            DiffusionBlockSpec,
            SparseMatrixBlockSpec,
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
                drift_template=jnp.zeros((2, 2)),
                cint_mask=zero_vector_mask(2),
                cint_template=jnp.zeros(2),
            ),
            diffusion_block=DiffusionBlockSpec(
                n_latent=2,
                diffusion_chol_mask=np.diag(full_diagonal_mask(2)),
                diffusion_chol_template=jnp.eye(2),
            ),
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
        diag_vals = jnp.array([0.4, 0.6])
        expected = spec.assemble_diffusion(diag_vals, None)

        block = DiffusionBlockSpec(
            n_latent=2,
            diffusion_chol_mask=spec.diffusion_block.diffusion_chol_mask,
            diffusion_chol_template=jnp.asarray(
                spec.diffusion_block.diffusion_chol_template
            ),
            time_invariant_mask=spec.diffusion_block.time_invariant_mask,
            diag_prior=ndist.LogNormal(0.0, 1.0),
        )
        drift_component, cint_component = spec.structural_drift_components()
        composite = CompositeSpec(
            n_latent=2,
            components=(
                StructuralDenseLinearSpec(
                    n_latent=2,
                    drift_diag_mask=drift_component.drift_diag_mask,
                    drift_offdiag_mask=drift_component.drift_offdiag_mask,
                    drift_template=jnp.asarray(drift_component.drift_template),
                ),
                StructuralInterceptSpec(
                    n_latent=2,
                    cint_mask=cint_component.cint_mask,
                    cint_template=jnp.asarray(cint_component.cint_template),
                ),
            ),
        )
        compile_composite(composite)  # smoke; not used here directly

        with seed(rng_seed=0), condition(data={"diffusion_diag_free": diag_vals}):
            block_assembled = block.sample_params(prefix="")["diffusion"]

        np.testing.assert_allclose(block_assembled, expected, atol=1e-12)

    def test_sparse_vector_block_matches_spec_assembly(self):
        from numpyro.handlers import condition, seed

        from nof1_causal_lab.models.ssm import SSMSpec
        from nof1_causal_lab.models.ssm.dynamics import default_linear_drift_spec
        from nof1_causal_lab.models.ssm.model import full_vector_mask
        from nof1_causal_lab.models.ssm.structure import (
            SparseMatrixBlockSpec,
            SparseVectorBlockSpec,
            default_diffusion_block,
            default_input_effect_block,
            default_manifest_chol_block,
            default_manifest_means_block,
            default_static_state_sd_block,
            default_t0_chol_block,
        )

        spec = SSMSpec(
            n_latent=3,
            n_manifest=1,
            drift_spec=default_linear_drift_spec(3),
            diffusion_block=default_diffusion_block(3),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=1, n_cols=3,
                mask=np.zeros((1, 3), dtype=bool),
                template=jnp.array([[0.0, 1.0, 0.0]]),
                free_site_name="lambda_free", det_site_name="lambda",
            ),
            manifest_means_block=default_manifest_means_block(1),
            manifest_chol_block=default_manifest_chol_block(1),
            t0_means_block=SparseVectorBlockSpec(
                n=3,
                mask=full_vector_mask(3),
                template=jnp.zeros(3),
                free_site_name="t0_means_free", det_site_name="t0_means",
            ),
            t0_chol_block=default_t0_chol_block(3),
            input_effect_block=default_input_effect_block(3),
            static_state_sd_block=default_static_state_sd_block(),
        )
        free = jnp.array([0.1, -0.2, 0.3])
        expected = spec.assemble_t0_means(free)

        block = SparseVectorBlockSpec(
            n=3,
            mask=spec.t0_means_block.mask,
            template=jnp.asarray(spec.t0_means_block.template),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            prior=ndist.Normal(jnp.zeros(3), 1.0),
        )
        with seed(rng_seed=0), condition(data={"t0_means_free": free}):
            assembled = block.sample_params(prefix="")["t0_means"]
        np.testing.assert_allclose(assembled, expected, atol=1e-12)


class TestCompositeSpecRoundTrip:
    """``composite_spec_to_dict`` is the inverse of
    ``composite_spec_from_dict``. Round-trip must preserve the structural
    info and dict-config priors so persistence works end-to-end —
    Stage 4 can emit a composite spec, persist it as JSON, and downstream
    deserialise + materialise via the same chain.
    """

    def test_ssri_chain_round_trips(self):
        from nof1_causal_lab.models.ssm.dynamics import composite_spec_to_dict

        DOSE, ADHERENCE, C_P, C_E, AFFECTIVE = 0, 1, 2, 3, 4
        config = {
            "n_latent": 5,
            "components": [
                {
                    "kind": "DiagonalDecay",
                    "priors": {
                        "decay": {
                            "family": "Gamma",
                            "params": {"concentration": 2.0, "rate": 4.0},
                            "shape": [5],
                        },
                    },
                },
                {
                    "kind": "Intercept",
                    "priors": {
                        "cint": {
                            "family": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "shape": [5],
                        }
                    },
                },
                {
                    "kind": "MultiplicativeEdge",
                    "source_a": DOSE,
                    "source_b": ADHERENCE,
                    "target": C_P,
                    "priors": {
                        "weight": {"family": "Normal", "params": {"mu": 0.0, "sigma": 1.0}}
                    },
                },
                {
                    "kind": "LinearEdge",
                    "source": C_P,
                    "target": C_E,
                    "priors": {
                        "weight": {"family": "LogNormal", "params": {"mu": 0.0, "sigma": 0.5}}
                    },
                },
                {
                    "kind": "HillEdge",
                    "source": C_E,
                    "target": AFFECTIVE,
                    "priors": {
                        "Emax": {"family": "LogNormal", "params": {"mu": 0.0, "sigma": 0.5}},
                        "EC50": {"family": "LogNormal", "params": {"mu": 0.0, "sigma": 0.5}},
                        "n": {
                            "family": "TruncatedNormal",
                            "params": {"mu": 2.0, "sigma": 0.5, "lower": 1.0, "upper": 4.0},
                        },
                    },
                },
            ],
        }
        spec = composite_spec_from_dict(config)
        round_tripped = composite_spec_to_dict(spec)
        assert round_tripped == config

    def test_structural_dense_linear_round_trips(self):
        import jax.numpy as jnp
        import numpy as np

        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            StructuralDenseLinearSpec,
            StructuralInterceptSpec,
            composite_spec_from_dict,
            composite_spec_to_dict,
        )

        spec = CompositeSpec(
            n_latent=2,
            components=(
                StructuralDenseLinearSpec(
                    n_latent=2,
                    drift_diag_mask=np.array([True, True], dtype=bool),
                    drift_offdiag_mask=np.array([[False, True], [True, False]], dtype=bool),
                    drift_template=jnp.zeros((2, 2)),
                    stability_margin=0.05,
                    time_invariant_mask=None,
                    base_decay_prior={
                        "family": "Gamma",
                        "params": {"concentration": 2.0, "rate": 4.0},
                        "shape": [2],
                    },
                    offdiag_prior={
                        "family": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.5},
                        "shape": [2],
                    },
                ),
                StructuralInterceptSpec(
                    n_latent=2,
                    cint_mask=np.array([True, False], dtype=bool),
                    cint_template=jnp.zeros(2),
                    cint_prior={
                        "family": "Normal",
                        "params": {"mu": 0.0, "sigma": 1.0},
                        "shape": [1],
                    },
                ),
            ),
        )
        payload = composite_spec_to_dict(spec)
        restored = composite_spec_from_dict(payload)

        # Structural fields round-trip
        c0 = restored.components[0]
        np.testing.assert_array_equal(c0.drift_diag_mask, [True, True])
        np.testing.assert_array_equal(c0.drift_offdiag_mask, [[False, True], [True, False]])
        np.testing.assert_allclose(c0.drift_template, np.zeros((2, 2)))
        assert c0.stability_margin == 0.05
        assert c0.time_invariant_mask is None

        c1 = restored.components[1]
        np.testing.assert_array_equal(c1.cint_mask, [True, False])
        np.testing.assert_allclose(c1.cint_template, np.zeros(2))

        # And the dict→dict round-trip is exact
        assert composite_spec_to_dict(restored) == payload

    def test_ssm_compiler_serializes_drift_spec(self):
        """``serialize_ssm_spec`` / ``deserialize_ssm_spec`` round-trip a
        populated ``drift_spec`` via the dict-config layer."""
        import jax.numpy as jnp

        from nof1_causal_lab.models.ssm import SSMSpec
        from nof1_causal_lab.models.ssm.compile.artifact import (
            deserialize_ssm_spec,
            serialize_ssm_spec,
        )
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            DiagonalDecaySpec,
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

        drift_spec = CompositeSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(
                    decay_prior={
                        "family": "Gamma",
                        "params": {"concentration": 2.0, "rate": 4.0},
                        "shape": [2],
                    }
                ),
            ),
        )
        spec = SSMSpec(
            n_latent=2,
            n_manifest=1,
            drift_spec=drift_spec,
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
        payload = serialize_ssm_spec(spec)
        assert isinstance(payload["drift_spec"], dict)
        assert payload["drift_spec"]["n_latent"] == 2
        assert payload["drift_spec"]["components"][0]["kind"] == "DiagonalDecay"

        restored = deserialize_ssm_spec(payload)
        assert restored.drift_spec is not None
        assert restored.drift_spec.n_latent == 2
        from nof1_causal_lab.models.ssm.dynamics import DiagonalDecaySpec as _DD

        assert isinstance(restored.drift_spec.components[0], _DD)
