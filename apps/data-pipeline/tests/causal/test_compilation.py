"""Tests for the spec → ``VectorField`` compiler.

Three layers of checking:

1. Each ``ComponentSpec`` builds the right ``VectorFieldComponent`` (pure
   structural test, no NumPyro context).
2. ``DynamicsSpec`` end-to-end: component-native linear specs compile to
   the expected vector field; an SSRI-style spec compiles to the same
   components tuple a hand-construction would.
3. ``sample_params`` produces correctly-shaped per-component slices when
   called inside a NumPyro model context.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as ndist
from numpyro.handlers import seed, trace

from nof1_causal_lab.models.ssm.dynamics import (
    DiagonalDecay,
    DiagonalDecaySpec,
    DynamicsSpec,
    HillEdge,
    HillEdgeSpec,
    Intercept,
    InterceptSpec,
    Intervention,
    LinearEdge,
    LinearEdgeSpec,
    MultiplicativeEdge,
    MultiplicativeEdgeSpec,
    VectorFieldArgs,
    compile_dynamics,
)
from nof1_causal_lab.models.ssm.dynamics.edges import StateDecay
from nof1_causal_lab.models.ssm.dynamics.spec import StateDecaySpec


def _delta_prior_fn(compiled, values: dict[str, object] | None = None):
    values = values or {}
    shapes = {site.name: site.shape for site in compiled.site_registry}

    def _prior(site_name: str):
        value = values.get(site_name, 1.0)
        return ndist.Delta(jnp.broadcast_to(jnp.asarray(value), shapes[site_name]))

    return _prior


# =============================================================================
# Per-spec build() tests
# =============================================================================


class TestComponentSpecBuild:
    def test_state_decay_spec_builds_state_decay(self):
        spec = StateDecaySpec(target=1)
        built = spec.build()
        assert isinstance(built, StateDecay)
        assert built.target == 1

    def test_diagonal_decay_spec_builds_diagonal_decay(self):
        spec = DiagonalDecaySpec()
        assert isinstance(spec.build(), DiagonalDecay)

    def test_intercept_spec_builds_intercept(self):
        spec = InterceptSpec()
        assert isinstance(spec.build(), Intercept)

    def test_linear_edge_spec_carries_indices(self):
        spec = LinearEdgeSpec(source=2, target=5)
        built = spec.build()
        assert isinstance(built, LinearEdge)
        assert built.source == 2
        assert built.target == 5

    def test_hill_edge_spec_carries_indices(self):
        spec = HillEdgeSpec(
            source=0,
            target=1,
        )
        built = spec.build()
        assert isinstance(built, HillEdge)
        assert (built.source, built.target) == (0, 1)

    def test_multiplicative_edge_spec_carries_indices(self):
        spec = MultiplicativeEdgeSpec(
            source_a=0,
            source_b=1,
            target=2,
        )
        built = spec.build()
        assert isinstance(built, MultiplicativeEdge)
        assert (built.source_a, built.source_b, built.target) == (0, 1, 2)


# =============================================================================
# Compilation: spec → VectorField
# =============================================================================


class TestCompileDynamics:
    def test_component_linear_spec_compiles_to_expected_field(self):
        """Component-native linear specs produce the expected vector field."""
        decay = jnp.array([1.0, 2.0])
        weight = jnp.array(0.5)
        cint = jnp.array([0.1, -0.2])

        spec = DynamicsSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(),
                LinearEdgeSpec(source=1, target=0),
                InterceptSpec(),
            ),
        )
        compiled = compile_dynamics(spec)

        assert compiled.vector_field.n_latent == 2
        assert len(compiled.vector_field.components) == 3

        params = ({"decay": decay}, {"weight": weight}, {"cint": cint})
        eta = jnp.array([1.5, 0.8])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        compiled_dynamics = compiled.vector_field(jnp.asarray(0.0), eta, args)

        expected = jnp.array([-decay[0] * eta[0] + weight * eta[1], -decay[1] * eta[1]]) + cint
        assert jnp.allclose(compiled_dynamics, expected, atol=1e-6)

    def test_ssri_chain_spec_compiles_to_expected_components(self):
        """SSRI-style spec compiles to (DiagonalDecay, Intercept,
        Multiplicative, Linear, Hill) in order."""
        DOSE, ADHERENCE, C_P, C_E, AFFECTIVE = 0, 1, 2, 3, 4
        spec = DynamicsSpec(
            n_latent=5,
            components=(
                DiagonalDecaySpec(),
                InterceptSpec(),
                MultiplicativeEdgeSpec(
                    source_a=DOSE,
                    source_b=ADHERENCE,
                    target=C_P,
                ),
                LinearEdgeSpec(
                    source=C_P,
                    target=C_E,
                ),
                HillEdgeSpec(
                    source=C_E,
                    target=AFFECTIVE,
                ),
            ),
        )
        compiled = compile_dynamics(spec)
        kinds = [type(c).__name__ for c in compiled.vector_field.components]
        assert kinds == [
            "DiagonalDecay",
            "Intercept",
            "MultiplicativeEdge",
            "LinearEdge",
            "HillEdge",
        ]
        # Spot-check indices made it through
        mult = compiled.vector_field.components[2]
        lin = compiled.vector_field.components[3]
        hill = compiled.vector_field.components[4]
        assert (mult.source_a, mult.source_b, mult.target) == (DOSE, ADHERENCE, C_P)
        assert (lin.source, lin.target) == (C_P, C_E)
        assert (hill.source, hill.target) == (C_E, AFFECTIVE)


# =============================================================================
# sample_params inside a NumPyro context
# =============================================================================


class TestSampleParams:
    def test_component_linear_sample_shape(self):
        spec = DynamicsSpec(
            n_latent=2,
            components=(
                StateDecaySpec(target=0),
                LinearEdgeSpec(source=1, target=0),
            ),
        )
        compiled = compile_dynamics(spec)

        with seed(rng_seed=0):
            params = compiled.sample_params(_delta_prior_fn(compiled))

        assert len(params) == 2
        assert params[0]["decay"].shape == ()
        assert params[1]["weight"].shape == ()

    def test_ssri_chain_sample_shape(self):
        """The full SSRI chain spec produces a 5-tuple of param dicts
        with the expected per-component keys."""
        DOSE, ADHERENCE, C_P, C_E, AFFECTIVE = 0, 1, 2, 3, 4
        spec = DynamicsSpec(
            n_latent=5,
            components=(
                DiagonalDecaySpec(),
                InterceptSpec(),
                MultiplicativeEdgeSpec(
                    source_a=DOSE,
                    source_b=ADHERENCE,
                    target=C_P,
                ),
                LinearEdgeSpec(
                    source=C_P,
                    target=C_E,
                ),
                HillEdgeSpec(
                    source=C_E,
                    target=AFFECTIVE,
                ),
            ),
        )
        compiled = compile_dynamics(spec)

        with seed(rng_seed=42):
            params = compiled.sample_params(_delta_prior_fn(compiled))

        assert len(params) == 5
        assert params[0]["decay"].shape == (5,)
        assert params[1]["cint"].shape == (5,)
        assert params[2]["weight"].shape == ()
        assert params[3]["weight"].shape == ()
        assert params[4]["Emax"].shape == ()
        assert params[4]["EC50"].shape == ()
        assert params[4]["n"].shape == ()

    def test_sample_names_are_prefixed(self):
        """All NumPyro sample sites must be prefixed and disambiguated by
        component index, so multiple compiled dynamics specs in one model
        do not collide."""
        spec = DynamicsSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(),
                LinearEdgeSpec(source=0, target=1),
            ),
        )
        compiled = compile_dynamics(spec, prefix="ssri")

        def _model():
            return compiled.sample_params(_delta_prior_fn(compiled))

        tr = trace(seed(_model, rng_seed=0)).get_trace()
        site_names = set(tr.keys())
        assert "ssri_0_decay" in site_names
        assert "ssri_1_weight" in site_names

    def test_compiled_params_drive_vector_field_end_to_end(self):
        """Sample params from a compiled spec, plug them into the
        compiled vector field, and verify the resulting dynamics makes
        numerical sense."""
        spec = DynamicsSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(),
                LinearEdgeSpec(source=0, target=1),
            ),
        )
        compiled = compile_dynamics(spec)

        with seed(rng_seed=0):
            params = compiled.sample_params(
                _delta_prior_fn(compiled, {"vf_0_decay": jnp.array([1.0, 0.5]), "vf_1_weight": 0.3})
            )

        eta = jnp.array([2.0, 1.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        dynamics = compiled.vector_field(jnp.asarray(0.0), eta, args)
        # Expected: dynamics[0] = -1·2 = -2; dynamics[1] = -0.5·1 + 0.3·2 = 0.1
        assert jnp.allclose(dynamics, jnp.array([-2.0, 0.1]), atol=1e-6)


# =============================================================================
# Empty spec
# =============================================================================


class TestEdgeCases:
    def test_empty_components_yields_zero_dynamics(self):
        """A spec with no components → constant-zero dynamics, no sampling."""
        spec = DynamicsSpec(n_latent=2, components=())
        compiled = compile_dynamics(spec)

        params = compiled.sample_params(_delta_prior_fn(compiled))
        assert params == ()

        eta = jnp.array([1.0, -1.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        dynamics = compiled.vector_field(jnp.asarray(0.0), eta, args)
        assert jnp.allclose(dynamics, jnp.zeros(2), atol=1e-6)
