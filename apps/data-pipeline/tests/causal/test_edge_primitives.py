"""Tests for the drift component primitives and CompositeVectorField.

Three layers:

1. Individual primitives (LinearEdge, HillEdge, MultiplicativeEdge,
   DenseLinear, DiagonalDecay, Intercept) — each in isolation, checking
   their per-component contribution to the drift.
2. CompositeVectorField equivalence — a single ``DenseLinear`` component
   reproduces ``f(t, η) = A·η + c`` exactly (proves the unified path
   subsumes the previous ``LinearVectorField`` regime).
3. SSRI chain integration: ``dose × adherence → C_p → C_e → Hill →
   affective`` exercises MultiplicativeEdge, LinearEdge-as-effect-
   compartment, HillEdge, plus DiagonalDecay and Intercept components.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nof1_causal_lab.models.ssm.counterfactual import (
    CompositeVectorField,
    DenseLinear,
    DiagonalDecay,
    EdgeInputOverride,
    HillEdge,
    Intercept,
    Intervention,
    LinearEdge,
    MultiplicativeEdge,
    VariableOverride,
    VectorFieldArgs,
    compute_steady_state,
    constant_value,
    linear_vector_field,
    simulate,
)

# =============================================================================
# Individual primitives
# =============================================================================


class TestLinearEdge:
    def test_adds_weighted_source_at_target(self):
        edge = LinearEdge(source=0, target=1)
        drift = jnp.zeros(2)
        eta_per_edge = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # eta_per_edge[1, 0] = 3.0
        out = edge.contribute_to_drift(
            drift,
            jnp.zeros(2),
            eta_per_edge,
            jnp.asarray(0.0),
            {"weight": jnp.asarray(0.5)},
        )
        assert float(out[1]) == pytest.approx(1.5)
        assert float(out[0]) == pytest.approx(0.0)


class TestHillEdge:
    def test_zero_at_zero(self):
        edge = HillEdge(source=0, target=1)
        drift = jnp.zeros(2)
        eta_per_edge = jnp.zeros((2, 2))
        params = {"Emax": jnp.asarray(2.0), "EC50": jnp.asarray(1.0), "n": jnp.asarray(2.0)}
        out = edge.contribute_to_drift(drift, jnp.zeros(2), eta_per_edge, jnp.asarray(0.0), params)
        assert float(out[1]) == pytest.approx(0.0, abs=1e-10)

    def test_half_emax_at_ec50(self):
        edge = HillEdge(source=0, target=1)
        drift = jnp.zeros(2)
        eta_per_edge = jnp.array([[0.0, 0.0], [1.0, 0.0]])  # source-as-seen-by-1 == EC50
        params = {"Emax": jnp.asarray(2.0), "EC50": jnp.asarray(1.0), "n": jnp.asarray(2.0)}
        out = edge.contribute_to_drift(drift, jnp.zeros(2), eta_per_edge, jnp.asarray(0.0), params)
        assert float(out[1]) == pytest.approx(1.0, abs=1e-6)

    def test_saturates_at_emax(self):
        edge = HillEdge(source=0, target=1)
        drift = jnp.zeros(2)
        eta_per_edge = jnp.array([[0.0, 0.0], [1000.0, 0.0]])
        params = {"Emax": jnp.asarray(2.0), "EC50": jnp.asarray(1.0), "n": jnp.asarray(2.0)}
        out = edge.contribute_to_drift(drift, jnp.zeros(2), eta_per_edge, jnp.asarray(0.0), params)
        assert float(out[1]) == pytest.approx(2.0, abs=1e-4)

    def test_clamps_negative_source(self):
        edge = HillEdge(source=0, target=1)
        drift = jnp.zeros(2)
        eta_per_edge = jnp.array([[0.0, 0.0], [-5.0, 0.0]])
        params = {"Emax": jnp.asarray(2.0), "EC50": jnp.asarray(1.0), "n": jnp.asarray(2.0)}
        out = edge.contribute_to_drift(drift, jnp.zeros(2), eta_per_edge, jnp.asarray(0.0), params)
        assert float(out[1]) == pytest.approx(0.0, abs=1e-10)


class TestMultiplicativeEdge:
    def test_product_of_two_sources(self):
        edge = MultiplicativeEdge(source_a=0, source_b=1, target=2)
        drift = jnp.zeros(3)
        eta_per_edge = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
            ]
        )
        params = {"weight": jnp.asarray(0.5)}
        out = edge.contribute_to_drift(drift, jnp.zeros(3), eta_per_edge, jnp.asarray(0.0), params)
        assert float(out[2]) == pytest.approx(6.0)


class TestDenseLinear:
    def test_matrix_vector_plus_intercept(self):
        component = DenseLinear()
        A = jnp.array([[-1.0, 0.5], [0.3, -2.0]])
        eta = jnp.array([1.5, 0.8])
        eta_per_edge = jnp.broadcast_to(eta[None, :], (2, 2))
        params = {"drift": A, "cint": jnp.array([0.1, -0.2])}
        out = component.contribute_to_drift(
            jnp.zeros(2), eta, eta_per_edge, jnp.asarray(0.0), params
        )
        expected = A @ eta + params["cint"]
        assert jnp.allclose(out, expected, atol=1e-6)


class TestDiagonalDecay:
    def test_negative_decay_times_state(self):
        component = DiagonalDecay()
        eta = jnp.array([2.0, -1.0])
        params = {"decay": jnp.array([0.5, 1.0])}
        out = component.contribute_to_drift(
            jnp.zeros(2), eta, jnp.zeros((2, 2)), jnp.asarray(0.0), params
        )
        assert jnp.allclose(out, jnp.array([-1.0, 1.0]), atol=1e-6)


class TestInterceptComponent:
    def test_adds_intercept(self):
        component = Intercept()
        params = {"cint": jnp.array([0.3, -0.1])}
        out = component.contribute_to_drift(
            jnp.zeros(2), jnp.zeros(2), jnp.zeros((2, 2)), jnp.asarray(0.0), params
        )
        assert jnp.allclose(out, params["cint"], atol=1e-6)


# =============================================================================
# CompositeVectorField — structural equivalence + intervention semantics
# =============================================================================


class TestCompositeEquivalence:
    """A single ``DenseLinear`` component reproduces ``f(t, η) = A·η + c``
    exactly, proving the unified path subsumes the previous
    ``LinearVectorField`` regime."""

    def test_dense_linear_matches_explicit_factory(self):
        A = jnp.array([[-1.0, 0.5], [0.3, -2.0]])
        cint = jnp.array([0.1, -0.2])
        eta = jnp.array([1.5, 0.8])

        dense_vf = CompositeVectorField(n_latent=2, components=(DenseLinear(),))
        factory_vf = linear_vector_field(n_latent=2)

        params = ({"drift": A, "cint": cint},)
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        dense_drift = dense_vf(jnp.asarray(0.0), eta, args)
        factory_drift = factory_vf(jnp.asarray(0.0), eta, args)
        expected = A @ eta + cint

        assert jnp.allclose(dense_drift, expected, atol=1e-6)
        assert jnp.allclose(factory_drift, expected, atol=1e-6)


class TestCompositeInterventions:
    """Override semantics on the composite path."""

    def test_variable_override_pins_state(self):
        vf = CompositeVectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                Intercept(),
                LinearEdge(source=0, target=1),
            ),
        )
        params = (
            {"decay": jnp.array([1.0, 1.0])},
            {"cint": jnp.array([2.0, 0.0])},
            {"weight": jnp.asarray(0.5)},
        )
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=constant_value(jnp.asarray(5.0))),)
        )
        steady = compute_steady_state(vf, params, intervention)
        assert float(steady[0]) == pytest.approx(5.0, abs=1e-4)
        # eta[1] satisfies -1·eta[1] + 0.5·5 = 0 → eta[1] = 2.5
        assert float(steady[1]) == pytest.approx(2.5, abs=1e-4)

    def test_edge_input_override_changes_target_only(self):
        vf = CompositeVectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                Intercept(),
                LinearEdge(source=0, target=1),
            ),
        )
        params = (
            {"decay": jnp.array([1.0, 1.0])},
            {"cint": jnp.array([2.0, 0.0])},
            {"weight": jnp.asarray(0.5)},
        )
        eta = jnp.array([2.0, 1.0])
        baseline_args = VectorFieldArgs(params=params, intervention=Intervention.none())
        baseline_drift = vf(jnp.asarray(0.0), eta, baseline_args)

        # Replace the source-as-seen-by-target-1 with 10 → drift[1] += 0.5*(10-2)
        intervention = Intervention(
            overrides=(
                EdgeInputOverride(
                    source=0, target=1, value_fn=constant_value(jnp.asarray(10.0))
                ),
            )
        )
        intervened_drift = vf(
            jnp.asarray(0.0),
            eta,
            VectorFieldArgs(params=params, intervention=intervention),
        )
        assert jnp.isclose(intervened_drift[0], baseline_drift[0], atol=1e-6)
        assert intervened_drift[1] > baseline_drift[1]


# =============================================================================
# Effect compartment via matched-rate LinearEdge
# =============================================================================


class TestEffectCompartment:
    """LinearEdge with weight matching the target's DiagonalDecay rate
    implements first-order lag ``dC_e/dt = k_e0 · (C_p − C_e)``. At
    steady state, C_e equals C_p; the half-life is ``ln(2) / k_e0``."""

    def test_steady_state_matches_input(self):
        # Latents: 0 = C_p (pinned via intervention), 1 = C_e
        vf = CompositeVectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                LinearEdge(source=0, target=1),
            ),
        )
        k_e0 = 0.1
        params = (
            {"decay": jnp.array([1.0, k_e0])},
            {"weight": jnp.asarray(k_e0)},
        )
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=constant_value(jnp.asarray(3.0))),)
        )
        steady = compute_steady_state(vf, params, intervention)
        assert float(steady[0]) == pytest.approx(3.0, abs=1e-4)
        assert float(steady[1]) == pytest.approx(3.0, abs=1e-4)

    def test_transient_half_life(self):
        vf = CompositeVectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                LinearEdge(source=0, target=1),
            ),
        )
        k_e0 = 0.1
        params = (
            {"decay": jnp.array([1.0, k_e0])},
            {"weight": jnp.asarray(k_e0)},
        )
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=constant_value(jnp.asarray(3.0))),)
        )
        half_life = jnp.log(2.0) / k_e0
        time_grid = jnp.linspace(0.0, float(half_life) * 4.0, 41)
        traj = simulate(vf, params, intervention, jnp.array([3.0, 0.0]), time_grid)
        ce_at_half_life = float(jnp.interp(half_life, time_grid, traj[:, 1]))
        assert ce_at_half_life == pytest.approx(1.5, abs=0.05)


# =============================================================================
# SSRI integration test: dose × adherence → C_p → C_e → Hill → affective
# =============================================================================


class TestSSRIChain:
    """Full pharmacological chain — multiplicative coupling, effect
    compartment, Hill saturation in series. ``do(dose = 2)`` produces a
    sub-linear affective effect because of Hill curvature."""

    DOSE = 0
    ADHERENCE = 1
    C_P = 2
    C_E = 3
    AFFECTIVE = 4

    def _build(self):
        vf = CompositeVectorField(
            n_latent=5,
            components=(
                DiagonalDecay(),
                Intercept(),
                MultiplicativeEdge(
                    source_a=self.DOSE, source_b=self.ADHERENCE, target=self.C_P
                ),
                LinearEdge(source=self.C_P, target=self.C_E),
                HillEdge(source=self.C_E, target=self.AFFECTIVE),
            ),
        )
        k_p = 1.0
        k_e0 = 0.1
        decay_aff = 1.0
        params = (
            {"decay": jnp.array([1.0, 1.0, k_p, k_e0, decay_aff])},
            {"cint": jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])},
            {"weight": jnp.asarray(k_p)},
            {"weight": jnp.asarray(k_e0)},
            {
                "Emax": jnp.asarray(2.0),
                "EC50": jnp.asarray(1.0),
                "n": jnp.asarray(2.0),
            },
        )
        return vf, params

    def test_baseline_steady_state(self):
        vf, params = self._build()
        steady = compute_steady_state(vf, params, Intervention.none())
        assert float(steady[self.DOSE]) == pytest.approx(1.0, abs=1e-4)
        assert float(steady[self.ADHERENCE]) == pytest.approx(1.0, abs=1e-4)
        assert float(steady[self.C_P]) == pytest.approx(1.0, abs=1e-4)
        assert float(steady[self.C_E]) == pytest.approx(1.0, abs=1e-4)
        assert float(steady[self.AFFECTIVE]) == pytest.approx(1.0, abs=1e-4)

    def test_dose_intervention_propagates_through_hill(self):
        vf, params = self._build()
        intervention = Intervention(
            overrides=(
                VariableOverride(
                    index=self.DOSE, value_fn=constant_value(jnp.asarray(2.0))
                ),
            )
        )
        steady = compute_steady_state(vf, params, intervention)
        assert float(steady[self.DOSE]) == pytest.approx(2.0, abs=1e-4)
        assert float(steady[self.C_P]) == pytest.approx(2.0, abs=1e-3)
        assert float(steady[self.C_E]) == pytest.approx(2.0, abs=1e-3)
        # Hill(2; Emax=2, EC50=1, n=2) = 2 · 4 / (1 + 4) = 1.6
        assert float(steady[self.AFFECTIVE]) == pytest.approx(1.6, abs=1e-3)

    def test_effect_is_sublinear_due_to_hill(self):
        vf, params = self._build()
        baseline_steady = compute_steady_state(vf, params, Intervention.none())
        do_dose_2 = Intervention(
            overrides=(
                VariableOverride(
                    index=self.DOSE, value_fn=constant_value(jnp.asarray(2.0))
                ),
            )
        )
        intervened_steady = compute_steady_state(vf, params, do_dose_2)
        effect = float(
            intervened_steady[self.AFFECTIVE] - baseline_steady[self.AFFECTIVE]
        )
        # Hill(2) - Hill(1) = 1.6 - 1.0 = 0.6 (NOT 1.0 like a linear chain)
        assert effect == pytest.approx(0.6, abs=1e-3)
        assert effect < 1.0

    def test_trajectory_shows_delayed_onset(self):
        vf, params = self._build()
        baseline_steady = compute_steady_state(vf, params, Intervention.none())
        intervention = Intervention(
            overrides=(
                VariableOverride(
                    index=self.DOSE, value_fn=constant_value(jnp.asarray(2.0))
                ),
            )
        )
        time_grid = jnp.linspace(0.0, 60.0, 121)
        traj = simulate(vf, params, intervention, baseline_steady, time_grid)
        affective_traj = traj[:, self.AFFECTIVE]
        day_1 = float(jnp.interp(1.0, time_grid, affective_traj))
        day_60 = float(jnp.interp(60.0, time_grid, affective_traj))
        assert day_1 < 1.15, f"Day-1 affective should still be near baseline 1.0, got {day_1}"
        assert day_60 == pytest.approx(1.6, abs=0.02)
        diffs = jnp.diff(affective_traj)
        assert bool(jnp.all(diffs >= -1e-3)), "Affective response should be monotone"
