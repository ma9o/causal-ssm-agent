"""Tests for the Diffrax/Optimistix counterfactual API.

Covers: LinearVectorField + Intervention DSL, simulate / simulate_pair,
compute_steady_state, summarize_draws / summarize_temporal_effect /
resolve_action_value, compute_interventions, approximate_abducted_state.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nof1_causal_lab.models.ssm.counterfactual import (
    approximate_abducted_state,
    build_time_grid,
    compute_interventions,
    linear_vector_field,
    resolve_action_value,
    summarize_draws,
    summarize_temporal_effect,
    vmap_steady_state_effect_composite,
)
from nof1_causal_lab.models.ssm.dynamics import (
    EdgeInputOverride,
    Intervention,
    SimulationConfig,
    VariableOverride,
    VectorFieldArgs,
    compute_steady_state,
    constant_value,
    linear_ramp,
    simulate,
    simulate_pair,
)

# =============================================================================
# LinearVectorField + Intervention DSL
# =============================================================================


class TestLinearVectorField:
    def test_drift_matches_matrix_form(self):
        vf = linear_vector_field(n_latent=2)
        params = ({"drift": jnp.array([[-1.0, 0.5], [0.0, -2.0]]), "cint": jnp.array([0.3, -0.1])},)
        eta = jnp.array([1.0, 2.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        out = vf(jnp.asarray(0.0), eta, args)
        expected = params[0]["drift"] @ eta + params[0]["cint"]
        assert jnp.allclose(out, expected, atol=1e-6)

    def test_variable_override_forces_zero_drift_for_constant(self):
        vf = linear_vector_field(n_latent=2)
        params = ({"drift": jnp.array([[-1.0, 0.5], [0.0, -2.0]]), "cint": jnp.zeros(2)},)
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=constant_value(jnp.asarray(5.0))),)
        )
        args = VectorFieldArgs(params=params, intervention=intervention)
        out = vf(jnp.asarray(0.0), jnp.array([5.0, 2.0]), args)
        assert jnp.isclose(out[0], 0.0, atol=1e-6)

    def test_edge_input_override_changes_target_only(self):
        vf = linear_vector_field(n_latent=2)
        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        params = ({"drift": A, "cint": jnp.zeros(2)},)
        eta = jnp.array([1.0, 0.0])
        baseline_args = VectorFieldArgs(params=params, intervention=Intervention.none())
        baseline_drift = vf(jnp.asarray(0.0), eta, baseline_args)
        intervention = Intervention(
            overrides=(EdgeInputOverride(source=0, target=1, value_fn=constant_value(jnp.asarray(10.0))),)
        )
        intervened_drift = vf(jnp.asarray(0.0), eta, VectorFieldArgs(params=params, intervention=intervention))
        assert jnp.isclose(intervened_drift[0], baseline_drift[0], atol=1e-6)
        assert intervened_drift[1] > baseline_drift[1]

    def test_initial_condition_applies_variable_overrides(self):
        vf = linear_vector_field(n_latent=2)
        intervention = Intervention(
            overrides=(VariableOverride(index=1, value_fn=constant_value(jnp.asarray(3.0))),)
        )
        args = VectorFieldArgs(params=({"drift": -jnp.eye(2), "cint": jnp.zeros(2)},), intervention=intervention)
        y0 = vf.initial_condition(jnp.array([1.0, 2.0]), args)
        assert jnp.isclose(y0[0], 1.0)
        assert jnp.isclose(y0[1], 3.0)


# =============================================================================
# compute_steady_state (numerical, replaces closed-form -A^{-1} c)
# =============================================================================


class TestComputeSteadyState:
    def test_matches_inverse_for_diagonal_drift(self):
        vf = linear_vector_field(n_latent=2)
        params = ({"drift": -jnp.eye(2), "cint": jnp.array([1.0, 2.0])},)
        ss = compute_steady_state(vf, params, Intervention.none())
        assert jnp.allclose(ss, jnp.array([1.0, 2.0]), atol=1e-5)

    def test_satisfies_residual(self):
        vf = linear_vector_field(n_latent=2)
        params = ({"drift": jnp.array([[-2.0, 0.5], [0.3, -1.5]]), "cint": jnp.array([1.0, -0.5])},)
        ss = compute_steady_state(vf, params, Intervention.none())
        residual = params[0]["drift"] @ ss + params[0]["cint"]
        assert jnp.allclose(residual, 0.0, atol=1e-4)

    def test_intervention_propagates_downstream(self):
        vf = linear_vector_field(n_latent=2)
        params = (
            {
                "drift": jnp.array([[-1.0, 0.0], [0.5, -1.0]]),
                "cint": jnp.array([2.0, 1.0]),
            },
        )
        baseline = compute_steady_state(vf, params, Intervention.none())
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=constant_value(jnp.asarray(5.0))),)
        )
        intervened = compute_steady_state(vf, params, intervention, initial_guess=baseline)
        assert jnp.isclose(intervened[0], 5.0, atol=1e-4)
        assert not jnp.isclose(intervened[1], baseline[1], atol=0.1)


# =============================================================================
# simulate / simulate_pair
# =============================================================================


class TestSimulate:
    def test_no_coupling_no_propagation(self):
        vf = linear_vector_field(n_latent=2)
        params = ({"drift": -jnp.eye(2), "cint": jnp.array([1.0, 2.0])},)
        time_grid = jnp.linspace(0.0, 5.0, 11)
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=constant_value(jnp.asarray(0.0))),)
        )
        baseline = compute_steady_state(vf, params, Intervention.none())
        _, _action, effect = simulate_pair(
            vf, params, Intervention.none(), intervention, baseline, time_grid
        )
        assert jnp.all(jnp.abs(effect[:, 1]) < 1e-3)

    def test_positive_coupling_yields_positive_effect(self):
        vf = linear_vector_field(n_latent=2)
        params = (
            {
                "drift": jnp.array([[-1.0, 0.0], [0.5, -1.0]]),
                "cint": jnp.array([1.0, 0.5]),
            },
        )
        baseline = compute_steady_state(vf, params, Intervention.none())
        time_grid = jnp.linspace(0.0, 20.0, 41)
        intervention = Intervention(
            overrides=(
                VariableOverride(
                    index=0, value_fn=constant_value(baseline[0] + jnp.asarray(1.0))
                ),
            )
        )
        _, _, effect = simulate_pair(
            vf, params, Intervention.none(), intervention, baseline, time_grid
        )
        assert float(effect[-1, 1]) > 0

    def test_clamped_state_tracks_constant(self):
        vf = linear_vector_field(n_latent=2)
        params = ({"drift": jnp.array([[-1.0, 0.0], [0.5, -1.0]]), "cint": jnp.zeros(2)},)
        time_grid = jnp.linspace(0.0, 5.0, 11)
        intervention = Intervention(
            overrides=(VariableOverride(index=0, value_fn=constant_value(jnp.asarray(3.0))),)
        )
        traj = simulate(vf, params, intervention, jnp.array([0.0, 0.0]), time_grid)
        assert jnp.allclose(traj[:, 0], 3.0, atol=1e-3)


# =============================================================================
# Estimand helpers
# =============================================================================


class TestSummarizeTemporalEffect:
    def test_monotonic_increasing_trajectory(self):
        time_grid = jnp.linspace(0.0, 30.0, 301)
        traj = jnp.linspace(0.0, 3.0, 301)
        result = summarize_temporal_effect(traj, time_grid)
        assert result["effect_30d"] > result["effect_7d"]
        assert result["effect_7d"] > result["effect_1d"]

    def test_constant_trajectory(self):
        time_grid = jnp.linspace(0.0, 10.0, 101)
        traj = jnp.ones(101) * 2.5
        result = summarize_temporal_effect(traj, time_grid)
        assert abs(result["effect_1d"] - 2.5) < 1e-4
        assert abs(result["effect_7d"] - 2.5) < 1e-4
        assert abs(result["peak_effect"] - 2.5) < 1e-4

    def test_peak_at_end_for_monotonic(self):
        time_grid = jnp.linspace(0.0, 5.0, 51)
        traj = jnp.linspace(0.0, 5.0, 51)
        result = summarize_temporal_effect(traj, time_grid)
        assert result["peak_effect"] == pytest.approx(5.0, abs=1e-5)


class TestSummarizeDraws:
    def test_reports_mean_interval_and_prob_positive(self):
        draws = jnp.array([-1.0, 0.0, 2.0, 3.0])
        summary = summarize_draws(draws)
        assert summary["mean"] == 1.0
        assert summary["median"] == 1.0
        assert summary["prob_positive"] == 0.5
        assert summary["lower_95"] <= summary["upper_95"]


class TestResolveActionValue:
    def test_set_mode_uses_absolute_value(self):
        resolved = resolve_action_value(jnp.asarray(2.0), mode="set", value=5.0)
        assert float(resolved) == 5.0

    def test_shift_mode_offsets_baseline(self):
        resolved = resolve_action_value(jnp.asarray(2.0), mode="shift", amount=-0.5)
        assert float(resolved) == 1.5


class TestLinearRamp:
    def test_holds_endpoints(self):
        ramp = linear_ramp(
            t_start=jnp.asarray(1.0),
            t_end=jnp.asarray(3.0),
            value_start=jnp.asarray(10.0),
            value_end=jnp.asarray(0.0),
        )
        assert float(ramp(jnp.asarray(0.0))) == 10.0
        assert float(ramp(jnp.asarray(5.0))) == 0.0
        assert float(ramp(jnp.asarray(2.0))) == pytest.approx(5.0, abs=1e-6)


class TestBuildTimeGrid:
    def test_inclusive_uniform_grid(self):
        grid = build_time_grid(0.0, 10.0, 1.0)
        assert float(grid[0]) == 0.0
        assert float(grid[-1]) == pytest.approx(10.0)
        assert len(grid) == 11


# =============================================================================
# compute_interventions orchestrator
# =============================================================================


class TestComputeInterventions:
    def _make_samples(self, n_draws=4, n_latent=3):
        drift = jnp.broadcast_to(-jnp.eye(n_latent), (n_draws, n_latent, n_latent))
        cint = jnp.broadcast_to(jnp.ones(n_latent), (n_draws, n_latent))
        return [({"drift": d, "cint": c},) for d, c in zip(drift, cint, strict=True)]

    def test_diagonal_drift_yields_zero_effects(self):
        samples = self._make_samples()
        results = compute_interventions(
            samples,
            linear_vector_field(n_latent=3),
            treatments=["A", "B"],
            outcome="C",
            latent_names=["A", "B", "C"],
        )
        assert len(results) == 2
        for r in results:
            assert "treatment" in r
            assert r["posterior_draws"] is not None
            mean_effect = sum(r["posterior_draws"]) / len(r["posterior_draws"])
            assert abs(mean_effect) < 1e-3, (
                f"{r['treatment']} should have ~zero effect with diagonal drift"
            )

    def test_outcome_not_in_latent_names(self):
        samples = self._make_samples()
        results = compute_interventions(
            samples,
            linear_vector_field(n_latent=3),
            treatments=["A"],
            outcome="MISSING",
            latent_names=["A", "B", "C"],
        )
        assert len(results) == 1
        assert results[0].get("posterior_draws") is None

    def test_treatment_not_in_latent_names(self):
        samples = self._make_samples()
        results = compute_interventions(
            samples,
            linear_vector_field(n_latent=3),
            treatments=["UNKNOWN"],
            outcome="C",
            latent_names=["A", "B", "C"],
        )
        assert results[0].get("posterior_draws") is None

    def test_no_drift_samples(self):
        results = compute_interventions(
            [],
            linear_vector_field(n_latent=3),
            treatments=["A"],
            outcome="C",
            latent_names=["A", "B", "C"],
        )
        assert results[0].get("posterior_draws") is None

    def test_sorted_by_abs_effect(self):
        n = 4
        A = jnp.array([[-1.0, 0.0, 0.0], [0.8, -1.0, 0.0], [0.1, 0.0, -1.0]])
        samples = [({"drift": A, "cint": jnp.ones(3)},) for _ in range(n)]
        results = compute_interventions(
            samples,
            linear_vector_field(n_latent=3),
            treatments=["A", "B"],
            outcome="C",
            latent_names=["A", "B", "C"],
        )
        means = [
            abs(sum(r["posterior_draws"]) / len(r["posterior_draws"]))
            for r in results
            if r.get("posterior_draws")
        ]
        assert means == sorted(means, reverse=True)

    def test_parameter_samples_are_required(self):
        results = compute_interventions(
            [],
            linear_vector_field(n_latent=2),
            treatments=["A"],
            outcome="B",
            latent_names=["A", "B"],
        )
        assert results[0].get("posterior_draws") is None

    def test_manifest_effects_include_interval_supported_outcome_indicators(self):
        n = 3
        drift = jnp.broadcast_to(
            jnp.array([[-1.0, 0.0], [0.5, -1.0]]),
            (n, 2, 2),
        )
        cint = jnp.broadcast_to(jnp.zeros(2), (n, 2))
        lambda_draws = jnp.broadcast_to(
            jnp.array([[0.0, -1.0]]),
            (n, 1, 2),
        )
        results = compute_interventions(
            [({"drift": d, "cint": c},) for d, c in zip(drift, cint, strict=True)],
            linear_vector_field(n_latent=2),
            treatments=["A"],
            outcome="B",
            latent_names=["A", "B"],
            causal_spec={"measurement": {"model_clock": "1d"}},
            manifest_names=["sleep_problem_search_count"],
            lambda_mean=jnp.mean(lambda_draws, axis=0),
        )

        manifest_effects = results[0].get("manifest_effects")
        assert manifest_effects is not None
        assert "sleep_problem_search_count" in manifest_effects
        mean_effect = sum(results[0]["posterior_draws"]) / len(results[0]["posterior_draws"])
        assert manifest_effects["sleep_problem_search_count"] == pytest.approx(
            -1.0 * mean_effect, abs=1e-4
        )


# =============================================================================
# approximate_abducted_state
# =============================================================================


class TestNumericalCorrectness:
    """Regression tests pinning the numerical Diffrax+Optimistix paths to the
    closed-form linear math they replace. Catches solver-tolerance or step-size
    regressions in the linear path before they get blamed on non-linear
    primitives in Phase 2."""

    def test_trajectory_matches_closed_form(self):
        """``simulate`` vs ``exp(A·t)·η₀ + A⁻¹·(exp(A·t) - I)·c``."""
        import jax.scipy.linalg as jla
        from jax import vmap

        A = jnp.array([[-1.0, 0.0], [0.5, -2.0]])
        c = jnp.array([1.0, -0.5])
        eta0 = jnp.array([0.5, 1.5])
        time_grid = jnp.linspace(0.0, 5.0, 21)
        identity = jnp.eye(A.shape[0])

        def analytic(t):
            expAt = jla.expm(A * t)
            return expAt @ eta0 + jla.solve(A, expAt - identity) @ c

        closed_form = vmap(analytic)(time_grid)
        numerical = simulate(
            linear_vector_field(n_latent=A.shape[0]),
            ({"drift": A, "cint": c},),
            Intervention.none(),
            eta0,
            time_grid,
            config=SimulationConfig(rtol=1e-8, atol=1e-10),
        )

        assert jnp.allclose(numerical, closed_form, atol=1e-6)

    def test_steady_state_matches_inverse_for_coupled_system(self):
        """``compute_steady_state`` vs ``-A⁻¹·c`` for a non-trivial 3-coupled system."""
        import jax.scipy.linalg as jla

        A = jnp.array(
            [
                [-1.5, 0.3, 0.1],
                [0.4, -2.0, 0.2],
                [0.1, 0.5, -1.8],
            ]
        )
        c = jnp.array([1.0, -0.5, 0.2])

        closed_form = -jla.solve(A, c)
        numerical = compute_steady_state(
            linear_vector_field(n_latent=3),
            ({"drift": A, "cint": c},),
            Intervention.none(),
        )

        assert jnp.allclose(numerical, closed_form, atol=1e-6)

    def test_treatment_effect_matches_closed_form_do(self):
        """Composite vmap helper vs hand-computed ``do(η_j=v) - baseline``."""
        import jax.scipy.linalg as jla

        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        c = jnp.array([1.0, 0.5])
        treat_idx, outcome_idx = 0, 1
        shift_size = 1.5

        baseline = -jla.solve(A, c)
        do_value = baseline[treat_idx] + shift_size
        A_mod = A.at[treat_idx, :].set(0.0).at[treat_idx, treat_idx].set(1.0)
        rhs = (-c).at[treat_idx].set(do_value)
        intervened = jla.solve(A_mod, rhs)
        expected = float(intervened[outcome_idx] - baseline[outcome_idx])

        vf = linear_vector_field(n_latent=A.shape[0])
        numerical = float(
            vmap_steady_state_effect_composite(
                vf,
                [({"drift": A, "cint": c},)],
                treat_idx=treat_idx,
                outcome_idx=outcome_idx,
                mode="shift",
                amount=shift_size,
            )[0]
        )

        assert abs(numerical - expected) < 1e-5


class TestApproximateAbductedState:
    def test_smoother_uses_selected_evidence_window(self, monkeypatch):
        captured = {}

        def fake_try_smoother(_ssm_model, observations, times, _site_values, _det_values):
            captured["observations"] = observations
            captured["times"] = times
            return jnp.array([[0.1], [0.2]])

        def fake_assemble_single_deterministics(_posterior_means, _spec):
            return {
                "lambda": jnp.array([[1.0]]),
            }

        monkeypatch.setattr(
            "nof1_causal_lab.models.ssm.counterfactual.abduction._try_smoother",
            fake_try_smoother,
        )
        monkeypatch.setattr(
            "nof1_causal_lab.models.ssm.inference.utils._assemble_single_deterministics",
            fake_assemble_single_deterministics,
        )

        class DummySpec:
            n_manifest = 1
            n_latent = 1

            class ManifestMeansBlock:
                template = jnp.zeros(1)

            manifest_means_block = ManifestMeansBlock()

        observations = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        times = jnp.array([0.0, 1.0, 2.0, 3.0])
        result = approximate_abducted_state(
            samples={"vf_0_decay": jnp.ones((2, 1))},
            ssm_model=object(),
            spec=DummySpec(),
            observations=observations,
            times=times,
            evidence_start_idx=1,
            evidence_end_idx=2,
        )

        assert result["method"] == "kalman_smoother"
        assert jnp.allclose(result["state"], jnp.array([0.2]))
        assert jnp.array_equal(captured["observations"], observations[1:3])
        assert jnp.array_equal(captured["times"], times[1:3])
