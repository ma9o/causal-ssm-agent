"""Tests for composable, windowed latent clamps (Stage 6 unified simulation).

Covers ``vmap_simulate_clamps_from_state`` + ``build_segment_bounds``: full-horizon
clamps, finite windows that release to natural dynamics, mid-rollout onset (the
segmentation correctness check), multiple simultaneous clamps, trajectory clamps,
and baseline vs abducted start states.
"""

from __future__ import annotations

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.counterfactual import (
    ClampSpec,
    build_segment_bounds,
    vmap_simulate_clamps_from_state,
)
from nof1_causal_lab.models.ssm.dynamics import VectorField
from nof1_causal_lab.models.ssm.dynamics.edges import DenseLinear

# var1 is driven by var0; both stable. Baseline steady state is η* = -A⁻¹c = [1, 1].
_PARAMS = ({"drift": jnp.array([[-1.0, 0.0], [0.5, -1.0]]), "cint": jnp.array([1.0, 0.5])},)
_TIME_GRID = jnp.linspace(0.0, 30.0, 31)  # daily grid, day == index


def _vf() -> VectorField:
    return VectorField(n_latent=2, components=(DenseLinear(),))


def _run(clamps, initial_states=None):
    baseline, action, effect = vmap_simulate_clamps_from_state(
        _vf(), [_PARAMS], initial_states=initial_states, clamps=clamps, time_grid=_TIME_GRID
    )
    return baseline[0], action[0], effect[0]  # single draw → (T, n_latent)


def test_segment_bounds_split_at_window_edges():
    clamps = [
        ClampSpec(index=0, mode="set", value=0.5, from_day=0.0, to_day=14.0),
        ClampSpec(index=1, mode="shift", amount=1.0, from_day=20.0),
    ]
    assert build_segment_bounds(_TIME_GRID, clamps) == [(0, 14), (14, 20), (20, 30)]
    # No windows → a single segment.
    assert build_segment_bounds(_TIME_GRID, [ClampSpec(index=0, mode="shift", amount=1.0)]) == [
        (0, 30)
    ]


def test_full_horizon_shift_holds_and_propagates():
    _baseline, action, effect = _run([ClampSpec(index=0, mode="shift", amount=1.0)])
    # var0 held at baseline(1) + 1 = 2 across the whole horizon.
    assert jnp.allclose(action[:, 0], 2.0, atol=0.05)
    # Positive coupling lifts var1 by the end.
    assert float(effect[-1, 1]) > 0.1
    assert jnp.isclose(effect[-1, 0], 1.0, atol=0.05)


def test_finite_window_releases_to_natural():
    # Clamp var0 := 3 over [0, 10), then release.
    _baseline, action, _effect = _run(
        [ClampSpec(index=0, mode="set", value=3.0, from_day=0.0, to_day=10.0)]
    )
    assert jnp.isclose(action[5, 0], 3.0, atol=0.05)  # inside window
    assert jnp.isclose(action[30, 0], 1.0, atol=0.1)  # relaxed back to steady state


def test_mid_rollout_onset_pins_exactly():
    # The segmentation correctness check: a set clamp opening at day 10 must JUMP to
    # the value at day 10 (not merely hold its slope from the natural value).
    _baseline, action, _effect = _run(
        [ClampSpec(index=0, mode="set", value=3.0, from_day=10.0, to_day=20.0)]
    )
    assert jnp.isclose(action[5, 0], 1.0, atol=0.05)  # natural before the window
    assert jnp.isclose(action[15, 0], 3.0, atol=0.05)  # pinned inside the window
    assert jnp.isclose(action[30, 0], 1.0, atol=0.1)  # released after the window


def test_multiple_simultaneous_clamps():
    _baseline, action, _effect = _run(
        [
            ClampSpec(index=0, mode="set", value=3.0),
            ClampSpec(index=1, mode="set", value=5.0),
        ]
    )
    assert jnp.allclose(action[:, 0], 3.0, atol=0.05)
    assert jnp.allclose(action[:, 1], 5.0, atol=0.05)


def test_trajectory_clamp_tracks_values():
    _baseline, action, _effect = _run(
        [ClampSpec(index=0, mode="trajectory", values=(1.0, 3.0), from_day=0.0, to_day=10.0)]
    )
    # Linear interpolation 1 → 3 across [0, 10].
    assert jnp.isclose(action[0, 0], 1.0, atol=0.05)
    assert jnp.isclose(action[5, 0], 2.0, atol=0.1)
    assert jnp.isclose(action[10, 0], 3.0, atol=0.05)


def test_abducted_start_evolves_from_given_state():
    initial_states = jnp.array([[0.0, 0.0]])
    baseline, _action, _effect = _run(
        [ClampSpec(index=0, mode="shift", amount=1.0)], initial_states=initial_states
    )
    # Reference path starts at the abducted state and relaxes toward steady state [1, 1].
    assert jnp.allclose(baseline[0], jnp.array([0.0, 0.0]), atol=0.05)
    assert jnp.allclose(baseline[-1], jnp.array([1.0, 1.0]), atol=0.1)
