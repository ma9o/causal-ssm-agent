"""Tests for ``compute_interventions`` with component-parameter dynamics.

- ``compute_interventions`` runs on a Hill chain.
- The returned dicts match the public schema (``treatment``,
  ``posterior_draws``, optional ``temporal``).
- Sign of the steady-state effect matches the deterministic prediction
  for a single posterior draw at the true parameters.
"""

from __future__ import annotations

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.counterfactual import compute_interventions
from nof1_causal_lab.models.ssm.dynamics import (
    DiagonalDecaySpec,
    DynamicsSpec,
    HillEdgeSpec,
    compile_dynamics,
)


class TestComputeInterventionsDynamics:
    def test_runs_on_hill_chain(self):
        """A 2-latent Hill chain: state[0] decays freely, state[1] is
        Hill-driven by state[0]. Treating state[0] should produce a
        positive steady-state shift on state[1]."""
        spec = DynamicsSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(),
                HillEdgeSpec(
                    source=0,
                    target=1,
                ),
            ),
        )
        compiled = compile_dynamics(spec)
        n_draws = 5
        param_samples = [
            (
                {"decay": jnp.array([0.5, 0.5])},
                {
                    "Emax": jnp.asarray(1.5),
                    "EC50": jnp.asarray(1.0),
                    "n": jnp.asarray(2.0),
                },
            )
            for _ in range(n_draws)
        ]
        results = compute_interventions(
            param_samples=param_samples,
            vector_field=compiled.vector_field,
            treatments=["src"],
            outcome="tgt",
            latent_names=["src", "tgt"],
            shift_size=0.5,
        )
        assert len(results) == 1
        entry = results[0]
        assert entry["treatment"] == "src"
        assert "posterior_draws" in entry
        assert len(entry["posterior_draws"]) == n_draws
        # Increasing 'src' should increase 'tgt' via Hill → positive draws
        assert all(d > 0 for d in entry["posterior_draws"])

    def test_skips_unknown_treatment(self):
        spec = DynamicsSpec(n_latent=1, components=(DiagonalDecaySpec(),))
        compiled = compile_dynamics(spec)
        param_samples = [({"decay": jnp.array([0.5])},)]
        results = compute_interventions(
            param_samples=param_samples,
            vector_field=compiled.vector_field,
            treatments=["nonexistent"],
            outcome="x",
            latent_names=["x"],
        )
        assert len(results) == 1
        assert results[0] == {"treatment": "nonexistent"}

    def test_empty_param_samples_returns_skeletons(self):
        """Dynamics path must not crash on an empty posterior."""
        spec = DynamicsSpec(n_latent=1, components=(DiagonalDecaySpec(),))
        compiled = compile_dynamics(spec)
        results = compute_interventions(
            param_samples=[],
            vector_field=compiled.vector_field,
            treatments=["x"],
            outcome="x",
            latent_names=["x"],
        )
        assert results == [{"treatment": "x"}]
