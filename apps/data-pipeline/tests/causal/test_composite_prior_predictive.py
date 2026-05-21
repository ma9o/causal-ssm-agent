"""Tests for the composite prior-predictive validation surface.

Pins the integration that gives the composite path the same
``validate_*`` shape Stage 4 already uses for the linear path:

- A spec known to be stable produces ``is_valid=True`` and finite trajectories.
- A spec known to be unstable (no decay, strong Hill self-feedback at the
  Hill curve's steep region) produces ``is_valid=False``.
- Stability + finiteness flags are per-draw.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as ndist

from nof1_causal_lab.models.ssm.dynamics import (
    CompositeSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    compile_composite,
)
from nof1_causal_lab.models.ssm.predictive import (
    sample_composite_prior_predictive,
    validate_composite_dynamics,
)


class TestSampleCompositePriorPredictive:
    def test_stable_spec_yields_finite_trajectories(self):
        """A spec with strict decay + bounded Hill produces finite,
        stable draws under generic priors."""
        spec = CompositeSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(decay_prior=ndist.Gamma(2.0, 4.0)),
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
        times = jnp.linspace(0.0, 5.0, 20)
        pp = sample_composite_prior_predictive(compiled, jnp.array([1.0, 0.5]), times, n_draws=10)
        assert pp.trajectories.shape == (10, 20, 2)
        assert bool(jnp.all(pp.finite))
        # Strictly positive decay → all draws should be stable
        assert bool(jnp.all(pp.stable))

    def test_unstable_self_feedback_is_flagged(self):
        """A Hill self-feedback with no decay drives a positive Jacobian
        at the Hill curve's steep region — most draws should be flagged
        as unstable."""
        spec = CompositeSpec(
            n_latent=1,
            components=(
                HillEdgeSpec(
                    source=0,
                    target=0,
                    emax_prior=ndist.LogNormal(2.0, 0.1),  # large Emax
                    ec50_prior=ndist.LogNormal(-1.0, 0.1),  # small EC50
                    n_prior=ndist.TruncatedNormal(loc=3.5, scale=0.1, low=3.0, high=4.0),
                ),
            ),
        )
        compiled = compile_composite(spec)
        times = jnp.linspace(0.0, 1.0, 10)
        pp = sample_composite_prior_predictive(
            compiled,
            jnp.array([0.4]),
            times,
            n_draws=20,
            x_lin=jnp.array([0.4]),  # near the steep region
        )
        # At least half should be flagged unstable
        assert float(jnp.mean(~pp.stable)) >= 0.5


class TestValidateCompositeAssembly:
    """Stage-4-shape assembly validator for composite specs.

    Drives the bridge end-to-end: a dict-config goes in, an
    AssemblyValidation-shaped object comes out — exactly what a Stage 4
    LLM tool or the agentic repair flow needs to call when handed a
    composite spec instead of the linear prior registry runtime.
    """

    def test_valid_dict_config_returns_is_valid_true(self):
        from nof1_causal_lab.models.ssm.predictive import validate_composite_assembly

        config = {
            "n_latent": 1,
            "components": [
                {
                    "kind": "DiagonalDecay",
                    "priors": {
                        "decay": {
                            "family": "Gamma",
                            "params": {"concentration": 2.0, "rate": 4.0},
                            "shape": [1],
                        }
                    },
                },
            ],
        }
        result = validate_composite_assembly(
            config, jnp.array([1.0]), jnp.linspace(0.0, 1.0, 5), n_draws=10
        )
        assert result.compile_ok is True
        assert result.pp_valid is True
        assert result.is_valid is True
        assert result.compiled is not None

    def test_malformed_config_surfaces_compile_error(self):
        from nof1_causal_lab.models.ssm.predictive import validate_composite_assembly

        config = {"n_latent": 1, "components": [{"kind": "Bogus"}]}
        result = validate_composite_assembly(
            config, jnp.array([1.0]), jnp.linspace(0.0, 1.0, 5), n_draws=3
        )
        assert result.compile_ok is False
        assert "Bogus" in (result.compile_error or "")
        assert result.is_valid is False


class TestValidateCompositeDynamics:
    def test_stable_spec_returns_is_valid_true(self):
        spec = CompositeSpec(
            n_latent=2,
            components=(DiagonalDecaySpec(decay_prior=ndist.Gamma(2.0, 4.0)),),
        )
        compiled = compile_composite(spec)
        times = jnp.linspace(0.0, 2.0, 5)
        result = validate_composite_dynamics(compiled, jnp.array([1.0, 1.0]), times, n_draws=10)
        assert result["code"] == "dynamics_stability"
        assert result["is_valid"] is True
        assert result["n_unstable"] == 0
        assert result["primary_score"] == 0.0

    def test_unstable_spec_returns_is_valid_false(self):
        """Spec with no decay and explosive Hill self-feedback fails
        the majority-stable threshold."""
        spec = CompositeSpec(
            n_latent=1,
            components=(
                HillEdgeSpec(
                    source=0,
                    target=0,
                    emax_prior=ndist.LogNormal(2.0, 0.1),
                    ec50_prior=ndist.LogNormal(-1.0, 0.1),
                    n_prior=ndist.TruncatedNormal(loc=3.5, scale=0.1, low=3.0, high=4.0),
                ),
            ),
        )
        compiled = compile_composite(spec)
        times = jnp.linspace(0.0, 0.5, 5)
        result = validate_composite_dynamics(compiled, jnp.array([0.4]), times, n_draws=20)
        assert result["is_valid"] is False
        assert result["primary_score"] > 0.5
        assert len(result["failing_draw_indices"]) >= 10
