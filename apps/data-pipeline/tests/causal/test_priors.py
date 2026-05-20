"""Tests for non-linear component prior factories.

Each factory returns a NumPyro Distribution. Tests verify:
- The returned object is a Distribution.
- Sampling from it inside a seeded NumPyro context gives values in the
  documented range (positive support for LogNormal, [1, 4] for Hill n,
  etc.).
- The factories integrate with ``compile_composite`` to produce
  fittable spec → vector-field pipelines.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as ndist
import pytest
from numpyro.handlers import seed

from nof1_causal_lab.models.ssm.dynamics import (
    CompositeSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    LinearEdgeSpec,
    MultiplicativeEdgeSpec,
    compile_composite,
)
from nof1_causal_lab.models.ssm.dynamics.priors import (
    diagonal_decay_prior,
    effect_compartment_rate_prior,
    hill_ec50_prior,
    hill_emax_prior,
    hill_n_prior,
    linear_edge_weight_prior,
    multiplicative_weight_prior,
)


class TestPriorFactoriesReturnDistributions:
    def test_hill_emax_is_lognormal(self):
        d = hill_emax_prior()
        assert isinstance(d, ndist.LogNormal)

    def test_hill_ec50_is_lognormal(self):
        d = hill_ec50_prior()
        assert isinstance(d, ndist.LogNormal)

    def test_hill_n_is_a_distribution_with_truncated_support(self):
        # NumPyro's TruncatedNormal is a callable factory, not a class, so
        # we check for Distribution-ness + correct support bounds instead.
        d = hill_n_prior()
        assert isinstance(d, ndist.Distribution)
        # Truncated to [1, 4] — samples should respect that.
        import jax.random as jr

        samples = d.sample(jr.PRNGKey(0), sample_shape=(50,))
        assert bool(jnp.all(samples >= 1.0))
        assert bool(jnp.all(samples <= 4.0))

    def test_multiplicative_weight_is_normal(self):
        d = multiplicative_weight_prior()
        assert isinstance(d, ndist.Normal)

    def test_linear_edge_weight_is_normal(self):
        d = linear_edge_weight_prior()
        assert isinstance(d, ndist.Normal)

    def test_effect_compartment_rate_is_lognormal(self):
        d = effect_compartment_rate_prior()
        assert isinstance(d, ndist.LogNormal)

    def test_diagonal_decay_is_gamma(self):
        d = diagonal_decay_prior()
        assert isinstance(d, ndist.Gamma)


class TestPriorSamplesRespectSupport:
    def test_hill_n_stays_within_bounds(self):
        """Many samples from Hill n should all lie in [1, 4]."""
        import jax.random as jr

        key = jr.PRNGKey(0)
        samples = hill_n_prior().sample(key, sample_shape=(200,))
        assert bool(jnp.all(samples >= 1.0))
        assert bool(jnp.all(samples <= 4.0))

    def test_hill_emax_is_positive(self):
        import jax.random as jr

        samples = hill_emax_prior().sample(jr.PRNGKey(1), sample_shape=(200,))
        assert bool(jnp.all(samples > 0))

    def test_hill_ec50_is_positive(self):
        import jax.random as jr

        samples = hill_ec50_prior().sample(jr.PRNGKey(2), sample_shape=(200,))
        assert bool(jnp.all(samples > 0))

    def test_effect_compartment_rate_is_positive(self):
        import jax.random as jr

        samples = effect_compartment_rate_prior().sample(
            jr.PRNGKey(3), sample_shape=(200,)
        )
        assert bool(jnp.all(samples > 0))

    def test_diagonal_decay_is_positive(self):
        import jax.random as jr

        samples = diagonal_decay_prior().sample(jr.PRNGKey(4), sample_shape=(200,))
        assert bool(jnp.all(samples > 0))


class TestPriorsIntegrateWithCompiler:
    """The end-to-end pipeline: defaults → spec → compile_composite →
    sample inside a seeded NumPyro context → finite param values."""

    def test_ssri_spec_with_default_priors_samples(self):
        DOSE, ADHERENCE, C_P, C_E, AFFECTIVE = 0, 1, 2, 3, 4

        spec = CompositeSpec(
            n_latent=5,
            components=(
                DiagonalDecaySpec(
                    decay_prior=ndist.LogNormal(jnp.zeros(5), 0.5)
                ),
                MultiplicativeEdgeSpec(
                    source_a=DOSE,
                    source_b=ADHERENCE,
                    target=C_P,
                    weight_prior=multiplicative_weight_prior(scale=0.5),
                ),
                LinearEdgeSpec(
                    source=C_P,
                    target=C_E,
                    weight_prior=effect_compartment_rate_prior(),
                ),
                HillEdgeSpec(
                    source=C_E,
                    target=AFFECTIVE,
                    emax_prior=hill_emax_prior(),
                    ec50_prior=hill_ec50_prior(),
                    n_prior=hill_n_prior(),
                ),
            ),
        )
        compiled = compile_composite(spec)

        with seed(rng_seed=0):
            params = compiled.sample_params()

        # All sampled params should be finite
        for slice_params in params:
            for value in slice_params.values():
                assert bool(jnp.all(jnp.isfinite(value))), (
                    f"Non-finite sample from defaults: {value}"
                )

        # Spot-check pharmacological constraints
        hill_n = params[3]["n"]
        assert 1.0 <= float(hill_n) <= 4.0
        emax = params[3]["Emax"]
        assert float(emax) > 0
        ec50 = params[3]["EC50"]
        assert float(ec50) > 0


class TestPriorParameterOverrides:
    def test_hill_emax_loc_shifts_median(self):
        """Overriding ``loc`` should shift the LogNormal median."""
        import jax.numpy as jnp
        import jax.random as jr

        base = hill_emax_prior(loc=0.0).sample(jr.PRNGKey(0), sample_shape=(2000,))
        shifted = hill_emax_prior(loc=jnp.log(5.0)).sample(
            jr.PRNGKey(0), sample_shape=(2000,)
        )
        assert float(jnp.median(shifted)) == pytest.approx(5.0, rel=0.1)
        assert float(jnp.median(base)) == pytest.approx(1.0, rel=0.1)

    def test_hill_n_tighter_scale_concentrates(self):
        """Smaller ``scale`` should give a tighter distribution."""
        import jax.random as jr

        wide = hill_n_prior(scale=1.0).sample(jr.PRNGKey(0), sample_shape=(2000,))
        narrow = hill_n_prior(scale=0.1).sample(jr.PRNGKey(0), sample_shape=(2000,))
        assert float(jnp.std(narrow)) < float(jnp.std(wide))
