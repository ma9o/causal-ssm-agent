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
                    n_prior=ndist.TruncatedNormal(
                        loc=2.0, scale=0.5, low=1.0, high=4.0
                    ),
                ),
            ),
        )
        compiled = compile_composite(spec)
        times = jnp.linspace(0.0, 5.0, 20)
        pp = sample_composite_prior_predictive(
            compiled, jnp.array([1.0, 0.5]), times, n_draws=10
        )
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
                    n_prior=ndist.TruncatedNormal(
                        loc=3.5, scale=0.1, low=3.0, high=4.0
                    ),
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


class TestSampleObservationsFromLatents:
    """The composite path can now emit observations via the ObservationKernel,
    closing the parity gap with the linear ``prior_predictive_runtime``.
    Gaussian is the implemented family; non-Gaussian raises with a clear
    message."""

    def _gaussian_canonical(self, n_latent: int = 2):
        from nof1_causal_lab.artifacts.model_spec import (
            DistributionFamily,
            LinkFunction,
        )
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            DiagonalDecaySpec,
            compile_composite,
            runtime_from_composite,
        )
        from nof1_causal_lab.models.ssm.inference.targets.kernels import (
            build_observation_kernel,
        )

        spec = CompositeSpec(
            n_latent=n_latent,
            components=(DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.5)),),
        )
        compiled = compile_composite(spec)
        H = jnp.array([[1.0, 0.0]])
        d_meas = jnp.array([0.0])
        R = jnp.array([[0.05]])
        import numpy as np

        kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=np.asarray(R),
        )
        return runtime_from_composite(
            compiled,
            init_mean=jnp.zeros(n_latent),
            init_cov=jnp.eye(n_latent) * 0.1,
            diffusion_cov=jnp.eye(n_latent) * 0.01,
            H=H,
            d_meas=d_meas,
            R=R,
            obs_kernel=kernel,
        )

    def test_gaussian_emits_observations_with_correct_shape(self):
        import jax.random as jr

        from nof1_causal_lab.models.ssm.dynamics import (
            sample_observations_from_latents,
        )

        canonical = self._gaussian_canonical(n_latent=2)
        latents = jnp.ones((4, 5, 2))  # (n_draws=4, T=5, n_latent=2)
        observations = sample_observations_from_latents(
            canonical, latents, jr.PRNGKey(0)
        )
        # H is (1, 2), so observation channel dim is 1
        assert observations.shape == (4, 5, 1)
        assert bool(jnp.all(jnp.isfinite(observations)))

    def test_gaussian_observations_centered_on_linear_predictor(self):
        """Sample many draws; the empirical mean should converge to
        ``H @ x + d`` at each timestep."""
        import jax.random as jr

        from nof1_causal_lab.models.ssm.dynamics import (
            sample_observations_from_latents,
        )

        canonical = self._gaussian_canonical(n_latent=2)
        # Fix the latent so any spread comes from observation noise alone.
        x = jnp.array([[2.0, -1.0], [1.0, 0.5]])  # (T=2, n_latent=2)
        latents = jnp.broadcast_to(x, (500, 2, 2))
        observations = sample_observations_from_latents(
            canonical, latents, jr.PRNGKey(1)
        )
        empirical_mean = jnp.mean(observations, axis=0)
        expected = jnp.einsum("ij,tj->ti", canonical.H, x) + canonical.d_meas
        assert jnp.allclose(empirical_mean, expected, atol=0.05)

    def test_beta_works_when_canonical_has_predictive_sampler(self):
        """When the canonical is built with manifest_dists/links/extra_params,
        non-Gaussian observation sampling works via the existing
        ``build_predictive_observation_sampler`` factory — no per-family
        re-implementation needed."""
        import jax.random as jr
        import numpy as np

        from nof1_causal_lab.artifacts.model_spec import (
            DistributionFamily,
            LinkFunction,
        )
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            DiagonalDecaySpec,
            compile_composite,
            runtime_from_composite,
            sample_observations_from_latents,
        )
        from nof1_causal_lab.models.ssm.inference.targets.kernels import (
            build_observation_kernel,
        )

        spec = CompositeSpec(
            n_latent=2,
            components=(DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.5)),),
        )
        compiled = compile_composite(spec)
        R = jnp.eye(1) * 0.05
        beta_kernel = build_observation_kernel(
            DistributionFamily.BETA,
            LinkFunction.LOGIT,
            extra_params={"obs_concentration": jnp.asarray([50.0])},
            manifest_cov=np.asarray(R),
        )
        canonical = runtime_from_composite(
            compiled,
            init_mean=jnp.zeros(2),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.01,
            H=jnp.array([[1.0, 0.0]]),
            d_meas=jnp.zeros(1),
            R=R,
            obs_kernel=beta_kernel,
            manifest_dists=(DistributionFamily.BETA,),
            manifest_links=(LinkFunction.LOGIT,),
            obs_extra_params={"obs_concentration": jnp.asarray([50.0])},
        )
        assert canonical.predictive_sampler is not None
        # 5 draws × 4 timesteps × 2 latent dims
        latents = jnp.zeros((5, 4, 2))
        observations = sample_observations_from_latents(
            canonical, latents, jr.PRNGKey(0)
        )
        assert observations.shape == (5, 4, 1)
        # Beta is supported on (0, 1)
        assert bool(jnp.all(observations >= 0.0))
        assert bool(jnp.all(observations <= 1.0))

    def test_non_gaussian_raises_not_implemented(self):
        import jax.random as jr
        import numpy as np

        from nof1_causal_lab.artifacts.model_spec import (
            DistributionFamily,
            LinkFunction,
        )
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            DiagonalDecaySpec,
            compile_composite,
            runtime_from_composite,
            sample_observations_from_latents,
        )
        from nof1_causal_lab.models.ssm.inference.targets.kernels import (
            build_observation_kernel,
        )

        spec = CompositeSpec(
            n_latent=2,
            components=(DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.5)),),
        )
        compiled = compile_composite(spec)
        beta_kernel = build_observation_kernel(
            DistributionFamily.BETA,
            LinkFunction.LOGIT,
            extra_params={"obs_concentration": jnp.asarray([50.0])},
            manifest_cov=np.asarray(jnp.eye(1) * 0.05),
        )
        canonical = runtime_from_composite(
            compiled,
            init_mean=jnp.zeros(2),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.01,
            H=jnp.array([[1.0, 0.0]]),
            d_meas=jnp.zeros(1),
            R=jnp.eye(1) * 0.05,
            obs_kernel=beta_kernel,
        )
        import pytest

        with pytest.raises(NotImplementedError, match="Gaussian"):
            sample_observations_from_latents(
                canonical, jnp.zeros((2, 3, 2)), jr.PRNGKey(0)
            )


class TestSampleCompositePriorPredictiveFull:
    """The canonical-keyed convenience wrapper: latents + observations
    in one call, mirroring the linear ``sample_prior_predictive_from_priors``
    shape."""

    def test_returns_observations_alongside_latents(self):
        import jax.numpy as jnp
        import numpy as np

        from nof1_causal_lab.artifacts.model_spec import (
            DistributionFamily,
            LinkFunction,
        )
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeSpec,
            DiagonalDecaySpec,
            compile_composite,
            runtime_from_composite,
            sample_composite_prior_predictive_full,
        )
        from nof1_causal_lab.models.ssm.inference.targets.kernels import (
            build_observation_kernel,
        )

        spec = CompositeSpec(
            n_latent=2,
            components=(DiagonalDecaySpec(decay_prior=ndist.Gamma(2.0, 4.0)),),
        )
        compiled = compile_composite(spec)
        R = jnp.eye(1) * 0.05
        kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=np.asarray(R),
        )
        canonical = runtime_from_composite(
            compiled,
            init_mean=jnp.array([1.0, 0.5]),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.01,
            H=jnp.array([[1.0, 0.0]]),
            d_meas=jnp.zeros(1),
            R=R,
            obs_kernel=kernel,
        )
        n_draws, T = 8, 5
        times = jnp.linspace(0.0, 2.0, T)
        pp = sample_composite_prior_predictive_full(
            canonical, times, n_draws=n_draws, rng_seed=42
        )
        # Latents
        assert pp.trajectories.shape == (n_draws, T, 2)
        # Observations
        assert pp.observations is not None
        assert pp.observations.shape == (n_draws, T, 1)
        assert bool(jnp.all(jnp.isfinite(pp.observations)))
        # Stability flags still populated
        assert pp.stable.shape == (n_draws,)


class TestValidateCompositeAssembly:
    """Stage-4-shape assembly validator for composite specs.

    Drives the bridge end-to-end: a dict-config goes in, an
    AssemblyValidation-shaped object comes out — exactly what a Stage 4
    LLM tool or the agentic repair flow needs to call when handed a
    composite spec instead of an SSMPriors instance.
    """

    def test_valid_dict_config_returns_is_valid_true(self):
        from nof1_causal_lab.models.ssm.dynamics import validate_composite_assembly

        config = {
            "n_latent": 1,
            "components": [
                {"kind": "DiagonalDecay",
                 "priors": {"decay": {"family": "Gamma",
                                      "params": {"concentration": 2.0, "rate": 4.0},
                                      "shape": [1]}}},
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
        from nof1_causal_lab.models.ssm.dynamics import validate_composite_assembly

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
            components=(
                DiagonalDecaySpec(decay_prior=ndist.Gamma(2.0, 4.0)),
            ),
        )
        compiled = compile_composite(spec)
        times = jnp.linspace(0.0, 2.0, 5)
        result = validate_composite_dynamics(
            compiled, jnp.array([1.0, 1.0]), times, n_draws=10
        )
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
                    n_prior=ndist.TruncatedNormal(
                        loc=3.5, scale=0.1, low=3.0, high=4.0
                    ),
                ),
            ),
        )
        compiled = compile_composite(spec)
        times = jnp.linspace(0.0, 0.5, 5)
        result = validate_composite_dynamics(
            compiled, jnp.array([0.4]), times, n_draws=20
        )
        assert result["is_valid"] is False
        assert result["primary_score"] > 0.5
        assert len(result["failing_draw_indices"]) >= 10
