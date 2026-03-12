"""Tests for two-level greedy Rao-Blackwellization.

Pipeline integration tests for make_likelihood_backend dispatching.
Graph analysis and partition tests live in test_graph_analysis.py.
"""

import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.models.ssm.model import SSMSpec
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

# =============================================================================
# Helpers
# =============================================================================


def _make_separable_spec(
    n_g: int = 2,
    n_s: int = 1,
    n_obs_g: int = 2,
    n_obs_s: int = 1,
    cross_coupling: bool = False,
) -> SSMSpec:
    """Build an SSMSpec with block-diagonal drift/lambda and mixed diffusion_dists.

    Variables 0..n_g-1 are Gaussian, n_g..n_g+n_s-1 are Student-t.
    Observations 0..n_obs_g-1 map to Gaussian vars, n_obs_g..n_obs_g+n_obs_s-1 to Student-t.
    """
    n = n_g + n_s
    m = n_obs_g + n_obs_s

    # Block-diagonal drift: stable diagonal
    drift = np.diag(np.full(n, -0.5))
    if cross_coupling and n_g > 0 and n_s > 0:
        drift[n_g, 0] = 0.2  # S <- G
        drift[0, n_g] = 0.15  # G <- S
    drift = jnp.array(drift, dtype=jnp.float32)

    # Block-diagonal lambda: obs_g -> latent_g, obs_s -> latent_s
    lam = np.zeros((m, n))
    for i in range(min(n_obs_g, n_g)):
        lam[i, i] = 1.0
    for i in range(min(n_obs_s, n_s)):
        lam[n_obs_g + i, n_g + i] = 1.0
    lambda_mat = jnp.array(lam, dtype=jnp.float32)

    # Per-variable diffusion dists
    diffusion_dists = [DistributionFamily.GAUSSIAN] * n_g + [DistributionFamily.STUDENT_T] * n_s

    return SSMSpec(
        n_latent=n,
        n_manifest=m,
        drift=drift,
        diffusion=jnp.eye(n) * 0.3,
        lambda_mat=lambda_mat,
        manifest_var=jnp.eye(m) * 0.1,
        manifest_means=jnp.zeros(m),
        t0_means=jnp.zeros(n),
        t0_var=jnp.eye(n),
        diffusion_dist=DistributionFamily.GAUSSIAN,
        manifest_dist=DistributionFamily.GAUSSIAN,
        diffusion_dists=diffusion_dists,
    )


# =============================================================================
# Pipeline Integration
# =============================================================================


class TestMakeLikelihoodBackend:
    """Test make_likelihood_backend dispatching."""

    def test_creates_composed_for_mixed_separable(self):
        """Mixed separable spec -> ComposedLikelihood."""
        from causal_ssm_agent.models.likelihoods.composed import ComposedLikelihood
        from causal_ssm_agent.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        assert isinstance(backend, ComposedLikelihood)

    def test_first_pass_rb_false_skips_analysis(self):
        """first_pass_rb=False -> no ComposedLikelihood, just ParticleLikelihood."""
        from causal_ssm_agent.models.likelihoods.composed import ComposedLikelihood
        from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood
        from causal_ssm_agent.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        # Override first_pass_rb
        spec = SSMSpec(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            drift=spec.drift,
            lambda_mat=spec.lambda_mat,
            diffusion=spec.diffusion,
            manifest_var=spec.manifest_var,
            manifest_means=spec.manifest_means,
            t0_means=spec.t0_means,
            t0_var=spec.t0_var,
            diffusion_dist=spec.diffusion_dist,
            manifest_dist=spec.manifest_dist,
            diffusion_dists=spec.diffusion_dists,
            first_pass_rb=False,
        )
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        assert isinstance(backend, ParticleLikelihood)
        assert not isinstance(backend, ComposedLikelihood)

    def test_second_pass_rb_false_forces_bootstrap(self):
        """second_pass_rb=False -> ParticleLikelihood with block_rb=False."""
        from causal_ssm_agent.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        spec = SSMSpec(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            drift=spec.drift,
            lambda_mat=spec.lambda_mat,
            diffusion=spec.diffusion,
            manifest_var=spec.manifest_var,
            manifest_means=spec.manifest_means,
            t0_means=spec.t0_means,
            t0_var=spec.t0_var,
            diffusion_dist=spec.diffusion_dist,
            manifest_dist=spec.manifest_dist,
            diffusion_dists=spec.diffusion_dists,
            second_pass_rb=False,
        )
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        # Should still create composed (first pass is on) but particle sub-backend
        # has block_rb=False
        from causal_ssm_agent.models.likelihoods.composed import ComposedLikelihood

        assert isinstance(backend, ComposedLikelihood)
        # The particle sub-backend should not use block RBPF
        assert not backend.particle_backend._block_rb

    def test_both_toggles_off_pure_bootstrap(self):
        """Both toggles off -> pure bootstrap PF."""
        from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood
        from causal_ssm_agent.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        spec = SSMSpec(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            drift=spec.drift,
            lambda_mat=spec.lambda_mat,
            diffusion=spec.diffusion,
            manifest_var=spec.manifest_var,
            manifest_means=spec.manifest_means,
            t0_means=spec.t0_means,
            t0_var=spec.t0_var,
            diffusion_dist=spec.diffusion_dist,
            manifest_dist=spec.manifest_dist,
            diffusion_dists=spec.diffusion_dists,
            first_pass_rb=False,
            second_pass_rb=False,
        )
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        assert isinstance(backend, ParticleLikelihood)
        # Should not use Rao-Blackwellization at all
        assert backend.diffusion_dist != "mixed"
        assert not backend._block_rb

    def test_kalman_override_bypasses_analysis(self):
        """likelihood="kalman" bypasses first-pass analysis entirely."""
        from causal_ssm_agent.models.likelihoods.kalman import KalmanLikelihood
        from causal_ssm_agent.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        model = SSMModel(spec=spec, likelihood="kalman")
        backend = model.make_likelihood_backend()
        assert isinstance(backend, KalmanLikelihood)
