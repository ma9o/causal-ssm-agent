"""Tests for two-level greedy Rao-Blackwellization.

Pipeline integration tests for make_likelihood_backend dispatching.
Graph analysis and partition tests live in test_graph_analysis.py.
"""

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.model import SSMSpec
from nof1_causal_lab.models.ssm_observation_metadata import ObservationSupportRuntime
from tests.ssm_test_utils import make_ssm_spec

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

    return make_ssm_spec(
        n_latent=n,
        n_manifest=m,
        drift=drift,
        diffusion=jnp.eye(n) * 0.3,
        lambda_mat=lambda_mat,
        manifest_var=jnp.eye(m) * 0.1,
        manifest_means=jnp.zeros(m),
        t0_means=jnp.zeros(n),
        t0_var=jnp.eye(n),
        diffusion_dists=diffusion_dists,
        manifest_dists=[DistributionFamily.GAUSSIAN] * m,
    )


# =============================================================================
# Pipeline Integration
# =============================================================================


class TestMakeLikelihoodBackend:
    """Test make_likelihood_backend dispatching."""

    def test_creates_composed_for_mixed_separable(self):
        """Mixed separable spec -> ComposedLikelihood."""
        from nof1_causal_lab.models.ssm.inference.targets.composed import ComposedLikelihood
        from nof1_causal_lab.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        assert isinstance(backend, ComposedLikelihood)

    def test_first_pass_rb_false_skips_analysis(self):
        """first_pass_rb=False -> no ComposedLikelihood, just ParticleLikelihood."""
        from nof1_causal_lab.models.ssm.inference.targets.composed import ComposedLikelihood
        from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleLikelihood
        from nof1_causal_lab.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        # Override first_pass_rb
        spec = make_ssm_spec(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            drift=spec.drift,
            lambda_mat=spec.lambda_mat,
            diffusion=spec.diffusion_chol,
            manifest_var=spec.manifest_chol,
            manifest_means=spec.manifest_means,
            t0_means=spec.t0_means,
            t0_var=spec.t0_chol,
            diffusion_dists=spec.diffusion_dists,
            manifest_dists=spec.manifest_dists,
            first_pass_rb=False,
        )
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        assert isinstance(backend, ParticleLikelihood)
        assert not isinstance(backend, ComposedLikelihood)

    def test_second_pass_rb_false_forces_bootstrap(self):
        """second_pass_rb=False -> ParticleLikelihood with block_rb=False."""
        from nof1_causal_lab.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        spec = make_ssm_spec(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            drift=spec.drift,
            lambda_mat=spec.lambda_mat,
            diffusion=spec.diffusion_chol,
            manifest_var=spec.manifest_chol,
            manifest_means=spec.manifest_means,
            t0_means=spec.t0_means,
            t0_var=spec.t0_chol,
            diffusion_dists=spec.diffusion_dists,
            manifest_dists=spec.manifest_dists,
            second_pass_rb=False,
        )
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        # Should still create composed (first pass is on) but particle sub-backend
        # has block_rb=False
        from nof1_causal_lab.models.ssm.inference.targets.composed import ComposedLikelihood

        assert isinstance(backend, ComposedLikelihood)
        # The particle sub-backend should not use block RBPF
        assert not backend.particle_backend._block_rb

    def test_both_toggles_off_pure_bootstrap(self):
        """Both toggles off -> pure bootstrap PF."""
        from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleLikelihood
        from nof1_causal_lab.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        spec = make_ssm_spec(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            drift=spec.drift,
            lambda_mat=spec.lambda_mat,
            diffusion=spec.diffusion_chol,
            manifest_var=spec.manifest_chol,
            manifest_means=spec.manifest_means,
            t0_means=spec.t0_means,
            t0_var=spec.t0_chol,
            diffusion_dists=spec.diffusion_dists,
            manifest_dists=spec.manifest_dists,
            first_pass_rb=False,
            second_pass_rb=False,
        )
        model = SSMModel(spec=spec)
        backend = model.make_likelihood_backend()
        assert isinstance(backend, ParticleLikelihood)
        # Should not use Rao-Blackwellization at all, but should preserve mixed diffusion semantics.
        assert backend.transition_dispatch_mode == "mixed"
        assert not backend._block_rb

    def test_kalman_override_bypasses_analysis(self):
        """likelihood="kalman" bypasses first-pass analysis entirely."""
        from nof1_causal_lab.models.ssm.inference.targets.kalman import KalmanLikelihood
        from nof1_causal_lab.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        model = SSMModel(spec=spec, likelihood="kalman")
        backend = model.make_likelihood_backend()
        assert isinstance(backend, KalmanLikelihood)

    def test_non_point_support_disables_first_pass_rb_and_uses_full_particle(self):
        """Interval-summary support should bypass composed/RB dispatch."""
        from nof1_causal_lab.models.ssm.inference.targets.composed import ComposedLikelihood
        from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleLikelihood
        from nof1_causal_lab.models.ssm.model import SSMModel

        spec = _make_separable_spec(n_g=2, n_s=1, n_obs_g=2, n_obs_s=1)
        support = ObservationSupportRuntime(
            anchor_times=np.array([0.0, 1.0]),
            manifest_names=["y0", "y1", "y2"],
            support_kinds=["point", "point", "interval"],
            summary_operators=["last", "last", "mean"],
            anchor_policies=["support_end", "support_end", "support_end"],
            observation_windows=["1d", "1d", "1d"],
            support_start_times=np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, 0.0]]),
            support_end_times=np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, 1.0]]),
            interval_prev_coeffs=np.array([[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.5]]]),
            interval_curr_coeffs=np.array([[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.5]]]),
            interval_weights=np.array([[[0.0], [0.0], [0.0]], [[0.0], [0.0], [1.0]]]),
            emission_slot_indices=np.array([[-1, -1, -1], [-1, -1, 0]], dtype=np.int64),
        )

        model = SSMModel(spec=spec)
        model.set_observation_support(support)
        backend = model.make_likelihood_backend()

        assert isinstance(backend, ParticleLikelihood)
        assert not isinstance(backend, ComposedLikelihood)
        assert backend.observation_support is support
        assert not backend._block_rb
