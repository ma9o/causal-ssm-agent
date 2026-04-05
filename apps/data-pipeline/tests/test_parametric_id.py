"""Tests for parametric identifiability diagnostics.

Tests:
0. T-rule: counting condition for necessary identification
1. Forward simulator shape and finiteness
2. Profile likelihood: identified model has well-shaped profiles
3. Profile likelihood: non-identified model flags issues
4. Profile likelihood result classification
5. SBC: basic structure and uniform ranks for identified model
6. Power-scaling: prior-dominated params flagged correctly
7. Stage 4b flow: smoke test
8. Recovery: simulate_ssm produces data recoverable by Kalman fit
9. Recovery: profile_likelihood correctly classifies identified vs non-identified
"""

from unittest.mock import patch

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from causal_ssm_agent.models.ssm.model import SSMModel, SSMPriors, SSMSpec
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

pytestmark = pytest.mark.slow


def _make_identified_model(n_latent=2, n_manifest=2, likelihood="kalman"):
    """Build a well-identified 2-latent, 2-manifest Gaussian SSM."""
    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift="free",
        diffusion="diag",
        cint=None,
        lambda_mat=jnp.eye(n_manifest, n_latent),
        manifest_means=None,
        manifest_var="diag",
        t0_means="free",
        t0_var="diag",
    )
    priors = SSMPriors(
        drift_diag={"mu": -0.5, "sigma": 0.5},
        drift_offdiag={"mu": 0.0, "sigma": 0.3},
        diffusion_diag={"sigma": 0.5},
        t0_means={"mu": 0.0, "sigma": 1.0},
        t0_var_diag={"sigma": 1.0},
        manifest_var_diag={"sigma": 0.5},
    )
    return SSMModel(spec, priors, n_particles=50, likelihood=likelihood)


def _make_nonidentified_model():
    """Build a non-identified model: 2 latent, 1 manifest -> rank deficient."""
    spec = SSMSpec(
        n_latent=2,
        n_manifest=1,
        drift="free",
        diffusion="diag",
        cint=None,
        lambda_mat=jnp.ones((1, 2)) * 0.5,  # Both latents map identically to 1 manifest
        manifest_means=None,
        manifest_var="diag",
        t0_means="free",
        t0_var="diag",
    )
    priors = SSMPriors(
        drift_diag={"mu": -0.5, "sigma": 0.5},
        drift_offdiag={"mu": 0.0, "sigma": 0.3},
        diffusion_diag={"sigma": 0.5},
        t0_means={"mu": 0.0, "sigma": 1.0},
        t0_var_diag={"sigma": 1.0},
        manifest_var_diag={"sigma": 0.5},
    )
    return SSMModel(spec, priors, n_particles=50, likelihood="kalman")


def _make_interval_support_runtime(
    manifest_names: list[str],
    *,
    summary_operators: list[str] | None = None,
) -> ObservationSupportRuntime:
    n_manifest = len(manifest_names)
    if summary_operators is None:
        summary_operators = ["sum"] * n_manifest
    if len(summary_operators) != n_manifest:
        raise ValueError("summary_operators must align with manifest_names")

    return ObservationSupportRuntime(
        anchor_times=np.array([0.0, 1.0, 2.0]),
        manifest_names=manifest_names,
        support_kinds=["interval"] * n_manifest,
        summary_operators=summary_operators,
        anchor_policies=["end"] * n_manifest,
        observation_windows=["1d"] * n_manifest,
        support_start_times=np.array(
            [
                [np.nan] * n_manifest,
                [0.0] * n_manifest,
                [1.0] * n_manifest,
            ],
        ),
        support_end_times=np.array(
            [
                [np.nan] * n_manifest,
                [1.0] * n_manifest,
                [2.0] * n_manifest,
            ],
        ),
        interval_prev_coeffs=np.array(
            [
                [[0.0]] * n_manifest,
                [[0.5]] * n_manifest,
                [[0.5]] * n_manifest,
            ],
        ),
        interval_curr_coeffs=np.array(
            [
                [[0.0]] * n_manifest,
                [[0.5]] * n_manifest,
                [[0.5]] * n_manifest,
            ],
        ),
        interval_weights=np.array(
            [
                [[0.0]] * n_manifest,
                [[1.0]] * n_manifest,
                [[1.0]] * n_manifest,
            ],
        ),
        emission_slot_indices=np.array(
            [[-1] * n_manifest, [0] * n_manifest, [0] * n_manifest],
            dtype=np.int32,
        ),
    )


class TestTRule:
    """Test t-rule (counting condition) for identification."""

    def test_identified_model_passes(self):
        """Well-identified 2L/2M model should pass t-rule with time series."""
        from causal_ssm_agent.utils.parametric_id import check_t_rule

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift="free",
            diffusion="diag",
            cint=None,
            lambda_mat=jnp.eye(2),
            manifest_means=None,
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
        )
        result = check_t_rule(spec, T=50)
        assert result.satisfies
        # p=2, T=50: moments = 2 + 3 + 49*2 = 103, plenty for ~12 params
        assert result.n_moments > result.n_free_params

    def test_overparameterized_model_fails_without_T(self):
        """Model with many params fails cross-sectional t-rule (no T)."""
        from causal_ssm_agent.utils.parametric_id import check_t_rule

        # 3 latent, 2 manifest: lots of drift params relative to cross-sectional moments
        spec = SSMSpec(
            n_latent=3,
            n_manifest=2,
            drift="free",
            diffusion="free",
            cint="free",
            lambda_mat=jnp.eye(2, 3),
            manifest_means="free",
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
        )
        result = check_t_rule(spec, T=None)
        # Cross-sectional only: p=2 -> 2 + 3 = 5 moments, way fewer than params
        assert not result.satisfies

    def test_overparameterized_rescued_by_time_series(self):
        """Same model passes when T is large enough (autocovariance helps)."""
        from causal_ssm_agent.utils.parametric_id import check_t_rule

        spec = SSMSpec(
            n_latent=3,
            n_manifest=2,
            drift="free",
            diffusion="free",
            cint="free",
            lambda_mat=jnp.eye(2, 3),
            manifest_means="free",
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
        )
        result = check_t_rule(spec, T=50)
        # With T=50: moments = 2 + 3 + 49*2 = 103, should be enough
        assert result.satisfies

    def test_count_free_params_fixed_lambda(self):
        """Fixed lambda should contribute 0 free params."""
        from causal_ssm_agent.utils.parametric_id import count_free_params

        spec = SSMSpec(
            n_latent=2,
            n_manifest=3,
            drift="free",
            diffusion="diag",
            lambda_mat=jnp.eye(3, 2),  # fixed
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
        )
        counts = count_free_params(spec)
        assert "lambda_free" not in counts

    def test_count_free_params_free_lambda(self):
        """Free lambda with n_m > n_l should have (n_m - n_l) * n_l free entries."""
        from causal_ssm_agent.utils.parametric_id import count_free_params

        spec = SSMSpec(
            n_latent=2,
            n_manifest=4,
            drift="free",
            diffusion="diag",
            lambda_mat="free",
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
        )
        counts = count_free_params(spec)
        # (4 - 2) * 2 = 4 free loadings
        assert counts["lambda_free"] == 4

    def test_count_free_params_drift_components(self):
        """Drift should have n_l diagonal + n_l*(n_l-1) off-diagonal."""
        from causal_ssm_agent.utils.parametric_id import count_free_params

        spec = SSMSpec(
            n_latent=3,
            n_manifest=3,
            drift="free",
            diffusion="diag",
            lambda_mat=jnp.eye(3),
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
        )
        counts = count_free_params(spec)
        assert counts["drift_diag_pop"] == 3
        assert counts["drift_offdiag_pop"] == 6  # 3*3 - 3

    def test_count_free_params_noise_hyperparams(self):
        """Student-t manifest noise should add obs_df parameter."""
        from causal_ssm_agent.utils.parametric_id import count_free_params

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            drift="free",
            diffusion="diag",
            lambda_mat=jnp.eye(1),
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
            manifest_dist=DistributionFamily.STUDENT_T,
        )
        counts = count_free_params(spec)
        assert counts.get("obs_df") == 1

    def test_count_free_params_per_channel_noise_hyperparams(self):
        """Per-channel manifest_dists should contribute registry-defined noise sites."""
        from causal_ssm_agent.utils.parametric_id import count_free_params

        spec = SSMSpec(
            n_latent=1,
            n_manifest=2,
            drift="free",
            diffusion="diag",
            lambda_mat=jnp.eye(2, 1),
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
            manifest_dist=DistributionFamily.GAUSSIAN,
            manifest_dists=[
                DistributionFamily.GAUSSIAN,
                DistributionFamily.STUDENT_T,
            ],
        )
        counts = count_free_params(spec)
        assert counts.get("obs_df") == 1


class TestSimulateSSM:
    """Test forward simulator."""

    def test_simulate_ssm_shape_and_finite(self):
        """Forward sim produces correct shape, finite values."""
        from causal_ssm_agent.utils.parametric_id import simulate_ssm

        n_latent, n_manifest, T = 2, 2, 20
        drift = jnp.array([[-0.5, 0.1], [0.05, -0.8]])
        diffusion_chol = jnp.eye(n_latent) * 0.3
        lambda_mat = jnp.eye(n_manifest, n_latent)
        manifest_chol = jnp.eye(n_manifest) * 0.2
        t0_means = jnp.zeros(n_latent)
        t0_chol = jnp.eye(n_latent) * 0.5
        times = jnp.linspace(0, 10, T)

        y = simulate_ssm(
            drift=drift,
            diffusion_chol=diffusion_chol,
            lambda_mat=lambda_mat,
            manifest_chol=manifest_chol,
            t0_means=t0_means,
            t0_chol=t0_chol,
            times=times,
            rng_key=random.PRNGKey(0),
        )

        assert y.shape == (T, n_manifest)
        assert jnp.all(jnp.isfinite(y))

    def test_simulate_ssm_with_cint(self):
        """Forward sim works with continuous intercept."""
        from causal_ssm_agent.utils.parametric_id import simulate_ssm

        n_latent, n_manifest, T = 2, 2, 15
        y = simulate_ssm(
            drift=jnp.array([[-0.5, 0.0], [0.0, -0.8]]),
            diffusion_chol=jnp.eye(n_latent) * 0.3,
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_chol=jnp.eye(n_manifest) * 0.2,
            t0_means=jnp.zeros(n_latent),
            t0_chol=jnp.eye(n_latent) * 0.5,
            times=jnp.linspace(0, 10, T),
            rng_key=random.PRNGKey(1),
            cint=jnp.array([0.1, -0.1]),
        )

        assert y.shape == (T, n_manifest)
        assert jnp.all(jnp.isfinite(y))

    def test_simulate_ssm_poisson(self):
        """Forward sim produces non-negative integers for Poisson noise."""
        from causal_ssm_agent.utils.parametric_id import simulate_ssm

        n_latent, n_manifest, T = 1, 1, 10
        y = simulate_ssm(
            drift=jnp.array([[-0.5]]),
            diffusion_chol=jnp.eye(n_latent) * 0.3,
            lambda_mat=jnp.eye(n_manifest, n_latent),
            manifest_chol=jnp.eye(n_manifest) * 0.2,
            t0_means=jnp.array([1.0]),
            t0_chol=jnp.eye(n_latent) * 0.1,
            times=jnp.linspace(0, 5, T),
            rng_key=random.PRNGKey(2),
            manifest_dist="poisson",
        )

        assert y.shape == (T, n_manifest)
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(y >= 0)


class TestOutputSensitivity:
    """Test output sensitivity analysis."""

    def test_stage4b_sweep_context_reuses_cached_topology_bundle(self):
        """Identical topology should reuse one cached Stage 4b sweep context."""
        from causal_ssm_agent.utils import parametric_id as pid

        model = _make_identified_model(n_latent=1, n_manifest=1, likelihood="kalman")
        pid.clear_stage4b_sweep_context_cache()

        with (
            patch.object(
                pid, "build_site_runtime_bundle", wraps=pid.build_site_runtime_bundle
            ) as build_site_runtime,
            patch.object(
                pid,
                "_build_runtime_eval_fns_from_registry",
                wraps=pid._build_runtime_eval_fns_from_registry,
            ) as build_eval_fns,
        ):
            ctx_1 = pid.get_stage4b_sweep_context(model)
            ctx_2 = pid.get_stage4b_sweep_context(model)

        assert ctx_1 is ctx_2
        assert build_site_runtime.call_count == 1
        assert build_eval_fns.call_count == 1

    def test_stage4b_sweep_context_separates_distinct_topologies(self):
        """A topology change should create a different cached sweep context."""
        from causal_ssm_agent.utils.parametric_id import (
            clear_stage4b_sweep_context_cache,
            get_stage4b_sweep_context,
        )

        clear_stage4b_sweep_context_cache()
        ctx_1 = get_stage4b_sweep_context(
            _make_identified_model(n_latent=1, n_manifest=1, likelihood="kalman")
        )
        ctx_2 = get_stage4b_sweep_context(
            _make_identified_model(n_latent=2, n_manifest=2, likelihood="kalman")
        )

        assert ctx_1 is not ctx_2
        assert ctx_1.cache_key != ctx_2.cache_key

    def test_identified_model_mostly_identifiable(self):
        """Well-identified 1D LGSS: all params should be flagged identifiable."""
        from causal_ssm_agent.utils.parametric_id import output_sensitivity_analysis

        # 1D model (3 free params: drift_diag, diffusion_diag, manifest_var_diag)
        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            manifest_means=jnp.zeros(1),
            diffusion="diag",
            t0_means=jnp.zeros(1),
            t0_var=jnp.eye(1),
        )
        priors = SSMPriors(
            drift_diag={"mu": -0.5, "sigma": 0.3},
            diffusion_diag={"sigma": 0.3},
            manifest_var_diag={"sigma": 0.3},
        )
        model = SSMModel(spec, priors, n_particles=50, likelihood="kalman")
        T = 100
        times = jnp.arange(T, dtype=jnp.float32)

        result = output_sensitivity_analysis(model, times, n_draws=5, seed=42)

        assert result.n_parameters > 0
        assert result.n_observations == 2 * T * 1  # 2*T*D for mean + variance
        assert result.n_draws == 5
        assert len(result.singular_values) > 0
        assert result.condition_number > 0
        assert all(jnp.isfinite(jnp.array(result.singular_values)))

        # All params should be identifiable for a well-specified 1D LGSS
        for entry in result.per_parameter:
            assert entry["identifiable"], (
                f"Parameter {entry['parameter']} flagged as non-identifiable "
                f"(norm={entry['sensitivity_norm']:.4f})"
            )

    def test_non_identified_model_flags_issues(self):
        """Non-identified model (2 latent, 1 manifest): should flag issues."""
        from causal_ssm_agent.utils.parametric_id import output_sensitivity_analysis

        model = _make_nonidentified_model()
        T = 50
        times = jnp.linspace(0, 25, T)

        result = output_sensitivity_analysis(model, times, n_draws=3, seed=42)

        # Should have high condition number (near-singular directions)
        assert result.condition_number > 100, (
            f"Non-identified model should have high condition number, got {result.condition_number:.1f}"
        )

        # At least some parameters should be flagged as non-identifiable
        n_non_id = sum(1 for e in result.per_parameter if not e["identifiable"])
        assert n_non_id > 0, (
            "Non-identified model should flag some parameters, all are identifiable"
        )

    def test_mixed_family_interval_observation_models_produce_finite_sensitivity(self):
        """Stage 4b should handle mixed-family interval summaries on the observation scale."""
        from causal_ssm_agent.utils.parametric_id import output_sensitivity_analysis

        spec = SSMSpec(
            n_latent=1,
            n_manifest=2,
            drift="free",
            diffusion="diag",
            lambda_mat=jnp.ones((2, 1), dtype=jnp.float32),
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
            manifest_dist=DistributionFamily.GAUSSIAN,
            manifest_dists=[
                DistributionFamily.NEGATIVE_BINOMIAL,
                DistributionFamily.GAUSSIAN,
            ],
            manifest_links=[
                LinkFunction.LOG,
                LinkFunction.IDENTITY,
            ],
            manifest_names=["count_sum", "score_mean"],
        )
        priors = SSMPriors(
            drift_diag={"mu": -0.4, "sigma": 0.1},
            diffusion_diag={"sigma": 0.15},
            manifest_var_diag={"sigma": 0.2},
            obs_r={"concentration": 4.0, "rate": 1.0},
            t0_means={"mu": 0.1, "sigma": 0.2},
            t0_var_diag={"sigma": 0.2},
        )
        model = SSMModel(spec, priors, n_particles=50, likelihood="particle")
        model.observation_support = _make_interval_support_runtime(
            spec.manifest_names,
            summary_operators=["sum", "mean"],
        )
        times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
        observations = jnp.array(
            [
                [jnp.nan, jnp.nan],
                [3.0, 0.2],
                [2.0, 0.4],
            ],
            dtype=jnp.float32,
        )

        result = output_sensitivity_analysis(
            model,
            times,
            observations=observations,
            n_draws=3,
            seed=3,
        )

        assert result.n_draws >= 1
        assert result.n_observations == 8
        assert len(result.singular_values) > 0
        assert all(jnp.isfinite(jnp.asarray(result.singular_values)))
        assert jnp.isfinite(result.condition_number)

    def test_discrete_point_observation_models_produce_finite_sensitivity(self):
        """Point-like ordered-logistic and categorical channels should be supported."""
        from causal_ssm_agent.utils.parametric_id import output_sensitivity_analysis

        spec = SSMSpec(
            n_latent=1,
            n_manifest=2,
            drift="free",
            diffusion="diag",
            lambda_mat=jnp.ones((2, 1), dtype=jnp.float32),
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
            manifest_dist=DistributionFamily.GAUSSIAN,
            manifest_dists=[
                DistributionFamily.ORDERED_LOGISTIC,
                DistributionFamily.CATEGORICAL,
            ],
            manifest_links=[
                LinkFunction.CUMULATIVE_LOGIT,
                LinkFunction.SOFTMAX,
            ],
            manifest_level_counts=[3, 3],
            manifest_names=["sleep_level", "activity_type"],
        )
        priors = SSMPriors(
            drift_diag={"mu": -0.5, "sigma": 0.1},
            diffusion_diag={"sigma": 0.15},
            manifest_var_diag={"sigma": 0.2},
            obs_ordered_base={"mu": 0.0, "sigma": 0.5},
            obs_ordered_gaps={"sigma": 0.3},
            obs_cat_intercepts={"mu": 0.0, "sigma": 0.4},
            obs_cat_slopes={"mu": 0.0, "sigma": 0.3},
            t0_means={"mu": 0.0, "sigma": 0.2},
            t0_var_diag={"sigma": 0.2},
        )
        model = SSMModel(spec, priors, n_particles=50, likelihood="particle")
        times = jnp.arange(6, dtype=jnp.float32)
        observations = jnp.array(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 0.0],
                [1.0, 1.0],
                [0.0, 2.0],
                [1.0, 1.0],
            ],
            dtype=jnp.float32,
        )

        result = output_sensitivity_analysis(
            model,
            times,
            observations=observations,
            n_draws=2,
            seed=5,
        )

        assert result.n_draws >= 1
        assert result.n_observations == 2 * times.shape[0] * spec.n_manifest
        assert len(result.singular_values) > 0
        assert all(jnp.isfinite(jnp.asarray(result.singular_values)))
        assert jnp.isfinite(result.condition_number)

    def test_interval_std_observation_models_produce_finite_sensitivity(self):
        """Interval-summary standard-deviation channels should produce finite sensitivities."""
        from causal_ssm_agent.utils.parametric_id import output_sensitivity_analysis

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            drift="free",
            diffusion="diag",
            lambda_mat=jnp.ones((1, 1), dtype=jnp.float32),
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
            manifest_dist=DistributionFamily.GAUSSIAN,
            manifest_names=["score_std"],
        )
        priors = SSMPriors(
            drift_diag={"mu": -0.4, "sigma": 0.1},
            diffusion_diag={"sigma": 0.1},
            manifest_var_diag={"sigma": 0.2},
            t0_means={"mu": 0.1, "sigma": 0.2},
            t0_var_diag={"sigma": 0.2},
        )
        model = SSMModel(spec, priors, n_particles=50, likelihood="kalman")
        model.observation_support = _make_interval_support_runtime(
            spec.manifest_names,
            summary_operators=["std"],
        )
        times = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
        observations = jnp.array(
            [
                [jnp.nan],
                [0.4],
                [0.6],
            ],
            dtype=jnp.float32,
        )

        result = output_sensitivity_analysis(
            model,
            times,
            observations=observations,
            n_draws=2,
            seed=7,
        )

        assert result.n_draws >= 1
        assert result.n_observations == 4
        assert len(result.singular_values) > 0
        assert all(jnp.isfinite(jnp.asarray(result.singular_values)))
        assert jnp.isfinite(result.condition_number)

    def test_interval_discrete_observation_models_raise_unsupported_error(self):
        """Interval summaries over discrete families should fail explicitly."""
        from causal_ssm_agent.utils.parametric_id import (
            OutputSensitivityUnsupportedError,
            output_sensitivity_analysis,
        )

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            drift="free",
            diffusion="diag",
            lambda_mat=jnp.ones((1, 1), dtype=jnp.float32),
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
            manifest_dist=DistributionFamily.ORDERED_LOGISTIC,
            manifest_link=LinkFunction.CUMULATIVE_LOGIT,
            manifest_level_counts=[3],
            manifest_names=["sleep_level"],
        )
        priors = SSMPriors(
            drift_diag={"mu": -0.5, "sigma": 0.1},
            diffusion_diag={"sigma": 0.15},
            manifest_var_diag={"sigma": 0.2},
            obs_ordered_base={"mu": 0.0, "sigma": 0.5},
            obs_ordered_gaps={"sigma": 0.3},
            t0_means={"mu": 0.0, "sigma": 0.2},
            t0_var_diag={"sigma": 0.2},
        )
        model = SSMModel(spec, priors, n_particles=50, likelihood="particle")
        model.observation_support = _make_interval_support_runtime(
            spec.manifest_names,
            summary_operators=["mean"],
        )

        with pytest.raises(OutputSensitivityUnsupportedError, match="mean-parameter likelihood"):
            output_sensitivity_analysis(
                model,
                jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32),
                observations=jnp.array([[jnp.nan], [1.0], [2.0]], dtype=jnp.float32),
                n_draws=2,
                seed=9,
            )

    def test_skips_nonfinite_prior_draws_instead_of_poisoning_median(self, monkeypatch):
        """One bad Jacobian draw should not turn the whole sensitivity result into NaN."""
        from causal_ssm_agent.utils import parametric_id as pid

        model = _make_identified_model(n_latent=1, n_manifest=1, likelihood="kalman")
        times = jnp.arange(6, dtype=jnp.float32)
        context = pid.get_stage4b_sweep_context(model)
        real_jacobian = context.jacobian_fn

        def fake_jacobian(z_flat, time_grid):
            if z_flat[0] < 0:
                return jnp.full((2 * time_grid.shape[0], context.flat_dim), jnp.nan)
            return real_jacobian(z_flat, time_grid)

        object.__setattr__(context, "jacobian_fn", fake_jacobian)

        def fake_sample_prior_unconstrained(rng_key, registry, prior_state, n_samples):
            draws = jnp.ones((n_samples, context.flat_dim), dtype=jnp.float32)
            if n_samples >= 2:
                draws = draws.at[1, 0].set(-1.0)
            return draws, rng_key

        monkeypatch.setattr(
            pid,
            "sample_prior_unconstrained",
            fake_sample_prior_unconstrained,
        )

        result = pid.output_sensitivity_analysis(
            model,
            times,
            n_draws=2,
            seed=13,
            sweep_context=context,
        )

        assert result.n_draws == 1
        assert all(jnp.isfinite(jnp.asarray(result.singular_values)))
        assert jnp.isfinite(result.condition_number)

    def test_output_sensitivity_counts_only_observed_outputs_when_masked(self):
        """Missing observations should be excluded from the sensitivity output count."""
        from causal_ssm_agent.utils.parametric_id import output_sensitivity_analysis

        model = _make_identified_model(n_latent=1, n_manifest=1, likelihood="kalman")
        times = jnp.linspace(0, 6, 4)
        observations = jnp.array([[0.0], [1.0], [jnp.nan], [2.0]], dtype=jnp.float32)

        result = output_sensitivity_analysis(
            model,
            times,
            observations=observations,
            n_draws=2,
            seed=7,
        )

        assert result.n_observations == 2 * 3

    def test_output_sensitivity_exposes_interpretable_parameter_names(self):
        """Sensitivity rows should carry semantic names resolved from bindings or spec metadata."""
        from causal_ssm_agent.utils.parametric_id import output_sensitivity_analysis

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            drift="free",
            diffusion="diag",
            lambda_mat=jnp.eye(1),
            manifest_means=None,
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
            latent_names=["mood"],
            manifest_names=["heart_rate"],
        )
        priors = SSMPriors(
            drift_diag={"mu": -0.5, "sigma": 0.3},
            diffusion_diag={"sigma": 0.3},
            manifest_var_diag={"sigma": 0.3},
            t0_means={"mu": 0.0, "sigma": 1.0},
            t0_var_diag={"sigma": 1.0},
        )
        model = SSMModel(spec, priors, n_particles=50, likelihood="kalman")
        model.parameter_bindings = [
            {"site_name": "drift_diag_pop", "flat_index": 0, "parameter": "rho_mood"}
        ]

        result = output_sensitivity_analysis(
            model,
            jnp.arange(8, dtype=jnp.float32),
            n_draws=2,
            seed=11,
        )

        names_by_parameter = {
            entry["parameter"]: entry["interpretable_parameter"] for entry in result.per_parameter
        }
        assert names_by_parameter["drift_diag_pop"] == "rho_mood"
        assert names_by_parameter["diffusion_diag_pop"] == "sigma_mood"
        assert names_by_parameter["manifest_var_diag"] == "obs_sd_heart_rate"
        assert names_by_parameter["t0_means_pop"] == "t0_mean_mood"

    def test_manifest_var_alias_uses_sparse_free_positions(self):
        """Sparse manifest-noise sites should resolve to the correct manifest channel."""
        from causal_ssm_agent.utils.parametric_id import _fallback_interpretable_parameter_name

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            manifest_names=["heart_rate", "sleep_quality"],
            lambda_mat=jnp.eye(2),
            manifest_var=jnp.diag(jnp.array([0.3, 0.0], dtype=jnp.float32)),
            manifest_var_mask=jnp.array([False, True]),
        )
        model = SSMModel(spec, SSMPriors(), likelihood="kalman")

        alias = _fallback_interpretable_parameter_name(
            spec,
            "manifest_var_diag",
            0,
            assembler=model._assembler,
        )

        assert alias == "obs_sd_sleep_quality"


class TestProfileLikelihood:
    """Test profile likelihood function."""

    def test_profile_likelihood_accepts_cached_sweep_context(self):
        """Explicitly reusing a cached Stage 4b context should still work."""
        from causal_ssm_agent.utils.parametric_id import (
            get_stage4b_sweep_context,
            output_sensitivity_analysis,
            profile_likelihood,
            simulate_ssm,
        )

        model = _make_identified_model(n_latent=1, n_manifest=1, likelihood="kalman")
        times = jnp.linspace(0, 6, 8)
        observations = simulate_ssm(
            drift=jnp.array([[-0.4]]),
            diffusion_chol=jnp.array([[0.3]]),
            lambda_mat=jnp.eye(1),
            manifest_chol=jnp.array([[0.2]]),
            t0_means=jnp.zeros(1),
            t0_chol=jnp.eye(1) * 0.5,
            times=times,
            rng_key=random.PRNGKey(7),
        )
        sweep_context = get_stage4b_sweep_context(model)

        sa_result = output_sensitivity_analysis(
            model,
            times,
            n_draws=2,
            seed=7,
            sweep_context=sweep_context,
        )
        result = profile_likelihood(
            model=model,
            observations=observations,
            times=times,
            n_grid=3,
            seed=7,
            sweep_context=sweep_context,
        )

        assert sa_result.n_parameters > 0
        assert result.parameter_names
        assert jnp.isfinite(result.mle_ll)

    @pytest.mark.slow
    def test_identified_model(self):
        """Well-identified model: all params should be classified as identified."""
        from causal_ssm_agent.utils.parametric_id import profile_likelihood

        model = _make_identified_model()
        T = 50
        times = jnp.linspace(0, 25, T)

        # Simulate real data from known params
        from causal_ssm_agent.utils.parametric_id import simulate_ssm

        obs = simulate_ssm(
            drift=jnp.array([[-0.5, 0.1], [0.05, -0.8]]),
            diffusion_chol=jnp.eye(2) * 0.3,
            lambda_mat=jnp.eye(2),
            manifest_chol=jnp.eye(2) * 0.2,
            t0_means=jnp.zeros(2),
            t0_chol=jnp.eye(2) * 0.5,
            times=times,
            rng_key=random.PRNGKey(42),
        )

        result = profile_likelihood(
            model=model,
            observations=obs,
            times=times,
            n_grid=15,
            seed=42,
        )

        assert len(result.parameter_profiles) > 0
        assert len(result.parameter_names) > 0
        assert jnp.isfinite(result.mle_ll)

        summary = result.summary()
        # Well-identified model should not have structurally unidentifiable params
        n_struct = sum(1 for v in summary.values() if v == "structurally_unidentifiable")
        assert n_struct == 0, f"Unexpected structural non-identifiability: {summary}"

    @pytest.mark.slow
    def test_non_identified_model(self):
        """Non-identified model (2 latent, 1 manifest) should flag issues."""
        from causal_ssm_agent.utils.parametric_id import profile_likelihood, simulate_ssm

        model = _make_nonidentified_model()
        T = 50
        times = jnp.linspace(0, 25, T)

        obs = simulate_ssm(
            drift=jnp.array([[-0.5, 0.1], [0.05, -0.8]]),
            diffusion_chol=jnp.eye(2) * 0.3,
            lambda_mat=jnp.ones((1, 2)) * 0.5,
            manifest_chol=jnp.eye(1) * 0.2,
            t0_means=jnp.zeros(2),
            t0_chol=jnp.eye(2) * 0.5,
            times=times,
            rng_key=random.PRNGKey(42),
        )

        result = profile_likelihood(
            model=model,
            observations=obs,
            times=times,
            n_grid=15,
            seed=42,
        )

        summary = result.summary()
        # With 2 latent and 1 manifest (identical loadings),
        # some params should be non-identifiable
        has_issues = any(
            v in ("structurally_unidentifiable", "practically_unidentifiable")
            for v in summary.values()
        )
        assert has_issues, f"Non-identified model should flag issues: {summary}"


class TestProfileLikelihoodResult:
    """Test ProfileLikelihoodResult dataclass methods."""

    def test_summary_keys(self):
        """Summary should return per-parameter classification strings."""
        from causal_ssm_agent.utils.parametric_id import ProfileLikelihoodResult

        result = ProfileLikelihoodResult(
            parameter_profiles={
                "param_a": {
                    "grid_unc": jnp.linspace(-3, 3, 10),
                    "grid_con": jnp.linspace(-3, 3, 10),
                    "profile_ll": -(jnp.linspace(-3, 3, 10) ** 2),  # parabola
                    "mle_value": 0.0,
                },
            },
            mle_ll=0.0,
            mle_params={"param_a": jnp.array(0.0)},
            threshold=1.92,
            parameter_names=["param_a"],
        )

        summary = result.summary()
        assert "param_a" in summary
        assert summary["param_a"] in (
            "identified",
            "practically_unidentifiable",
            "structurally_unidentifiable",
        )

    def test_identified_classification(self):
        """Parabolic profile (strong curvature) should be classified as identified."""
        from causal_ssm_agent.utils.parametric_id import ProfileLikelihoodResult

        grid = jnp.linspace(-3, 3, 20)
        # Strong parabola: -2*x^2, drops by >1.92 within grid
        profile = -2.0 * grid**2

        result = ProfileLikelihoodResult(
            parameter_profiles={
                "p": {
                    "grid_unc": grid,
                    "grid_con": grid,
                    "profile_ll": profile,
                    "mle_value": 0.0,
                },
            },
            mle_ll=0.0,
            mle_params={"p": jnp.array(0.0)},
            threshold=1.92,
            parameter_names=["p"],
        )

        assert result.summary()["p"] == "identified"

    def test_flat_profile_detection(self):
        """Flat profile should be classified as structurally_unidentifiable."""
        from causal_ssm_agent.utils.parametric_id import ProfileLikelihoodResult

        grid = jnp.linspace(-3, 3, 20)
        profile = jnp.zeros(20) - 10.0  # flat

        result = ProfileLikelihoodResult(
            parameter_profiles={
                "p": {
                    "grid_unc": grid,
                    "grid_con": grid,
                    "profile_ll": profile,
                    "mle_value": 0.0,
                },
            },
            mle_ll=-10.0,
            mle_params={"p": jnp.array(0.0)},
            threshold=1.92,
            parameter_names=["p"],
        )

        assert result.summary()["p"] == "structurally_unidentifiable"


class TestSBCCheck:
    """Test simulation-based calibration."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_sbc_identified_model_uniform_ranks(self):
        """Well-identified 1D LGSS with enough replicates should have uniform ranks."""
        from causal_ssm_agent.utils.parametric_id import sbc_check

        spec = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            manifest_means=jnp.zeros(1),
            diffusion="diag",
            t0_means=jnp.zeros(1),
            t0_var=jnp.eye(1),
        )
        priors = SSMPriors(
            drift_diag={"mu": -0.5, "sigma": 0.3},
            diffusion_diag={"sigma": 0.3},
            manifest_var_diag={"sigma": 0.3},
        )
        model = SSMModel(spec, priors, n_particles=50, likelihood="kalman")

        result = sbc_check(
            model,
            T=50,
            dt=1.0,
            n_sbc=20,
            method="nuts",
            num_warmup=200,
            num_samples=200,
            num_chains=1,
            seed=42,
        )

        result.print_report()
        summary = result.summary()

        # With a well-identified model and enough replicates,
        # we expect p > 0.01 (no strong evidence of miscalibration)
        # This is a soft check — SBC is stochastic
        n_failing = sum(
            1 for name, info in summary.items() if name != "_likelihood" and not info["uniform"]
        )
        # Allow at most 1 parameter to fail by chance
        assert n_failing <= 1, f"Too many SBC failures: {summary}"


class TestPowerScalingSensitivity:
    """Test post-fit power-scaling sensitivity."""

    @pytest.mark.slow
    def test_power_scaling_basic(self):
        """After fitting with simple data, power scaling should produce valid output."""
        from causal_ssm_agent.models.ssm.inference import InferenceResult
        from causal_ssm_agent.utils.parametric_id_postfit import power_scaling_sensitivity

        model = _make_identified_model()
        T = 20
        times = jnp.linspace(0, 10, T)

        # Create mock posterior samples that look reasonable
        n_samples = 50
        rng = random.PRNGKey(123)
        samples = {}
        rng, k1, k2, k3, k4, k5 = random.split(rng, 6)
        samples["drift_diag_pop"] = jnp.abs(random.normal(k1, (n_samples, 2))) * 0.5
        samples["drift_offdiag_pop"] = random.normal(k2, (n_samples, 2)) * 0.1
        samples["diffusion_diag_pop"] = jnp.abs(random.normal(k3, (n_samples, 2))) * 0.3
        samples["t0_means_pop"] = random.normal(k4, (n_samples, 2)) * 0.5
        samples["t0_var_diag"] = jnp.abs(random.normal(k5, (n_samples, 2))) * 0.5
        # Add manifest_var_diag
        rng, k6 = random.split(rng)
        samples["manifest_var_diag"] = jnp.abs(random.normal(k6, (n_samples, 2))) * 0.3

        mock_result = InferenceResult(
            _samples=samples,
            method="hessmc2",
            diagnostics={},
        )

        obs = jnp.zeros((T, 2))

        ps_result = power_scaling_sensitivity(
            model=model,
            observations=obs,
            times=times,
            result=mock_result,
            seed=42,
        )

        # Check structure
        assert isinstance(ps_result.prior_sensitivity, dict)
        assert isinstance(ps_result.likelihood_sensitivity, dict)
        assert isinstance(ps_result.diagnosis, dict)

        # All diagnosed params should have valid diagnosis values
        valid_diagnoses = {"prior_dominated", "well_identified", "prior_data_conflict"}
        for name, diag in ps_result.diagnosis.items():
            assert diag in valid_diagnoses, f"Invalid diagnosis for {name}: {diag}"


# ---------------------------------------------------------------------------
# Recovery tests: verify simulate_ssm + parametric ID against ground truth
# ---------------------------------------------------------------------------


class TestSimulateSSMRecovery:
    """Recovery tests: simulate from known params, fit, check posterior covers truth.

    Follows the same pattern as test_inference_strategies.py recovery tests.
    Uses 1D LGSS (D=3 params) for fast verification.
    """

    @pytest.fixture
    def lgss_ground_truth(self):
        """1D Linear Gaussian SSM ground truth + simulated data via simulate_ssm."""
        from causal_ssm_agent.utils.parametric_id import simulate_ssm

        n_latent, n_manifest = 1, 1
        T = 100

        true_drift_diag = -0.3
        true_diff_diag = 0.3
        true_obs_sd = 0.5

        drift = jnp.array([[true_drift_diag]])
        diffusion_chol = jnp.array([[true_diff_diag]])
        lambda_mat = jnp.eye(n_manifest, n_latent)
        manifest_chol = jnp.array([[true_obs_sd]])
        t0_means = jnp.zeros(n_latent)
        t0_chol = jnp.eye(n_latent)
        times = jnp.arange(T, dtype=jnp.float32)

        observations = simulate_ssm(
            drift=drift,
            diffusion_chol=diffusion_chol,
            lambda_mat=lambda_mat,
            manifest_chol=manifest_chol,
            t0_means=t0_means,
            t0_chol=t0_chol,
            times=times,
            rng_key=random.PRNGKey(42),
        )

        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=lambda_mat,
            manifest_means=jnp.zeros(n_manifest),
            diffusion="diag",
            t0_means=jnp.zeros(n_latent),
            t0_var=jnp.eye(n_latent),
        )

        return {
            "observations": observations,
            "times": times,
            "spec": spec,
            "true_drift_diag": true_drift_diag,
            "true_diff_diag": true_diff_diag,
            "true_obs_sd": true_obs_sd,
        }

    @pytest.mark.slow
    @pytest.mark.timeout(180)
    def test_simulate_ssm_kalman_recovery(self, lgss_ground_truth):
        """Data from simulate_ssm is recoverable by Kalman-based inference.

        Validates that simulate_ssm produces data consistent with the model's
        generative process: fit with NUTS+Kalman, check 90% CI coverage.
        """
        from causal_ssm_agent.models.ssm.inference import fit

        data = lgss_ground_truth
        model = SSMModel(data["spec"], n_particles=50, likelihood="kalman")

        result = fit(
            model,
            observations=data["observations"],
            times=data["times"],
            method="nuts",
            num_warmup=500,
            num_samples=500,
            num_chains=1,
            seed=0,
        )

        samples = result.get_samples()

        # drift_diag_pop: model applies -abs(), so recovered drift = -abs(sample)
        drift_samples = -jnp.abs(samples["drift_diag_pop"][:, 0])
        drift_q5 = float(jnp.percentile(drift_samples, 5))
        drift_q95 = float(jnp.percentile(drift_samples, 95))
        assert drift_q5 <= data["true_drift_diag"] <= drift_q95, (
            f"Drift {data['true_drift_diag']:.2f} outside 90% CI [{drift_q5:.3f}, {drift_q95:.3f}]"
        )

        # diffusion_diag_pop: HalfNormal, positive
        diff_samples = samples["diffusion_diag_pop"][:, 0]
        diff_q5 = float(jnp.percentile(diff_samples, 5))
        diff_q95 = float(jnp.percentile(diff_samples, 95))
        assert diff_q5 <= data["true_diff_diag"] <= diff_q95, (
            f"Diffusion {data['true_diff_diag']:.2f} outside 90% CI [{diff_q5:.3f}, {diff_q95:.3f}]"
        )

        # manifest_var_diag: observation noise SD
        obs_samples = samples["manifest_var_diag"][:, 0]
        obs_q5 = float(jnp.percentile(obs_samples, 5))
        obs_q95 = float(jnp.percentile(obs_samples, 95))
        assert obs_q5 <= data["true_obs_sd"] <= obs_q95, (
            f"Obs SD {data['true_obs_sd']:.2f} outside 90% CI [{obs_q5:.3f}, {obs_q95:.3f}]"
        )
