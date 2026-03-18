"""Tests for posterior predictive checks (PPCs)."""

import jax.numpy as jnp
import jax.random as random
import numpy as np

import causal_ssm_agent.models.posterior_predictive as posterior_predictive_module
from causal_ssm_agent.models.likelihoods.observation_families import (
    POSTERIOR_PREDICTIVE_SWITCH_ORDER,
    get_posterior_predictive_switch_index,
)
from causal_ssm_agent.models.posterior_predictive import (
    PPCResult,
    _check_calibration,
    _check_residual_autocorrelation,
    _check_variance_ratio,
    _compute_overlays,
    _compute_test_stats,
    get_relevant_manifest_variables,
    run_posterior_predictive_checks,
    simulate_posterior_predictive,
)


def _make_samples(
    n_draws: int = 20,
    n_latent: int = 2,
    n_manifest: int = 3,
    seed: int = 0,
    drift_diag: float = -0.3,
    diff_sd: float = 0.3,
    obs_sd: float = 0.5,
    with_cint: bool = False,
) -> dict[str, jnp.ndarray]:
    """Build synthetic posterior samples for testing."""
    key = random.PRNGKey(seed)

    # Drift: diagonal negative, small off-diagonal
    k1, *_ = random.split(key, 6)
    drift_base = jnp.eye(n_latent) * drift_diag
    offdiag = random.normal(k1, (n_draws, n_latent, n_latent)) * 0.01
    drift_draws = jnp.broadcast_to(drift_base, (n_draws, n_latent, n_latent)) + offdiag
    # Keep diagonal negative
    diag_idx = jnp.arange(n_latent)
    drift_draws = drift_draws.at[:, diag_idx, diag_idx].set(
        -jnp.abs(drift_draws[:, diag_idx, diag_idx])
    )

    # Diffusion: cholesky factor (diagonal)
    diff_chol = jnp.eye(n_latent) * diff_sd
    diffusion_draws = jnp.broadcast_to(diff_chol, (n_draws, n_latent, n_latent))

    # Lambda: identity-like with extra rows
    lambda_mat = jnp.zeros((n_manifest, n_latent))
    for i in range(min(n_manifest, n_latent)):
        lambda_mat = lambda_mat.at[i, i].set(1.0)
    # Extra manifest variables load on first latent
    for i in range(n_latent, n_manifest):
        lambda_mat = lambda_mat.at[i, 0].set(0.5)

    # Manifest cov: diagonal
    manifest_cov = jnp.eye(n_manifest) * obs_sd**2

    # t0
    t0_means = jnp.zeros((n_draws, n_latent))
    t0_cov = jnp.eye(n_latent) * 1.0

    samples = {
        "drift": drift_draws,
        "diffusion": diffusion_draws,
        "lambda": lambda_mat,
        "manifest_cov": manifest_cov,
        "t0_means": t0_means,
        "t0_cov": t0_cov,
    }

    if with_cint:
        cint_draws = jnp.zeros((n_draws, n_latent))
        samples["cint"] = cint_draws

    return samples


class TestForwardSimulation:
    """Tests for simulate_posterior_predictive."""

    def test_switch_indices_follow_registry_order(self):
        """Posterior predictive dispatch indices come from the shared registry order."""
        for idx, (dist, link) in enumerate(POSTERIOR_PREDICTIVE_SWITCH_ORDER):
            assert get_posterior_predictive_switch_index(dist, link=link) == idx

    def test_switch_default_link_uses_explicit_family_default(self):
        """Omitting the link should resolve to the first registered branch for that family."""
        first_index_by_dist = {}
        for idx, (dist, _link) in enumerate(POSTERIOR_PREDICTIVE_SWITCH_ORDER):
            first_index_by_dist.setdefault(dist, idx)

        for dist, first_idx in first_index_by_dist.items():
            assert get_posterior_predictive_switch_index(dist) == first_idx

    def test_forward_simulate_shape(self):
        """Output shape is (n_subsample, T, n_manifest)."""
        n_draws, T, n_latent, n_manifest = 10, 20, 2, 3
        samples = _make_samples(n_draws=n_draws, n_latent=n_latent, n_manifest=n_manifest)
        times = jnp.arange(T, dtype=float)

        y_sim = simulate_posterior_predictive(samples=samples, times=times, n_subsample=n_draws)

        assert y_sim.shape == (n_draws, T, n_manifest)
        assert jnp.all(jnp.isfinite(y_sim))

    def test_forward_simulate_subsample(self):
        """Subsampling returns fewer draws than total."""
        samples = _make_samples(n_draws=50, n_latent=2, n_manifest=2)
        times = jnp.arange(15, dtype=float)

        y_sim = simulate_posterior_predictive(samples=samples, times=times, n_subsample=10)

        assert y_sim.shape[0] == 10

    def test_forward_simulate_uses_t0_for_first_observation(self, monkeypatch):
        """The first observation is emitted from the initial state, not a fake transition."""
        samples = {
            "drift": jnp.array([[[-0.1]]]),
            "diffusion": jnp.zeros((1, 1, 1)),
            "lambda": jnp.array([[[1.0]]]),
            "manifest_cov": jnp.array([[[0.0]]]),
            "t0_means": jnp.array([[2.0]]),
            "t0_cov": jnp.array([[[0.0]]]),
        }
        times = jnp.array([0.0, 1.0, 2.0])

        def fake_discretize_system_batched(drift, diffusion_cov, cint, dt_array):
            del drift, diffusion_cov, cint
            assert dt_array.shape == (2,)
            Ad = jnp.array([[[10.0]], [[1.0]]])
            Qd = jnp.zeros((2, 1, 1))
            cd = jnp.array([[5.0], [0.0]])
            return Ad, Qd, cd

        monkeypatch.setattr(
            posterior_predictive_module,
            "discretize_system_batched",
            fake_discretize_system_batched,
        )

        y_sim = posterior_predictive_module.simulate_posterior_predictive(
            samples=samples,
            times=times,
            n_subsample=1,
            rng_seed=0,
        )

        assert y_sim.shape == (1, 3, 1)
        assert abs(float(y_sim[0, 0, 0]) - 2.0) < 0.01

    def test_forward_simulate_mixed_repairs_slightly_indefinite_process_covariance(
        self, monkeypatch
    ):
        """Mixed-family simulation stays finite when discretization is numerically indefinite."""
        samples = _make_samples(n_draws=3, n_latent=2, n_manifest=3, obs_sd=0.1)
        samples["lambda"] = jnp.broadcast_to(samples["lambda"], (3, *samples["lambda"].shape))
        samples["manifest_cov"] = jnp.broadcast_to(
            samples["manifest_cov"], (3, *samples["manifest_cov"].shape)
        )
        samples["t0_cov"] = jnp.broadcast_to(samples["t0_cov"], (3, *samples["t0_cov"].shape))
        times = jnp.array([0.0, 1.0, 2.0])

        def fake_discretize_system_batched(drift, diffusion_cov, cint, dt_array):
            del drift, diffusion_cov, cint
            assert dt_array.shape == (2,)
            Ad = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
            Qd = jnp.array(
                [
                    [[1.0e-8, 0.0], [0.0, -1.5e-8]],
                    [[1.0e-3, 0.0], [0.0, 1.0e-3]],
                ]
            )
            cd = jnp.zeros((2, 2))
            return Ad, Qd, cd

        monkeypatch.setattr(
            posterior_predictive_module,
            "discretize_system_batched",
            fake_discretize_system_batched,
        )

        y_sim = posterior_predictive_module.simulate_posterior_predictive(
            samples=samples,
            times=times,
            manifest_dists=["gaussian", "bernoulli", "gaussian"],
            n_subsample=3,
            rng_seed=0,
        )

        assert y_sim.shape == (3, 3, 3)
        assert jnp.all(jnp.isfinite(y_sim))

    def test_forward_simulate_poisson(self):
        """Poisson noise family produces non-negative observations."""
        samples = _make_samples(n_draws=10, n_latent=2, n_manifest=2, obs_sd=0.1)
        times = jnp.arange(15, dtype=float)

        y_sim = simulate_posterior_predictive(
            samples=samples, times=times, manifest_dist="poisson", n_subsample=10
        )

        assert y_sim.shape == (10, 15, 2)
        # Poisson samples are non-negative integers
        assert jnp.all(y_sim >= 0)

    def test_forward_simulate_student_t(self):
        """Student-t noise family produces finite values with heavier tails."""
        samples = _make_samples(n_draws=10, n_latent=2, n_manifest=2, obs_sd=0.5)
        samples["obs_df"] = jnp.array(3.0)  # low df = heavy tails
        times = jnp.arange(15, dtype=float)

        y_sim = simulate_posterior_predictive(
            samples=samples, times=times, manifest_dist="student_t", n_subsample=10
        )

        assert y_sim.shape == (10, 15, 2)
        assert jnp.all(jnp.isfinite(y_sim))

    def test_forward_simulate_gamma(self):
        """Gamma noise family produces positive observations."""
        samples = _make_samples(n_draws=10, n_latent=2, n_manifest=2)
        samples["obs_shape"] = jnp.array(2.0)
        times = jnp.arange(15, dtype=float)

        y_sim = simulate_posterior_predictive(
            samples=samples, times=times, manifest_dist="gamma", n_subsample=10
        )

        assert y_sim.shape == (10, 15, 2)
        assert jnp.all(y_sim > 0)

    def test_forward_simulate_ordered_logistic(self):
        """Ordered-logistic simulation returns encoded category indices."""
        samples = _make_samples(n_draws=10, n_latent=2, n_manifest=2)
        samples["obs_ordered_cutpoints"] = jnp.array([[-1.0, 1.0, 0.0], [-1.5, 0.0, 1.5]])
        times = jnp.arange(15, dtype=float)

        y_sim = simulate_posterior_predictive(
            samples=samples,
            times=times,
            manifest_dist="ordered_logistic",
            manifest_level_counts=[3, 4],
            n_subsample=10,
        )

        assert y_sim.shape == (10, 15, 2)
        assert jnp.all(jnp.isfinite(y_sim))
        assert jnp.all((y_sim[:, :, 0] >= 0) & (y_sim[:, :, 0] <= 2))
        assert jnp.all((y_sim[:, :, 1] >= 0) & (y_sim[:, :, 1] <= 3))

    def test_forward_simulate_categorical(self):
        """Categorical simulation returns encoded category indices."""
        samples = _make_samples(n_draws=10, n_latent=2, n_manifest=2)
        samples["obs_cat_intercepts"] = jnp.array([[-1.0, 0.5], [0.2, -0.3]])
        samples["obs_cat_slopes"] = jnp.array([[0.2, -0.4], [0.5, 0.1]])
        times = jnp.arange(15, dtype=float)

        y_sim = simulate_posterior_predictive(
            samples=samples,
            times=times,
            manifest_dist="categorical",
            manifest_level_counts=[3, 3],
            n_subsample=10,
        )

        assert y_sim.shape == (10, 15, 2)
        assert jnp.all(jnp.isfinite(y_sim))
        assert jnp.all((y_sim >= 0) & (y_sim <= 2))


class TestDiagnosticChecks:
    """Tests for individual diagnostic checks."""

    def test_calibration_well_specified(self):
        """Data generated from same model should have good calibration."""
        n_draws, T, n_manifest = 100, 50, 2
        samples = _make_samples(n_draws=n_draws, n_latent=2, n_manifest=n_manifest)
        times = jnp.arange(T, dtype=float)

        y_sim = simulate_posterior_predictive(
            samples=samples, times=times, n_subsample=n_draws, rng_seed=0
        )
        # Use one draw as "observed data" — should be well-calibrated
        key = random.PRNGKey(99)
        obs_idx = random.randint(key, (), 0, n_draws)
        observations = y_sim[obs_idx]  # (T, m)

        warnings = _check_calibration(y_sim, observations, [f"var_{j}" for j in range(n_manifest)])

        # Well-specified: no undercoverage warnings (overcoverage OK since
        # using one of the draws as "observed" biases coverage upward)
        undercoverage = [w for w in warnings if "Undercoverage" in w.message]
        assert len(undercoverage) == 0

    def test_calibration_misspecified(self):
        """Wrong parameters should trigger calibration warning."""
        T, n_manifest = 50, 2
        manifest_names = [f"var_{j}" for j in range(n_manifest)]

        # Simulate from one model
        samples_true = _make_samples(
            n_draws=100, n_latent=2, n_manifest=n_manifest, drift_diag=-0.3
        )
        times = jnp.arange(T, dtype=float)
        y_sim_true = simulate_posterior_predictive(
            samples=samples_true, times=times, n_subsample=100, rng_seed=0
        )

        # Observations from a very different model (large shift)
        observations = jnp.ones((T, n_manifest)) * 100.0  # way outside PPC range

        warnings = _check_calibration(y_sim_true, observations, manifest_names)

        # Should flag at least one variable
        assert len(warnings) > 0
        assert any(w.check_type == "calibration" for w in warnings)

    def test_autocorrelation_detection(self):
        """Correlated residuals should be flagged."""
        T, n_manifest = 100, 1
        manifest_names = ["y"]

        # Create simulated data with zero-mean
        key = random.PRNGKey(42)
        y_sim = random.normal(key, (50, T, n_manifest)) * 0.5

        # Create observations with strong autocorrelation in residuals
        pp_mean = jnp.mean(y_sim, axis=0)  # (T, 1)
        # AR(1) residuals with rho=0.8
        key2 = random.PRNGKey(123)
        noise = random.normal(key2, (T,)) * 0.1
        resid = jnp.zeros(T)
        for t in range(1, T):
            resid = resid.at[t].set(0.8 * resid[t - 1] + noise[t])
        observations = pp_mean + resid[:, None]

        warnings = _check_residual_autocorrelation(y_sim, observations, manifest_names)

        assert len(warnings) > 0
        assert any(w.check_type == "autocorrelation" for w in warnings)

    def test_variance_ratio_detection(self):
        """Scale mismatch should be flagged."""
        T, n_manifest = 50, 1
        manifest_names = ["y"]

        # Simulated data with small variance
        key = random.PRNGKey(0)
        y_sim = random.normal(key, (50, T, n_manifest)) * 0.1

        # Observations with much larger variance
        key2 = random.PRNGKey(1)
        observations = random.normal(key2, (T, n_manifest)) * 10.0

        warnings = _check_variance_ratio(y_sim, observations, manifest_names)

        assert len(warnings) > 0
        assert any(w.check_type == "variance" for w in warnings)

    def test_nan_handling(self):
        """NaN observations should be skipped without errors."""
        T, n_manifest = 30, 2
        manifest_names = ["x", "y"]

        key = random.PRNGKey(0)
        y_sim = random.normal(key, (20, T, n_manifest))

        # Observations with some NaNs
        key2 = random.PRNGKey(1)
        observations = random.normal(key2, (T, n_manifest))
        observations = observations.at[:5, 0].set(jnp.nan)  # first 5 timepoints of var 0
        observations = observations.at[10:15, 1].set(jnp.nan)

        # Should not raise
        cal_warnings = _check_calibration(y_sim, observations, manifest_names)
        ac_warnings = _check_residual_autocorrelation(y_sim, observations, manifest_names)
        vr_warnings = _check_variance_ratio(y_sim, observations, manifest_names)

        # All should return lists (possibly empty)
        assert isinstance(cal_warnings, list)
        assert isinstance(ac_warnings, list)
        assert isinstance(vr_warnings, list)


class TestGetRelevantManifestVariables:
    """Tests for get_relevant_manifest_variables."""

    def test_identity_lambda(self):
        """Identity lambda maps each manifest to its latent."""
        lambda_mat = jnp.eye(3)
        names = ["x", "y", "z"]

        result = get_relevant_manifest_variables(lambda_mat, 0, 1, names)
        assert result == {"x", "y"}

    def test_extra_loadings(self):
        """Extra manifest variables with nonzero loadings are included."""
        # 4 manifest, 2 latent
        lambda_mat = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.0],  # loads on latent 0
                [0.0, 0.3],  # loads on latent 1
            ]
        )
        names = ["a", "b", "c", "d"]

        result = get_relevant_manifest_variables(lambda_mat, 0, 1, names)
        assert result == {"a", "b", "c", "d"}

    def test_threshold_filtering(self):
        """Loadings below threshold are excluded."""
        lambda_mat = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.005, 0.0],  # below default threshold 0.01
            ]
        )
        names = ["a", "b", "c"]

        result = get_relevant_manifest_variables(lambda_mat, 0, 1, names)
        assert result == {"a", "b"}

    def test_none_indices(self):
        """None indices should be safely skipped."""
        lambda_mat = jnp.eye(2)
        names = ["x", "y"]

        result = get_relevant_manifest_variables(lambda_mat, None, 1, names)
        assert result == {"y"}

        result = get_relevant_manifest_variables(lambda_mat, None, None, names)
        assert result == set()


class TestLinkFunctionSimulation:
    """Tests for forward simulation with non-default link functions."""

    def test_forward_simulate_bernoulli_probit(self):
        """Probit Bernoulli produces valid binary-range observations."""
        samples = _make_samples(n_draws=10, n_latent=2, n_manifest=2, obs_sd=0.1)
        times = jnp.arange(15, dtype=float)

        y_sim = simulate_posterior_predictive(
            samples=samples,
            times=times,
            manifest_dist="bernoulli",
            manifest_links=["probit", "probit"],
            n_subsample=10,
        )

        assert y_sim.shape == (10, 15, 2)
        assert jnp.all(jnp.isfinite(y_sim))
        # Bernoulli samples should be 0 or 1
        assert jnp.all((y_sim == 0) | (y_sim == 1))

    def test_forward_simulate_gamma_inverse(self):
        """Inverse Gamma produces positive observations."""
        samples = _make_samples(n_draws=10, n_latent=2, n_manifest=2)
        samples["obs_shape"] = jnp.array(2.0)
        times = jnp.arange(15, dtype=float)

        y_sim = simulate_posterior_predictive(
            samples=samples,
            times=times,
            manifest_dist="gamma",
            manifest_links=["inverse", "inverse"],
            n_subsample=10,
        )

        assert y_sim.shape == (10, 15, 2)
        assert jnp.all(jnp.isfinite(y_sim))
        assert jnp.all(y_sim > 0)

    def test_forward_simulate_beta_probit(self):
        """Probit Beta produces observations in (0, 1)."""
        samples = _make_samples(n_draws=10, n_latent=2, n_manifest=2, obs_sd=0.1)
        times = jnp.arange(15, dtype=float)

        y_sim = simulate_posterior_predictive(
            samples=samples,
            times=times,
            manifest_dist="beta",
            manifest_links=["probit", "probit"],
            n_subsample=10,
        )

        assert y_sim.shape == (10, 15, 2)
        assert jnp.all(jnp.isfinite(y_sim))
        # Beta samples should be in [0, 1] (clipping may produce boundary values)
        assert jnp.all((y_sim >= 0) & (y_sim <= 1))

    def test_mixed_links_dispatch(self):
        """Mixed distribution with non-default links uses correct dispatch."""
        samples = _make_samples(n_draws=10, n_latent=2, n_manifest=2, obs_sd=0.1)
        times = jnp.arange(10, dtype=float)

        # Channel 0: Bernoulli probit, Channel 1: Bernoulli logit (default)
        y_sim = simulate_posterior_predictive(
            samples=samples,
            times=times,
            manifest_dists=["bernoulli", "bernoulli"],
            manifest_links=["probit", "logit"],
            n_subsample=10,
        )

        assert y_sim.shape == (10, 10, 2)
        assert jnp.all(jnp.isfinite(y_sim))
        assert jnp.all((y_sim == 0) | (y_sim == 1))


class TestRunPPC:
    """Integration test for run_posterior_predictive_checks."""

    def test_basic_run(self):
        """Full PPC pipeline runs without errors."""
        T, n_latent, n_manifest = 30, 2, 2
        samples = _make_samples(n_draws=20, n_latent=n_latent, n_manifest=n_manifest)
        times = jnp.arange(T, dtype=float)

        key = random.PRNGKey(7)
        observations = random.normal(key, (T, n_manifest))
        manifest_names = ["x", "y"]

        result = run_posterior_predictive_checks(
            samples=samples,
            observations=observations,
            times=times,
            manifest_names=manifest_names,
            n_subsample=20,
        )

        assert isinstance(result, PPCResult)
        assert result.checked is True
        assert result.n_subsample == 20
        assert isinstance(result.per_variable_warnings, list)


# =============================================================================
# _compute_overlays
# =============================================================================


class TestComputeOverlays:
    def _make_data(self, n_draws=10, T=5, n_manifest=2):
        """Create synthetic simulation data."""
        rng = np.random.default_rng(42)
        y_sim = jnp.array(rng.normal(0, 1, (n_draws, T, n_manifest)))
        observations = jnp.array(rng.normal(0, 1, (T, n_manifest)))
        return y_sim, observations

    def test_returns_one_overlay_per_variable(self):
        y_sim, obs = self._make_data(n_manifest=3)
        result = _compute_overlays(y_sim, obs, ["x", "y", "z"])
        assert len(result) == 3
        assert result[0].variable == "x"
        assert result[1].variable == "y"
        assert result[2].variable == "z"

    def test_overlay_has_correct_length(self):
        y_sim, obs = self._make_data(T=7, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"])
        assert len(result[0].q025) == 7
        assert len(result[0].median) == 7
        assert len(result[0].observed) == 7

    def test_quantile_ordering(self):
        """q025 <= q25 <= median <= q75 <= q975 at each time step."""
        y_sim, obs = self._make_data(n_draws=50, T=10, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"])
        overlay = result[0]
        for t in range(10):
            assert overlay.q025[t] <= overlay.q25[t]
            assert overlay.q25[t] <= overlay.median[t]
            assert overlay.median[t] <= overlay.q75[t]
            assert overlay.q75[t] <= overlay.q975[t]

    def test_spaghetti_draws_count(self):
        y_sim, obs = self._make_data(n_draws=30, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"], n_spaghetti=5)
        assert len(result[0].spaghetti_draws) == 5

    def test_spaghetti_count_capped_at_n_draws(self):
        y_sim, obs = self._make_data(n_draws=3, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"], n_spaghetti=100)
        assert len(result[0].spaghetti_draws) == 3

    def test_spaghetti_draw_length_matches_T(self):
        y_sim, obs = self._make_data(n_draws=5, T=8, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"])
        for draw in result[0].spaghetti_draws:
            assert len(draw) == 8

    def test_nan_observations_become_none(self):
        y_sim, obs = self._make_data(T=3, n_manifest=1)
        obs_with_nan = obs.at[1, 0].set(float("nan"))
        result = _compute_overlays(y_sim, obs_with_nan, ["x"])
        assert result[0].observed[1] is None
        assert result[0].observed[0] is not None

    def test_fallback_variable_name(self):
        """When manifest_names is shorter than n_manifest, use fallback."""
        y_sim, obs = self._make_data(n_manifest=2)
        result = _compute_overlays(y_sim, obs, ["x"])  # only 1 name for 2 vars
        assert result[0].variable == "x"
        assert result[1].variable == "var_1"

    def test_single_draw(self):
        y_sim, obs = self._make_data(n_draws=1, T=3, n_manifest=1)
        result = _compute_overlays(y_sim, obs, ["x"])
        # All quantiles should be the same value
        assert result[0].q025 == result[0].q975


# =============================================================================
# _compute_test_stats
# =============================================================================


class TestComputeTestStats:
    def _make_data(self, n_draws=10, T=20, n_manifest=2):
        rng = np.random.default_rng(42)
        y_sim = jnp.array(rng.normal(0, 1, (n_draws, T, n_manifest)))
        observations = jnp.array(rng.normal(0, 1, (T, n_manifest)))
        return y_sim, observations

    def test_returns_4_stats_per_variable(self):
        y_sim, obs = self._make_data(n_manifest=1)
        result = _compute_test_stats(y_sim, obs, ["x"])
        assert len(result) == 4
        stat_names = {r.stat_name for r in result}
        assert stat_names == {"mean", "sd", "min", "max"}

    def test_multiple_variables(self):
        y_sim, obs = self._make_data(n_manifest=2)
        result = _compute_test_stats(y_sim, obs, ["x", "y"])
        assert len(result) == 8  # 4 stats * 2 variables

    def test_rep_values_length_matches_n_draws(self):
        y_sim, obs = self._make_data(n_draws=15, n_manifest=1)
        result = _compute_test_stats(y_sim, obs, ["x"])
        for stat in result:
            assert len(stat.rep_values) == 15

    def test_observed_value_is_float(self):
        y_sim, obs = self._make_data(n_manifest=1)
        result = _compute_test_stats(y_sim, obs, ["x"])
        for stat in result:
            assert isinstance(stat.observed_value, float)

    def test_skips_variables_with_too_few_valid_obs(self):
        """Variables with < 3 valid observations should be skipped."""
        y_sim, obs = self._make_data(T=5, n_manifest=1)
        # Make all but 2 observations NaN
        obs_sparse = jnp.full_like(obs, float("nan"))
        obs_sparse = obs_sparse.at[0, 0].set(1.0)
        obs_sparse = obs_sparse.at[1, 0].set(2.0)
        result = _compute_test_stats(y_sim, obs_sparse, ["x"])
        assert len(result) == 0

    def test_handles_nan_masking(self):
        """NaN values in observations should be excluded from stats."""
        y_sim, obs = self._make_data(T=10, n_manifest=1)
        obs_with_nan = obs.at[0, 0].set(float("nan"))
        result = _compute_test_stats(y_sim, obs_with_nan, ["x"])
        # Should still produce results (9 valid obs > 3)
        assert len(result) == 4

    def test_mean_stat_is_reasonable(self):
        """Mean of replicated data should be near 0 for N(0,1) draws."""
        rng = np.random.default_rng(0)
        y_sim = jnp.array(rng.normal(0, 1, (50, 100, 1)))
        obs = jnp.array(rng.normal(0, 1, (100, 1)))
        result = _compute_test_stats(y_sim, obs, ["x"])
        mean_stat = next(r for r in result if r.stat_name == "mean")
        # Observed mean should be within range of replicated means
        rep_min = min(mean_stat.rep_values)
        rep_max = max(mean_stat.rep_values)
        # Generous bounds since data is random
        assert rep_min < 0.5
        assert rep_max > -0.5

    def test_fallback_variable_name(self):
        y_sim, obs = self._make_data(n_manifest=2)
        result = _compute_test_stats(y_sim, obs, ["x"])
        vars_seen = {r.variable for r in result}
        assert "x" in vars_seen
        assert "var_1" in vars_seen
