"""Tests for posterior predictive checks (PPCs)."""

import jax.numpy as jnp
import jax.random as random

from causal_ssm_agent.models.posterior_predictive import (
    PPCResult,
    PPCWarning,
    _check_calibration,
    _check_residual_autocorrelation,
    _check_variance_ratio,
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


class TestPPCDataclasses:
    """Tests for PPCWarning and PPCResult Pydantic models."""

    def test_ppc_warning_to_dict(self):
        w = PPCWarning(variable="x", check_type="calibration", message="bad", value=0.5)
        d = w.model_dump()
        assert d == {
            "variable": "x",
            "check_type": "calibration",
            "message": "bad",
            "value": 0.5,
            "passed": True,
        }

    def test_ppc_warning_to_dict_failed(self):
        w = PPCWarning(variable="x", check_type="calibration", message="bad", value=0.5, passed=False)
        d = w.model_dump()
        assert d["passed"] is False
        assert d["check_type"] == "calibration"

    def test_ppc_result_to_dict(self):
        w = PPCWarning(variable="x", check_type="calibration", message="bad", value=0.5)
        r = PPCResult(per_variable_warnings=[w], checked=True, n_subsample=50)
        d = r.model_dump()
        assert d["checked"] is True
        assert d["n_subsample"] == 50
        assert len(d["per_variable_warnings"]) == 1
        assert d["per_variable_warnings"][0]["variable"] == "x"
        assert d["overlays"] == []
        assert d["test_stats"] == []

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
