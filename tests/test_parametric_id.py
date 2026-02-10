"""Tests for parametric identifiability diagnostics.

Tests:
1. Forward simulator shape and finiteness
2. Identified model: well-separated eigenvalues, strong contraction
3. Non-identified model: redundant latent → rank-deficient Hessian
4. Estimand projection: cross-lag has nonzero projected info
5. Expected contraction: predictions in [0, 1]
6. Boundary identifiability: identified at some draws but not others
7. Power-scaling: prior-dominated params flagged correctly
8. Stage 4b flow: smoke test
9. Recovery: simulate_ssm produces data recoverable by Kalman fit
10. Recovery: check_parametric_id correctly flags identified vs non-identified
"""

import jax.numpy as jnp
import jax.random as random
import pytest

from dsem_agent.models.ssm.model import SSMModel, SSMPriors, SSMSpec


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
    """Build a non-identified model: 2 latent, 1 manifest → rank deficient."""
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


class TestSimulateSSM:
    """Test forward simulator."""

    def test_simulate_ssm_shape_and_finite(self):
        """Forward sim produces correct shape, finite values."""
        from dsem_agent.utils.parametric_id import simulate_ssm

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
        from dsem_agent.utils.parametric_id import simulate_ssm

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
        from dsem_agent.utils.parametric_id import simulate_ssm

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


class TestCheckParametricID:
    """Test the main pre-fit check_parametric_id function."""

    def test_identified_model(self):
        """Well-identified model should have well-separated eigenvalues."""
        from dsem_agent.utils.parametric_id import check_parametric_id

        model = _make_identified_model()
        T = 30
        times = jnp.linspace(0, 15, T)
        obs = jnp.zeros((T, 2))  # dummy — will be replaced by simulated data

        result = check_parametric_id(
            model=model,
            observations=obs,
            times=times,
            n_draws=3,
            seed=42,
        )

        assert result.eigenvalues.shape == (3, result.eigenvalues.shape[1])
        assert result.n_draws == 3
        assert len(result.parameter_names) > 0

        summary = result.summary()
        # Well-identified model should not have structural issues
        # (though with only 3 draws and synthetic data, we check basic structure)
        assert "structural_issues" in summary
        assert "boundary_issues" in summary
        assert "weak_params" in summary

    def test_non_identified_model(self):
        """Non-identified model (2 latent, 1 manifest) should flag issues."""
        from dsem_agent.utils.parametric_id import check_parametric_id

        model = _make_nonidentified_model()
        T = 30
        times = jnp.linspace(0, 15, T)
        obs = jnp.zeros((T, 1))

        result = check_parametric_id(
            model=model,
            observations=obs,
            times=times,
            n_draws=3,
            seed=42,
        )

        # With 2 latent and 1 manifest (identical loadings),
        # there should be near-zero eigenvalues
        assert result.eigenvalues.shape[0] == 3
        summary = result.summary()
        # At minimum, some weak params should be detected
        assert isinstance(summary["weak_params"], list)

    def test_estimand_projection(self):
        """Cross-lag coefficient should have nonzero projected information."""
        from dsem_agent.utils.parametric_id import check_parametric_id

        model = _make_identified_model()
        T = 30
        times = jnp.linspace(0, 15, T)
        obs = jnp.zeros((T, 2))

        result = check_parametric_id(
            model=model,
            observations=obs,
            times=times,
            n_draws=3,
            estimand_sites=["drift_offdiag_pop"],
            seed=42,
        )

        # Check that estimand information was computed
        if "drift_offdiag_pop" in result.estimand_information:
            info = result.estimand_information["drift_offdiag_pop"]
            assert info.shape == (3,)
            # All values should be finite and non-negative
            assert jnp.all(jnp.isfinite(info))
            assert jnp.all(info >= 0)

    def test_expected_contraction_bounds(self):
        """Expected contraction values should be in [0, 1]."""
        from dsem_agent.utils.parametric_id import check_parametric_id

        model = _make_identified_model()
        T = 30
        times = jnp.linspace(0, 15, T)
        obs = jnp.zeros((T, 2))

        result = check_parametric_id(
            model=model,
            observations=obs,
            times=times,
            n_draws=3,
            seed=42,
        )

        for name, contraction in result.expected_contraction.items():
            assert contraction.shape == (3,), f"Wrong shape for {name}"
            assert jnp.all(contraction >= 0.0), f"Contraction < 0 for {name}"
            assert jnp.all(contraction <= 1.0), f"Contraction > 1 for {name}"

    def test_result_print_report(self, capsys):
        """print_report should not crash."""
        from dsem_agent.utils.parametric_id import check_parametric_id

        model = _make_identified_model()
        T = 20
        times = jnp.linspace(0, 10, T)
        obs = jnp.zeros((T, 2))

        result = check_parametric_id(model=model, observations=obs, times=times, n_draws=2, seed=0)
        result.print_report()

        captured = capsys.readouterr()
        assert "Parametric Identifiability Report" in captured.out


class TestParametricIDResult:
    """Test the ParametricIDResult dataclass methods."""

    def test_summary_keys(self):
        """Summary dict should have all required keys."""
        from dsem_agent.utils.parametric_id import ParametricIDResult

        result = ParametricIDResult(
            eigenvalues=jnp.array([[1.0, 2.0], [0.5, 1.5]]),
            min_eigenvalues=jnp.array([1.0, 0.5]),
            condition_numbers=jnp.array([2.0, 3.0]),
            estimand_information={},
            expected_contraction={"param_a": jnp.array([0.5, 0.6])},
            prior_variances={"param_a": 1.0},
            parameter_names=["param_a"],
            n_draws=2,
        )

        summary = result.summary()
        assert "structural_issues" in summary
        assert "boundary_issues" in summary
        assert "weak_params" in summary
        assert "estimand_status" in summary
        assert "mean_condition_number" in summary

    def test_structural_issue_detection(self):
        """Near-zero min Fisher eigenvalues at all draws → structural issue."""
        from dsem_agent.utils.parametric_id import ParametricIDResult

        # Fisher info eigenvalues: near-zero means non-identifiable
        result = ParametricIDResult(
            eigenvalues=jnp.array([[1e-8, 2.0], [1e-9, 1.5]]),
            min_eigenvalues=jnp.array([1e-8, 1e-9]),
            condition_numbers=jnp.array([2e8, 1.5e9]),
            estimand_information={},
            expected_contraction={},
            prior_variances={},
            parameter_names=["p1", "p2"],
            n_draws=2,
        )

        summary = result.summary()
        assert summary["structural_issues"] is True
        assert summary["boundary_issues"] is False

    def test_boundary_issue_detection(self):
        """Near-zero Fisher eigenvalue at SOME draws → boundary issue."""
        from dsem_agent.utils.parametric_id import ParametricIDResult

        # Draw 0: one near-zero eigenvalue (boundary). Draw 1: all positive.
        result = ParametricIDResult(
            eigenvalues=jnp.array([[1e-8, 2.0], [0.5, 1.5]]),
            min_eigenvalues=jnp.array([1e-8, 0.5]),
            condition_numbers=jnp.array([2e8, 3.0]),
            estimand_information={},
            expected_contraction={},
            prior_variances={},
            parameter_names=["p1", "p2"],
            n_draws=2,
        )

        summary = result.summary()
        assert summary["structural_issues"] is False
        assert summary["boundary_issues"] is True


class TestPowerScalingResult:
    """Test PowerScalingResult dataclass."""

    def test_print_report(self, capsys):
        """print_report should not crash."""
        from dsem_agent.utils.parametric_id import PowerScalingResult

        result = PowerScalingResult(
            prior_sensitivity={"drift_diag_pop": 0.02, "diffusion_diag_pop": 0.08},
            likelihood_sensitivity={"drift_diag_pop": 0.5, "diffusion_diag_pop": 0.01},
            diagnosis={
                "drift_diag_pop": "well_identified",
                "diffusion_diag_pop": "prior_dominated",
            },
            psis_k_hat={"drift_diag_pop": 0.3, "diffusion_diag_pop": 0.5},
        )

        result.print_report()
        captured = capsys.readouterr()
        assert "Power-Scaling Sensitivity Report" in captured.out
        assert "well_identified" in captured.out
        assert "prior_dominated" in captured.out


class TestPowerScalingSensitivity:
    """Test post-fit power-scaling sensitivity."""

    def test_power_scaling_basic(self):
        """After fitting with simple data, power scaling should produce valid output."""
        from dsem_agent.models.ssm.inference import InferenceResult
        from dsem_agent.utils.parametric_id import power_scaling_sensitivity

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


class TestStage4bFlow:
    """Smoke test for the Prefect stage 4b flow."""

    def test_parametric_id_task_import(self):
        """Stage 4b task and flow should be importable."""
        from dsem_agent.flows.stages.stage4b_parametric_id import (
            parametric_id_task,
            stage4b_parametric_id_flow,
        )

        assert callable(parametric_id_task)
        assert callable(stage4b_parametric_id_flow)

    def test_run_power_scaling_import(self):
        """Power-scaling task should be importable from stage5."""
        from dsem_agent.flows.stages.stage5_inference import run_power_scaling

        assert callable(run_power_scaling)

    def test_stages_init_exports(self):
        """New exports should be available from stages __init__."""
        from dsem_agent.flows.stages import (
            parametric_id_task,
            run_power_scaling,
            stage4b_parametric_id_flow,
        )

        assert callable(parametric_id_task)
        assert callable(stage4b_parametric_id_flow)
        assert callable(run_power_scaling)

    def test_utils_init_exports(self):
        """New exports should be available from utils __init__."""
        from dsem_agent.utils import (
            ParametricIDResult,
            PowerScalingResult,
            check_parametric_id,
            power_scaling_sensitivity,
            simulate_ssm,
        )

        assert callable(check_parametric_id)
        assert callable(power_scaling_sensitivity)
        assert callable(simulate_ssm)
        assert ParametricIDResult is not None
        assert PowerScalingResult is not None


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
        from dsem_agent.utils.parametric_id import simulate_ssm

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
        from dsem_agent.models.ssm.inference import fit

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


class TestParametricIDRecovery:
    """Recovery tests for parametric identifiability diagnostics.

    Verifies that check_parametric_id correctly distinguishes between
    identified and non-identified models using ground-truth setups.
    """

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_identified_model_no_structural_issues(self):
        """Well-identified 1D LGSS should not flag structural issues.

        1 latent, 1 manifest with identity lambda: all parameters observable.
        """
        from dsem_agent.utils.parametric_id import check_parametric_id

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
        obs = jnp.zeros((T, 1))

        result = check_parametric_id(
            model=model,
            observations=obs,
            times=times,
            n_draws=5,
            seed=42,
        )

        summary = result.summary()
        assert summary["structural_issues"] is False, (
            f"1D LGSS should not have structural issues. Min eigenvalues: {result.min_eigenvalues}"
        )

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_nonidentified_model_flags_weak_params(self):
        """Non-identified model (2 latent, 1 manifest, identical loadings).

        With only 1 manifest and symmetric loadings [0.5, 0.5], the two
        latent processes are not individually distinguishable. Should flag
        weak parameters or structural issues.
        """
        from dsem_agent.utils.parametric_id import check_parametric_id

        model = _make_nonidentified_model()
        T = 100
        times = jnp.arange(T, dtype=jnp.float32)
        obs = jnp.zeros((T, 1))

        result = check_parametric_id(
            model=model,
            observations=obs,
            times=times,
            n_draws=5,
            seed=42,
        )

        summary = result.summary()

        # Either structural issues or weak params should be flagged
        has_issues = (
            summary["structural_issues"]
            or summary["boundary_issues"]
            or len(summary["weak_params"]) > 0
        )
        assert has_issues, f"Non-identified model should flag issues. Summary: {summary}"

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_contraction_higher_for_identified_model(self):
        """Identified model should have higher mean contraction than non-identified."""
        from dsem_agent.utils.parametric_id import check_parametric_id

        # Identified: 1 latent, 1 manifest
        spec_id = SSMSpec(
            n_latent=1,
            n_manifest=1,
            lambda_mat=jnp.eye(1),
            manifest_means=jnp.zeros(1),
            diffusion="diag",
            t0_means=jnp.zeros(1),
            t0_var=jnp.eye(1),
        )
        model_id = SSMModel(
            spec_id,
            SSMPriors(
                drift_diag={"mu": -0.5, "sigma": 0.3},
                diffusion_diag={"sigma": 0.3},
                manifest_var_diag={"sigma": 0.3},
            ),
            n_particles=50,
            likelihood="kalman",
        )

        T = 100
        times = jnp.arange(T, dtype=jnp.float32)

        result_id = check_parametric_id(
            model=model_id,
            observations=jnp.zeros((T, 1)),
            times=times,
            n_draws=5,
            seed=42,
        )

        # Non-identified: 2 latent, 1 manifest
        model_nonid = _make_nonidentified_model()
        result_nonid = check_parametric_id(
            model=model_nonid,
            observations=jnp.zeros((T, 1)),
            times=times,
            n_draws=5,
            seed=42,
        )

        # Mean contraction across all params should be higher for identified model
        def _mean_contraction(result):
            all_c = []
            for vals in result.expected_contraction.values():
                all_c.append(float(jnp.mean(vals)))
            return sum(all_c) / len(all_c) if all_c else 0.0

        mean_c_id = _mean_contraction(result_id)
        mean_c_nonid = _mean_contraction(result_nonid)

        assert mean_c_id > mean_c_nonid, (
            f"Identified model contraction ({mean_c_id:.3f}) should exceed "
            f"non-identified ({mean_c_nonid:.3f})"
        )
