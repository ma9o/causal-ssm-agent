"""Recovery tests for Laplace-EM, Structured VI, and DPF.

Smoke tests verify pipeline correctness (small settings, fast).
Recovery tests verify parameter recovery within 90% CIs (slow).

All tests share the lgss_data fixture from conftest.py.
"""

import jax.numpy as jnp
import pytest

from causal_ssm_agent.models.ssm import InferenceResult, SSMModel, fit
from tests.helpers import assert_recovery_ci

# =============================================================================
# Laplace-EM
# =============================================================================


class TestLaplaceEM:
    """Laplace-EM smoke and recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_laplace_em_smoke(self, lgss_data):
        """Laplace-EM pipeline check on 1D LGSS (D=3).

        Verifies: instantiation, inference completes, correct output structure.
        """
        import time

        t0 = time.perf_counter()

        model = SSMModel(lgss_data["spec"], n_particles=50)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="laplace_em",
            n_outer=6,
            n_csmc_particles=8,
            n_mh_steps=3,
            param_step_size=0.1,
            n_warmup=3,
            n_ieks_iters=3,
            adaptive_tempering=False,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "laplace_em"
        samples = result.get_samples()

        for site in ["drift_diag_pop", "diffusion_diag_pop", "manifest_var_diag"]:
            assert site in samples, f"Missing sample site: {site}"

        # Post-warmup samples: n_outer - n_warmup = 3
        assert samples["drift_diag_pop"].shape == (3, 1)
        assert samples["diffusion_diag_pop"].shape == (3, 1)
        assert samples["manifest_var_diag"].shape == (3, 1)

        # Diagnostics present
        assert "accept_rates" in result.diagnostics
        assert "n_ieks_iters" in result.diagnostics
        assert len(result.diagnostics["accept_rates"]) == 6

        elapsed = time.perf_counter() - t0
        assert elapsed < 120.0, f"Laplace-EM smoke took {elapsed:.1f}s, must be under 120s"

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_laplace_em_recovery(self, lgss_data):
        """Laplace-EM recovers 1D LGSS params (D=3) within 90% CIs.

        Uses Kalman likelihood backend (exact for linear Gaussian) for fast
        evaluation. The Laplace-EM outer loop (tempered SMC over parameters)
        is the same as tempered_smc — the method's value is for non-Gaussian
        emissions where Laplace approximation replaces the PF.
        """
        model = SSMModel(lgss_data["spec"], likelihood="kalman")

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="laplace_em",
            n_outer=100,
            n_csmc_particles=20,
            n_mh_steps=10,
            param_step_size=0.1,
            n_warmup=50,
            n_ieks_iters=5,
            adaptive_tempering=False,
            seed=0,
        )

        samples = result.get_samples()

        assert_recovery_ci(
            samples["drift_diag_pop"][:, 0],
            lgss_data["true_drift_diag"],
            "Drift",
            transform=lambda s: -jnp.abs(s),
        )
        assert_recovery_ci(
            samples["diffusion_diag_pop"][:, 0],
            lgss_data["true_diff_diag"],
            "Diffusion",
        )
        assert_recovery_ci(
            samples["manifest_var_diag"][:, 0],
            lgss_data["true_obs_sd"],
            "Obs SD",
        )


# =============================================================================
# Structured VI
# =============================================================================


class TestStructuredVI:
    """Structured VI smoke and recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_structured_vi_smoke(self, lgss_data):
        """Structured VI pipeline check on 1D LGSS (D=3).

        Verifies: instantiation, inference completes, correct output structure.
        """
        import time

        t0 = time.perf_counter()

        model = SSMModel(lgss_data["spec"], n_particles=50)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="structured_vi",
            n_outer=6,
            n_csmc_particles=8,
            n_mh_steps=3,
            param_step_size=0.1,
            n_warmup=3,
            adaptive_tempering=False,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "structured_vi"
        samples = result.get_samples()

        for site in ["drift_diag_pop", "diffusion_diag_pop", "manifest_var_diag"]:
            assert site in samples, f"Missing sample site: {site}"

        # Post-warmup samples: n_outer - n_warmup = 3
        assert samples["drift_diag_pop"].shape == (3, 1)
        assert samples["diffusion_diag_pop"].shape == (3, 1)
        assert samples["manifest_var_diag"].shape == (3, 1)

        # Diagnostics present
        assert "accept_rates" in result.diagnostics
        assert len(result.diagnostics["accept_rates"]) == 6

        elapsed = time.perf_counter() - t0
        assert elapsed < 120.0, f"Structured VI smoke took {elapsed:.1f}s, must be under 120s"

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_structured_vi_recovery(self, lgss_data):
        """Structured VI recovers 1D LGSS params (D=3) within 90% CIs."""
        model = SSMModel(lgss_data["spec"], likelihood="kalman")

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="structured_vi",
            n_outer=100,
            n_csmc_particles=20,
            n_mh_steps=10,
            param_step_size=0.1,
            n_warmup=50,
            adaptive_tempering=False,
            seed=0,
        )

        samples = result.get_samples()

        assert_recovery_ci(
            samples["drift_diag_pop"][:, 0],
            lgss_data["true_drift_diag"],
            "Drift",
            transform=lambda s: -jnp.abs(s),
        )
        assert_recovery_ci(
            samples["diffusion_diag_pop"][:, 0],
            lgss_data["true_diff_diag"],
            "Diffusion",
        )
        assert_recovery_ci(
            samples["manifest_var_diag"][:, 0],
            lgss_data["true_obs_sd"],
            "Obs SD",
        )


# =============================================================================
# DPF (Differentiable Particle Filter)
# =============================================================================


class TestDPF:
    """DPF smoke and recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(180)
    def test_dpf_smoke(self, lgss_data):
        """DPF pipeline check on 1D LGSS (D=3).

        Verifies: proposal training, inference pipeline, correct output structure.
        Longer timeout due to proposal training phase.
        """
        import time

        t0 = time.perf_counter()

        model = SSMModel(lgss_data["spec"], n_particles=50)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="dpf",
            n_outer=6,
            n_csmc_particles=8,
            n_mh_steps=3,
            param_step_size=0.1,
            n_warmup=3,
            adaptive_tempering=False,
            # DPF-specific: small training for smoke test
            n_train_seqs=5,
            n_train_steps=20,
            n_particles_train=8,
            n_pf_particles=20,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "dpf"
        samples = result.get_samples()

        for site in ["drift_diag_pop", "diffusion_diag_pop", "manifest_var_diag"]:
            assert site in samples, f"Missing sample site: {site}"

        # Post-warmup samples: n_outer - n_warmup = 3
        assert samples["drift_diag_pop"].shape == (3, 1)
        assert samples["diffusion_diag_pop"].shape == (3, 1)
        assert samples["manifest_var_diag"].shape == (3, 1)

        # Diagnostics present
        assert "accept_rates" in result.diagnostics
        assert "proposal_net" in result.diagnostics
        assert len(result.diagnostics["accept_rates"]) == 6

        elapsed = time.perf_counter() - t0
        assert elapsed < 180.0, f"DPF smoke took {elapsed:.1f}s, must be under 180s"

    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_dpf_recovery(self, lgss_data):
        """DPF recovers 1D LGSS params (D=3) within 90% CIs.

        Uses Kalman likelihood for fast evaluation in the outer loop.
        The DPF's value is the trained proposal for non-Gaussian models.
        """
        model = SSMModel(lgss_data["spec"], likelihood="kalman")

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="dpf",
            n_outer=100,
            n_csmc_particles=20,
            n_mh_steps=10,
            param_step_size=0.1,
            n_warmup=50,
            adaptive_tempering=False,
            n_train_seqs=5,
            n_train_steps=20,
            n_particles_train=8,
            n_pf_particles=20,
            seed=0,
        )

        samples = result.get_samples()

        assert_recovery_ci(
            samples["drift_diag_pop"][:, 0],
            lgss_data["true_drift_diag"],
            "Drift",
            transform=lambda s: -jnp.abs(s),
        )
        assert_recovery_ci(
            samples["diffusion_diag_pop"][:, 0],
            lgss_data["true_diff_diag"],
            "Diffusion",
        )
        assert_recovery_ci(
            samples["manifest_var_diag"][:, 0],
            lgss_data["true_obs_sd"],
            "Obs SD",
        )
