"""Recovery tests for HessMC2, PGAS, and Tempered SMC.

Smoke tests verify pipeline correctness (small settings, fast).
Recovery tests verify parameter recovery within 90% CIs (slow).

All tests share the lgss_data fixture from conftest.py.
"""

import jax.numpy as jnp
import pytest

from dsem_agent.models.ssm import InferenceResult, SSMModel, fit
from tests.helpers import assert_recovery_ci

# =============================================================================
# NUTS Data Augmentation Recovery
# =============================================================================


class TestNutsDARecovery:
    """NUTS-DA smoke and recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_nuts_da_smoke(self, lgss_data):
        """NUTS-DA pipeline check on 1D LGSS (D=3).

        Verifies: instantiation, inference completes, correct output structure.
        """
        model = SSMModel(lgss_data["spec"])

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="nuts_da",
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "nuts_da"
        samples = result.get_samples()

        for site in ["drift_diag_pop", "diffusion_diag_pop", "manifest_var_diag"]:
            assert site in samples, f"Missing sample site: {site}"

        # innovations should be excluded from returned samples
        assert "innovations" not in samples

        # Should have 50 posterior samples
        assert samples["drift_diag_pop"].shape == (50, 1)
        assert samples["diffusion_diag_pop"].shape == (50, 1)

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_nuts_da_recovery(self, lgss_data):
        """NUTS-DA recovers 1D LGSS params (D=3) within 90% CIs."""
        model = SSMModel(lgss_data["spec"])

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="nuts_da",
            num_warmup=500,
            num_samples=500,
            num_chains=1,
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
# Hess-MC2 Recovery
# =============================================================================


class TestHessMC2Recovery:
    """Hess-MC2 Hessian proposal recovery on 1D LGSS (D=3)."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_lgss_hessian_recovery(self, lgss_data):
        """Hess-MC2 Hessian proposal recovers 1D LGSS params (D=3).

        Paper reference: Section IV-A. With D=3 and proper settings,
        the SO proposal should recover parameters within 90% CIs.

        Uses tempered warmup (warmup_iters=10) to avoid initial particle
        collapse with diffuse HalfNormal priors. N=256 for reliable
        posterior approximation.
        """
        model = SSMModel(lgss_data["spec"], n_particles=200)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="hessmc2",
            n_smc_particles=256,
            n_iterations=20,
            proposal="hessian",
            step_size=0.5,
            warmup_iters=10,
            warmup_step_size=0.5,
            adapt_step_size=False,
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
# PGAS Recovery
# =============================================================================


class TestPGASRecovery:
    """PGAS smoke and recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_pgas_smoke(self, lgss_data):
        """PGAS pipeline check on 1D LGSS (D=3).

        Verifies: instantiation, inference completes, correct output structure.
        Performance gate: must complete within 60s.
        """
        import time

        t0 = time.perf_counter()

        model = SSMModel(lgss_data["spec"], n_particles=50)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="pgas",
            n_outer=6,
            n_csmc_particles=8,
            n_mh_steps=3,
            param_step_size=0.1,
            n_warmup=3,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "pgas"
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
        assert elapsed < 60.0, f"PGAS smoke took {elapsed:.1f}s, must be under 60s"

    @pytest.mark.slow
    @pytest.mark.timeout(180)
    def test_pgas_recovery(self, lgss_data):
        """PGAS recovers 1D LGSS params (D=3) within 90% CIs."""
        model = SSMModel(lgss_data["spec"], n_particles=50)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="pgas",
            n_outer=150,
            n_csmc_particles=30,
            n_mh_steps=10,
            langevin_step_size=0.0,
            param_step_size=0.1,
            n_warmup=75,
            block_sampling=False,
            n_leapfrog=1,
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
# Tempered SMC Recovery
# =============================================================================


class TestTemperedSMCRecovery:
    """Tempered SMC smoke and recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_tempered_smc_smoke(self, lgss_data):
        """Tempered SMC pipeline check on 1D LGSS (D=3).

        Verifies: instantiation, inference completes, correct output structure.
        Performance gate: must complete within 60s.
        """
        import time

        t0 = time.perf_counter()

        model = SSMModel(lgss_data["spec"], n_particles=50)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="tempered_smc",
            n_outer=6,
            n_csmc_particles=8,
            n_mh_steps=3,
            param_step_size=0.01,
            n_warmup=3,
            adaptive_tempering=False,
            waste_free=False,
            seed=0,
        )

        assert isinstance(result, InferenceResult)
        assert result.method == "tempered_smc"
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
        assert elapsed < 60.0, f"Tempered SMC smoke took {elapsed:.1f}s, must be under 60s"

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    @pytest.mark.xfail(reason="MCMC convergence sensitive to seed; needs tuning")
    def test_tempered_smc_recovery(self, lgss_data):
        """Tempered SMC recovers 1D LGSS params (D=3) within 90% CIs."""
        model = SSMModel(lgss_data["spec"], n_particles=50)

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="tempered_smc",
            n_outer=100,
            n_csmc_particles=20,
            n_mh_steps=10,
            param_step_size=0.1,
            n_warmup=50,
            adaptive_tempering=False,
            waste_free=False,
            n_leapfrog=1,
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
