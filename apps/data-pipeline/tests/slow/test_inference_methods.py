"""Recovery tests for all inference methods.

Smoke tests verify pipeline correctness (small settings, fast).
Recovery tests verify parameter recovery within 90% CIs (slow).

All tests share the lgss_data fixture from conftest.py.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
import polars as pl
import pytest

from causal_ssm_agent.models.ssm import (
    InferenceResult,
    SSMModel,
    SSMPriors,
    SSMSpec,
    discretize_system,
    fit,
    full_diagonal_mask,
    full_drift_offdiag_mask,
    zero_diagonal_mask,
    zero_loading_mask,
    zero_square_mask,
    zero_vector_mask,
)
from causal_ssm_agent.models.ssm.diagnostics import simulate_ssm
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
from tests.helpers import assert_recovery_ci

pytestmark = pytest.mark.slow

DOCTOLIB_FIXTURE_DIR = Path(__file__).resolve().parents[4] / "data" / "DOCTOLIB" / "run"


def _load_doctolib_fixture(name: str) -> dict:
    """Load the shared Doctolib mock fixture used by the web app."""
    return json.loads((DOCTOLIB_FIXTURE_DIR / name).read_text())


def _assert_core_smoke_result(
    result: InferenceResult,
    *,
    method: str,
    n_draws: int,
    extra_diag_keys: tuple[str, ...] = (),
) -> dict[str, jnp.ndarray]:
    assert isinstance(result, InferenceResult)
    assert result.method == method
    samples = result.get_samples()

    for site in ["drift_diag_free", "diffusion_diag_free", "manifest_var_diag_free"]:
        assert site in samples, f"Missing sample site: {site}"

    assert samples["drift_diag_free"].shape == (n_draws, 1)
    assert samples["diffusion_diag_free"].shape == (n_draws, 1)
    assert samples["manifest_var_diag_free"].shape == (n_draws, 1)

    for key in extra_diag_keys:
        assert key in result.diagnostics

    return samples


def _assert_lgss_recovery(samples: dict[str, jnp.ndarray], lgss_data) -> None:
    assert_recovery_ci(
        samples["drift_diag_free"][:, 0],
        lgss_data["true_drift_diag"],
        "Drift",
        transform=lambda s: -jnp.abs(s),
    )
    assert_recovery_ci(
        samples["diffusion_diag_free"][:, 0],
        lgss_data["true_diff_diag"],
        "Diffusion",
    )
    assert_recovery_ci(
        samples["manifest_var_diag_free"][:, 0],
        lgss_data["true_obs_sd"],
        "Obs SD",
    )


def _make_laplace_em_recovery_data() -> dict:
    """Build a more informative 1D LGSS for Laplace-EM recovery checks.

    The shared ``lgss_data`` fixture is adequate for CI-coverage tests but the
    optimizer status can be noisy for the canonical Laplace-EM path. This
    synthetic dataset increases the signal-to-noise ratio and sample size so
    the mode-finding and local Gaussian approximation are both meaningfully
    exercised.
    """
    n_latent, n_manifest = 1, 1
    T, dt = 250, 1.0
    true_drift_diag = -0.3
    true_diff_diag = 0.2
    true_obs_sd = 0.25

    true_drift = jnp.array([[true_drift_diag]])
    true_diff_cov = jnp.array([[true_diff_diag**2]])
    true_obs_var = jnp.array([[true_obs_sd**2]])

    Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
    Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
    R_chol = jla.cholesky(true_obs_var, lower=True)

    key = random.PRNGKey(42)
    states = [jnp.zeros(n_latent)]
    for _ in range(T - 1):
        key, nk = random.split(key)
        states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
    latent = jnp.stack(states)

    key, obs_key = random.split(key)
    observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
    times = jnp.arange(T, dtype=float) * dt

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift_diag_mask=full_diagonal_mask(n_latent),
        drift_offdiag_mask=full_drift_offdiag_mask(n_latent),
        drift=jnp.zeros((n_latent, n_latent)),
        cint_mask=zero_vector_mask(n_latent),
        cint=jnp.zeros(n_latent),
        lambda_mask=zero_loading_mask(n_manifest, n_latent),
        lambda_mat=jnp.eye(n_manifest, n_latent),
        diffusion_chol_mask=np.diag(full_diagonal_mask(n_latent)),
        diffusion_chol=jnp.eye(n_latent),
        manifest_means_mask=zero_vector_mask(n_manifest),
        manifest_means=jnp.zeros(n_manifest),
        manifest_chol_diag_mask=full_diagonal_mask(n_manifest),
        manifest_chol=jnp.zeros((n_manifest, n_manifest)),
        t0_means_mask=zero_vector_mask(n_latent),
        t0_means=jnp.zeros(n_latent),
        t0_chol_diag_mask=zero_diagonal_mask(n_latent),
        t0_correlation_mask=zero_square_mask(n_latent),
        t0_chol=jnp.eye(n_latent),
    )

    return {
        "observations": observations,
        "times": times,
        "spec": spec,
        "true_drift_diag": true_drift_diag,
        "true_diff_diag": true_diff_diag,
        "true_obs_sd": true_obs_sd,
    }


def _build_mixed_support_runtime(
    times: jnp.ndarray,
    manifest_names: list[str],
) -> ObservationSupportRuntime:
    """One point channel and one interval-mean channel per latent."""
    times_np = np.asarray(times, dtype=np.float64)
    n_time = int(times_np.shape[0])
    n_manifest = len(manifest_names)
    n_point = n_manifest // 2

    support_start = np.full((n_time, n_manifest), np.nan, dtype=np.float64)
    support_end = np.full((n_time, n_manifest), np.nan, dtype=np.float64)
    prev_coeffs = np.zeros((n_time, n_manifest, 1), dtype=np.float64)
    curr_coeffs = np.zeros((n_time, n_manifest, 1), dtype=np.float64)
    weights = np.zeros((n_time, n_manifest, 1), dtype=np.float64)
    emission_slots = np.full((n_time, n_manifest), -1, dtype=np.int64)

    interval_slice = slice(n_point, n_manifest)
    for t in range(1, n_time):
        dt = times_np[t] - times_np[t - 1]
        support_start[t, interval_slice] = times_np[t - 1]
        support_end[t, interval_slice] = times_np[t]
        prev_coeffs[t, interval_slice, 0] = 0.5 * dt
        curr_coeffs[t, interval_slice, 0] = 0.5 * dt
        weights[t, interval_slice, 0] = dt
        emission_slots[t, interval_slice] = 0

    return ObservationSupportRuntime(
        anchor_times=times_np,
        manifest_names=manifest_names,
        support_kinds=["point"] * n_point + ["interval"] * n_point,
        summary_operators=[None] * n_point + ["mean"] * n_point,
        anchor_policies=[None] * n_point + ["support_end"] * n_point,
        observation_windows=[None] * n_point + ["1d"] * n_point,
        support_start_times=support_start,
        support_end_times=support_end,
        interval_prev_coeffs=prev_coeffs,
        interval_curr_coeffs=curr_coeffs,
        interval_weights=weights,
        emission_slot_indices=emission_slots,
    )


def _build_mixed_support_observations(point_observations: jnp.ndarray) -> jnp.ndarray:
    """Keep the first half point-like and average the second half over each interval."""
    point_np = np.asarray(point_observations, dtype=np.float32)
    n_point = point_np.shape[1] // 2
    interval_means = np.full_like(point_np, np.nan)
    interval_means[1:] = 0.5 * (point_np[:-1] + point_np[1:])
    mixed = point_np.copy()
    mixed[:, n_point:] = interval_means[:, n_point:]
    return jnp.asarray(mixed)


def _make_laplace_em_mixed_support_recovery_data() -> dict:
    """Build a recoverable mixed-support 10-latent benchmark for Laplace-EM."""
    n_latent, T = 10, 40
    n_manifest = 2 * n_latent
    true_drift_diag = -jnp.linspace(0.18, 0.45, n_latent, dtype=jnp.float32)
    true_diff_diag = jnp.linspace(0.10, 0.18, n_latent, dtype=jnp.float32)
    point_obs_sd = jnp.linspace(0.08, 0.14, n_latent, dtype=jnp.float32)
    interval_obs_sd = jnp.linspace(0.08, 0.14, n_latent, dtype=jnp.float32)
    true_obs_sd = jnp.concatenate([point_obs_sd, interval_obs_sd])
    true_t0_sd = jnp.linspace(0.20, 0.32, n_latent, dtype=jnp.float32)

    times = jnp.arange(T, dtype=jnp.float32)
    lambda_mat = jnp.concatenate(
        [jnp.eye(n_latent, dtype=jnp.float32), jnp.eye(n_latent, dtype=jnp.float32)],
        axis=0,
    )
    manifest_names = [
        *(f"y{i}_point" for i in range(n_latent)),
        *(f"y{i}_interval" for i in range(n_latent)),
    ]
    point_observations = simulate_ssm(
        drift=jnp.diag(true_drift_diag),
        diffusion_chol=jnp.diag(true_diff_diag),
        lambda_mat=lambda_mat,
        manifest_chol=jnp.diag(true_obs_sd),
        t0_means=jnp.zeros(n_latent, dtype=jnp.float32),
        t0_chol=jnp.diag(true_t0_sd),
        times=times,
        rng_key=random.PRNGKey(0),
    )
    observations = _build_mixed_support_observations(point_observations)
    observation_support = _build_mixed_support_runtime(times, manifest_names)

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift_diag_mask=full_diagonal_mask(n_latent),
        drift_offdiag_mask=zero_square_mask(n_latent),
        drift=jnp.zeros((n_latent, n_latent), dtype=jnp.float32),
        cint_mask=zero_vector_mask(n_latent),
        cint=jnp.zeros(n_latent, dtype=jnp.float32),
        lambda_mask=zero_loading_mask(n_manifest, n_latent),
        lambda_mat=lambda_mat,
        diffusion_chol_mask=np.diag(full_diagonal_mask(n_latent)),
        diffusion_chol=jnp.eye(n_latent, dtype=jnp.float32),
        manifest_means_mask=zero_vector_mask(n_manifest),
        manifest_means=jnp.zeros(n_manifest, dtype=jnp.float32),
        manifest_chol_diag_mask=full_diagonal_mask(n_manifest),
        manifest_chol=jnp.zeros((n_manifest, n_manifest), dtype=jnp.float32),
        t0_means_mask=zero_vector_mask(n_latent),
        t0_means=jnp.zeros(n_latent, dtype=jnp.float32),
        t0_chol_diag_mask=zero_diagonal_mask(n_latent),
        t0_correlation_mask=zero_square_mask(n_latent),
        t0_chol=jnp.diag(true_t0_sd),
        latent_names=[f"x{i}" for i in range(n_latent)],
        manifest_names=manifest_names,
    )
    priors = SSMPriors(
        drift_diag={"mu": -0.35, "sigma": 0.15},
        diffusion_diag={"sigma": 0.15},
        manifest_var_diag={"sigma": 0.15},
    )

    return {
        "observations": observations,
        "times": times,
        "spec": spec,
        "priors": priors,
        "observation_support": observation_support,
        "true_drift_diag": true_drift_diag,
        "true_diff_diag": true_diff_diag,
        "true_obs_sd": true_obs_sd,
    }


def _summarize_family_recovery(samples: dict[str, jnp.ndarray], data: dict) -> dict[str, dict[str, float]]:
    """Summarize mean error, interval width, and empirical coverage by parameter family."""
    families = [
        ("drift", -jnp.abs(samples["drift_diag_free"]), data["true_drift_diag"]),
        ("diffusion_sd", samples["diffusion_diag_free"], data["true_diff_diag"]),
        ("obs_sd", samples["manifest_var_diag_free"], data["true_obs_sd"]),
    ]
    summary: dict[str, dict[str, float]] = {}
    for family, draws, truth in families:
        means = jnp.mean(draws, axis=0)
        q05 = jnp.quantile(draws, 0.05, axis=0)
        q95 = jnp.quantile(draws, 0.95, axis=0)
        summary[family] = {
            "coverage": float(jnp.mean((q05 <= truth) & (truth <= q95))),
            "mean_abs_error": float(jnp.mean(jnp.abs(means - truth))),
            "mean_ci_width": float(jnp.mean(q95 - q05)),
        }
    return summary


# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    (
        "method",
        "model_factory",
        "fit_kwargs",
        "n_draws",
        "extra_diag_keys",
        "expected_accept_rate_len",
    ),
    [
        pytest.param(
            "nuts_da",
            lambda spec: SSMModel(spec),
            {
                "num_warmup": 50,
                "num_samples": 50,
                "num_chains": 1,
                "seed": 0,
            },
            50,
            (),
            None,
            id="nuts_da",
        ),
        pytest.param(
            "laplace_smc",
            lambda spec: SSMModel(spec, n_particles=50),
            {
                "n_outer": 6,
                "n_csmc_particles": 8,
                "n_mh_steps": 3,
                "param_step_size": 0.1,
                "n_warmup": 3,
                "n_ieks_iters": 3,
                "adaptive_tempering": False,
                "seed": 0,
            },
            8,
            ("accept_rates", "n_ieks_iters"),
            6,
            id="laplace_smc",
        ),
        pytest.param(
            "structured_vi",
            lambda spec: SSMModel(spec, n_particles=50),
            {
                "n_outer": 6,
                "n_csmc_particles": 8,
                "n_mh_steps": 3,
                "param_step_size": 0.1,
                "n_warmup": 3,
                "adaptive_tempering": False,
                "seed": 0,
            },
            8,
            ("accept_rates",),
            6,
            id="structured_vi",
        ),
        pytest.param(
            "dpf",
            lambda spec: SSMModel(spec, n_particles=50),
            {
                "n_outer": 6,
                "n_csmc_particles": 8,
                "n_mh_steps": 3,
                "param_step_size": 0.1,
                "n_warmup": 3,
                "adaptive_tempering": False,
                "n_train_seqs": 5,
                "n_train_steps": 20,
                "n_particles_train": 8,
                "n_pf_particles": 20,
                "seed": 0,
            },
            8,
            ("accept_rates", "proposal_net"),
            6,
            id="dpf",
        ),
    ],
)
def test_core_inference_smoke_matrix(
    lgss_data,
    method,
    model_factory,
    fit_kwargs,
    n_draws,
    extra_diag_keys,
    expected_accept_rate_len,
):
    """Backend smoke tests share one fixture and one output-shape contract."""
    model = model_factory(lgss_data["spec"])

    result = fit(
        model,
        observations=lgss_data["observations"],
        times=lgss_data["times"],
        method=method,
        **fit_kwargs,
    )

    samples = _assert_core_smoke_result(
        result,
        method=method,
        n_draws=n_draws,
        extra_diag_keys=extra_diag_keys,
    )

    if method == "nuts_da":
        assert "innovations" not in samples
    if expected_accept_rate_len is not None:
        assert len(result.diagnostics["accept_rates"]) == expected_accept_rate_len


# =============================================================================
# NUTS Data Augmentation
# =============================================================================


class TestNutsDARecovery:
    """NUTS-DA recovery tests on 1D LGSS."""

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

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# Hess-MC2
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

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# Tempered SMC
# =============================================================================


class TestTemperedSMCRecovery:
    """Tempered SMC recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
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

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# Laplace-SMC
# =============================================================================


class TestLaplaceSMC:
    """Laplace-SMC recovery tests on 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_laplace_smc_recovery(self, lgss_data):
        """Laplace-SMC recovers 1D LGSS params (D=3) within 90% CIs.

        Uses Kalman likelihood backend (exact for linear Gaussian) for fast
        evaluation. The Laplace-SMC outer loop (tempered SMC over parameters)
        is the same as tempered_smc -- the method's value is for non-Gaussian
        emissions where Laplace approximation replaces the PF.
        """
        model = SSMModel(lgss_data["spec"], likelihood="kalman")

        result = fit(
            model,
            observations=lgss_data["observations"],
            times=lgss_data["times"],
            method="laplace_smc",
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

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# Laplace-EM
# =============================================================================


class TestLaplaceEM:
    """Canonical Laplace-EM recovery tests on an informative 1D LGSS."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_laplace_em_recovery(self):
        """Laplace-EM recovers a well-identified 1D LGSS under Kalman likelihood.

        This checks more than execution:
        1. BFGS converges on a genuinely informative dataset.
        2. The Gaussian parameter-space approximation contains the truth in its
           90% intervals.
        3. Posterior means stay close to the generating parameters, so the
           approximation is not passing only because the intervals are overly
           wide.
        """
        data = _make_laplace_em_recovery_data()
        model = SSMModel(data["spec"], likelihood="kalman")

        result = fit(
            model,
            observations=data["observations"],
            times=data["times"],
            method="laplace_em",
            num_samples=1000,
            n_ieks_iters=5,
            maxiter=100,
            tol=1e-5,
            n_init_samples=64,
            parameter_covariance_method="exact_hessian",
            seed=0,
        )

        assert result.diagnostics["optimizer"] == "L-BFGS-B"
        assert result.diagnostics["success"] is True
        assert result.diagnostics["status"] == 0

        samples = result.get_samples()
        _assert_lgss_recovery(samples, data)

        drift_mean = float(jnp.mean(-jnp.abs(samples["drift_diag_free"][:, 0])))
        diff_mean = float(jnp.mean(samples["diffusion_diag_free"][:, 0]))
        obs_mean = float(jnp.mean(samples["manifest_var_diag_free"][:, 0]))

        assert abs(drift_mean - data["true_drift_diag"]) < 0.12
        assert abs(diff_mean - data["true_diff_diag"]) < 0.08
        assert abs(obs_mean - data["true_obs_sd"]) < 0.05

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_laplace_em_mixed_support_particle_recovery(self):
        """Laplace-EM recovers a mixed point/interval 10-latent particle model.

        This is the benchmark that exercises the support-aware Laplace path while
        still requiring useful recovery, not just optimizer termination.
        """
        data = _make_laplace_em_mixed_support_recovery_data()
        model = SSMModel(data["spec"], data["priors"], likelihood="particle")
        model.set_observation_support(data["observation_support"])

        result = fit(
            model,
            observations=data["observations"],
            times=data["times"],
            method="laplace_em",
            num_samples=150,
            n_ieks_iters=5,
            maxiter=60,
            tol=1e-4,
            seed=0,
        )

        assert data["observation_support"].requires_interval_summary_handling is True
        assert result.diagnostics["optimizer"] == "L-BFGS-B"
        assert result.diagnostics["success"] is True
        assert result.diagnostics["status"] == 0

        summary = _summarize_family_recovery(result.get_samples(), data)

        assert summary["drift"]["coverage"] >= 0.9
        assert summary["diffusion_sd"]["coverage"] >= 0.9
        assert summary["obs_sd"]["coverage"] >= 0.8

        assert summary["drift"]["mean_abs_error"] < 0.12
        assert summary["diffusion_sd"]["mean_abs_error"] < 0.16
        assert summary["obs_sd"]["mean_abs_error"] < 0.10

        assert summary["drift"]["mean_ci_width"] < 0.75
        assert summary["diffusion_sd"]["mean_ci_width"] < 1.0
        assert summary["obs_sd"]["mean_ci_width"] < 0.5


# =============================================================================
# Structured VI
# =============================================================================


class TestStructuredVI:
    """Structured VI recovery tests on 1D LGSS."""

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

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# DPF (Differentiable Particle Filter)
# =============================================================================


class TestDPF:
    """DPF recovery tests on 1D LGSS."""

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

        _assert_lgss_recovery(samples, lgss_data)


# =============================================================================
# Laplace-EM Doctolib Fixture
# =============================================================================


def _build_executable_doctolib_fixture_v2() -> tuple[dict, dict, dict, pl.DataFrame]:
    """Normalize the shared Doctolib web fixture into an executable pipeline artifact.

    The shared web fixture predates the stricter compiler contract: it uses
    shorthand parameter names and a latent graph that is not directly executable
    under the current builder. For inference smoke tests we normalize the
    parameter names into the compiler naming convention and use the latent graph
    implied by the stage-4 fixed effects.
    """
    stage4 = _load_doctolib_fixture("stage-4.json")
    stage1b = _load_doctolib_fixture("stage-1b.json")["causal_spec"]
    data_for_model = pl.read_parquet(DOCTOLIB_FIXTURE_DIR / "stage2-raw-data.parquet")

    name_map = {
        "beta_lipid_cv": "beta_lipid_burden_cardiovascular_risk",
        "beta_pressure_cv": "beta_arterial_pressure_cardiovascular_risk",
        "beta_glycemic_cv": "beta_glycemic_control_cardiovascular_risk",
        "beta_lipid_inflammation": "beta_lipid_burden_vascular_inflammation",
        "beta_inflammation_cv": "beta_vascular_inflammation_cardiovascular_risk",
        "beta_adherence_lipid": "beta_medication_adherence_lipid_burden",
        "beta_adherence_pressure": "beta_medication_adherence_arterial_pressure",
        "rho_lipid": "rho_lipid_burden",
        "rho_pressure": "rho_arterial_pressure",
        "sigma_lipid": "sigma_lipid_burden",
        "sigma_pressure": "sigma_arterial_pressure",
        "rho_inflammation": "rho_vascular_inflammation",
    }

    model_spec = json.loads(json.dumps(stage4["model_spec"]))
    for parameter in model_spec["parameters"]:
        parameter["name"] = name_map.get(parameter["name"], parameter["name"])
        if parameter["role"] == "ar_coefficient":
            parameter["constraint"] = "unit_interval"

    priors = json.loads(json.dumps(stage4["authored_priors"]))
    for old_name, new_name in name_map.items():
        if old_name in priors:
            priors[new_name] = priors.pop(old_name)

    beta_variables = {
        likelihood["variable"]
        for likelihood in model_spec["likelihoods"]
        if likelihood["distribution"] == "beta"
    }
    executable_manifest_names = {likelihood["variable"] for likelihood in model_spec["likelihoods"]}
    if beta_variables:
        eps = 1e-3
        data_for_model = data_for_model.with_columns(
            pl.when(pl.col("indicator").is_in(sorted(beta_variables)))
            .then(
                pl.col("value")
                .cast(pl.Float64, strict=False)
                .clip(lower_bound=eps, upper_bound=1.0 - eps)
                .cast(pl.Utf8)
            )
            .otherwise(pl.col("value"))
            .alias("value")
        )

    stage4_construct_names = {
        "medication_adherence",
        "lipid_burden",
        "vascular_inflammation",
        "glycemic_control",
        "arterial_pressure",
        "cardiovascular_risk",
    }
    measurement = {
        "model_clock": "1d",
        "indicators": [
            {
                **indicator,
                "construct_polarity": (
                    "negative" if indicator["name"] == "hdl_cholesterol" else "positive"
                ),
            }
            for indicator in stage1b["measurement"]["indicators"]
            if indicator["construct_name"] in stage4_construct_names
        ],
    }
    latent_constructs = [
        {
            "name": "medication_adherence",
            "description": "Prescription refill and appointment follow-through.",
            "role": "exogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "lipid_burden",
            "description": "Atherogenic lipid profile.",
            "role": "endogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "vascular_inflammation",
            "description": "Inflammatory state relevant to cardiovascular risk.",
            "role": "endogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "glycemic_control",
            "description": "Blood-glucose regulation quality.",
            "role": "endogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "arterial_pressure",
            "description": "Blood-pressure burden.",
            "role": "endogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "cardiovascular_risk",
            "description": "Overall cardiovascular risk trajectory.",
            "role": "endogenous",
            "is_outcome": True,
            "temporal_status": "time_varying",
        },
    ]
    latent_edges = [
        {
            "cause": "medication_adherence",
            "effect": "lipid_burden",
            "description": "Medication adherence improves lipid control.",
        },
        {
            "cause": "medication_adherence",
            "effect": "arterial_pressure",
            "description": "Medication adherence improves blood-pressure control.",
        },
        {
            "cause": "lipid_burden",
            "effect": "vascular_inflammation",
            "description": "Higher lipid burden raises vascular inflammation.",
        },
        {
            "cause": "lipid_burden",
            "effect": "cardiovascular_risk",
            "description": "Higher lipid burden raises cardiovascular risk.",
        },
        {
            "cause": "vascular_inflammation",
            "effect": "cardiovascular_risk",
            "description": "Higher vascular inflammation raises cardiovascular risk.",
        },
        {
            "cause": "glycemic_control",
            "effect": "cardiovascular_risk",
            "description": "Poorer glycemic control raises cardiovascular risk.",
        },
        {
            "cause": "arterial_pressure",
            "effect": "cardiovascular_risk",
            "description": "Higher arterial pressure raises cardiovascular risk.",
        },
    ]
    retained_state_order = [
        construct["name"]
        for construct in latent_constructs
        if any(
            indicator["construct_name"] == construct["name"]
            and indicator["name"] in executable_manifest_names
            for indicator in measurement["indicators"]
        )
    ]
    excluded_effect_suffixes = tuple(
        f"_{construct_name}"
        for construct_name in stage4_construct_names
        if construct_name not in retained_state_order
    )
    if excluded_effect_suffixes:
        model_spec["parameters"] = [
            parameter
            for parameter in model_spec["parameters"]
            if not (
                parameter["role"] == "fixed_effect"
                and any(parameter["name"].endswith(suffix) for suffix in excluded_effect_suffixes)
            )
        ]
        priors = {
            name: payload
            for name, payload in priors.items()
            if not (
                name.startswith("beta_")
                and any(name.endswith(suffix) for suffix in excluded_effect_suffixes)
            )
        }
    retained_parameter_names = {parameter["name"] for parameter in model_spec["parameters"]}
    priors = {name: payload for name, payload in priors.items() if name in retained_parameter_names}
    causal_spec = {
        "latent": {
            "constructs": latent_constructs,
            "edges": latent_edges,
        },
        "estimation": {
            "state_order": retained_state_order,
            "edges": [
                edge
                for edge in latent_edges
                if edge["cause"] in retained_state_order and edge["effect"] in retained_state_order
            ],
            "induced_dependencies": [],
        },
        "measurement": measurement,
    }
    return causal_spec, model_spec, priors, data_for_model


class TestLaplaceSMCDoctolib:
    """Fixture-backed Laplace-SMC smoke tests on the Doctolib mock data."""

    @pytest.mark.slow
    @pytest.mark.timeout(180)
    def test_laplace_smc_doctolib_fixture_smoke(self):
        """Laplace-SMC fits the executable Doctolib fixture end-to-end."""
        from causal_ssm_agent.distributions import DistributionFamily
        from causal_ssm_agent.models.ssm.inference import select_default_method
        from causal_ssm_agent.models.ssm_builder import build_ssm_builder
        from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact
        from causal_ssm_agent.utils.data import pivot_to_wide

        causal_spec, model_spec, priors, data_for_model = _build_executable_doctolib_fixture_v2()

        assert data_for_model.schema["anchor_time"] == pl.String

        compiled = compile_ssm_artifact(
            model_spec,
            priors,
            causal_spec=causal_spec,
        )
        builder = build_ssm_builder(
            wide_data=pivot_to_wide(data_for_model),
            compiled_ssm=compiled,
            sampler_config={
                "method": "laplace_smc",
                "n_outer": 6,
                "n_csmc_particles": 8,
                "n_mh_steps": 3,
                "param_step_size": 0.05,
                "n_warmup": 3,
                "n_ieks_iters": 3,
                "adaptive_tempering": False,
                "seed": 0,
            },
        )

        spec = builder.spec
        assert spec.manifest_dists is not None
        assert DistributionFamily.BETA in spec.manifest_dists
        assert select_default_method(spec) == "laplace_em"

        wide = pivot_to_wide(data_for_model)
        assert wide.schema["time"] == pl.Float64

        result = builder.fit(wide)

        assert isinstance(result, InferenceResult)
        assert result.method == "laplace_smc"

        samples = result.get_samples()
        assert "drift_diag_free" in samples
        assert "diffusion_diag_free" in samples
        assert "manifest_var_diag_free" in samples
        assert samples["drift_diag_free"].shape == (8, spec.n_latent)
        assert samples["diffusion_diag_free"].shape == (8, spec.n_latent)
        assert samples["manifest_var_diag_free"].shape == (8, spec.n_manifest)
        assert bool(jnp.isfinite(samples["drift_diag_free"]).all())
        assert bool(jnp.isfinite(samples["diffusion_diag_free"]).all())
        assert bool(jnp.isfinite(samples["manifest_var_diag_free"]).all())

        assert "accept_rates" in result.diagnostics
        assert "n_ieks_iters" in result.diagnostics
        assert result.diagnostics["n_ieks_iters"] == 3
        assert len(result.diagnostics["accept_rates"]) == 6
