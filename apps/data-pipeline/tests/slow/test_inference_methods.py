"""Recovery tests for all inference methods.

Smoke tests verify pipeline correctness (small settings, fast).
Recovery tests verify parameter recovery within 90% CIs (slow).

All tests share the lgss_data fixture from conftest.py.
"""

import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
import pytest

from causal_ssm_agent.artifacts.model_spec import DistributionFamily
from causal_ssm_agent.models.ssm import (
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
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
from tests.helpers import assert_recovery_ci

pytestmark = pytest.mark.slow


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
    """One point channel and one interval-mean channel per latent.

    Interval channels use monthly windows, except the final channel which emits
    non-overlapping yearly means.
    """
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

    monthly_interval_slice = slice(n_point, n_manifest - 1)
    yearly_interval_idx = n_manifest - 1
    for t in range(1, n_time):
        dt = times_np[t] - times_np[t - 1]
        support_start[t, monthly_interval_slice] = times_np[t - 1]
        support_end[t, monthly_interval_slice] = times_np[t]
        prev_coeffs[t, monthly_interval_slice, 0] = 0.5 * dt
        curr_coeffs[t, monthly_interval_slice, 0] = 0.5 * dt
        weights[t, monthly_interval_slice, 0] = dt
        emission_slots[t, monthly_interval_slice] = 0

    yearly_window = 12
    for t in range(yearly_window, n_time, yearly_window):
        window_start = t - yearly_window
        support_start[t, yearly_interval_idx] = times_np[window_start]
        support_end[t, yearly_interval_idx] = times_np[t]
        emission_slots[t, yearly_interval_idx] = 0
        for step_idx in range(window_start + 1, t + 1):
            dt = times_np[step_idx] - times_np[step_idx - 1]
            prev_coeffs[step_idx, yearly_interval_idx, 0] = 0.5 * dt
            curr_coeffs[step_idx, yearly_interval_idx, 0] = 0.5 * dt
            weights[step_idx, yearly_interval_idx, 0] = dt

    interval_windows = ["1mo"] * (n_point - 1) + ["1y"]

    return ObservationSupportRuntime(
        anchor_times=times_np,
        manifest_names=manifest_names,
        support_kinds=["point"] * n_point + ["interval"] * n_point,
        summary_operators=[None] * n_point + ["mean"] * n_point,
        anchor_policies=[None] * n_point + ["support_end"] * n_point,
        observation_windows=[None] * n_point + interval_windows,
        support_start_times=support_start,
        support_end_times=support_end,
        interval_prev_coeffs=prev_coeffs,
        interval_curr_coeffs=curr_coeffs,
        interval_weights=weights,
        emission_slot_indices=emission_slots,
    )


def _build_mixed_support_observations(point_observations: jnp.ndarray) -> jnp.ndarray:
    """Keep point channels local and aggregate interval channels over their windows."""
    point_np = np.asarray(point_observations, dtype=np.float32)
    n_point = point_np.shape[1] // 2
    mixed = point_np.copy()
    mixed[:, n_point:] = np.nan

    monthly_interval_slice = slice(n_point, point_np.shape[1] - 1)
    mixed[1:, monthly_interval_slice] = 0.5 * (
        point_np[:-1, monthly_interval_slice] + point_np[1:, monthly_interval_slice]
    )

    yearly_interval_idx = point_np.shape[1] - 1
    yearly_window = 12
    for t in range(yearly_window, point_np.shape[0], yearly_window):
        window = point_np[t - yearly_window : t + 1, yearly_interval_idx]
        mixed[t, yearly_interval_idx] = (
            0.5 * window[0] + np.sum(window[1:-1]) + 0.5 * window[-1]
        ) / yearly_window

    return jnp.asarray(mixed)


def _sample_student_t_noise(
    rng_key: jnp.ndarray,
    *,
    df: float,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jnp.ndarray:
    normal_key, gamma_key = random.split(rng_key)
    z = random.normal(normal_key, shape, dtype=dtype)
    chi2 = 2.0 * random.gamma(gamma_key, df / 2.0, shape=shape, dtype=dtype)
    return z * jnp.sqrt(jnp.asarray(df, dtype=dtype) / chi2)


def _simulate_mixed_continuous_observations(
    *,
    drift_diag: jnp.ndarray,
    diffusion_diag: jnp.ndarray,
    lambda_mat: jnp.ndarray,
    manifest_scales: jnp.ndarray,
    manifest_dists: list[DistributionFamily],
    t0_sd: jnp.ndarray,
    times: jnp.ndarray,
    rng_key: jnp.ndarray,
    obs_df: float,
) -> jnp.ndarray:
    """Simulate continuous observations with mixed Gaussian and Student-t noise."""
    n_latent = int(drift_diag.shape[0])
    n_manifest = int(lambda_mat.shape[0])
    dt = float(times[1] - times[0]) if times.shape[0] > 1 else 1.0

    Ad, Qd, _ = discretize_system(
        jnp.diag(drift_diag),
        jnp.diag(diffusion_diag**2),
        None,
        dt,
    )
    qd_chol = jla.cholesky(Qd + jnp.eye(n_latent, dtype=times.dtype) * 1e-8, lower=True)

    rng_key, init_key = random.split(rng_key)
    states = [t0_sd * random.normal(init_key, (n_latent,), dtype=times.dtype)]
    for _ in range(times.shape[0] - 1):
        rng_key, state_key = random.split(rng_key)
        states.append(
            states[-1] @ Ad.T + qd_chol @ random.normal(state_key, (n_latent,), dtype=times.dtype)
        )
    latent = jnp.stack(states)
    means = latent @ lambda_mat.T

    student_mask = jnp.asarray(
        [dist == DistributionFamily.STUDENT_T for dist in manifest_dists],
        dtype=bool,
    )
    obs_keys = random.split(rng_key, times.shape[0])
    draws: list[jnp.ndarray] = []
    for obs_key, mean in zip(obs_keys, means, strict=False):
        gaussian_key, student_key = random.split(obs_key)
        gaussian_noise = random.normal(gaussian_key, (n_manifest,), dtype=times.dtype)
        student_noise = _sample_student_t_noise(
            student_key,
            df=obs_df,
            shape=(n_manifest,),
            dtype=times.dtype,
        )
        base_noise = jnp.where(student_mask, student_noise, gaussian_noise)
        draws.append(mean + manifest_scales * base_noise)
    return jnp.stack(draws)


def _make_laplace_em_mixed_support_recovery_data() -> dict:
    """Build a recoverable mixed-support mixed-family 10-latent benchmark."""
    n_latent, T = 10, 40
    n_manifest = 2 * n_latent
    true_drift_diag = -jnp.linspace(0.18, 0.45, n_latent, dtype=jnp.float32)
    true_diff_diag = jnp.linspace(0.10, 0.18, n_latent, dtype=jnp.float32)
    point_obs_scale = jnp.linspace(0.08, 0.14, n_latent, dtype=jnp.float32)
    interval_obs_scale = jnp.linspace(0.08, 0.14, n_latent, dtype=jnp.float32)
    true_obs_scale = jnp.concatenate([point_obs_scale, interval_obs_scale])
    true_obs_df = 3.0
    true_t0_sd = jnp.linspace(0.20, 0.32, n_latent, dtype=jnp.float32)

    times = jnp.arange(T, dtype=jnp.float32)
    lambda_mat = jnp.concatenate(
        [jnp.eye(n_latent, dtype=jnp.float32), jnp.eye(n_latent, dtype=jnp.float32)],
        axis=0,
    )
    manifest_dists = [DistributionFamily.STUDENT_T] * n_latent + [
        DistributionFamily.GAUSSIAN
    ] * n_latent
    manifest_names = [
        *(f"y{i}_point" for i in range(n_latent)),
        *(f"y{i}_interval" for i in range(n_latent)),
    ]
    point_observations = _simulate_mixed_continuous_observations(
        drift_diag=true_drift_diag,
        diffusion_diag=true_diff_diag,
        lambda_mat=lambda_mat,
        manifest_scales=true_obs_scale,
        manifest_dists=manifest_dists,
        t0_sd=true_t0_sd,
        times=times,
        rng_key=random.PRNGKey(0),
        obs_df=true_obs_df,
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
        manifest_dists=manifest_dists,
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
        "true_obs_scale": true_obs_scale,
        "true_obs_df": true_obs_df,
    }


def _summarize_family_recovery(
    samples: dict[str, jnp.ndarray], data: dict
) -> dict[str, dict[str, float]]:
    """Summarize mean error, interval width, and empirical coverage by parameter family."""
    families = [
        ("drift", -jnp.abs(samples["drift_diag_free"]), data["true_drift_diag"]),
        ("diffusion_sd", samples["diffusion_diag_free"], data["true_diff_diag"]),
        ("obs_scale", samples["manifest_var_diag_free"], data["true_obs_scale"]),
        ("obs_df", samples["obs_df"], data["true_obs_df"]),
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
            method="map",
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
            method="map",
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
        assert summary["obs_scale"]["coverage"] >= 0.8
        assert summary["obs_df"]["coverage"] == 1.0

        assert summary["drift"]["mean_abs_error"] < 0.12
        assert summary["diffusion_sd"]["mean_abs_error"] < 0.16
        assert summary["obs_scale"]["mean_abs_error"] < 0.10
        assert summary["obs_df"]["mean_abs_error"] < 3.5

        assert summary["drift"]["mean_ci_width"] < 0.75
        assert summary["diffusion_sd"]["mean_ci_width"] < 1.0
        assert summary["obs_scale"]["mean_ci_width"] < 0.5
        assert summary["obs_df"]["mean_ci_width"] < 20.0


class TestNUTS:
    """Recovery tests for NUTS with IEKS/Laplace marginalization."""

    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_nuts_mixed_support_particle_recovery(self):
        """NUTS + IEKS runs on the mixed-support 10-latent benchmark.

        The sampler budget is intentionally cheap, so this focuses on posterior
        mean recovery and sampler health rather than demanding tight empirical
        interval coverage for every family.
        """
        data = _make_laplace_em_mixed_support_recovery_data()
        model = SSMModel(data["spec"], data["priors"], likelihood="particle")
        model.set_observation_support(data["observation_support"])

        result = fit(
            model,
            observations=data["observations"],
            times=data["times"],
            method="nuts",
            num_warmup=25,
            num_samples=25,
            num_chains=4,
            seed=0,
            dense_mass=False,
            target_accept_prob=0.9,
            max_tree_depth=6,
            n_ieks_iters=6,
            pathfinder_num_elbo_samples=20,
            pathfinder_maxiter=20,
            progress_bar=False,
        )

        summary = _summarize_family_recovery(result.get_samples(), data)
        extra = result.diagnostics["mcmc"].get_extra_fields()

        assert result.diagnostics["init_method"] == "pathfinder"
        assert data["observation_support"].requires_interval_summary_handling is True
        assert int(jnp.sum(extra["diverging"])) <= 3
        assert float(jnp.mean(extra["accept_prob"])) >= 0.65

        assert summary["drift"]["coverage"] >= 0.8
        assert summary["diffusion_sd"]["coverage"] >= 0.8

        assert summary["drift"]["mean_abs_error"] < 0.10
        assert summary["diffusion_sd"]["mean_abs_error"] < 0.03
        assert summary["obs_scale"]["mean_abs_error"] < 0.07
        assert summary["obs_df"]["mean_abs_error"] < 2.0

        assert summary["drift"]["mean_ci_width"] < 0.4
        assert summary["diffusion_sd"]["mean_ci_width"] < 0.1
        assert summary["obs_scale"]["mean_ci_width"] < 0.1
        assert summary["obs_df"]["mean_ci_width"] < 4.0
