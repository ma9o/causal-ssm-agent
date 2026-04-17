"""Run the 10-latent mixed-support NUTS recovery benchmark.

Local:
    uv run python scripts/run_nuts_mixed_support_recovery.py

Modal:
    uv run modal run scripts/run_nuts_mixed_support_recovery.py --gpu A100
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np

from causal_ssm_agent.artifacts.model_spec import DistributionFamily
from causal_ssm_agent.models.ssm import (
    SSMModel,
    SSMPriors,
    SSMSpec,
    discretize_system,
    fit,
    full_diagonal_mask,
    zero_diagonal_mask,
    zero_loading_mask,
    zero_square_mask,
    zero_vector_mask,
)
from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime

DEFAULT_CONFIG: dict[str, Any] = {
    "n_time": 40,
    "num_warmup": 25,
    "num_samples": 25,
    "num_chains": 4,
    "seed": 0,
    "dense_mass": False,
    "target_accept_prob": 0.9,
    "max_tree_depth": 6,
    "n_ieks_iters": 6,
    "pathfinder_num_elbo_samples": 20,
    "pathfinder_maxiter": 20,
    "progress_bar": True,
}

THRESHOLDS: dict[str, float] = {
    "max_diverging": 3,
    "min_accept_prob_mean": 0.65,
    "min_drift_coverage": 0.8,
    "min_diffusion_sd_coverage": 0.8,
    "max_drift_mean_abs_error": 0.10,
    "max_diffusion_sd_mean_abs_error": 0.03,
    "max_obs_scale_mean_abs_error": 0.07,
    "max_obs_df_mean_abs_error": 2.0,
    "max_drift_mean_ci_width": 0.4,
    "max_diffusion_sd_mean_ci_width": 0.1,
    "max_obs_scale_mean_ci_width": 0.1,
    "max_obs_df_mean_ci_width": 4.0,
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


def _make_mixed_support_recovery_data(*, n_time: int) -> dict[str, Any]:
    """Build the mixed-support mixed-family 10-latent recovery benchmark."""
    n_latent = 10
    n_manifest = 2 * n_latent
    true_drift_diag = -jnp.linspace(0.18, 0.45, n_latent, dtype=jnp.float32)
    true_diff_diag = jnp.linspace(0.10, 0.18, n_latent, dtype=jnp.float32)
    point_obs_scale = jnp.linspace(0.08, 0.14, n_latent, dtype=jnp.float32)
    interval_obs_scale = jnp.linspace(0.08, 0.14, n_latent, dtype=jnp.float32)
    true_obs_scale = jnp.concatenate([point_obs_scale, interval_obs_scale])
    true_obs_df = 3.0
    true_t0_sd = jnp.linspace(0.20, 0.32, n_latent, dtype=jnp.float32)

    times = jnp.arange(n_time, dtype=jnp.float32)
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
    samples: dict[str, jnp.ndarray],
    data: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Summarize mean error, interval width, and empirical coverage by family."""
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


def _evaluate_thresholds(
    summary: dict[str, dict[str, float]],
    diagnostics: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if diagnostics["diverging"] > THRESHOLDS["max_diverging"]:
        failures.append(
            f"diverging={diagnostics['diverging']} > {int(THRESHOLDS['max_diverging'])}"
        )
    if diagnostics["accept_prob_mean"] < THRESHOLDS["min_accept_prob_mean"]:
        failures.append(
            f"accept_prob_mean={diagnostics['accept_prob_mean']:.4f} < "
            f"{THRESHOLDS['min_accept_prob_mean']:.2f}"
        )
    if summary["drift"]["coverage"] < THRESHOLDS["min_drift_coverage"]:
        failures.append(
            f"drift.coverage={summary['drift']['coverage']:.4f} < "
            f"{THRESHOLDS['min_drift_coverage']:.2f}"
        )
    if summary["diffusion_sd"]["coverage"] < THRESHOLDS["min_diffusion_sd_coverage"]:
        failures.append(
            f"diffusion_sd.coverage={summary['diffusion_sd']['coverage']:.4f} < "
            f"{THRESHOLDS['min_diffusion_sd_coverage']:.2f}"
        )

    max_checks = {
        ("drift", "mean_abs_error"): THRESHOLDS["max_drift_mean_abs_error"],
        ("diffusion_sd", "mean_abs_error"): THRESHOLDS["max_diffusion_sd_mean_abs_error"],
        ("obs_scale", "mean_abs_error"): THRESHOLDS["max_obs_scale_mean_abs_error"],
        ("obs_df", "mean_abs_error"): THRESHOLDS["max_obs_df_mean_abs_error"],
        ("drift", "mean_ci_width"): THRESHOLDS["max_drift_mean_ci_width"],
        ("diffusion_sd", "mean_ci_width"): THRESHOLDS["max_diffusion_sd_mean_ci_width"],
        ("obs_scale", "mean_ci_width"): THRESHOLDS["max_obs_scale_mean_ci_width"],
        ("obs_df", "mean_ci_width"): THRESHOLDS["max_obs_df_mean_ci_width"],
    }
    for (family, metric), limit in max_checks.items():
        value = summary[family][metric]
        if value > limit:
            failures.append(f"{family}.{metric}={value:.4f} > {limit:.4f}")
    return failures


def run_benchmark(config: dict[str, Any], *, check: bool) -> dict[str, Any]:
    """Run the benchmark and return a JSON-serializable summary."""
    data = _make_mixed_support_recovery_data(n_time=config["n_time"])
    model = SSMModel(data["spec"], data["priors"], likelihood="particle")
    model.set_observation_support(data["observation_support"])

    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        while not heartbeat_stop.wait(30.0):
            print(
                f"[benchmark] fit still running ({time.perf_counter() - t0:.1f}s elapsed)",
                file=sys.stderr,
                flush=True,
            )

    t0 = time.perf_counter()
    heartbeat_thread: threading.Thread | None = None
    if config["progress_bar"]:
        print(
            f"[benchmark] starting fit with config={json.dumps(config, sort_keys=True)}",
            file=sys.stderr,
            flush=True,
        )
        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()
    try:
        result = fit(
            model,
            observations=data["observations"],
            times=data["times"],
            method="nuts",
            num_warmup=config["num_warmup"],
            num_samples=config["num_samples"],
            num_chains=config["num_chains"],
            seed=config["seed"],
            dense_mass=config["dense_mass"],
            target_accept_prob=config["target_accept_prob"],
            max_tree_depth=config["max_tree_depth"],
            n_ieks_iters=config["n_ieks_iters"],
            pathfinder_num_elbo_samples=config["pathfinder_num_elbo_samples"],
            pathfinder_maxiter=config["pathfinder_maxiter"],
            progress_bar=config["progress_bar"],
        )
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
    elapsed = time.perf_counter() - t0
    if config["progress_bar"]:
        print(
            f"[benchmark] fit finished in {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    summary = _summarize_family_recovery(result.get_samples(), data)
    extra = result.diagnostics["mcmc"].get_extra_fields()
    diagnostics = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "elapsed_seconds": elapsed,
        "init_method": result.diagnostics.get("init_method"),
        "pathfinder_elbo": float(result.diagnostics.get("pathfinder_elbo", float("nan"))),
        "diverging": int(jnp.sum(extra["diverging"])),
        "accept_prob_mean": float(jnp.mean(extra["accept_prob"])),
        "num_steps_mean": float(jnp.mean(extra["num_steps"])),
        "energy_mean": float(jnp.mean(extra["energy"])),
        "observation_support_interval_handling": bool(
            data["observation_support"].requires_interval_summary_handling
        ),
    }
    fit_timings = result.diagnostics.get("timings")
    if fit_timings is not None:
        diagnostics["fit_timings"] = {
            name: float(value) for name, value in fit_timings.items()
        }
    failures = _evaluate_thresholds(summary, diagnostics)

    payload = {
        "config": dict(config),
        "summary": summary,
        "diagnostics": diagnostics,
        "passed": not failures,
        "failures": failures,
    }
    if check and failures:
        raise AssertionError("Benchmark thresholds failed:\n- " + "\n- ".join(failures))
    return payload


def _resolve_modal_gpu(default: str = "A100") -> str:
    """Resolve the Modal GPU override from CLI args at import time."""
    import sys

    args = sys.argv
    for idx, arg in enumerate(args):
        if arg == "--gpu" and idx + 1 < len(args):
            return args[idx + 1]
    return default


try:
    from benchmarks.modal_infra import make_modal_app

    _MODAL_GPU = _resolve_modal_gpu()
    app, GPU = make_modal_app("causal-ssm-nuts-mixed-support-recovery", _MODAL_GPU)
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False


if HAS_MODAL:

    @app.function(gpu=GPU, timeout=7200)
    def benchmark_remote(config: dict[str, Any], check: bool = True) -> dict[str, Any]:
        return run_benchmark(config, check=check)

    @app.local_entrypoint()
    def modal_main(
        gpu: str = "A100",  # noqa: ARG001 (consumed at import time)
        n_time: int = 40,
        num_warmup: int = 25,
        num_samples: int = 25,
        num_chains: int = 4,
        seed: int = 0,
        dense_mass: bool = False,
        target_accept_prob: float = 0.9,
        max_tree_depth: int = 6,
        n_ieks_iters: int = 6,
        pathfinder_num_elbo_samples: int = 20,
        pathfinder_maxiter: int = 20,
        progress_bar: bool = False,
        check: bool = True,
    ) -> None:
        config = {
            "n_time": n_time,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
            "seed": seed,
            "dense_mass": dense_mass,
            "target_accept_prob": target_accept_prob,
            "max_tree_depth": max_tree_depth,
            "n_ieks_iters": n_ieks_iters,
            "pathfinder_num_elbo_samples": pathfinder_num_elbo_samples,
            "pathfinder_maxiter": pathfinder_maxiter,
            "progress_bar": progress_bar,
        }
        result = benchmark_remote.remote(config, check=check)
        print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-time", type=int, default=DEFAULT_CONFIG["n_time"])
    parser.add_argument("--num-warmup", type=int, default=DEFAULT_CONFIG["num_warmup"])
    parser.add_argument("--num-samples", type=int, default=DEFAULT_CONFIG["num_samples"])
    parser.add_argument("--num-chains", type=int, default=DEFAULT_CONFIG["num_chains"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument(
        "--dense-mass",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["dense_mass"],
    )
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        default=DEFAULT_CONFIG["target_accept_prob"],
    )
    parser.add_argument("--max-tree-depth", type=int, default=DEFAULT_CONFIG["max_tree_depth"])
    parser.add_argument("--n-ieks-iters", type=int, default=DEFAULT_CONFIG["n_ieks_iters"])
    parser.add_argument(
        "--pathfinder-num-elbo-samples",
        type=int,
        default=DEFAULT_CONFIG["pathfinder_num_elbo_samples"],
    )
    parser.add_argument(
        "--pathfinder-maxiter",
        type=int,
        default=DEFAULT_CONFIG["pathfinder_maxiter"],
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["progress_bar"],
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise if the benchmark misses the recovery thresholds.",
    )
    args = parser.parse_args()

    config = {
        "n_time": args.n_time,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
        "seed": args.seed,
        "dense_mass": args.dense_mass,
        "target_accept_prob": args.target_accept_prob,
        "max_tree_depth": args.max_tree_depth,
        "n_ieks_iters": args.n_ieks_iters,
        "pathfinder_num_elbo_samples": args.pathfinder_num_elbo_samples,
        "pathfinder_maxiter": args.pathfinder_maxiter,
        "progress_bar": args.progress_bar,
    }
    result = run_benchmark(config, check=args.check)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
