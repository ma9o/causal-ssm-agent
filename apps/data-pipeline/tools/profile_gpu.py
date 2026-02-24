"""Profile PF+NUTS on Modal GPU — sweep particle counts to find GPU saturation.

Usage:
    modal run profile_gpu.py
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(uv_project_dir=str(ROOT), groups=["dev"], frozen=True)
    .uv_pip_install("jax[cuda12]", gpu="B200")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(ROOT / "src" / "causal_ssm_agent", remote_path="/root/src/causal_ssm_agent")
)

app = modal.App("causal-ssm-gpu-profile", image=image)


@app.function(gpu="B200", timeout=600)
def profile():
    import subprocess
    import threading
    import time

    import jax
    import jax.numpy as jnp
    import jax.random as random
    import numpyro

    from causal_ssm_agent.models.likelihoods.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )
    from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood
    from causal_ssm_agent.models.ssm import SSMModel, SSMSpec

    # ── GPU monitor: polls nvidia-smi every 100ms in background ─────────
    # NOTE: nvidia-smi "utilization.gpu" = % of time any kernel was active.
    # It does NOT measure SM occupancy or compute throughput. Two workloads
    # can both show 100% while one uses 1% of SMs and the other 90%.
    # Still useful for relative comparison across particle counts.
    class GpuMonitor:
        def __init__(self, interval=0.1):
            self.interval = interval
            self.readings: list[int] = []
            self._stop = threading.Event()
            self._thread: threading.Thread | None = None

        def start(self):
            self.readings.clear()
            self._stop.clear()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

        def stop(self) -> str:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=2)
            if not self.readings:
                return "n/a"
            return (
                f"mean={sum(self.readings) / len(self.readings):.0f}% "
                f"peak={max(self.readings)}% (n={len(self.readings)})"
            )

        def _poll(self):
            while not self._stop.is_set():
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if r.returncode == 0 and r.stdout.strip().isdigit():
                    self.readings.append(int(r.stdout.strip()))
                self._stop.wait(self.interval)

    monitor = GpuMonitor()

    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"JAX {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
        check=False,
    )
    print()

    # ── Simulate stable AR data ─────────────────────────────────────────
    true_drift_diag = jnp.array([-0.6, -0.9])
    key = random.PRNGKey(42)
    T, n_latent, dt = 60, 2, 0.5
    discrete_coef = jnp.diag(jnp.exp(true_drift_diag * dt))

    states = [jnp.zeros(n_latent)]
    for _ in range(T - 1):
        key, subkey = random.split(key)
        states.append(discrete_coef @ states[-1] + random.normal(subkey, (n_latent,)) * 0.3)

    key, subkey = random.split(key)
    observations = jnp.stack(states) + random.normal(subkey, (T, n_latent)) * 0.1
    times = jnp.arange(T, dtype=float) * dt
    time_intervals = jnp.diff(times, prepend=times[0])
    time_intervals = time_intervals.at[0].set(1e-6)

    meas_params = MeasurementParams(
        lambda_mat=jnp.eye(2), manifest_means=jnp.zeros(2), manifest_cov=jnp.eye(2) * 0.01
    )
    init = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

    print(f"Data: T={T}, n_latent={n_latent}")
    print()

    # ── Particle count sweep ────────────────────────────────────────────
    print("=" * 60)
    print("PARTICLE COUNT SWEEP")
    print("=" * 60)
    print(f"{'particles':>10} {'LL':>10} {'LL(ms)':>10} {'Grad(ms)':>10} {'GPU util':>20}")
    print("-" * 66)

    sweep_results = []
    for n_particles in [200, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]:
        backend = ParticleLikelihood(n_latent=2, n_manifest=2, n_particles=n_particles)

        def ll_fn(drift_diag, _backend=backend):
            ct = CTParams(drift=jnp.diag(drift_diag), diffusion_cov=jnp.eye(2) * 0.09, cint=None)
            return _backend.compute_log_likelihood(
                ct, meas_params, init, observations, time_intervals
            )

        # Compile (don't measure)
        try:
            jax.block_until_ready(ll_fn(true_drift_diag))
            jax.block_until_ready(jax.grad(ll_fn)(true_drift_diag))
        except Exception as e:
            print(f"{n_particles:>10} OOM or error: {e}")
            break

        # Benchmark LL with GPU monitoring
        monitor.start()
        t0 = time.perf_counter()
        n_iters = 5
        for _ in range(n_iters):
            jax.block_until_ready(ll_fn(true_drift_diag))
        t_ll = (time.perf_counter() - t0) / n_iters * 1000
        monitor.stop()
        ll_val = float(ll_fn(true_drift_diag))

        # Benchmark grad with GPU monitoring
        monitor.start()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            jax.block_until_ready(jax.grad(ll_fn)(true_drift_diag))
        t_grad = (time.perf_counter() - t0) / n_iters * 1000
        gpu_grad = monitor.stop()

        print(f"{n_particles:>10} {ll_val:>10.2f} {t_ll:>10.1f} {t_grad:>10.1f} {gpu_grad:>20}")
        sweep_results.append((n_particles, t_grad))

    print()

    # Pick the particle count with best grad throughput per particle
    # (highest particles where grad time is still reasonable for MCMC)
    # Target: grad eval < 500ms so 100-step tree < 50s
    feasible = [(n, t) for n, t in sweep_results if t < 5000]
    best_n = max(feasible, key=lambda x: x[0])[0] if feasible else 10_000
    print(f"Selected {best_n} particles for MCMC (grad < 5s)")
    print()

    # ── MCMC ────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"MCMC ({best_n} particles, 50 warmup + 50 samples)")
    print("=" * 60)

    spec = SSMSpec(n_latent=2, n_manifest=2, lambda_mat=jnp.eye(2), diffusion="diag")
    model = SSMModel(spec, n_particles=best_n)
    numpyro.set_host_device_count(1)

    monitor.start()
    t0 = time.perf_counter()
    mcmc = model.fit(
        observations=observations,
        times=times,
        num_warmup=50,
        num_samples=50,
        num_chains=1,
    )
    t_mcmc = time.perf_counter() - t0
    gpu_mcmc = monitor.stop()

    print(f"\nTotal: {t_mcmc:.1f}s  ({t_mcmc / 100:.2f}s/step)")
    print(f"GPU during MCMC: {gpu_mcmc}")

    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()

    print(f"Divergences: {int(jnp.sum(extra.get('diverging', jnp.array(0))))}")
    if "accept_prob" in extra:
        print(f"Accept prob: {float(jnp.mean(extra['accept_prob'])):.3f}")
    if "num_steps" in extra:
        steps = extra["num_steps"]
        print(
            f"Tree steps: mean={float(jnp.mean(steps)):.1f}, "
            f"max={int(jnp.max(steps))}, median={float(jnp.median(steps)):.0f}"
        )

    print("\nPosterior vs True:")
    drift_samples = samples.get("drift_diag_pop")
    if drift_samples is not None:
        mean = jnp.mean(drift_samples, axis=0)
        std = jnp.std(drift_samples, axis=0)
        print(f"  drift_diag: mean={mean} std={std} (true: {true_drift_diag})")
    for name in ["diffusion_diag_pop", "manifest_var_diag"]:
        if name in samples:
            print(f"  {name}: mean={jnp.mean(samples[name], axis=0)}")

    # Final memory snapshot
    print()
    subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
        check=False,
    )


@app.local_entrypoint()
def main():
    profile.remote()
