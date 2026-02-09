"""Parameter recovery via SMC² (Hess-MC²) on a 4-latent SSM.

Ground truth: 4 latent processes (Stress -> Fatigue -> Focus -> Performance)
with off-diagonal drift, a 6-indicator measurement model, and continuous
intercepts. Fits with SMC² using the Hessian (second-order) proposal.

Usage:
    modal run tools/recovery_hessmc2.py            # remote GPU
    uv run python tools/recovery_hessmc2.py        # local CPU (smoke test)
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root

# ---------------------------------------------------------------------------
# Modal setup (only used when invoked via `modal run`)
# ---------------------------------------------------------------------------
try:
    import modal

    GPU = "B200"

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install("uv")
        .uv_sync(uv_project_dir=str(ROOT), groups=["dev"], frozen=True)
        .uv_pip_install("jax[cuda12]", gpu=GPU)
        .env({"PYTHONPATH": "/root/src"})
        .add_local_dir(ROOT / "src" / "dsem_agent", remote_path="/root/src/dsem_agent")
    )
    app = modal.App("dsem-recovery-hessmc2", image=image)
    HAS_MODAL = True
except Exception:
    HAS_MODAL = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def header(title: str):
    w = 70
    print("=" * w)
    print(f" {title}")
    print("=" * w)


def print_recovery(name: str, true_val, samples_arr) -> bool:
    import jax.numpy as jnp

    mean = float(jnp.mean(samples_arr))
    std = float(jnp.std(samples_arr))
    q5 = float(jnp.percentile(samples_arr, 5))
    q95 = float(jnp.percentile(samples_arr, 95))
    true = float(true_val)
    covered = q5 <= true <= q95
    bias = mean - true
    tag = "OK" if covered else "MISS"
    print(
        f"  {name:<20s}  true={true:+.3f}  post={mean:+.3f}+-{std:.3f}"
        f"  90%CI=[{q5:+.3f},{q95:+.3f}]  {tag}  bias={bias:+.3f}"
    )
    return covered


def print_matrix(m, row_labels, col_labels):
    import numpy as np

    a = np.array(m)
    w = 8
    print(f"{'':>{w}s}", "".join(f"{c:>{w}s}" for c in col_labels))
    for i, rl in enumerate(row_labels):
        vals = "".join(f"{a[i, j]:>{w}.3f}" for j in range(a.shape[1]))
        print(f"{rl:>{w}s}{vals}")


def extract_drift(samples, n_latent):
    import jax.numpy as jnp

    if "drift" in samples and samples["drift"].ndim == 3:
        return samples["drift"]
    drift_diag = samples["drift_diag_pop"]
    drift_offdiag = samples["drift_offdiag_pop"]

    def assemble_one(diag, offdiag):
        d = jnp.diag(-jnp.abs(diag))
        idx = 0
        for i in range(n_latent):
            for j in range(n_latent):
                if i != j:
                    d = d.at[i, j].set(offdiag[idx])
                    idx += 1
        return d

    from jax import vmap

    return vmap(assemble_one)(drift_diag, drift_offdiag)


def extract_diffusion_diag(samples, n_latent):
    import jax.numpy as jnp

    if "diffusion" in samples and samples["diffusion"].ndim == 3:
        return jnp.array([samples["diffusion"][:, i, i] for i in range(n_latent)]).T
    return samples["diffusion_diag_pop"]


def extract_cint(samples):
    if "cint" in samples and samples["cint"].ndim == 2:
        return samples["cint"]
    return samples["cint_pop"]


def report_recovery(method_name, samples, true_drift, true_diff_diag, true_cint, names, n_latent):
    import jax.numpy as jnp
    import numpy as np

    header(f"RESULTS: {method_name}")

    offdiag_pairs = [(i, j) for i in range(n_latent) for j in range(n_latent) if i != j]

    drift_s = extract_drift(samples, n_latent)
    diff_diag_s = extract_diffusion_diag(samples, n_latent)
    cint_s = extract_cint(samples)

    print("  DRIFT DIAGONAL (auto-effects)")
    drift_diag_ok = 0
    for i in range(n_latent):
        drift_diag_ok += print_recovery(names[i], true_drift[i, i], drift_s[:, i, i])
    print(f"  Coverage: {drift_diag_ok}/{n_latent}")
    print()

    print("  DRIFT OFF-DIAGONAL (cross-effects)")
    drift_off_ok = 0
    for i, j in offdiag_pairs:
        drift_off_ok += print_recovery(
            f"{names[j]}->{names[i]}", true_drift[i, j], drift_s[:, i, j]
        )
    print(f"  Coverage: {drift_off_ok}/{len(offdiag_pairs)}")
    print()

    print("  DIFFUSION (process noise SD)")
    diff_ok = 0
    for i in range(n_latent):
        diff_ok += print_recovery(names[i], true_diff_diag[i], diff_diag_s[:, i])
    print(f"  Coverage: {diff_ok}/{n_latent}")
    print()

    print("  CONTINUOUS INTERCEPT")
    cint_ok = 0
    for i in range(n_latent):
        cint_ok += print_recovery(names[i], true_cint[i], cint_s[:, i])
    print(f"  Coverage: {cint_ok}/{n_latent}")
    print()

    # Summary stats
    pairs = []
    for i in range(n_latent):
        pairs.append((float(true_drift[i, i]), float(jnp.mean(drift_s[:, i, i]))))
    for i, j in offdiag_pairs:
        pairs.append((float(true_drift[i, j]), float(jnp.mean(drift_s[:, i, j]))))
    for i in range(n_latent):
        pairs.append((float(true_diff_diag[i]), float(jnp.mean(diff_diag_s[:, i]))))
    for i in range(n_latent):
        pairs.append((float(true_cint[i]), float(jnp.mean(cint_s[:, i]))))

    trues = np.array([p[0] for p in pairs])
    posts = np.array([p[1] for p in pairs])
    rmse = float(np.sqrt(np.mean((posts - trues) ** 2)))
    corr = float(np.corrcoef(trues, posts)[0, 1]) if len(trues) > 2 else 0.0

    print(f"  RMSE={rmse:.4f}  Corr={corr:.4f}")
    print()
    return rmse, corr


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def run(local: bool = False):
    import time

    import jax
    import jax.numpy as jnp
    import jax.random as random
    import jax.scipy.linalg as jla
    import numpy as np

    from dsem_agent.models.ssm import SSMModel, SSMPriors, SSMSpec, discretize_system, fit

    header("ENVIRONMENT")
    print(f"JAX {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")

    if local:
        T = 80
        N_SMC = 64
        K_ITER = 15
        K_WARMUP = 5
        N_PF = 200
    else:
        T = 200
        N_SMC = 512
        K_ITER = 30
        K_WARMUP = 10
        N_PF = 500

    print(f"T={T}, N_smc={N_SMC}, K={K_ITER}, warmup={K_WARMUP}, N_pf={N_PF}")
    print()

    # ==================================================================
    # 1. Ground truth
    # ==================================================================
    header("GROUND TRUTH")

    n_latent, n_manifest, dt = 4, 6, 0.5
    names = ["Stress", "Fatigue", "Focus", "Perf"]

    true_drift = jnp.array(
        [
            [-0.8, 0.0, 0.0, 0.0],
            [0.3, -0.5, 0.0, 0.0],
            [-0.2, -0.3, -0.6, 0.0],
            [0.0, 0.0, 0.4, -0.7],
        ]
    )
    true_diff_diag = jnp.array([0.4, 0.3, 0.3, 0.25])
    true_cint = jnp.array([0.5, 0.0, 0.3, 0.0])

    true_lambda = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.6, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.7],
        ]
    )
    true_manifest_means = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    true_mvar_diag = jnp.array([0.1, 0.1, 0.1, 0.1, 0.15, 0.15])

    eigs = np.linalg.eigvals(np.array(true_drift))
    print("Drift matrix A (continuous-time):")
    print_matrix(true_drift, names, names)
    print()
    print("Eigenvalues:", "  ".join(f"{e.real:+.3f}" for e in eigs))
    assert all(e.real < 0 for e in eigs), "Drift is not stable!"
    print()

    # ==================================================================
    # 2. Simulate
    # ==================================================================
    header("SIMULATE")

    diff_cov = jnp.diag(true_diff_diag) @ jnp.diag(true_diff_diag).T
    Ad, Qd, cd = discretize_system(true_drift, diff_cov, true_cint, dt)
    Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
    R_chol = jla.cholesky(jnp.diag(true_mvar_diag), lower=True)

    key = random.PRNGKey(42)
    key, init_key = random.split(key)
    x0 = jla.cholesky(jnp.eye(n_latent) * 0.5, lower=True) @ random.normal(init_key, (n_latent,))

    states = [x0]
    for _ in range(T - 1):
        key, nk = random.split(key)
        states.append(Ad @ states[-1] + cd.flatten() + Qd_chol @ random.normal(nk, (n_latent,)))
    latent = jnp.stack(states)

    key, ok = random.split(key)
    obs = (
        jax.vmap(lambda x: true_lambda @ x + true_manifest_means)(latent)
        + random.normal(ok, (T, n_manifest)) @ R_chol.T
    )
    times = jnp.arange(T, dtype=float) * dt

    print(f"Latent  shape={latent.shape}  mean={np.array(latent.mean(0)).round(2)}")
    print(f"Obs     shape={obs.shape}  mean={np.array(obs.mean(0)).round(2)}")
    print()

    # ==================================================================
    # 3. Model spec
    # ==================================================================
    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift="free",
        diffusion="diag",
        cint="free",
        lambda_mat="free",
        manifest_means="free",
        manifest_var="diag",
        t0_means="free",
        t0_var="diag",
        latent_names=names,
    )
    priors = SSMPriors(
        drift_diag={"mu": -0.5, "sigma": 0.5},
        drift_offdiag={"mu": 0.0, "sigma": 0.5},
        diffusion_diag={"sigma": 0.5},
        cint={"mu": 0.0, "sigma": 1.0},
        lambda_free={"mu": 0.5, "sigma": 0.5},
        manifest_means={"mu": 0.0, "sigma": 2.0},
        manifest_var_diag={"sigma": 0.5},
    )

    report_args = {
        "true_drift": true_drift,
        "true_diff_diag": true_diff_diag,
        "true_cint": true_cint,
        "names": names,
        "n_latent": n_latent,
    }

    # ==================================================================
    # 4. SMC² with Hessian (second-order) proposal
    # ==================================================================
    header("FIT: SMC² HESSIAN")
    model = SSMModel(spec, priors=priors, n_particles=N_PF, pf_seed=42)

    t0 = time.perf_counter()
    result = fit(
        model,
        observations=obs,
        times=times,
        method="hessmc2",
        n_smc_particles=N_SMC,
        n_iterations=K_ITER,
        proposal="hessian",
        step_size=0.5,
        warmup_iters=K_WARMUP,
        warmup_step_size=0.5,
        adapt_step_size=False,
        seed=0,
    )
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")
    print()

    rmse, corr = report_recovery("SMC² Hessian", result.get_samples(), **report_args)
    print(f"  Time={elapsed:.1f}s  RMSE={rmse:.4f}  Corr={corr:.4f}")


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

if HAS_MODAL:

    @app.function(gpu=GPU, timeout=3600)
    def recovery_remote():
        run(local=False)

    @app.local_entrypoint()
    def main():
        recovery_remote.remote()


if __name__ == "__main__":
    # Direct invocation: local CPU smoke test
    run(local=True)
