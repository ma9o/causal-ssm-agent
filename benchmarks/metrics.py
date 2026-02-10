"""Recovery metrics and reporting for parameter recovery benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class RecoveryResult:
    """Structured result from a recovery benchmark."""

    method: str
    rmse: float
    corr: float
    elapsed: float
    coverage: dict[str, tuple[int, int]]  # category -> (covered, total)


def header(title: str):
    w = 70
    print("=" * w)
    print(f" {title}")
    print("=" * w)


def print_recovery(name: str, true_val, samples_arr) -> bool:
    """Print one row of recovery stats. Returns True if 90% CI covers truth."""
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
    """Print a matrix with row/col labels."""
    a = np.array(m)
    w = 8
    print(f"{'':>{w}s}", "".join(f"{c:>{w}s}" for c in col_labels))
    for i, rl in enumerate(row_labels):
        vals = "".join(f"{a[i, j]:>{w}.3f}" for j in range(a.shape[1]))
        print(f"{rl:>{w}s}{vals}")


def extract_drift(samples, n_latent):
    """Extract full drift matrix samples, handling both SVI and MCMC formats."""
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
    """Extract diffusion diagonal samples (Cholesky SD)."""
    if "diffusion" in samples and samples["diffusion"].ndim == 3:
        return jnp.array([samples["diffusion"][:, i, i] for i in range(n_latent)]).T
    return samples["diffusion_diag_pop"]


def extract_cint(samples):
    """Extract continuous intercept samples."""
    if "cint" in samples and samples["cint"].ndim == 2:
        return samples["cint"]
    return samples["cint_pop"]


def extract_lambda(samples):
    """Extract lambda matrix samples (or None if not available)."""
    if "lambda" in samples and samples["lambda"].ndim == 3:
        return samples["lambda"]
    return None


def report_recovery(
    method_name,
    samples,
    true_drift,
    true_diff_diag,
    true_cint,
    names,
    n_latent,
    true_lambda=None,
    true_manifest_means=None,  # noqa: ARG001
    n_manifest=None,
) -> tuple[float, float]:
    """Print recovery report and return (rmse, corr)."""
    header(f"RESULTS: {method_name}")

    offdiag_pairs = [(i, j) for i in range(n_latent) for j in range(n_latent) if i != j]

    drift_s = extract_drift(samples, n_latent)
    diff_diag_s = extract_diffusion_diag(samples, n_latent)
    cint_s = extract_cint(samples)
    lam_s = extract_lambda(samples)

    # --- Drift diagonal ---
    print("  DRIFT DIAGONAL (auto-effects)")
    drift_diag_ok = 0
    for i in range(n_latent):
        drift_diag_ok += print_recovery(names[i], true_drift[i, i], drift_s[:, i, i])
    print(f"  Coverage: {drift_diag_ok}/{n_latent}")
    print()

    # --- Drift off-diagonal ---
    print("  DRIFT OFF-DIAGONAL (cross-effects)")
    drift_off_ok = 0
    for i, j in offdiag_pairs:
        drift_off_ok += print_recovery(
            f"{names[j]}->{names[i]}", true_drift[i, j], drift_s[:, i, j]
        )
    print(f"  Coverage: {drift_off_ok}/{len(offdiag_pairs)}")
    print()

    # --- Diffusion ---
    print("  DIFFUSION (process noise SD)")
    diff_ok = 0
    for i in range(n_latent):
        diff_ok += print_recovery(names[i], true_diff_diag[i], diff_diag_s[:, i])
    print(f"  Coverage: {diff_ok}/{n_latent}")
    print()

    # --- CINT ---
    print("  CONTINUOUS INTERCEPT")
    cint_ok = 0
    for i in range(n_latent):
        cint_ok += print_recovery(names[i], true_cint[i], cint_s[:, i])
    print(f"  Coverage: {cint_ok}/{n_latent}")
    print()

    # --- Lambda ---
    if lam_s is not None and true_lambda is not None and n_manifest is not None:
        print("  LAMBDA (cross-loadings)")
        lam_ok, lam_n = 0, 0
        for i in range(n_latent, n_manifest):
            for j in range(n_latent):
                lam_ok += print_recovery(f"M{i}<-{names[j]}", true_lambda[i, j], lam_s[:, i, j])
                lam_n += 1
        print(f"  Coverage: {lam_ok}/{lam_n}")
        print()

    # --- Drift matrix comparison ---
    drift_post = jnp.mean(drift_s, axis=0)
    print("  True A:")
    print_matrix(true_drift, names, names)
    print()
    print("  Posterior mean A:")
    print_matrix(drift_post, names, names)
    print()
    print("  Bias (post - true):")
    print_matrix(drift_post - true_drift, names, names)
    print()

    # --- Summary stats ---
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
