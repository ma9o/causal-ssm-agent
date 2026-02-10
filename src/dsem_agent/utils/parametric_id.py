"""Parametric identifiability diagnostics for state-space models.

Pre-fit diagnostics (Stage 4b):
- Eigenspectrum analysis of the Fisher information (Hessian of log-likelihood)
- Estimand projection: does the data inform the quantity of interest?
- Expected contraction: BvM approximation of prior→posterior shrinkage

Post-fit diagnostics (Stage 5):
- Power-scaling sensitivity: detect prior-dominated or conflicting parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.flatten_util import ravel_pytree

from dsem_agent.models.ssm.discretization import discretize_system_batched
from dsem_agent.models.ssm.hessmc2 import (
    _assemble_deterministics,
    _build_eval_fns,
    _discover_sites,
)

if TYPE_CHECKING:
    from dsem_agent.models.ssm.inference import InferenceResult
    from dsem_agent.models.ssm.model import SSMModel

# ---------------------------------------------------------------------------
# Forward simulator
# ---------------------------------------------------------------------------


def simulate_ssm(
    drift: jnp.ndarray,
    diffusion_chol: jnp.ndarray,
    lambda_mat: jnp.ndarray,
    manifest_chol: jnp.ndarray,
    t0_means: jnp.ndarray,
    t0_chol: jnp.ndarray,
    times: jnp.ndarray,
    rng_key: jnp.ndarray,
    cint: jnp.ndarray | None = None,
    manifest_means: jnp.ndarray | None = None,
    manifest_dist: str = "gaussian",
) -> jnp.ndarray:
    """Generate synthetic observations from constrained SSM parameters.

    Uses discretize_system_batched for CT→DT conversion, then lax.scan
    for JAX-traceable forward simulation.

    Args:
        drift: (n_latent, n_latent) continuous-time drift matrix
        diffusion_chol: (n_latent, n_latent) lower Cholesky of diffusion
        lambda_mat: (n_manifest, n_latent) factor loadings
        manifest_chol: (n_manifest, n_manifest) lower Cholesky of obs noise
        t0_means: (n_latent,) initial state means
        t0_chol: (n_latent, n_latent) lower Cholesky of initial state cov
        times: (T,) observation times
        rng_key: JAX PRNG key
        cint: (n_latent,) continuous intercept (optional)
        manifest_means: (n_manifest,) manifest intercepts (optional)
        manifest_dist: observation noise family ("gaussian" or "poisson")

    Returns:
        observations: (T, n_manifest) simulated data
    """
    n_latent = drift.shape[0]
    n_manifest = lambda_mat.shape[0]
    T = times.shape[0]

    # Diffusion covariance
    diffusion_cov = diffusion_chol @ diffusion_chol.T

    # Discretize over all time intervals
    dt_array = jnp.diff(times)
    Ad, Qd, cd = discretize_system_batched(drift, diffusion_cov, cint, dt_array)

    # Initial state covariance
    t0_cov = t0_chol @ t0_chol.T

    # Manifest noise covariance
    manifest_cov = manifest_chol @ manifest_chol.T

    # Default manifest means
    if manifest_means is None:
        manifest_means = jnp.zeros(n_manifest)

    # Sample initial state
    rng_key, init_key = random.split(rng_key)
    t0_chol_safe = jnp.linalg.cholesky(t0_cov + jnp.eye(n_latent) * 1e-8)
    x_0 = t0_means + t0_chol_safe @ random.normal(init_key, (n_latent,))

    # First observation from x_0
    rng_key, obs_key = random.split(rng_key)
    mu_0 = lambda_mat @ x_0 + manifest_means
    if manifest_dist == "poisson":
        y_0 = random.poisson(obs_key, jax.nn.softplus(mu_0)).astype(jnp.float32)
    else:
        manifest_chol_safe = jnp.linalg.cholesky(manifest_cov + jnp.eye(n_manifest) * 1e-8)
        y_0 = mu_0 + manifest_chol_safe @ random.normal(obs_key, (n_manifest,))

    # Scan over remaining timesteps
    def scan_fn(carry, inputs):
        x_prev, rng = carry
        Ad_t, Qd_t = inputs[0], inputs[1]
        cd_t = inputs[2]

        # State transition
        rng, state_key, obs_key = random.split(rng, 3)
        Qd_chol = jnp.linalg.cholesky(Qd_t + jnp.eye(n_latent) * 1e-8)
        mean_x = Ad_t @ x_prev + cd_t
        x_t = mean_x + Qd_chol @ random.normal(state_key, (n_latent,))

        # Observation
        mu_t = lambda_mat @ x_t + manifest_means
        if manifest_dist == "poisson":
            y_t = random.poisson(obs_key, jax.nn.softplus(mu_t)).astype(jnp.float32)
        else:
            y_t = mu_t + manifest_chol_safe @ random.normal(obs_key, (n_manifest,))

        return (x_t, rng), y_t

    # Handle cd: if None, use zeros
    if cd is None:
        cd_scan = jnp.zeros((T - 1, n_latent))
    else:
        cd_scan = cd

    (_, _), y_rest = lax.scan(scan_fn, (x_0, rng_key), (Ad, Qd, cd_scan))

    # Stack: first obs + rest
    observations = jnp.concatenate([y_0[None, :], y_rest], axis=0)
    return observations


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParametricIDResult:
    """Results from pre-fit parametric identifiability analysis."""

    eigenvalues: jnp.ndarray  # (n_draws, D) — Fisher information eigenvalues
    min_eigenvalues: jnp.ndarray  # (n_draws,) — min Fisher eigenvalue per draw
    condition_numbers: jnp.ndarray  # (n_draws,)
    estimand_information: dict[str, jnp.ndarray]  # site -> (n_draws,)
    expected_contraction: dict[str, jnp.ndarray]  # site -> (n_draws,) per-param
    prior_variances: dict[str, float]
    parameter_names: list[str]
    n_draws: int

    def summary(self) -> dict:
        """Produce diagnostic flags from the analysis.

        Returns:
            Dict with keys: structural_issues, boundary_issues,
            weak_params, estimand_status
        """
        # Structural: min eigenvalue ≈ 0 across ALL draws
        struct_threshold = 1e-4
        structural_issues = bool(jnp.all(self.min_eigenvalues < struct_threshold))

        # Boundary: min eigenvalue ≈ 0 at SOME draws but not all
        boundary_issues = bool(
            jnp.any(self.min_eigenvalues < struct_threshold)
            and not jnp.all(self.min_eigenvalues < struct_threshold)
        )

        # Weak params: expected contraction < 0.1
        weak_params = []
        for name, contraction in self.expected_contraction.items():
            mean_contraction = float(jnp.mean(contraction))
            if mean_contraction < 0.1:
                weak_params.append(name)

        # Estimand status
        estimand_status = {}
        for name, info in self.estimand_information.items():
            mean_info = float(jnp.mean(info))
            if mean_info < 1e-4:
                estimand_status[name] = "unidentified"
            elif mean_info < 1.0:
                estimand_status[name] = "weakly_identified"
            else:
                estimand_status[name] = "identified"

        return {
            "structural_issues": structural_issues,
            "boundary_issues": boundary_issues,
            "weak_params": weak_params,
            "estimand_status": estimand_status,
            "mean_condition_number": float(jnp.mean(self.condition_numbers)),
        }

    def print_report(self) -> None:
        """Print a human-readable diagnostic report."""
        summary = self.summary()
        print("\n=== Parametric Identifiability Report ===")
        print(f"  Draws: {self.n_draws}")
        print(f"  Parameters: {len(self.parameter_names)}")
        print(f"  Mean condition number: {summary['mean_condition_number']:.2e}")

        if summary["structural_issues"]:
            print("  [!] STRUCTURAL non-identifiability detected")
            print("      Min eigenvalue ≈ 0 at all prior draws")
        elif summary["boundary_issues"]:
            print("  [~] BOUNDARY identifiability issues")
            print("      Min eigenvalue ≈ 0 at some prior draws")
        else:
            print("  [ok] No structural identifiability issues")

        if summary["weak_params"]:
            print(f"  [~] Weak parameters (contraction < 0.1): {summary['weak_params']}")

        if summary["estimand_status"]:
            print("  Estimand status:")
            for name, status in summary["estimand_status"].items():
                print(f"    {name}: {status}")


@dataclass
class PowerScalingResult:
    """Results from post-fit power-scaling sensitivity analysis."""

    prior_sensitivity: dict[str, float]
    likelihood_sensitivity: dict[str, float]
    diagnosis: dict[str, str]  # "prior_dominated" | "well_identified" | "prior_data_conflict"
    psis_k_hat: dict[str, float] = field(default_factory=dict)

    def print_report(self) -> None:
        """Print a human-readable power-scaling report."""
        print("\n=== Power-Scaling Sensitivity Report ===")
        for name in self.diagnosis:
            prior_s = self.prior_sensitivity.get(name, 0.0)
            lik_s = self.likelihood_sensitivity.get(name, 0.0)
            diag = self.diagnosis[name]
            k_hat = self.psis_k_hat.get(name, float("nan"))
            reliable = "reliable" if k_hat < 0.7 else "UNRELIABLE"
            print(
                f"  {name}: prior_sens={prior_s:.3f}, lik_sens={lik_s:.3f} "
                f"→ {diag} (k_hat={k_hat:.2f}, {reliable})"
            )


# ---------------------------------------------------------------------------
# Pre-fit: check_parametric_id
# ---------------------------------------------------------------------------


def check_parametric_id(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    n_draws: int = 5,
    estimand_sites: list[str] | None = None,
    seed: int = 42,
) -> ParametricIDResult:
    """Pre-fit parametric identifiability diagnostics.

    For each of n_draws prior draws:
    1. Draw z from prior, simulate synthetic data y
    2. Compute Hessian of log-likelihood at z with data y
    3. Analyse eigenspectrum, estimand projection, expected contraction

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data (used for shape/times only)
        times: (T,) observation times
        subject_ids: optional subject indices
        n_draws: number of prior draws to evaluate
        estimand_sites: parameter sites to project onto (e.g. ["drift_offdiag_pop"])
        seed: random seed

    Returns:
        ParametricIDResult with diagnostic information
    """
    rng_key = random.PRNGKey(seed)

    # 1. Discover sites
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, subject_ids, trace_key)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    D = flat_example.shape[0]

    param_names = sorted(site_info.keys())

    # Compute prior variances from distributions
    prior_variances = {}
    for name in param_names:
        d = site_info[name]["distribution"]
        try:
            var = float(jnp.sum(d.variance))
        except (NotImplementedError, AttributeError):
            var = float("inf")
        prior_variances[name] = var

    # Storage
    all_eigenvalues = []
    estimand_info_storage: dict[str, list[float]] = {s: [] for s in (estimand_sites or [])}
    contraction_storage: dict[str, list[float]] = {n: [] for n in param_names}

    for _i in range(n_draws):
        # 2. Draw z_i from prior
        parts = []
        for name in param_names:
            info = site_info[name]
            rng_key, sample_key = random.split(rng_key)
            prior_sample = info["distribution"].sample(sample_key, ())
            unc_sample = info["transform"].inv(prior_sample)
            parts.append(unc_sample.reshape(-1))
        z_i = jnp.concatenate(parts)

        # 3. Constrain z_i → simulate synthetic observations
        unc_dict = unravel_fn(z_i)
        transforms = {name: info["transform"] for name, info in site_info.items()}
        con_dict = {name: transforms[name](unc_dict[name]) for name in unc_dict}

        # Assemble SSM matrices from constrained samples for simulation
        det = _assemble_deterministics({k: v[None, ...] for k, v in con_dict.items()}, model.spec)
        # Squeeze batch dim
        det = {k: v[0] for k, v in det.items()}

        rng_key, sim_key = random.split(rng_key)
        y_i = simulate_ssm(
            drift=det.get("drift", jnp.zeros((model.spec.n_latent, model.spec.n_latent))),
            diffusion_chol=det.get("diffusion", jnp.eye(model.spec.n_latent)),
            lambda_mat=det.get("lambda", jnp.eye(model.spec.n_manifest, model.spec.n_latent)),
            manifest_chol=jnp.linalg.cholesky(
                det.get("manifest_cov", jnp.eye(model.spec.n_manifest))
                + jnp.eye(model.spec.n_manifest) * 1e-8
            ),
            t0_means=det.get("t0_means", jnp.zeros(model.spec.n_latent)),
            t0_chol=jnp.linalg.cholesky(
                det.get("t0_cov", jnp.eye(model.spec.n_latent))
                + jnp.eye(model.spec.n_latent) * 1e-8
            ),
            times=times,
            rng_key=sim_key,
            cint=det.get("cint"),
            manifest_dist=model.spec.manifest_dist.value,
        )

        # 4. Build log-likelihood function for simulated data
        log_lik_fn, _log_prior_unc_fn = _build_eval_fns(
            model, y_i, times, subject_ids, site_info, unravel_fn
        )

        # 5. Hessian of log-likelihood at z_i
        H_i = jax.hessian(log_lik_fn)(z_i)
        H_i = jnp.nan_to_num(H_i, nan=0.0, posinf=0.0, neginf=0.0)

        # Fisher information = -H (should be PSD for identified model)
        # Eigenspectrum of Fisher info: near-zero eigenvalues → non-identifiability
        fisher_eigvals_i = jnp.linalg.eigvalsh(-H_i)
        all_eigenvalues.append(fisher_eigvals_i)

        # 6. Estimand projection
        if estimand_sites:
            for site_name in estimand_sites:
                if site_name in param_names:
                    # Build projection function: g(z) extracts the site value
                    def _make_g(sname, _transforms=transforms):
                        def g(z):
                            unc = unravel_fn(z)
                            t = _transforms[sname]
                            return jnp.sum(t(unc[sname]))

                        return g  # noqa: B023

                    g = _make_g(site_name)
                    grad_g = jax.grad(g)(z_i)
                    grad_g = jnp.nan_to_num(grad_g, nan=0.0)
                    # Projected Fisher information: grad_g^T H grad_g
                    I_g = float(grad_g @ (-H_i) @ grad_g)
                    estimand_info_storage[site_name].append(max(I_g, 0.0))

        # 7. Expected contraction: BvM approximation
        # posterior_var ≈ diag((-H)^{-1}), contraction = 1 - post_var / prior_var
        neg_H = -H_i
        neg_H_reg = neg_H + jnp.eye(D) * 1e-6
        try:
            H_inv_diag = jnp.diag(jnp.linalg.inv(neg_H_reg))
        except Exception:
            H_inv_diag = jnp.full(D, float("inf"))

        # Map diagonal elements back to parameter names
        offset = 0
        for name in param_names:
            size = int(jnp.prod(jnp.array(site_info[name]["shape"])))
            post_var = float(jnp.mean(jnp.abs(H_inv_diag[offset : offset + size])))
            pv = prior_variances[name]
            if pv > 0 and jnp.isfinite(pv):
                contraction = max(0.0, min(1.0, 1.0 - post_var / pv))
            else:
                contraction = 0.0
            contraction_storage[name].append(contraction)
            offset += size

    # Assemble results
    eigenvalues = jnp.stack(all_eigenvalues)
    min_eigenvalues = jnp.min(eigenvalues, axis=1)
    max_eigenvalues = jnp.max(jnp.abs(eigenvalues), axis=1)
    condition_numbers = max_eigenvalues / jnp.maximum(jnp.abs(min_eigenvalues), 1e-30)

    estimand_information = {name: jnp.array(vals) for name, vals in estimand_info_storage.items()}
    expected_contraction = {name: jnp.array(vals) for name, vals in contraction_storage.items()}

    return ParametricIDResult(
        eigenvalues=eigenvalues,
        min_eigenvalues=min_eigenvalues,
        condition_numbers=condition_numbers,
        estimand_information=estimand_information,
        expected_contraction=expected_contraction,
        prior_variances=prior_variances,
        parameter_names=param_names,
        n_draws=n_draws,
    )


# ---------------------------------------------------------------------------
# Post-fit: power-scaling sensitivity
# ---------------------------------------------------------------------------


def power_scaling_sensitivity(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    result: InferenceResult,
    subject_ids: jnp.ndarray | None = None,
    seed: int = 0,
    alpha_delta: float = 0.01,
) -> PowerScalingResult:
    """Post-fit power-scaling sensitivity diagnostic.

    Detects whether posterior is driven by prior or likelihood by
    perturbing each component's contribution and measuring the
    resulting shift in posterior means.

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) real observed data
        times: (T,) observation times
        result: InferenceResult from fitting
        subject_ids: optional subject indices
        seed: random seed
        alpha_delta: perturbation size for power scaling (default 0.01)

    Returns:
        PowerScalingResult with per-parameter sensitivity diagnostics
    """
    rng_key = random.PRNGKey(seed)

    # 1. Discover sites and build eval functions
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, subject_ids, trace_key)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    _, unravel_fn = ravel_pytree(example_unc)

    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model, observations, times, subject_ids, site_info, unravel_fn
    )

    param_names = sorted(site_info.keys())

    # 2. Extract posterior samples -> unconstrained flat vectors
    samples = result.get_samples()
    n_samples = next(iter(samples.values())).shape[0]

    flat_samples = []
    for i in range(n_samples):
        parts = []
        for name in param_names:
            if name in samples:
                con_val = samples[name][i]
                unc_val = site_info[name]["transform"].inv(con_val)
                parts.append(unc_val.reshape(-1))
        if parts:
            flat_samples.append(jnp.concatenate(parts))

    if not flat_samples:
        return PowerScalingResult(
            prior_sensitivity={},
            likelihood_sensitivity={},
            diagnosis={},
        )

    z_samples = jnp.stack(flat_samples)  # (n_samples, D)

    # 3. Evaluate log-prior and log-likelihood for each sample
    batch_log_lik = jax.vmap(log_lik_fn)
    batch_log_prior = jax.vmap(log_prior_unc_fn)

    # Chunk to avoid OOM
    chunk_size = 32
    log_liks_parts = []
    log_priors_parts = []
    for start in range(0, n_samples, chunk_size):
        chunk = z_samples[start : start + chunk_size]
        log_liks_parts.append(batch_log_lik(chunk))
        log_priors_parts.append(batch_log_prior(chunk))

    log_liks = jnp.concatenate(log_liks_parts)
    log_priors = jnp.concatenate(log_priors_parts)

    # 4. Power-scaling: compute weighted means under perturbed distributions
    alpha = alpha_delta

    # Prior perturbation weights: w_i propto exp(alpha * log_prior_i)
    prior_log_weights = alpha * log_priors
    prior_log_weights = prior_log_weights - jax.nn.logsumexp(prior_log_weights)
    prior_weights = jnp.exp(prior_log_weights)

    # Likelihood perturbation weights: w_i propto exp(alpha * log_lik_i)
    lik_log_weights = alpha * log_liks
    lik_log_weights = lik_log_weights - jax.nn.logsumexp(lik_log_weights)
    lik_weights = jnp.exp(lik_log_weights)

    # 5. PSIS: Pareto-smoothed importance sampling for reliability
    # Simple Pareto k-hat diagnostic on the largest weights
    def _pareto_k_hat(log_w: jnp.ndarray) -> float:
        """Estimate Pareto k from the largest importance weights."""
        n = log_w.shape[0]
        # Use top 20% of weights
        m = max(int(0.2 * n), 5)
        sorted_w = jnp.sort(log_w)[-m:]
        # Simple moment-based k estimator
        mean_w = jnp.mean(sorted_w)
        var_w = jnp.var(sorted_w)
        k = 0.5 * (1.0 + mean_w**2 / jnp.maximum(var_w, 1e-10))
        return float(jnp.clip(k, 0.0, 2.0))

    # 6. Compute per-parameter sensitivity
    prior_sensitivity = {}
    likelihood_sensitivity = {}
    diagnosis = {}
    psis_k_hat = {}

    # Extract per-parameter values from flat samples
    offset = 0
    for name in param_names:
        if name not in samples:
            continue
        size = int(jnp.prod(jnp.array(site_info[name]["shape"])))
        param_vals = z_samples[:, offset : offset + size]  # (n_samples, size)
        param_mean = jnp.mean(param_vals, axis=0)

        # Weighted means under perturbation
        prior_weighted_mean = jnp.sum(prior_weights[:, None] * param_vals, axis=0)
        lik_weighted_mean = jnp.sum(lik_weights[:, None] * param_vals, axis=0)

        # Sensitivity = ||shift|| / alpha_delta
        prior_shift = float(jnp.mean(jnp.abs(prior_weighted_mean - param_mean))) / alpha_delta
        lik_shift = float(jnp.mean(jnp.abs(lik_weighted_mean - param_mean))) / alpha_delta

        prior_sensitivity[name] = prior_shift
        likelihood_sensitivity[name] = lik_shift

        # k-hat from prior weights
        psis_k_hat[name] = _pareto_k_hat(prior_log_weights)

        # Diagnosis
        if prior_shift > 0.05 and lik_shift < 0.05:
            diagnosis[name] = "prior_dominated"
        elif prior_shift > 0.05 and lik_shift > 0.05:
            diagnosis[name] = "prior_data_conflict"
        else:
            diagnosis[name] = "well_identified"

        offset += size

    return PowerScalingResult(
        prior_sensitivity=prior_sensitivity,
        likelihood_sensitivity=likelihood_sensitivity,
        diagnosis=diagnosis,
        psis_k_hat=psis_k_hat,
    )
