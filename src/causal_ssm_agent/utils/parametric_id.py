"""Parametric identifiability diagnostics for state-space models.

Pre-fit diagnostics (Stage 4b):
- T-rule (counting condition): necessary condition checking that the number
  of free parameters does not exceed available moment conditions.
- Profile likelihood: per-parameter identifiability classification via
  constrained optimization (Raue et al. 2009). Uses only 1st-order AD.
- Simulation-based calibration (SBC): posterior calibration validation
  with data-dependent test quantities (Modrak et al. 2023).

Post-fit diagnostics (Stage 5):
- Power-scaling sensitivity: detect prior-dominated or conflicting parameters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.model import NoiseFamily, SSMSpec
from causal_ssm_agent.models.ssm.utils import (
    _assemble_deterministics,
    _build_eval_fns,
    _discover_sites,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.inference import InferenceResult
    from causal_ssm_agent.models.ssm.model import SSMModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# T-rule (counting condition)
# ---------------------------------------------------------------------------


@dataclass
class TRuleResult:
    """Result of the t-rule (counting condition) check.

    The t-rule is a necessary condition for identification: if the number
    of free parameters exceeds the number of available moment conditions,
    the model is provably non-identified.

    For cross-sectional SEMs the constraint is n_params <= p(p+1)/2.
    For time series (SSMs), autocovariance at each lag provides p^2
    additional moment conditions, so the constraint is much weaker.
    """

    n_free_params: int
    n_manifest: int
    n_timepoints: int | None
    n_moments: int
    satisfies: bool
    param_counts: dict[str, int]
    hierarchical_warning: str | None = None

    def print_report(self) -> None:
        """Print a human-readable t-rule report."""
        tag = "[ok]" if self.satisfies else "[FAIL]"
        print("\n=== T-Rule (Counting Condition) ===")
        print(f"  {tag} {self.n_free_params} free params vs {self.n_moments} moment conditions")
        if self.n_timepoints is not None:
            print(f"  Time points: {self.n_timepoints}")
        print(f"  Manifest variables: {self.n_manifest}")
        print("  Parameter breakdown:")
        for name, count in sorted(self.param_counts.items()):
            print(f"    {name}: {count}")
        if self.hierarchical_warning:
            print(f"  WARNING: {self.hierarchical_warning}")


def count_free_params(spec: SSMSpec) -> dict[str, int]:
    """Count free parameters in an SSMSpec, matching the model's sampling logic.

    Returns a dict mapping parameter group name to the number of scalar
    free parameters in that group. Follows SSMModel._sample_* methods exactly.

    When drift_mask or lambda_mask is set, counts only the masked
    positions instead of assuming fully free matrices.
    """
    n_l, n_m = spec.n_latent, spec.n_manifest
    hier = spec.hierarchical and spec.n_subjects > 1
    n_s = spec.n_subjects if hier else 1
    counts: dict[str, int] = {}

    # -- Drift --
    if isinstance(spec.drift, str) and spec.drift == "free":
        counts["drift_diag_pop"] = n_l

        # Count off-diagonal entries from mask or assume all free
        if spec.drift_mask is not None:
            import numpy as np

            mask = np.asarray(spec.drift_mask)
            # Off-diagonal = mask True AND not on diagonal
            diag_mask = np.eye(n_l, dtype=bool)
            n_offdiag = int(np.sum(mask & ~diag_mask))
        else:
            n_offdiag = n_l * n_l - n_l

        if n_offdiag > 0:
            counts["drift_offdiag_pop"] = n_offdiag
        if hier and "drift_diag" in spec.indvarying:
            counts["drift_diag_sd"] = n_l
            counts["drift_diag_raw"] = n_s * n_l
        if hier and "drift_offdiag" in spec.indvarying and n_offdiag > 0:
            counts["drift_offdiag_sd"] = n_offdiag
            counts["drift_offdiag_raw"] = n_s * n_offdiag

    # -- Diffusion --
    if isinstance(spec.diffusion, str):
        counts["diffusion_diag_pop"] = n_l
        if spec.diffusion == "free":
            n_lower = n_l * (n_l - 1) // 2
            if n_lower > 0:
                counts["diffusion_lower"] = n_lower
        if hier and "diffusion" in spec.indvarying:
            counts["diffusion_diag_sd"] = n_l
            counts["diffusion_diag_raw"] = n_s * n_l

    # -- Continuous intercept --
    if spec.cint is not None and isinstance(spec.cint, str) and spec.cint == "free":
        counts["cint_pop"] = n_l
        if hier and "cint" in spec.indvarying:
            counts["cint_sd"] = n_l
            counts["cint_raw"] = n_s * n_l

    # -- Lambda (factor loadings) --
    if isinstance(spec.lambda_mat, str) and spec.lambda_mat == "free":
        n_free = max(0, n_m - n_l) * n_l
        if n_free > 0:
            counts["lambda_free"] = n_free
    elif spec.lambda_mask is not None:
        import numpy as np

        n_free = int(np.sum(spec.lambda_mask))
        if n_free > 0:
            counts["lambda_free"] = n_free

    # -- Manifest means --
    if isinstance(spec.manifest_means, str) and spec.manifest_means == "free":
        counts["manifest_means"] = n_m

    # -- Manifest variance (always diagonal in current impl) --
    if isinstance(spec.manifest_var, str):
        counts["manifest_var_diag"] = n_m

    # -- Initial state means --
    if isinstance(spec.t0_means, str) and spec.t0_means == "free":
        counts["t0_means_pop"] = n_l
        if hier and "t0_means" in spec.indvarying:
            counts["t0_means_sd"] = n_l
            counts["t0_means_raw"] = n_s * n_l

    # -- Initial state variance (always diagonal in current impl) --
    if isinstance(spec.t0_var, str):
        counts["t0_var_diag"] = n_l

    # -- Noise family hyperparameters --
    if spec.manifest_dist == NoiseFamily.STUDENT_T:
        counts["obs_df"] = 1
    if spec.manifest_dist == NoiseFamily.GAMMA:
        counts["obs_shape"] = 1
    if spec.diffusion_dist == NoiseFamily.STUDENT_T:
        counts["proc_df"] = 1

    return counts


def check_t_rule(spec: SSMSpec, T: int | None = None) -> TRuleResult:
    """Check the t-rule (necessary counting condition) for identification.

    The t-rule states that the number of free parameters must not exceed
    the number of independent moment conditions available from the data.

    For an SSM observed at T time points with p manifest variables:
    - Contemporaneous covariance: p(p+1)/2 unique entries
    - Mean structure: p equations
    - Autocovariance at each lag: p entries per lag (conservative), with T-1 lags

    This is a necessary but NOT sufficient condition. Passing does not
    guarantee identification; failing guarantees non-identification.

    Args:
        spec: SSMSpec instance
        T: Number of time points (if known). When None, uses only
           cross-sectional moments (conservative).

    Returns:
        TRuleResult with pass/fail and parameter breakdown
    """
    param_counts = count_free_params(spec)
    n_free = sum(param_counts.values())
    p = spec.n_manifest

    # Available moment conditions
    n_mean = p
    n_cov = p * (p + 1) // 2
    # Conservative bound: each lag contributes p distinct autocovariance
    # conditions. The full p*p cross-autocovariance matrix at each lag is
    # not fully independent due to symmetry constraints in the SSM structure,
    # so we use p per lag as a conservative lower bound.
    n_autocov = (T - 1) * p if T is not None and T > 1 else 0
    n_moments = n_mean + n_cov + n_autocov

    # Check for hierarchical parameter expansion (M13)
    hierarchical_warning = None
    if spec.hierarchical and spec.n_subjects > 1:
        hierarchical_warning = (
            f"Hierarchical model with {spec.n_subjects} subjects: the T-rule "
            f"parameter count ({n_free}) includes subject-level parameters from "
            f"count_free_params, but the moment condition count does not account "
            f"for the multi-subject data structure. Interpret with caution."
        )

    return TRuleResult(
        n_free_params=n_free,
        n_manifest=p,
        n_timepoints=T,
        n_moments=n_moments,
        satisfies=n_free <= n_moments,
        param_counts=param_counts,
        hierarchical_warning=hierarchical_warning,
    )


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

    Uses discretize_system_batched for CT->DT conversion, then lax.scan
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
# Helpers
# ---------------------------------------------------------------------------


def _simulate_from_params(con_dict, spec, times, rng_key):
    """Simulate observations from constrained parameter dict."""
    det = _assemble_deterministics({k: v[None, ...] for k, v in con_dict.items()}, spec)
    det = {k: v[0] for k, v in det.items()}
    n_l, n_m = spec.n_latent, spec.n_manifest
    return simulate_ssm(
        drift=det.get("drift", jnp.zeros((n_l, n_l))),
        diffusion_chol=det.get("diffusion", jnp.eye(n_l)),
        lambda_mat=det.get("lambda", jnp.eye(n_m, n_l)),
        manifest_chol=jnp.linalg.cholesky(
            det.get("manifest_cov", jnp.eye(n_m)) + jnp.eye(n_m) * 1e-8
        ),
        t0_means=det.get("t0_means", jnp.zeros(n_l)),
        t0_chol=jnp.linalg.cholesky(det.get("t0_cov", jnp.eye(n_l)) + jnp.eye(n_l) * 1e-8),
        times=times,
        rng_key=rng_key,
        cint=det.get("cint"),
        manifest_dist=spec.manifest_dist.value,
    )


def _chi_squared_uniformity_pvalue(ranks: jnp.ndarray, max_rank: int, n_bins: int) -> float:
    """Chi-squared uniformity test on discrete rank statistics.

    Uses regularized incomplete gamma for p-value (no scipy needed).
    """
    ranks = jnp.asarray(ranks, dtype=jnp.float32)
    n = ranks.shape[0]
    bin_width = (max_rank + 1) / n_bins
    bin_idx = jnp.clip((ranks / bin_width).astype(jnp.int32), 0, n_bins - 1)
    observed = jnp.array([float(jnp.sum(bin_idx == i)) for i in range(n_bins)], dtype=jnp.float32)
    expected = float(n) / n_bins
    chi2 = jnp.sum((observed - expected) ** 2 / jnp.maximum(expected, 1e-10))
    df = n_bins - 1
    return float(1.0 - jax.scipy.special.gammainc(df / 2.0, chi2 / 2.0))


def _build_scalar_names(param_names, site_info):
    """Build flat list of scalar element names from parameter groups."""
    names = []
    for name in param_names:
        size = int(jnp.prod(jnp.array(site_info[name]["shape"])))
        if size == 1:
            names.append(name)
        else:
            for k in range(size):
                names.append(f"{name}[{k}]")
    return names


def _build_param_index(param_names, site_info):
    """Build {param_name: (offset, size)} map into flat vector."""
    index = {}
    offset = 0
    for name in param_names:
        size = int(jnp.prod(jnp.array(site_info[name]["shape"])))
        index[name] = (offset, size)
        offset += size
    return index


def _sample_prior_unc(param_names, site_info, rng_key, n_samples=200):
    """Sample from prior in unconstrained space. Returns (n_samples, D) array."""
    samples = []
    for _ in range(n_samples):
        parts = []
        for name in param_names:
            info = site_info[name]
            rng_key, sk = random.split(rng_key)
            con = info["distribution"].sample(sk, ())
            unc = info["transform"].inv(con)
            parts.append(unc.reshape(-1))
        samples.append(jnp.concatenate(parts))
    return jnp.stack(samples), rng_key


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProfileLikelihoodResult:
    """Results from profile likelihood identifiability analysis."""

    parameter_profiles: dict[
        str, dict
    ]  # scalar_name -> {grid_unc, grid_con, profile_ll, mle_value}
    mle_ll: float  # MAP log-posterior
    mle_params: dict[str, jnp.ndarray]  # MAP parameter values (constrained)
    threshold: float  # chi-squared threshold (1.92 for 95%)
    parameter_names: list[str]  # scalar element names that were profiled

    def summary(self) -> dict[str, str]:
        """Per-parameter classification based on profile shape.

        Returns:
            Dict mapping scalar parameter name to one of:
            - "identified": profile drops below threshold on both sides
            - "practically_unidentifiable": doesn't cross threshold on one/both sides
            - "structurally_unidentifiable": profile is flat (range < 0.5)
        """
        eps = 0.5
        classifications = {}
        for name, prof in self.parameter_profiles.items():
            pll = jnp.asarray(prof["profile_ll"])
            pll_max = float(jnp.max(pll))
            ref = max(pll_max, self.mle_ll)
            ratio = pll - ref
            ll_range = float(pll_max - jnp.min(pll))

            if ll_range < eps:
                classifications[name] = "structurally_unidentifiable"
                continue

            peak = int(jnp.argmax(pll))
            left = ratio[:peak] if peak > 0 else jnp.array([0.0])
            right = ratio[peak + 1 :] if peak < len(pll) - 1 else jnp.array([0.0])
            left_ok = bool(jnp.any(left < -self.threshold))
            right_ok = bool(jnp.any(right < -self.threshold))

            if left_ok and right_ok:
                classifications[name] = "identified"
            else:
                classifications[name] = "practically_unidentifiable"

        return classifications

    def print_report(self) -> None:
        """Print a human-readable profile likelihood report."""
        summary = self.summary()
        markers = {
            "identified": "[ok]",
            "practically_unidentifiable": "[~]",
            "structurally_unidentifiable": "[!]",
        }
        print("\n=== Profile Likelihood Report ===")
        print(f"  Parameters profiled: {len(self.parameter_profiles)}")
        print(f"  Threshold: {self.threshold:.2f}")
        print(f"  MAP log-posterior: {self.mle_ll:.2f}")
        for name, cls in summary.items():
            print(f"  {markers.get(cls, '[?]')} {name}: {cls}")


@dataclass
class SBCResult:
    """Results from simulation-based calibration (Modrak et al. 2023)."""

    ranks: dict[str, jnp.ndarray]  # scalar_name -> (n_sbc,) rank stats
    likelihood_ranks: jnp.ndarray  # (n_sbc,) data-dependent test quantity
    n_sbc: int
    n_posterior_samples: int
    parameter_names: list[str]
    n_failed: int = 0
    n_attempted: int = 0

    def summary(self) -> dict[str, dict]:
        """Per-parameter uniformity test (chi-squared on binned ranks).

        Returns:
            Dict mapping param name -> {p_value, uniform, mean_rank, expected_mean}.
            Also includes "_likelihood" key for data-dependent test quantity.
        """
        result = {}
        n_bins = max(5, int(self.n_sbc**0.5))
        for name, r in self.ranks.items():
            pv = _chi_squared_uniformity_pvalue(r, self.n_posterior_samples, n_bins)
            result[name] = {
                "p_value": pv,
                "uniform": pv > 0.01,
                "mean_rank": float(jnp.mean(r)),
                "expected_mean": self.n_posterior_samples / 2.0,
            }
        ll_pv = _chi_squared_uniformity_pvalue(
            self.likelihood_ranks, self.n_posterior_samples, n_bins
        )
        result["_likelihood"] = {"p_value": ll_pv, "uniform": ll_pv > 0.01}
        return result

    def print_report(self) -> None:
        """Print a human-readable SBC report."""
        summary = self.summary()
        print(f"\n=== SBC Calibration Report (n={self.n_sbc}) ===")
        if self.n_failed > 0:
            print(
                f"  Replicates: {self.n_sbc} succeeded, {self.n_failed} failed "
                f"out of {self.n_attempted} attempted"
            )
        for name, info in summary.items():
            tag = "ok" if info["uniform"] else "FAIL"
            if name == "_likelihood":
                print(f"  [{tag}] likelihood: p={info['p_value']:.4f}")
            else:
                print(
                    f"  [{tag}] {name}: p={info['p_value']:.4f} (mean_rank={info['mean_rank']:.1f})"
                )


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
                f"-> {diag} (k_hat={k_hat:.2f}, {reliable})"
            )


# ---------------------------------------------------------------------------
# Pre-fit: profile_likelihood
# ---------------------------------------------------------------------------


def profile_likelihood(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    profile_params: list[str] | None = None,
    n_grid: int = 20,
    confidence: float = 0.95,
    seed: int = 42,
) -> ProfileLikelihoodResult:
    """Profile likelihood identifiability diagnostic.

    For each scalar parameter element:
    1. Fix the parameter at grid points around the MAP
    2. Optimize all other parameters (BFGS, 1st-order AD only)
    3. Classify based on profile shape vs chi-squared threshold

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        subject_ids: optional subject indices
        profile_params: parameter group names to profile (None = all)
        n_grid: number of grid points per parameter
        confidence: confidence level for threshold (0.95 or 0.99)
        seed: random seed

    Returns:
        ProfileLikelihoodResult with per-parameter profiles and classifications
    """
    rng_key = random.PRNGKey(seed)

    # 1. Discover sites
    backend = model.make_likelihood_backend()
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, subject_ids, trace_key, backend)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    D = flat_example.shape[0]
    param_names = sorted(site_info.keys())

    # 2. Build eval fns
    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model, observations, times, subject_ids, site_info, unravel_fn, backend
    )

    def neg_log_post(z):
        val = -(log_lik_fn(z) + log_prior_unc_fn(z))
        return jnp.where(jnp.isfinite(val), val, jnp.array(1e10))

    # 3. Prior stds in unconstrained space (for grid range)
    prior_z, rng_key = _sample_prior_unc(param_names, site_info, rng_key, n_samples=200)
    prior_stds = jnp.std(prior_z, axis=0)
    prior_stds = jnp.maximum(prior_stds, 0.1)

    # 4. Find MAP (optimize posterior for stability)
    z_init = jnp.median(prior_z, axis=0)
    map_result = jax.scipy.optimize.minimize(neg_log_post, z_init, method="BFGS")
    z_map = map_result.x
    if not jnp.all(jnp.isfinite(z_map)):
        z_map = z_init
    # Record log-LIKELIHOOD at MAP (not posterior) for profile comparison.
    # Raue et al. 2009: profile the likelihood to detect structural
    # non-identifiability; optimize the posterior for numerical stability.
    mle_ll = float(log_lik_fn(z_map))

    # 5. Parameter index map
    param_index = _build_param_index(param_names, site_info)
    scalar_names = _build_scalar_names(param_names, site_info)

    # 6. Determine which scalar indices to profile
    if profile_params is not None:
        indices = []
        for pname in profile_params:
            if pname in param_index:
                off, sz = param_index[pname]
                indices.extend(range(off, off + sz))
    else:
        indices = list(range(D))

    # Threshold: chi2(1, alpha)/2
    threshold = 3.32 if confidence >= 0.99 else 1.92

    # 7. Transforms for constrained mapping
    transforms = {name: site_info[name]["transform"] for name in site_info}
    unc_map = unravel_fn(z_map)

    # 8. Profile each scalar element
    parameter_profiles = {}

    for j in indices:
        sname = scalar_names[j]
        prior_std_j = float(prior_stds[j])
        z_map_j = float(z_map[j])

        grid_unc = jnp.linspace(
            z_map_j - 3 * prior_std_j,
            z_map_j + 3 * prior_std_j,
            n_grid,
        )

        profile_ll = []

        if D > 1:
            # Build JIT-compiled profiler for this j.
            # Optimize posterior (stable), return optimized z and LL.
            _j = j  # capture for closure

            @jax.jit
            def _profile_point(z_mj_init, z_j_val, _j=_j):
                def _obj(z_mj):
                    z_full = jnp.concatenate([z_mj[:_j], z_j_val[None], z_mj[_j:]])
                    return neg_log_post(z_full)

                res = jax.scipy.optimize.minimize(_obj, z_mj_init, method="BFGS")
                # Evaluate log-LIKELIHOOD (not posterior) at optimum
                z_opt = jnp.concatenate([res.x[:_j], z_j_val[None], res.x[_j:]])
                ll_val = log_lik_fn(z_opt)
                return res.x, ll_val

            z_mj_warm = jnp.concatenate([z_map[:j], z_map[j + 1 :]])

            for g_idx in range(n_grid):
                g_val = grid_unc[g_idx]
                z_mj_opt, ll_val = _profile_point(z_mj_warm, g_val)
                if jnp.all(jnp.isfinite(z_mj_opt)):
                    z_mj_warm = z_mj_opt
                profile_ll.append(float(ll_val))
        else:
            # D=1: no inner optimization, just evaluate likelihood
            for g_idx in range(n_grid):
                z_full = grid_unc[g_idx : g_idx + 1]
                profile_ll.append(float(log_lik_fn(z_full)))

        profile_ll = jnp.array(profile_ll)

        # Convert grid to constrained space
        # Find which param group owns this scalar index
        grid_con = grid_unc  # fallback
        mle_value = z_map_j
        for name in param_names:
            off, sz = param_index[name]
            if off <= j < off + sz:
                local_idx = j - off
                con_vals = []
                for g_val in grid_unc:
                    z_temp = z_map.at[j].set(g_val)
                    unc_dict = unravel_fn(z_temp)
                    con_val = transforms[name](unc_dict[name])
                    flat_con = con_val.reshape(-1)
                    con_vals.append(float(flat_con[local_idx]))
                grid_con = jnp.array(con_vals)
                # MLE value in constrained space
                con_map = transforms[name](unc_map[name])
                flat_map = con_map.reshape(-1)
                mle_value = float(flat_map[local_idx])
                break

        parameter_profiles[sname] = {
            "grid_unc": grid_unc,
            "grid_con": grid_con,
            "profile_ll": profile_ll,
            "mle_value": mle_value,
        }

    # MAP params in constrained space
    mle_params = {name: transforms[name](unc_map[name]) for name in unc_map}

    return ProfileLikelihoodResult(
        parameter_profiles=parameter_profiles,
        mle_ll=mle_ll,
        mle_params=mle_params,
        threshold=threshold,
        parameter_names=[scalar_names[j] for j in indices],
    )


# ---------------------------------------------------------------------------
# Pre-fit: sbc_check
# ---------------------------------------------------------------------------


def sbc_check(
    model: SSMModel,
    T: int = 100,
    dt: float = 0.5,
    n_sbc: int = 50,
    method: str = "laplace_em",
    seed: int = 42,
    **fit_kwargs,
) -> SBCResult:
    """Simulation-based calibration check (Modrak et al. 2023).

    For each replicate:
    1. Draw true params from prior
    2. Simulate data from true params
    3. Fit model to simulated data
    4. Compute rank of true value within posterior samples
    5. Compute rank of true log-likelihood among posterior log-likelihoods

    Well-calibrated posteriors produce uniform rank distributions.

    Args:
        model: SSMModel instance
        T: number of time points per replicate
        dt: time step between observations
        n_sbc: number of SBC replicates
        method: inference method for fitting
        seed: random seed
        **fit_kwargs: additional arguments passed to fit()

    Returns:
        SBCResult with rank statistics and uniformity tests
    """
    from causal_ssm_agent.models.ssm.inference import fit

    rng_key = random.PRNGKey(seed)
    times = jnp.arange(T, dtype=jnp.float32) * dt

    # Discover sites from dummy data
    backend = model.make_likelihood_backend()
    dummy_obs = jnp.zeros((T, model.spec.n_manifest))
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, dummy_obs, times, None, trace_key, backend)
    param_names = sorted(site_info.keys())

    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    _, unravel_fn = ravel_pytree(example_unc)

    param_index = _build_param_index(param_names, site_info)
    scalar_names = _build_scalar_names(param_names, site_info)

    all_ranks: dict[str, list[int]] = {sn: [] for sn in scalar_names}
    ll_ranks: list[int] = []
    n_post = 0
    n_failed = 0

    for _rep in range(n_sbc):
        # a. Draw true params from prior
        true_con = {}
        true_unc_parts = []
        for name in param_names:
            info = site_info[name]
            rng_key, sk = random.split(rng_key)
            con_sample = info["distribution"].sample(sk, ())
            true_con[name] = con_sample
            true_unc_parts.append(info["transform"].inv(con_sample).reshape(-1))
        true_z = jnp.concatenate(true_unc_parts)

        # b+c. Simulate data
        rng_key, sim_key = random.split(rng_key)
        try:
            y_star = _simulate_from_params(true_con, model.spec, times, sim_key)
        except Exception:
            n_failed += 1
            continue  # skip replicate on simulation failure

        if not jnp.all(jnp.isfinite(y_star)):
            n_failed += 1
            continue

        # d. Fit model
        rng_key, fit_key = random.split(rng_key)
        try:
            fit_result = fit(
                model, y_star, times, method=method, seed=int(fit_key[0]), **fit_kwargs
            )
        except Exception:
            n_failed += 1
            continue  # skip replicate on fit failure

        # e. Get posterior samples
        samples = fit_result.get_samples()
        if not samples:
            continue
        n_post = next(iter(samples.values())).shape[0]

        # Check which raw param names are available in samples
        available = [n for n in param_names if n in samples]

        # f. Compute parameter ranks (only for methods returning raw params)
        for name in available:
            _off, sz = param_index[name]
            true_flat = true_con[name].reshape(-1)
            post_flat = samples[name].reshape(n_post, -1)

            for k in range(sz):
                sname = name if sz == 1 else f"{name}[{k}]"
                rank = int(jnp.sum(post_flat[:, k] < true_flat[k]))
                all_ranks[sname].append(rank)

        # g. Likelihood rank (data-dependent test quantity)
        if available:
            # Can build unconstrained vectors from raw param samples
            log_lik_fn, _ = _build_eval_fns(
                model, y_star, times, None, site_info, unravel_fn, backend
            )
            true_ll = float(log_lik_fn(true_z))

            post_z_list = []
            for i in range(n_post):
                parts = []
                for name in param_names:
                    if name in samples:
                        unc = site_info[name]["transform"].inv(samples[name][i])
                        parts.append(unc.reshape(-1))
                if parts:
                    post_z_list.append(jnp.concatenate(parts))

            if post_z_list:
                post_z = jnp.stack(post_z_list)
                batch_ll = jax.vmap(log_lik_fn)
                post_lls = []
                chunk_size = 32
                for start in range(0, post_z.shape[0], chunk_size):
                    post_lls.append(batch_ll(post_z[start : start + chunk_size]))
                post_lls = jnp.concatenate(post_lls)
                ll_rank = int(jnp.sum(post_lls < true_ll))
            else:
                ll_rank = 0
        else:
            ll_rank = 0
        ll_ranks.append(ll_rank)

    # Warn if failure rate exceeds 20%
    n_attempted = n_sbc
    failure_rate = n_failed / n_attempted if n_attempted > 0 else 0.0
    if failure_rate > 0.2:
        logger.warning(
            "SBC: %d/%d replicates failed (%.0f%%). Results may be biased "
            "toward stable parameter regimes.",
            n_failed,
            n_attempted,
            failure_rate * 100,
        )

    # Filter out empty rank lists
    ranks_dict = {sn: jnp.array(v) for sn, v in all_ranks.items() if v}

    return SBCResult(
        ranks=ranks_dict,
        likelihood_ranks=jnp.array(ll_ranks) if ll_ranks else jnp.zeros(0),
        n_sbc=len(ll_ranks),
        n_posterior_samples=n_post,
        parameter_names=list(ranks_dict.keys()),
        n_failed=n_failed,
        n_attempted=n_attempted,
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
    backend = model.make_likelihood_backend()
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, subject_ids, trace_key, backend)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    _, unravel_fn = ravel_pytree(example_unc)

    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model, observations, times, subject_ids, site_info, unravel_fn, backend
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

    # 5. Compute per-parameter sensitivity
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

        # k-hat from prior weights via PSIS
        import arviz as az
        import numpy as np

        _, kss = az.psislw(np.asarray(prior_log_weights))
        psis_k_hat[name] = float(kss)

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
