"""Pre-fit parametric identifiability diagnostics for state-space models.

- T-rule (counting condition): necessary condition checking that the number
  of free parameters does not exceed available moment conditions.
- Output sensitivity analysis: Jacobian-based structural identifiability via SVD.
- Profile likelihood: per-parameter identifiability classification via
  constrained optimization (Raue et al. 2009). Uses only 1st-order AD.
- Simulation-based calibration (SBC): posterior calibration validation
  with data-dependent test quantities (Modrak et al. 2023).

Post-fit diagnostics (power-scaling sensitivity) are in parametric_id_postfit.py.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.optimize
import numpy as np
from jax import lax
from pydantic import BaseModel

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.likelihoods.base import CHOL_JITTER, NUMERICAL_EPSILON
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.parameterization import (
    SiteRuntimeBundle,
    assemble_deterministics_from_registry,
    build_prior_runtime_state,
    build_site_registry,
    build_site_runtime_bundle,
    sample_prior_unconstrained,
)
from causal_ssm_agent.models.ssm.utils import (
    _build_runtime_eval_fns_from_registry,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from causal_ssm_agent.models.ssm.model import SSMModel, SSMSpec

logger = get_prefect_logger(__name__)

# Chi-squared(1) critical values divided by 2, for profile likelihood thresholds.
# chi2(1, 0.05) / 2 = 3.84 / 2 = 1.92  (95% confidence)
# chi2(1, 0.01) / 2 = 6.635 / 2 ≈ 3.32 (99% confidence)
CHI2_THRESHOLD_95 = 1.92
CHI2_THRESHOLD_99 = 3.32
_STAGE4B_SWEEP_CONTEXT_CACHE_MAXSIZE = 8


@dataclass(frozen=True)
class Stage4bSweepContext:
    """Reusable topology-dependent Stage 4b runtime state.

    Delegates parameter-space metadata to :class:`SiteRuntimeBundle` to
    avoid duplicating registry, transforms, unravel_fn, etc.
    """

    cache_key: tuple[str, ...]
    spec: SSMSpec
    site_runtime: SiteRuntimeBundle
    predict_moments_fn: Callable
    jacobian_fn: Callable
    log_lik_fn: Callable
    log_prior_unc_fn: Callable

    # -- Convenience accessors delegating to site_runtime ------------------

    @property
    def registry(self):
        return self.site_runtime.registry

    @property
    def transforms(self):
        return self.site_runtime.transforms

    @property
    def flat_dim(self):
        return self.site_runtime.flat_dim

    @property
    def unravel_fn(self):
        return self.site_runtime.unravel_fn

    @property
    def param_names(self):
        return self.site_runtime.param_names

    @property
    def site_shapes(self):
        return self.site_runtime.site_shapes

    @property
    def scalar_names(self):
        return self.site_runtime.scalar_names

    @property
    def param_index(self):
        return self.site_runtime.param_index


_STAGE4B_SWEEP_CONTEXT_CACHE: OrderedDict[tuple[str, ...], Stage4bSweepContext] = OrderedDict()


def _normalize_sweep_cache_value(value: Any):
    """Convert spec/backend metadata into a stable JSON-serializable form."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jnp.ndarray):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, dict):
        return {
            str(key): _normalize_sweep_cache_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_sweep_cache_value(item) for item in value]
    return repr(value)


def _stage4b_context_key(model: SSMModel) -> tuple[str, ...]:
    """Build a process-local cache key for topology-stable Stage 4b sweeps."""
    spec_payload = {
        field_name: _normalize_sweep_cache_value(field_value)
        for field_name, field_value in vars(model.spec).items()
    }
    spec_fingerprint = hashlib.sha1(
        json.dumps(spec_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    pf_key = tuple(str(int(v)) for v in np.asarray(model.pf_key).reshape(-1))
    return (
        "stage4b-sweep",
        spec_fingerprint,
        str(model.likelihood),
        str(model.n_particles),
        "reparam:none",
        *pf_key,
    )


def clear_stage4b_sweep_context_cache() -> None:
    """Clear the process-local Stage 4b sweep context cache."""
    _STAGE4B_SWEEP_CONTEXT_CACHE.clear()


def get_stage4b_sweep_context(model: SSMModel) -> Stage4bSweepContext:
    """Build or reuse a topology-keyed Stage 4b sweep context."""
    cache_key = _stage4b_context_key(model)
    cached = _STAGE4B_SWEEP_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        _STAGE4B_SWEEP_CONTEXT_CACHE.move_to_end(cache_key)
        return cached

    site_runtime = build_site_runtime_bundle(model.spec, model._assembler)
    backend = model.make_likelihood_backend()
    log_lik_fn, log_prior_unc_fn = _build_runtime_eval_fns_from_registry(
        model.spec,
        site_runtime.registry,
        site_runtime.unravel_fn,
        site_runtime.transforms,
        backend,
    )

    def _predict(z_flat, times):
        return _predict_moments(
            z_flat,
            site_runtime.unravel_fn,
            site_runtime.transforms,
            model.spec,
            times,
            registry=site_runtime.registry,
        )

    context = Stage4bSweepContext(
        cache_key=cache_key,
        spec=model.spec,
        site_runtime=site_runtime,
        predict_moments_fn=_predict,
        jacobian_fn=jax.jit(jax.jacrev(_predict, argnums=0)),
        log_lik_fn=log_lik_fn,
        log_prior_unc_fn=log_prior_unc_fn,
    )
    _STAGE4B_SWEEP_CONTEXT_CACHE[cache_key] = context
    _STAGE4B_SWEEP_CONTEXT_CACHE.move_to_end(cache_key)
    while len(_STAGE4B_SWEEP_CONTEXT_CACHE) > _STAGE4B_SWEEP_CONTEXT_CACHE_MAXSIZE:
        _STAGE4B_SWEEP_CONTEXT_CACHE.popitem(last=False)
    return context


# ---------------------------------------------------------------------------
# T-rule (counting condition)
# ---------------------------------------------------------------------------


class TRuleResult(BaseModel):
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

    def print_report(self) -> None:
        """Log a human-readable t-rule report."""
        tag = "[ok]" if self.satisfies else "[FAIL]"
        lines = [
            "=== T-Rule (Counting Condition) ===",
            f"  {tag} {self.n_free_params} free params vs {self.n_moments} moment conditions",
        ]
        if self.n_timepoints is not None:
            lines.append(f"  Time points: {self.n_timepoints}")
        lines.append(f"  Manifest variables: {self.n_manifest}")
        lines.append("  Parameter breakdown:")
        for name, count in sorted(self.param_counts.items()):
            lines.append(f"    {name}: {count}")
        logger.info("\n%s", "\n".join(lines))


def count_free_params(spec: SSMSpec) -> dict[str, int]:
    """Count free parameters using the canonical site registry as authority."""
    counts: dict[str, int] = {}
    for site in build_site_registry(spec):
        counts[site.name] = int(np.prod(site.shape)) if site.shape else 1
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

    return TRuleResult(
        n_free_params=n_free,
        n_manifest=p,
        n_timepoints=T,
        n_moments=n_moments,
        satisfies=n_free <= n_moments,
        param_counts=param_counts,
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
    t0_chol_safe = jnp.linalg.cholesky(t0_cov + jnp.eye(n_latent) * CHOL_JITTER)
    x_0 = t0_means + t0_chol_safe @ random.normal(init_key, (n_latent,))

    # First observation from x_0
    rng_key, obs_key = random.split(rng_key)
    mu_0 = lambda_mat @ x_0 + manifest_means
    if manifest_dist == "poisson":
        y_0 = random.poisson(obs_key, jax.nn.softplus(mu_0)).astype(jnp.float32)
    else:
        manifest_chol_safe = jnp.linalg.cholesky(manifest_cov + jnp.eye(n_manifest) * CHOL_JITTER)
        y_0 = mu_0 + manifest_chol_safe @ random.normal(obs_key, (n_manifest,))

    # Scan over remaining timesteps
    def scan_fn(carry, inputs):
        x_prev, rng = carry
        Ad_t, Qd_t = inputs[0], inputs[1]
        cd_t = inputs[2]

        # State transition
        rng, state_key, obs_key = random.split(rng, 3)
        Qd_chol = jnp.linalg.cholesky(Qd_t + jnp.eye(n_latent) * CHOL_JITTER)
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


def _simulate_from_params(con_dict, spec, times, rng_key, *, registry=None):
    """Simulate observations from constrained parameter dict."""
    if registry is None:
        registry = build_site_registry(spec)
    det = assemble_deterministics_from_registry(
        {k: v[None, ...] for k, v in con_dict.items()}, spec, registry
    )
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
        t0_chol=jnp.linalg.cholesky(det.get("t0_cov", jnp.eye(n_l)) + jnp.eye(n_l) * CHOL_JITTER),
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
    chi2 = jnp.sum((observed - expected) ** 2 / jnp.maximum(expected, NUMERICAL_EPSILON))
    df = n_bins - 1
    return float(1.0 - jax.scipy.special.gammainc(df / 2.0, chi2 / 2.0))


# ---------------------------------------------------------------------------
# Output sensitivity analysis
# ---------------------------------------------------------------------------


def _predict_moments(z_flat, unravel_fn, transforms, spec, times, *, registry=None):
    """Predicted observation means and variances from unconstrained params.

    Runs Kalman prediction equations (no data update) to propagate state
    mean and covariance through time. Returns a flat vector of
    [means_flat, variances_flat] suitable for Jacobian computation.

    Captures both mean-dependent identifiability (drift, lambda, intercepts)
    and variance-dependent identifiability (diffusion, observation noise).
    Fully deterministic and JAX-differentiable.
    """
    if registry is None:
        registry = build_site_registry(spec)
    unc_dict = unravel_fn(z_flat)
    con_dict = {name: transforms[name](unc_dict[name]) for name in unc_dict}

    # Assemble matrices from constrained parameters (batch dim = 1)
    batched = {k: v[None, ...] for k, v in con_dict.items()}
    det = assemble_deterministics_from_registry(batched, spec, registry)
    det = {k: v[0] for k, v in det.items()}

    n_l, n_m = spec.n_latent, spec.n_manifest

    drift = det.get("drift", jnp.zeros((n_l, n_l)))
    diffusion_chol = det.get("diffusion", jnp.eye(n_l))
    diffusion_cov = diffusion_chol @ diffusion_chol.T
    t0_means = det.get("t0_means", jnp.zeros(n_l))
    t0_cov = det.get("t0_cov", jnp.eye(n_l))
    manifest_cov = det.get("manifest_cov", jnp.eye(n_m))

    # Lambda: may be in det (free) or fixed in spec
    lambda_val = det.get("lambda")
    if lambda_val is None:
        lambda_val = (
            spec.lambda_mat if isinstance(spec.lambda_mat, jnp.ndarray) else jnp.eye(n_m, n_l)
        )

    # Always provide cint for JAX-traceability
    cint = det.get("cint", jnp.zeros(n_l))

    # Discretize CT → DT
    dt_array = jnp.diff(times)
    Ad, Qd, cd = discretize_system_batched(drift, diffusion_cov, cint, dt_array)

    # Initial observation statistics
    x_0 = t0_means
    P_0 = t0_cov
    y_mean_0 = lambda_val @ x_0
    y_var_0 = jnp.diag(lambda_val @ P_0 @ lambda_val.T + manifest_cov)

    # Propagate state mean and covariance through time
    def scan_fn(carry, inputs):
        x_m, P = carry
        Ad_t, Qd_t, cd_t = inputs

        # State prediction
        x_m_next = Ad_t @ x_m + cd_t
        P_next = Ad_t @ P @ Ad_t.T + Qd_t

        # Observation statistics
        y_m = lambda_val @ x_m_next
        y_v = jnp.diag(lambda_val @ P_next @ lambda_val.T + manifest_cov)

        return (x_m_next, P_next), (y_m, y_v)

    _, (y_means_rest, y_vars_rest) = lax.scan(scan_fn, (x_0, P_0), (Ad, Qd, cd))

    y_means = jnp.concatenate([y_mean_0[None, :], y_means_rest], axis=0)
    y_vars = jnp.concatenate([y_var_0[None, :], y_vars_rest], axis=0)

    return jnp.concatenate([y_means.reshape(-1), y_vars.reshape(-1)])


@dataclass
class OutputSensitivityResult:
    """Results from output sensitivity analysis (pre-inference identifiability).

    Structural identifiability check via the Jacobian of the forward model's
    predicted means and variances. A full-rank sensitivity matrix indicates
    all parameters are locally identifiable. Near-zero singular values
    indicate parameter combinations that observations cannot distinguish.
    """

    singular_values: list[float]  # median SVD spectrum across draws (descending)
    condition_number: float  # median max_sv / min_sv
    per_parameter: list[dict]  # [{parameter, sensitivity_norm, identifiable}]
    n_draws: int
    n_observations: int  # output dimension (2 * T * D)
    n_parameters: int  # number of scalar free parameters

    def print_report(self) -> None:
        """Log a human-readable sensitivity analysis report."""
        n_nonsing = sum(1 for sv in self.singular_values if sv > NUMERICAL_EPSILON)
        lines = [
            "=== Output Sensitivity Analysis ===",
            f"  Parameters: {self.n_parameters}, Observations: {self.n_observations}",
            f"  Condition number: {self.condition_number:.2e}",
            f"  Prior draws: {self.n_draws}",
            f"  Rank: {n_nonsing}/{min(self.n_observations, self.n_parameters)}",
        ]
        for entry in self.per_parameter:
            tag = "[ok]" if entry["identifiable"] else "[!]"
            lines.append(f"  {tag} {entry['parameter']}: norm={entry['sensitivity_norm']:.4f}")
        logger.info("\n%s", "\n".join(lines))


def output_sensitivity_analysis(
    model: SSMModel,
    times: jnp.ndarray,
    n_draws: int = 8,
    seed: int = 42,
    sweep_context: Stage4bSweepContext | None = None,
) -> OutputSensitivityResult:
    """Pre-inference parametric identifiability via output sensitivity analysis.

    Computes the sensitivity matrix S[i,j] = dy_i / dtheta_j for the forward
    model's predicted observation means and variances (Kalman prediction
    equations without data update), then performs SVD to detect structurally
    non-identifiable parameter directions.

    Args:
        model: SSMModel instance
        times: (T,) observation times
        n_draws: Number of prior draws for robustness (default 8)
        seed: Random seed

    Returns:
        OutputSensitivityResult with SVD spectrum and per-parameter flags
    """
    rng_key = random.PRNGKey(seed)
    T_obs = times.shape[0]
    context = sweep_context or get_stage4b_sweep_context(model)
    n_manifest = context.spec.n_manifest

    # 1. Reuse topology-dependent registry metadata and rebuild only prior values.
    P = context.flat_dim
    scalar_names = context.scalar_names
    prior_state = build_prior_runtime_state(context.registry, model.priors)

    # 3. Sample from prior (Jacobian draws + larger batch for prior std)
    prior_z, rng_key = sample_prior_unconstrained(
        rng_key,
        context.registry,
        prior_state,
        n_samples=n_draws,
    )
    prior_z_std, rng_key = sample_prior_unconstrained(rng_key, context.registry, prior_state)
    prior_std = jnp.std(prior_z_std, axis=0)  # (P,) per-parameter prior SD
    # Guard against degenerate priors (zero std)
    prior_std = jnp.maximum(prior_std, NUMERICAL_EPSILON)

    N_out = 2 * T_obs * n_manifest

    # Helper to extract manifest_cov for a given parameter vector
    def _get_obs_noise_scales(z_0):
        unc_dict = context.unravel_fn(z_0)
        con_dict = {name: context.transforms[name](unc_dict[name]) for name in unc_dict}
        batched = {k: v[None, ...] for k, v in con_dict.items()}
        det = assemble_deterministics_from_registry(batched, context.spec, context.registry)
        det = {k: v[0] for k, v in det.items()}
        manifest_cov = det.get("manifest_cov", jnp.eye(n_manifest))
        obs_sd = jnp.sqrt(jnp.diag(manifest_cov))
        # Row scales: obs noise SD for mean rows, obs noise variance for var rows
        mean_scales = jnp.tile(obs_sd, T_obs)
        var_scales = jnp.tile(obs_sd**2, T_obs)
        return jnp.concatenate([mean_scales, var_scales])

    # Helper to compute per-parameter effective SVs from V matrix and sv vector
    def _per_param_effective_sv(V, sv):
        weight_threshold = 0.1
        effective = jnp.full(P, float(jnp.max(sv)))
        for k in range(P):
            significant = jnp.abs(V[k, :]) > weight_threshold
            if jnp.any(significant):
                effective = effective.at[k].set(
                    jnp.min(jnp.where(significant, sv[: V.shape[1]], jnp.inf))
                )
        return effective

    # 4. Compute Jacobian and SVD for each draw (raw + normalized)
    all_sv = []
    all_col_norms = []
    all_effective_sv = []
    all_norm_effective_sv = []  # normalized effective SVs

    for i in range(n_draws):
        z_0 = prior_z[i]
        S = context.jacobian_fn(z_0, times)  # (N_out, P) sensitivity matrix

        # --- Raw SVD ---
        _U, sv, Vt = jnp.linalg.svd(S, full_matrices=False)
        V = Vt.T  # (P, rank)
        all_sv.append(sv)
        all_col_norms.append(jnp.linalg.norm(S, axis=0))
        all_effective_sv.append(_per_param_effective_sv(V, sv))

        # --- Normalized SVD ---
        # S_norm[i,j] = (prior_std[j] / obs_scale[i]) * S[i,j]
        row_scales = _get_obs_noise_scales(z_0)
        row_scales = jnp.maximum(row_scales, NUMERICAL_EPSILON)
        S_norm = (prior_std[None, :] / row_scales[:, None]) * S
        _Un, sv_n, Vt_n = jnp.linalg.svd(S_norm, full_matrices=False)
        V_n = Vt_n.T
        all_norm_effective_sv.append(_per_param_effective_sv(V_n, sv_n))

    sv_matrix = jnp.stack(all_sv)  # (n_draws, rank)
    col_norm_matrix = jnp.stack(all_col_norms)  # (n_draws, P)
    eff_sv_matrix = jnp.stack(all_effective_sv)  # (n_draws, P)
    norm_eff_sv_matrix = jnp.stack(all_norm_effective_sv)  # (n_draws, P)

    # 5. Aggregate across draws (median for robustness to outlier draws)
    median_sv = jnp.median(sv_matrix, axis=0)
    median_col_norms = jnp.median(col_norm_matrix, axis=0)
    median_eff_sv = jnp.median(eff_sv_matrix, axis=0)
    median_norm_eff_sv = jnp.median(norm_eff_sv_matrix, axis=0)

    sv_max = float(jnp.max(median_sv))
    sv_min = float(jnp.min(median_sv))
    condition_number = sv_max / max(sv_min, 1e-30)

    # 6. Classify per-parameter identifiability
    # Raw effective SV: relative 3-decade gap thresholds (Joubert et al.)
    # Normalized effective SV: absolute thresholds (Fisher/prior scaling)
    per_param = []
    for k, sname in enumerate(scalar_names):
        norm_k = float(median_col_norms[k])
        eff_sv_k = float(median_eff_sv[k])
        norm_eff_sv_k = float(median_norm_eff_sv[k])

        # Raw: relative to max singular value
        if eff_sv_k > 1e-3 * sv_max:
            sv_status = "pass"
        elif eff_sv_k > 1e-6 * sv_max:
            sv_status = "warn"
        else:
            sv_status = "fail"

        # Normalized: absolute (units of prior-SD per noise-SD)
        if norm_eff_sv_k > 10:
            norm_sv_status = "pass"
        elif norm_eff_sv_k > 1:
            norm_sv_status = "warn"
        else:
            norm_sv_status = "fail"

        per_param.append(
            {
                "parameter": sname,
                "sensitivity_norm": norm_k,
                "effective_sv": eff_sv_k,
                "sv_status": sv_status,
                "normalized_effective_sv": norm_eff_sv_k,
                "normalized_sv_status": norm_sv_status,
                "identifiable": sv_status != "fail",
            }
        )

    return OutputSensitivityResult(
        singular_values=[float(v) for v in median_sv],
        condition_number=condition_number,
        per_parameter=per_param,
        n_draws=n_draws,
        n_observations=N_out,
        n_parameters=P,
    )


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
    threshold: float  # chi-squared threshold (CHI2_THRESHOLD_95 or CHI2_THRESHOLD_99)
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
        """Log a human-readable profile likelihood report."""
        summary = self.summary()
        markers = {
            "identified": "[ok]",
            "practically_unidentifiable": "[~]",
            "structurally_unidentifiable": "[!]",
        }
        lines = [
            "=== Profile Likelihood Report ===",
            f"  Parameters profiled: {len(self.parameter_profiles)}",
            f"  Threshold: {self.threshold:.2f}",
            f"  MAP log-posterior: {self.mle_ll:.2f}",
        ]
        for name, cls in summary.items():
            lines.append(f"  {markers.get(cls, '[?]')} {name}: {cls}")
        logger.info("\n%s", "\n".join(lines))


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
        """Log a human-readable SBC report."""
        summary = self.summary()
        lines = [f"=== SBC Calibration Report (n={self.n_sbc}) ==="]
        if self.n_failed > 0:
            lines.append(
                f"  Replicates: {self.n_sbc} succeeded, {self.n_failed} failed "
                f"out of {self.n_attempted} attempted"
            )
        for name, info in summary.items():
            tag = "ok" if info["uniform"] else "FAIL"
            if name == "_likelihood":
                lines.append(f"  [{tag}] likelihood: p={info['p_value']:.4f}")
            else:
                lines.append(
                    f"  [{tag}] {name}: p={info['p_value']:.4f} (mean_rank={info['mean_rank']:.1f})"
                )
        logger.info("\n%s", "\n".join(lines))


# ---------------------------------------------------------------------------
# Pre-fit: profile_likelihood
# ---------------------------------------------------------------------------


def profile_likelihood(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    profile_params: list[str] | None = None,
    profile_indices: list[int] | None = None,
    n_grid: int = 20,
    confidence: float = 0.95,
    seed: int = 42,
    sweep_context: Stage4bSweepContext | None = None,
) -> ProfileLikelihoodResult:
    """Profile likelihood identifiability diagnostic.

    For each scalar parameter element:
    1. Fix the parameter at grid points around the MAP
    2. Optimize all other parameters (BFGS, 1st-order AD only)
    3. Classify based on profile shape vs chi-squared threshold

    Uses the canonical site registry and compile-stable prior evaluation.
    Changing prior values or families does not trigger JAX recompilation
    as long as the model topology (shapes, support classes) is unchanged.

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        profile_params: parameter group names to profile (None = all)
        profile_indices: scalar indices into the flat parameter vector to
            profile. Overrides profile_params when set. Used by the RB
            partition to restrict profiling to Kalman-block parameters.
        n_grid: number of grid points per parameter
        confidence: confidence level for threshold (0.95 or 0.99)
        seed: random seed

    Returns:
        ProfileLikelihoodResult with per-parameter profiles and classifications
    """
    rng_key = random.PRNGKey(seed)
    context = sweep_context or get_stage4b_sweep_context(model)

    # 1. Reuse topology-dependent registry/evaluator state; rebuild only priors.
    D = context.flat_dim
    param_names = context.param_names
    prior_state = build_prior_runtime_state(context.registry, model.priors)

    def neg_log_post(z, ps):
        val = -(context.log_lik_fn(z, observations, times) + context.log_prior_unc_fn(z, ps))
        return jnp.where(jnp.isfinite(val), val, jnp.array(1e10))

    # 3. Prior stds in unconstrained space (for grid range)
    prior_z, rng_key = sample_prior_unconstrained(rng_key, context.registry, prior_state)
    prior_stds = jnp.std(prior_z, axis=0)
    prior_stds = jnp.maximum(prior_stds, 0.1)

    # 4. Find MAP (optimize posterior for stability)
    z_init = jnp.median(prior_z, axis=0)
    map_result = jax.scipy.optimize.minimize(
        lambda z: neg_log_post(z, prior_state), z_init, method="BFGS"
    )
    z_map = map_result.x
    if not jnp.all(jnp.isfinite(z_map)):
        z_map = z_init
    # Record log-LIKELIHOOD at MAP (not posterior) for profile comparison.
    # Raue et al. 2009: profile the likelihood to detect structural
    # non-identifiability; optimize the posterior for numerical stability.
    mle_ll = float(context.log_lik_fn(z_map, observations, times))

    # 5. Parameter index map
    param_index = context.param_index
    scalar_names = context.scalar_names

    # 6. Determine which scalar indices to profile
    if profile_indices is not None:
        indices = [i for i in profile_indices if i < D]
    elif profile_params is not None:
        indices = []
        for pname in profile_params:
            if pname in param_index:
                off, sz = param_index[pname]
                indices.extend(range(off, off + sz))
    else:
        indices = list(range(D))

    # Threshold: chi2(1, alpha)/2
    threshold = CHI2_THRESHOLD_99 if confidence >= 0.99 else CHI2_THRESHOLD_95

    # 7. Transforms for constrained mapping
    unc_map = context.unravel_fn(z_map)

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
            # prior_state is a JIT argument — changing it does NOT recompile.
            _j = j  # capture for closure

            @jax.jit
            def _profile_point(z_mj_init, z_j_val, ps, _j=_j):
                def _obj(z_mj):
                    z_full = jnp.concatenate([z_mj[:_j], z_j_val[None], z_mj[_j:]])
                    return neg_log_post(z_full, ps)

                res = jax.scipy.optimize.minimize(_obj, z_mj_init, method="BFGS")
                # Evaluate log-LIKELIHOOD (not posterior) at optimum
                z_opt = jnp.concatenate([res.x[:_j], z_j_val[None], res.x[_j:]])
                ll_val = context.log_lik_fn(z_opt, observations, times)
                return res.x, ll_val

            z_mj_warm = jnp.concatenate([z_map[:j], z_map[j + 1 :]])

            for g_idx in range(n_grid):
                g_val = grid_unc[g_idx]
                z_mj_opt, ll_val = _profile_point(z_mj_warm, g_val, prior_state)
                if jnp.all(jnp.isfinite(z_mj_opt)):
                    z_mj_warm = z_mj_opt
                profile_ll.append(float(ll_val))
        else:
            # D=1: no inner optimization, just evaluate likelihood
            for g_idx in range(n_grid):
                z_full = grid_unc[g_idx : g_idx + 1]
                profile_ll.append(float(context.log_lik_fn(z_full, observations, times)))

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
                    unc_dict = context.unravel_fn(z_temp)
                    con_val = context.transforms[name](unc_dict[name])
                    flat_con = con_val.reshape(-1)
                    con_vals.append(float(flat_con[local_idx]))
                grid_con = jnp.array(con_vals)
                # MLE value in constrained space
                con_map = context.transforms[name](unc_map[name])
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
    mle_params = {name: context.transforms[name](unc_map[name]) for name in unc_map}

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
    method: Literal[
        "svi",
        "nuts",
        "nuts_da",
        "hessmc2",
        "pgas",
        "tempered_smc",
        "laplace_em",
        "structured_vi",
        "dpf",
    ] = "laplace_em",
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

    # Build registry-based runtime (no model tracing needed).
    # The log_lik_fn is compiled once and reused across all replicates.
    site_runtime = build_site_runtime_bundle(model.spec, model._assembler)
    prior_state = build_prior_runtime_state(site_runtime.registry, model.priors)
    backend = model.make_likelihood_backend()
    log_lik_fn, _ = _build_runtime_eval_fns_from_registry(
        model.spec,
        site_runtime.registry,
        site_runtime.unravel_fn,
        site_runtime.transforms,
        backend,
    )

    param_names = site_runtime.param_names
    param_index = site_runtime.param_index
    scalar_names = site_runtime.scalar_names
    registry = site_runtime.registry

    all_ranks: dict[str, list[int]] = {sn: [] for sn in scalar_names}
    ll_ranks: list[int] = []
    n_post = 0
    n_failed = 0

    for rep in range(n_sbc):
        # a. Draw true params from prior (unconstrained, then constrain)
        prior_z, rng_key = sample_prior_unconstrained(rng_key, registry, prior_state, n_samples=1)
        true_z = prior_z[0]  # (D,)
        true_con = site_runtime.constrain(true_z)

        # b+c. Simulate data
        rng_key, sim_key = random.split(rng_key)
        try:
            y_star = _simulate_from_params(true_con, model.spec, times, sim_key, registry=registry)
        except Exception:
            logger.debug("SBC replicate %d: simulation failed", rep, exc_info=True)
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
            logger.debug("SBC replicate %d: fit failed", rep, exc_info=True)
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

        # g. Likelihood rank (reuse compile-stable log_lik_fn)
        if available:
            true_ll = float(log_lik_fn(true_z, y_star, times))

            post_z_list = []
            for i in range(n_post):
                parts = []
                for name in param_names:
                    if name in samples:
                        unc = site_runtime.transforms[name].inv(samples[name][i])
                        parts.append(unc.reshape(-1))
                if parts:
                    post_z_list.append(jnp.concatenate(parts))

            if post_z_list:
                post_z = jnp.stack(post_z_list)
                batch_ll = jax.vmap(log_lik_fn, in_axes=(0, None, None))
                post_lls = []
                chunk_size = 32
                for start in range(0, post_z.shape[0], chunk_size):
                    post_lls.append(batch_ll(post_z[start : start + chunk_size], y_star, times))
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
