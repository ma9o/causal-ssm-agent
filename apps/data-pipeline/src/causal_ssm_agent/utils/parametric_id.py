"""Pre-fit parametric identifiability diagnostics for state-space models.

- T-rule (counting screen): conservative parameter-vs-moment comparison used
  as a cheap pre-fit warning signal.
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
import re
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.optimize
import jax.scipy.stats as jstats
import numpy as np
from jax import lax
from pydantic import BaseModel

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference.targets.base import (
    CHOL_JITTER,
    NUMERICAL_EPSILON,
    PROB_CLIP_MIN,
)
from causal_ssm_agent.models.ssm.inference.utils import (
    _build_runtime_eval_fns_from_registry,
)
from causal_ssm_agent.models.ssm.parameterization import (
    SiteRuntimeBundle,
    assemble_deterministics_from_registry,
    build_site_registry,
    build_site_runtime_bundle,
    sample_prior_unconstrained,
)
from causal_ssm_agent.models.ssm.structure_runtime import SSMStructureRuntime
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

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
_SCALAR_PARAMETER_INDEX_RE = re.compile(r"^(?P<site>.+)\[(?P<index>\d+)\]$")


@dataclass(frozen=True)
class Stage4bSweepContext:
    """Reusable topology-dependent Stage 4b runtime state.

    Delegates parameter-space metadata to :class:`SiteRuntimeBundle` to
    avoid duplicating registry, transforms, unravel_fn, etc.
    """

    cache_key: tuple[str, ...]
    spec: SSMSpec
    structure_runtime: SSMStructureRuntime
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
    observation_support = getattr(model, "observation_support", None)
    support_payload = (
        None
        if observation_support is None
        else {
            field_name: _normalize_sweep_cache_value(field_value)
            for field_name, field_value in vars(observation_support).items()
        }
    )
    support_fingerprint = hashlib.sha1(
        json.dumps(support_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    pf_key = tuple(str(int(v)) for v in np.asarray(model.pf_key).reshape(-1))
    return (
        "stage4b-sweep",
        spec_fingerprint,
        support_fingerprint,
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

    site_runtime = build_site_runtime_bundle(model.spec, model._structure_runtime)
    backend = model.make_likelihood_backend()
    log_lik_fn, log_prior_unc_fn = _build_runtime_eval_fns_from_registry(
        model.spec,
        site_runtime.registry,
        site_runtime.unravel_fn,
        site_runtime.transforms,
        model._structure_runtime,
        backend,
    )

    def _predict(z_flat, times):
        return _predict_observation_moments(
            z_flat,
            site_runtime.unravel_fn,
            site_runtime.transforms,
            model.spec,
            times,
            structure_runtime=model._structure_runtime,
            observation_support=getattr(model, "observation_support", None),
            registry=site_runtime.registry,
        )

    context = Stage4bSweepContext(
        cache_key=cache_key,
        spec=model.spec,
        structure_runtime=model._structure_runtime,
        site_runtime=site_runtime,
        predict_moments_fn=_predict,
        jacobian_fn=jax.jit(jax.jacfwd(_predict, argnums=0)),
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

    This implementation uses a conservative lower bound on available moment
    conditions. If the number of free parameters exceeds that lower bound,
    the model is at high risk of non-identifiability and should be reviewed,
    but the result is not treated as a proof.

    For cross-sectional SEMs the constraint is n_params <= p(p+1)/2.
    For time series (SSMs), lagged autocovariance contributes additional
    information; this implementation counts only a conservative p moments
    per lag.
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

    This implementation is conservative: it uses a lower bound on the
    available lagged moment conditions. Passing still does not guarantee
    identification, and failing is treated as a warning signal rather than
    a proof.

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
    manifest_dists: list[DistributionFamily | str] | None = None,
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
        manifest_dists: per-channel observation families. Currently supports
            Gaussian and Poisson channels.

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

    resolved_manifest_dists = (
        [
            dist if isinstance(dist, DistributionFamily) else DistributionFamily(dist)
            for dist in manifest_dists
        ]
        if manifest_dists is not None
        else [DistributionFamily.GAUSSIAN] * n_manifest
    )
    if len(resolved_manifest_dists) != n_manifest:
        raise ValueError(
            "manifest_dists length must match n_manifest: "
            f"{len(resolved_manifest_dists)} vs {n_manifest}"
        )
    unsupported_manifest_dists = sorted(
        {
            dist.value
            for dist in resolved_manifest_dists
            if dist not in {DistributionFamily.GAUSSIAN, DistributionFamily.POISSON}
        }
    )
    if unsupported_manifest_dists:
        raise ValueError(
            "simulate_ssm only supports gaussian/poisson manifest_dists. "
            f"Got {unsupported_manifest_dists}."
        )
    poisson_mask = [dist == DistributionFamily.POISSON for dist in resolved_manifest_dists]
    poisson_mask_array = jnp.asarray(poisson_mask, dtype=bool)
    all_gaussian = not any(poisson_mask)

    # Sample initial state
    rng_key, init_key = random.split(rng_key)
    t0_chol_safe = jnp.linalg.cholesky(t0_cov + jnp.eye(n_latent) * CHOL_JITTER)
    x_0 = t0_means + t0_chol_safe @ random.normal(init_key, (n_latent,))

    if all_gaussian:
        manifest_chol_safe = jnp.linalg.cholesky(manifest_cov + jnp.eye(n_manifest) * CHOL_JITTER)

        def _sample_observation(key: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
            return mean + manifest_chol_safe @ random.normal(key, (n_manifest,))

    else:
        manifest_sd = jnp.sqrt(jnp.maximum(jnp.diag(manifest_cov), CHOL_JITTER))

        def _sample_observation(key: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
            gaussian_key, poisson_key = random.split(key)
            gaussian_obs = mean + manifest_sd * random.normal(gaussian_key, (n_manifest,))
            poisson_obs = random.poisson(poisson_key, jax.nn.softplus(mean)).astype(jnp.float32)
            return jnp.where(poisson_mask_array, poisson_obs, gaussian_obs)

    # First observation from x_0
    rng_key, obs_key = random.split(rng_key)
    mu_0 = lambda_mat @ x_0 + manifest_means
    y_0 = _sample_observation(obs_key, mu_0)

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
        y_t = _sample_observation(obs_key, mu_t)

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
        manifest_dists=[dist.value for dist in spec.manifest_dists],
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


def _assemble_sensitivity_measurement_state(
    z_flat,
    unravel_fn,
    transforms,
    spec,
    *,
    structure_runtime: SSMStructureRuntime,
    registry,
):
    """Assemble deterministic matrices and observation hyperparameters for one draw."""
    unc_dict = unravel_fn(z_flat)
    con_dict = {name: transforms[name](unc_dict[name]) for name in unc_dict}

    batched = {k: v[None, ...] for k, v in con_dict.items()}
    det = assemble_deterministics_from_registry(
        batched,
        spec,
        registry,
        structure_runtime=structure_runtime,
    )
    det = {k: v[0] for k, v in det.items()}
    from causal_ssm_agent.models.ssm.model import assemble_sampled_extra_params

    extra_params = assemble_sampled_extra_params(spec, con_dict)
    return det, extra_params


def _response_latent_variance_diag(
    eta_mean: jnp.ndarray,
    eta_cov: jnp.ndarray,
    response_mean: jnp.ndarray,
    *,
    obs_kernel,
    manifest_dists,
    manifest_links,
) -> jnp.ndarray:
    """Approximate latent-induced variance on the observation-mean scale."""
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

    unsupported_response_families = {
        DistributionFamily.ORDERED_LOGISTIC,
        DistributionFamily.CATEGORICAL,
    }
    if not any(dist in unsupported_response_families for dist in manifest_dists):
        eta_var_diag = jnp.maximum(jnp.diag(eta_cov), 0.0)
        deriv = []
        for idx, link in enumerate(manifest_links):
            eta_j = eta_mean[idx]
            mean_j = response_mean[idx]
            if link == LinkFunction.IDENTITY:
                deriv.append(jnp.asarray(1.0, dtype=eta_mean.dtype))
            elif link == LinkFunction.LOG:
                deriv.append(mean_j)
            elif link == LinkFunction.LOGIT:
                deriv.append(mean_j * (1.0 - mean_j))
            elif link == LinkFunction.PROBIT:
                deriv.append(jstats.norm.pdf(eta_j))
            elif link == LinkFunction.INVERSE:
                deriv.append(
                    jnp.where(eta_j > 0.0, -1.0 / (eta_j**2), jnp.nan).astype(eta_mean.dtype)
                )
            else:
                raise OutputSensitivityUnsupportedError(
                    f"output sensitivity does not support link={link.value!r}"
                )
        deriv_vec = jnp.stack(deriv)
        return jnp.square(deriv_vec) * eta_var_diag

    response_jacobian = jax.jacfwd(obs_kernel.response_fn)(eta_mean)
    response_cov = response_jacobian @ eta_cov @ response_jacobian.T
    return jnp.maximum(jnp.diag(response_cov), 0.0)


def _symmetrize_covariance(cov: jnp.ndarray) -> jnp.ndarray:
    """Numerically symmetrize one covariance-like matrix."""
    return 0.5 * (cov + cov.T)


def _response_latent_covariance(
    eta_mean: jnp.ndarray,
    eta_cov: jnp.ndarray,
    *,
    obs_kernel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Approximate response-scale covariance and Jacobian at one predictor mean."""
    response_jacobian = jax.jacfwd(obs_kernel.response_fn)(eta_mean)
    response_cov = response_jacobian @ eta_cov @ response_jacobian.T
    response_cov = _symmetrize_covariance(response_cov)
    diag_idx = jnp.diag_indices(response_cov.shape[0])
    response_cov = response_cov.at[diag_idx].set(jnp.maximum(jnp.diag(response_cov), 0.0))
    return response_cov, response_jacobian


def _observation_noise_variance_arguments(
    eta_mean: jnp.ndarray,
    response_mean: jnp.ndarray,
    *,
    manifest_dists,
) -> jnp.ndarray:
    """Build the per-channel input vector expected by ``obs_kernel.variance_fn``."""
    variance_args = []
    for idx, dist in enumerate(manifest_dists):
        if dist in {
            DistributionFamily.ORDERED_LOGISTIC,
            DistributionFamily.CATEGORICAL,
        }:
            variance_args.append(eta_mean[idx])
        else:
            variance_args.append(response_mean[idx])
    return jnp.stack(variance_args)


def _observation_noise_covariance(
    variance_args: jnp.ndarray,
    *,
    obs_kernel,
) -> jnp.ndarray:
    """Return one same-row observation-noise covariance matrix."""
    observation_noise_cov = _symmetrize_covariance(obs_kernel.variance_fn(variance_args))
    diag_idx = jnp.diag_indices(observation_noise_cov.shape[0])
    observation_noise_cov = observation_noise_cov.at[diag_idx].set(
        jnp.maximum(jnp.diag(observation_noise_cov), NUMERICAL_EPSILON)
    )
    return observation_noise_cov


def _extra_param_at(
    extra_params: dict,
    key: str,
    index: int,
    default: float,
) -> jnp.ndarray:
    """Return one scalar observation hyperparameter, broadcasting shared values."""
    value = extra_params.get(key, default)
    value_arr = jnp.asarray(value)
    if value_arr.ndim == 0:
        return value_arr
    return value_arr[index]


def _point_observation_noise_var_diag(
    eta_mean: jnp.ndarray,
    response_mean: jnp.ndarray,
    *,
    manifest_dists,
    manifest_cov: jnp.ndarray,
    extra_params: dict,
    allow_discrete_mean_space: bool,
) -> jnp.ndarray:
    """Return diagonal observation noise variances on the emitted observation scale."""
    from causal_ssm_agent.models.ssm.inference.targets.emissions import (
        categorical_moments,
        ordered_logistic_moments,
    )
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    variances = []
    for idx, dist in enumerate(manifest_dists):
        mean_j = response_mean[idx]
        eta_j = eta_mean[idx]
        if dist in {DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T}:
            variances.append(manifest_cov[idx, idx])
        elif dist == DistributionFamily.POISSON:
            variances.append(jnp.maximum(mean_j, NUMERICAL_EPSILON))
        elif dist == DistributionFamily.GAMMA:
            shape = _extra_param_at(extra_params, "obs_shape", idx, 1.0)
            mu = jnp.maximum(mean_j, NUMERICAL_EPSILON)
            variances.append(mu**2 / jnp.maximum(shape, NUMERICAL_EPSILON))
        elif dist == DistributionFamily.BERNOULLI:
            p = jnp.clip(mean_j, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
            variances.append(p * (1.0 - p))
        elif dist == DistributionFamily.NEGATIVE_BINOMIAL:
            r = _extra_param_at(extra_params, "obs_r", idx, 5.0)
            mu = jnp.maximum(mean_j, NUMERICAL_EPSILON)
            variances.append(mu + mu**2 / jnp.maximum(r, NUMERICAL_EPSILON))
        elif dist == DistributionFamily.BETA:
            concentration = _extra_param_at(extra_params, "obs_concentration", idx, 10.0)
            p = jnp.clip(mean_j, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
            variances.append(p * (1.0 - p) / (jnp.maximum(concentration, NUMERICAL_EPSILON) + 1.0))
        elif dist == DistributionFamily.ORDERED_LOGISTIC:
            if not allow_discrete_mean_space:
                raise OutputSensitivityUnsupportedError(
                    "interval-summary sensitivity is not defined for ordered_logistic observations"
                )
            level_counts = jnp.asarray(extra_params["obs_level_counts"], dtype=jnp.int32)[
                idx : idx + 1
            ]
            cutpoints = jnp.asarray(extra_params["obs_ordered_cutpoints"])[idx : idx + 1]
            _mean, variance = ordered_logistic_moments(
                jnp.asarray([eta_j]),
                cutpoints,
                level_counts,
            )
            variances.append(jnp.maximum(variance[0], NUMERICAL_EPSILON))
        elif dist == DistributionFamily.CATEGORICAL:
            if not allow_discrete_mean_space:
                raise OutputSensitivityUnsupportedError(
                    "interval-summary sensitivity is not defined for categorical observations"
                )
            level_counts = jnp.asarray(extra_params["obs_level_counts"], dtype=jnp.int32)[
                idx : idx + 1
            ]
            intercepts = jnp.asarray(extra_params["obs_cat_intercepts"])[idx : idx + 1]
            slopes = jnp.asarray(extra_params["obs_cat_slopes"])[idx : idx + 1]
            _mean, variance = categorical_moments(
                jnp.asarray([eta_j]),
                intercepts,
                slopes,
                level_counts,
            )
            variances.append(jnp.maximum(variance[0], NUMERICAL_EPSILON))
        else:
            raise OutputSensitivityUnsupportedError(
                f"output sensitivity does not support manifest_dist={dist.value!r}"
            )

    return jnp.stack(variances)


def _select_support_slot(stat: jnp.ndarray, emission_slot_indices: jnp.ndarray) -> jnp.ndarray:
    """Gather one accumulator statistic from the active interval slot."""
    safe_indices = jnp.clip(emission_slot_indices, 0, stat.shape[-1] - 1)
    selected = jnp.take_along_axis(
        stat,
        jnp.expand_dims(safe_indices, axis=-1),
        axis=-1,
    ).squeeze(-1)
    valid = emission_slot_indices >= 0
    return jnp.where(valid, selected, jnp.zeros_like(selected))


def _reset_support_stat(
    stat: jnp.ndarray,
    emission_slot_indices: jnp.ndarray,
    emit_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Reset one accumulator statistic after an interval-summary emission."""
    safe_indices = jnp.clip(emission_slot_indices, 0, stat.shape[-1] - 1)
    slot_ids = jnp.arange(stat.shape[-1], dtype=safe_indices.dtype)
    while slot_ids.ndim < stat.ndim:
        slot_ids = slot_ids.reshape((1,) * (stat.ndim - 1) + (-1,))
    reset_mask = jnp.expand_dims(emit_mask > 0.5, axis=-1) & (
        slot_ids == jnp.expand_dims(safe_indices, axis=-1)
    )
    return jnp.where(reset_mask, jnp.zeros_like(stat), stat)


def _project_response_moments(
    response_means: jnp.ndarray,
    response_vars: jnp.ndarray,
    observation_operator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project response-space first/second moments into emitted observation moments.

    The latent variance projection uses a diagonal approximation in time and a
    Gaussian moment closure for the squared-response statistics that feed the
    ``std`` summary operator.
    """
    from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
        _COUNT_OPERATOR_CODE,
        _MEAN_OPERATOR_CODE,
        _STD_OPERATOR_CODE,
        _SUM_OPERATOR_CODE,
    )

    dtype = response_means.dtype
    if not observation_operator.requires_interval_summary_handling:
        return (
            response_means,
            response_vars,
            jnp.ones_like(response_means, dtype=dtype),
        )

    support = observation_operator.observation_support
    assert support is not None
    assert observation_operator.summary_operator_codes is not None
    assert observation_operator.prev_coeffs is not None
    assert observation_operator.curr_coeffs is not None
    assert observation_operator.interval_weights is not None
    assert observation_operator.emission_slots is not None

    T, n_manifest = response_means.shape
    point_like_mask = observation_operator.point_like_mask(dtype)
    interval_summary_mask = observation_operator.interval_summary_mask(dtype)
    emission_slots = jnp.asarray(observation_operator.emission_slots, dtype=jnp.int32)
    summary_codes = observation_operator.summary_operator_codes
    semantic_mask_0 = point_like_mask + interval_summary_mask * (emission_slots[0] >= 0).astype(
        dtype
    )

    if T == 1:
        return response_means, response_vars, semantic_mask_0[None, :]

    prev_coeffs = jnp.asarray(observation_operator.prev_coeffs, dtype=dtype)
    curr_coeffs = jnp.asarray(observation_operator.curr_coeffs, dtype=dtype)
    interval_weights = jnp.asarray(observation_operator.interval_weights, dtype=dtype)
    zeros = observation_operator.empty_accumulators(dtype)
    full_obs_mask = jnp.ones((n_manifest,), dtype=dtype)

    def _second_stats(mean_t: jnp.ndarray, var_t: jnp.ndarray):
        second_mean = mean_t**2 + var_t
        second_var = jnp.maximum(2.0 * var_t**2 + 4.0 * mean_t**2 * var_t, 0.0)
        cov = 2.0 * mean_t * var_t
        return second_mean, second_var, cov

    weight_zeros = observation_operator.empty_accumulators(dtype)

    def _scan_step_with_weight(carry, inputs):
        (
            response_prev,
            response_var_prev,
            accum_sum_mean,
            accum_sum_var,
            accum_sumsq_mean,
            accum_sumsq_var,
            accum_sum_sumsq_cov,
            accum_weight,
        ) = carry
        response_t, response_var_t, prev_coeff_t, curr_coeff_t, weight_t, emission_slots_t = inputs

        response_prev_exp = jnp.expand_dims(response_prev, axis=-1)
        response_prev_var_exp = jnp.expand_dims(response_var_prev, axis=-1)
        response_t_exp = jnp.expand_dims(response_t, axis=-1)
        response_t_var_exp = jnp.expand_dims(response_var_t, axis=-1)

        prev_second_mean, prev_second_var, prev_cov = _second_stats(
            response_prev, response_var_prev
        )
        curr_second_mean, curr_second_var, curr_cov = _second_stats(response_t, response_var_t)

        obs_sum_mean = (
            accum_sum_mean + prev_coeff_t * response_prev_exp + curr_coeff_t * response_t_exp
        )
        obs_sum_var = (
            accum_sum_var
            + prev_coeff_t**2 * response_prev_var_exp
            + curr_coeff_t**2 * response_t_var_exp
        )
        obs_sumsq_mean = (
            accum_sumsq_mean
            + prev_coeff_t * jnp.expand_dims(prev_second_mean, axis=-1)
            + curr_coeff_t * jnp.expand_dims(curr_second_mean, axis=-1)
        )
        obs_sumsq_var = (
            accum_sumsq_var
            + prev_coeff_t**2 * jnp.expand_dims(prev_second_var, axis=-1)
            + curr_coeff_t**2 * jnp.expand_dims(curr_second_var, axis=-1)
        )
        obs_sum_sumsq_cov = (
            accum_sum_sumsq_cov
            + prev_coeff_t**2 * jnp.expand_dims(prev_cov, axis=-1)
            + curr_coeff_t**2 * jnp.expand_dims(curr_cov, axis=-1)
        )
        obs_weight = accum_weight + weight_t

        selected_sum_mean = _select_support_slot(obs_sum_mean, emission_slots_t)
        selected_sum_var = _select_support_slot(obs_sum_var, emission_slots_t)
        selected_sumsq_mean = _select_support_slot(obs_sumsq_mean, emission_slots_t)
        selected_sumsq_var = _select_support_slot(obs_sumsq_var, emission_slots_t)
        selected_sum_sumsq_cov = _select_support_slot(obs_sum_sumsq_cov, emission_slots_t)
        selected_weight = _select_support_slot(obs_weight, emission_slots_t)
        safe_weight = jnp.maximum(selected_weight, NUMERICAL_EPSILON)
        window_mean = selected_sum_mean / safe_weight
        window_mean_var = selected_sum_var / (safe_weight**2)
        window_second_mean = selected_sumsq_mean / safe_weight
        std_arg = jnp.maximum(window_second_mean - window_mean**2, NUMERICAL_EPSILON)
        std_mean = jnp.sqrt(std_arg)
        d_std_d_sum = -window_mean / (std_mean * safe_weight)
        d_std_d_sumsq = 1.0 / (2.0 * std_mean * safe_weight)
        std_var = (
            d_std_d_sum**2 * selected_sum_var
            + d_std_d_sumsq**2 * selected_sumsq_var
            + 2.0 * d_std_d_sum * d_std_d_sumsq * selected_sum_sumsq_cov
        )
        std_var = jnp.maximum(std_var, 0.0)

        expected_mean = response_t
        latent_var = response_var_t
        sum_like = jnp.logical_or(
            summary_codes == _SUM_OPERATOR_CODE,
            summary_codes == _COUNT_OPERATOR_CODE,
        )
        expected_mean = jnp.where(sum_like, selected_sum_mean, expected_mean)
        latent_var = jnp.where(sum_like, selected_sum_var, latent_var)
        expected_mean = jnp.where(summary_codes == _MEAN_OPERATOR_CODE, window_mean, expected_mean)
        latent_var = jnp.where(summary_codes == _MEAN_OPERATOR_CODE, window_mean_var, latent_var)
        expected_mean = jnp.where(summary_codes == _STD_OPERATOR_CODE, std_mean, expected_mean)
        latent_var = jnp.where(summary_codes == _STD_OPERATOR_CODE, std_var, latent_var)

        emitted_interval_summary_mask = (
            full_obs_mask * interval_summary_mask * (emission_slots_t >= 0).astype(dtype)
        )
        semantic_mask = point_like_mask + emitted_interval_summary_mask

        next_sum_mean = _reset_support_stat(
            obs_sum_mean, emission_slots_t, emitted_interval_summary_mask
        )
        next_sum_var = _reset_support_stat(
            obs_sum_var, emission_slots_t, emitted_interval_summary_mask
        )
        next_sumsq_mean = _reset_support_stat(
            obs_sumsq_mean, emission_slots_t, emitted_interval_summary_mask
        )
        next_sumsq_var = _reset_support_stat(
            obs_sumsq_var, emission_slots_t, emitted_interval_summary_mask
        )
        next_sum_sumsq_cov = _reset_support_stat(
            obs_sum_sumsq_cov,
            emission_slots_t,
            emitted_interval_summary_mask,
        )
        next_weight = _reset_support_stat(
            obs_weight, emission_slots_t, emitted_interval_summary_mask
        )

        return (
            response_t,
            response_var_t,
            next_sum_mean,
            next_sum_var,
            next_sumsq_mean,
            next_sumsq_var,
            next_sum_sumsq_cov,
            next_weight,
        ), (
            expected_mean,
            latent_var,
            semantic_mask,
        )

    _, (expected_rest, latent_var_rest, semantic_mask_rest) = lax.scan(
        _scan_step_with_weight,
        (
            response_means[0],
            response_vars[0],
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            weight_zeros,
        ),
        (
            response_means[1:],
            response_vars[1:],
            prev_coeffs[1:],
            curr_coeffs[1:],
            interval_weights[1:],
            emission_slots[1:],
        ),
    )

    return (
        jnp.concatenate([response_means[0][None, :], expected_rest], axis=0),
        jnp.concatenate([response_vars[0][None, :], latent_var_rest], axis=0),
        jnp.concatenate([semantic_mask_0[None, :], semantic_mask_rest], axis=0),
    )


def _response_second_moment_loading(
    response_means: jnp.ndarray,
    response_state_loading: jnp.ndarray,
) -> jnp.ndarray:
    """Linearized loading for squared-response deviations."""
    return 2.0 * jnp.expand_dims(response_means, axis=-1) * response_state_loading


def _support_accumulator_response_map(coeffs: jnp.ndarray) -> jnp.ndarray:
    """Map one response vector into flattened manifest-slot accumulators."""
    eye = jnp.eye(coeffs.shape[0], dtype=coeffs.dtype)
    return (jnp.expand_dims(coeffs, axis=-1) * jnp.expand_dims(eye, axis=1)).reshape(
        coeffs.shape[0] * coeffs.shape[1],
        coeffs.shape[0],
    )


def _support_selection_matrix(
    emission_slot_indices: jnp.ndarray,
    *,
    n_slots: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Select one active slot per manifest from flattened accumulator state."""
    eye = jnp.eye(emission_slot_indices.shape[0], dtype=dtype)
    safe_indices = jnp.clip(emission_slot_indices, 0, n_slots - 1)
    slot_one_hot = (
        jax.nn.one_hot(safe_indices, n_slots, dtype=dtype)
        * (emission_slot_indices >= 0).astype(dtype)[:, None]
    )
    return (jnp.expand_dims(eye, axis=2) * jnp.expand_dims(slot_one_hot, axis=1)).reshape(
        emission_slot_indices.shape[0],
        emission_slot_indices.shape[0] * n_slots,
    )


def _point_response_covariances(
    response_state_loadings: jnp.ndarray,
    state_covs: jnp.ndarray,
    transitions: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return same-row and lag-1 response covariances without building T x T blocks."""
    same_covs = jax.vmap(lambda loading_t, state_cov_t: loading_t @ state_cov_t @ loading_t.T)(
        response_state_loadings, state_covs
    )
    if response_state_loadings.shape[0] <= 1:
        return same_covs, jnp.zeros(
            (0, response_state_loadings.shape[1], response_state_loadings.shape[1]),
            dtype=response_state_loadings.dtype,
        )

    lag1_state_covs = jax.vmap(lambda transition_t, state_cov_prev: transition_t @ state_cov_prev)(
        transitions, state_covs[:-1]
    )
    lag1_covs = jax.vmap(
        lambda loading_t, lag1_state_cov_t, loading_prev: (
            loading_t @ lag1_state_cov_t @ loading_prev.T
        )
    )(response_state_loadings[1:], lag1_state_covs, response_state_loadings[:-1])
    return same_covs, lag1_covs


def _support_response_covariances(
    response_means: jnp.ndarray,
    response_state_loadings: jnp.ndarray,
    *,
    t0_cov: jnp.ndarray,
    transitions: jnp.ndarray,
    process_covs: jnp.ndarray,
    observation_operator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project support-aware same-row and lag-1 covariances with a local scan."""
    from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
        _COUNT_OPERATOR_CODE,
        _MEAN_OPERATOR_CODE,
        _STD_OPERATOR_CODE,
        _SUM_OPERATOR_CODE,
    )

    emitted_means, semantic_mask = observation_operator.project_response_trajectory(response_means)
    n_timepoints, n_manifest = response_means.shape
    dtype = response_means.dtype

    same_cov_0 = response_state_loadings[0] @ t0_cov @ response_state_loadings[0].T
    if n_timepoints <= 1:
        same_pair_mask = jnp.expand_dims(semantic_mask, axis=2) * jnp.expand_dims(
            semantic_mask, axis=1
        )
        return (
            emitted_means,
            same_cov_0[None, :, :] * same_pair_mask,
            jnp.zeros((0, n_manifest, n_manifest), dtype=dtype),
            semantic_mask,
        )

    assert observation_operator.summary_operator_codes is not None
    assert observation_operator.prev_coeffs is not None
    assert observation_operator.curr_coeffs is not None
    assert observation_operator.interval_weights is not None
    assert observation_operator.emission_slots is not None

    interval_mask = observation_operator.interval_summary_mask(dtype)
    summary_codes = observation_operator.summary_operator_codes
    transformed_interval_mask = interval_mask * (
        (summary_codes == _SUM_OPERATOR_CODE)
        | (summary_codes == _COUNT_OPERATOR_CODE)
        | (summary_codes == _MEAN_OPERATOR_CODE)
        | (summary_codes == _STD_OPERATOR_CODE)
    ).astype(dtype)
    direct_response_mask = 1.0 - transformed_interval_mask
    direct_response_diag = jnp.diag(direct_response_mask)

    n_latent = int(t0_cov.shape[0])
    n_slots = observation_operator.max_active_windows
    accum_dim = n_manifest * n_slots

    second_loadings = jax.vmap(_response_second_moment_loading)(
        response_means,
        response_state_loadings,
    )
    prev_coeffs = jnp.asarray(observation_operator.prev_coeffs, dtype=dtype)
    curr_coeffs = jnp.asarray(observation_operator.curr_coeffs, dtype=dtype)
    interval_weights = jnp.asarray(observation_operator.interval_weights, dtype=dtype)
    emission_slots = jnp.asarray(observation_operator.emission_slots, dtype=jnp.int32)
    eye_latent = jnp.eye(n_latent, dtype=dtype)
    eye_accum = jnp.eye(accum_dim, dtype=dtype)
    zeros_accum = observation_operator.empty_accumulators(dtype)
    zeros_accum_cov = jnp.zeros((accum_dim, accum_dim), dtype=dtype)
    zeros_latent_accum = jnp.zeros((n_latent, accum_dim), dtype=dtype)
    p_aug_0 = jnp.block(
        [
            [t0_cov, zeros_latent_accum, zeros_latent_accum],
            [zeros_latent_accum.T, zeros_accum_cov, zeros_accum_cov],
            [zeros_latent_accum.T, zeros_accum_cov, zeros_accum_cov],
        ]
    )
    y0 = jnp.concatenate(
        [
            response_state_loadings[0],
            jnp.zeros((n_manifest, accum_dim), dtype=dtype),
            jnp.zeros((n_manifest, accum_dim), dtype=dtype),
        ],
        axis=1,
    )
    cross_prev_0 = p_aug_0 @ y0.T

    def _scan_step(carry, inputs):
        accum_sum_mean_prev, accum_sumsq_mean_prev, accum_weight_prev, p_aug_prev, cross_prev = (
            carry
        )
        (
            response_mean_prev,
            response_mean_t,
            response_loading_prev,
            response_loading_t,
            second_loading_prev,
            second_loading_t,
            transition_t,
            process_cov_t,
            prev_coeff_t,
            curr_coeff_t,
            weight_t,
            emission_slots_t,
        ) = inputs

        obs_sum_mean = (
            accum_sum_mean_prev
            + prev_coeff_t * jnp.expand_dims(response_mean_prev, axis=-1)
            + curr_coeff_t * jnp.expand_dims(response_mean_t, axis=-1)
        )
        obs_sumsq_mean = (
            accum_sumsq_mean_prev
            + prev_coeff_t * jnp.expand_dims(response_mean_prev**2, axis=-1)
            + curr_coeff_t * jnp.expand_dims(response_mean_t**2, axis=-1)
        )
        obs_weight = accum_weight_prev + weight_t

        selected_sum_mean = _select_support_slot(obs_sum_mean, emission_slots_t)
        selected_sumsq_mean = _select_support_slot(obs_sumsq_mean, emission_slots_t)
        selected_weight = _select_support_slot(obs_weight, emission_slots_t)
        safe_weight = jnp.maximum(selected_weight, NUMERICAL_EPSILON)
        window_mean = selected_sum_mean / safe_weight
        window_second_mean = selected_sumsq_mean / safe_weight
        std_arg = jnp.maximum(window_second_mean - window_mean**2, NUMERICAL_EPSILON)
        std_mean = jnp.sqrt(std_arg)
        d_std_d_sum = -window_mean / (std_mean * safe_weight)
        d_std_d_sumsq = 1.0 / (2.0 * std_mean * safe_weight)

        alpha_sum = jnp.where(
            (summary_codes == _SUM_OPERATOR_CODE) | (summary_codes == _COUNT_OPERATOR_CODE),
            1.0,
            0.0,
        )
        alpha_sum = jnp.where(summary_codes == _MEAN_OPERATOR_CODE, 1.0 / safe_weight, alpha_sum)
        alpha_sum = jnp.where(summary_codes == _STD_OPERATOR_CODE, d_std_d_sum, alpha_sum)
        alpha_sumsq = jnp.where(summary_codes == _STD_OPERATOR_CODE, d_std_d_sumsq, 0.0)

        select_matrix = _support_selection_matrix(
            emission_slots_t,
            n_slots=n_slots,
            dtype=dtype,
        )
        prev_response_map = _support_accumulator_response_map(prev_coeff_t)
        curr_response_map = _support_accumulator_response_map(curr_coeff_t)
        curr_response_to_prev_x = response_loading_t @ transition_t
        curr_second_to_prev_x = second_loading_t @ transition_t
        sum_prev_x = (
            prev_response_map @ response_loading_prev + curr_response_map @ curr_response_to_prev_x
        )
        sumsq_prev_x = (
            prev_response_map @ second_loading_prev + curr_response_map @ curr_second_to_prev_x
        )
        sum_noise = curr_response_map @ response_loading_t
        sumsq_noise = curr_response_map @ second_loading_t

        emitted_interval_summary_mask = interval_mask * (emission_slots_t >= 0).astype(dtype)
        keep_mask = _reset_support_stat(
            jnp.ones((n_manifest, n_slots), dtype=dtype),
            emission_slots_t,
            emitted_interval_summary_mask,
        ).reshape(-1)
        keep_rows = jnp.expand_dims(keep_mask, axis=-1)
        identity_kept = keep_rows * eye_accum
        sum_select = jnp.expand_dims(alpha_sum, axis=1) * select_matrix
        sumsq_select = jnp.expand_dims(alpha_sumsq, axis=1) * select_matrix

        y_prev_x = (
            direct_response_diag @ curr_response_to_prev_x
            + sum_select @ sum_prev_x
            + sumsq_select @ sumsq_prev_x
        )
        y_noise = (
            direct_response_diag @ response_loading_t
            + sum_select @ sum_noise
            + sumsq_select @ sumsq_noise
        )
        y_t = jnp.concatenate([y_prev_x, sum_select, sumsq_select], axis=1)

        f_t = jnp.block(
            [
                [
                    transition_t,
                    jnp.zeros((n_latent, accum_dim), dtype=dtype),
                    jnp.zeros((n_latent, accum_dim), dtype=dtype),
                ],
                [
                    keep_rows * sum_prev_x,
                    identity_kept,
                    jnp.zeros((accum_dim, accum_dim), dtype=dtype),
                ],
                [
                    keep_rows * sumsq_prev_x,
                    jnp.zeros((accum_dim, accum_dim), dtype=dtype),
                    identity_kept,
                ],
            ]
        )
        g_t = jnp.concatenate(
            [
                eye_latent,
                keep_rows * sum_noise,
                keep_rows * sumsq_noise,
            ],
            axis=0,
        )

        same_cov_t = y_t @ p_aug_prev @ y_t.T + y_noise @ process_cov_t @ y_noise.T
        lag1_cov_t = y_t @ cross_prev
        p_aug_t = f_t @ p_aug_prev @ f_t.T + g_t @ process_cov_t @ g_t.T
        cross_t = f_t @ p_aug_prev @ y_t.T + g_t @ process_cov_t @ y_noise.T

        next_accum_sum_mean = _reset_support_stat(
            obs_sum_mean,
            emission_slots_t,
            emitted_interval_summary_mask,
        )
        next_accum_sumsq_mean = _reset_support_stat(
            obs_sumsq_mean,
            emission_slots_t,
            emitted_interval_summary_mask,
        )
        next_accum_weight = _reset_support_stat(
            obs_weight,
            emission_slots_t,
            emitted_interval_summary_mask,
        )

        return (
            next_accum_sum_mean,
            next_accum_sumsq_mean,
            next_accum_weight,
            p_aug_t,
            cross_t,
        ), (
            same_cov_t,
            lag1_cov_t,
        )

    _, (same_covs_rest, lag1_covs) = lax.scan(
        _scan_step,
        (
            zeros_accum,
            zeros_accum,
            zeros_accum,
            p_aug_0,
            cross_prev_0,
        ),
        (
            response_means[:-1],
            response_means[1:],
            response_state_loadings[:-1],
            response_state_loadings[1:],
            second_loadings[:-1],
            second_loadings[1:],
            transitions,
            process_covs,
            prev_coeffs[1:],
            curr_coeffs[1:],
            interval_weights[1:],
            emission_slots[1:],
        ),
    )

    same_covs = jnp.concatenate([same_cov_0[None, :, :], same_covs_rest], axis=0)
    same_pair_mask = jnp.expand_dims(semantic_mask, axis=2) * jnp.expand_dims(semantic_mask, axis=1)
    lag1_pair_mask = jnp.expand_dims(semantic_mask[1:], axis=2) * jnp.expand_dims(
        semantic_mask[:-1], axis=1
    )
    return (
        emitted_means,
        same_covs * same_pair_mask,
        lag1_covs * lag1_pair_mask,
        semantic_mask,
    )


def _build_sensitivity_measurement_semantics(
    spec,
    *,
    manifest_cov: jnp.ndarray,
    extra_params: dict,
    observation_support,
):
    """Compile measurement semantics for the observation-space sensitivity map."""
    from causal_ssm_agent.models.ssm.inference.targets.kernels import compile_measurement_semantics

    return compile_measurement_semantics(
        manifest_dists=spec.manifest_dists,
        manifest_cov=manifest_cov,
        extra_params=extra_params or None,
        manifest_links=spec.manifest_links,
        observation_support=observation_support,
    )


def _flatten_time_block_covariance(cov_blocks: jnp.ndarray) -> jnp.ndarray:
    """Flatten ``(T, T, M, M)`` covariance blocks to ``(T*M, T*M)`` order."""
    return cov_blocks.transpose(0, 2, 1, 3).reshape(
        cov_blocks.shape[0] * cov_blocks.shape[2],
        cov_blocks.shape[1] * cov_blocks.shape[3],
    )


def _unflatten_time_block_covariance(
    cov_flat: jnp.ndarray,
    *,
    n_timepoints: int,
    n_manifest: int,
) -> jnp.ndarray:
    """Restore ``(T, T, M, M)`` covariance blocks from flattened form."""
    return cov_flat.reshape(n_timepoints, n_manifest, n_timepoints, n_manifest).transpose(
        0, 2, 1, 3
    )


def _build_state_cross_covariance_blocks(
    state_covs: jnp.ndarray,
    transitions: jnp.ndarray,
) -> jnp.ndarray:
    """Build all pairwise latent-state covariance blocks from one-step transitions."""
    n_timepoints = int(state_covs.shape[0])
    blocks: list[list[jnp.ndarray | None]] = [[None] * n_timepoints for _ in range(n_timepoints)]

    for time_idx in range(n_timepoints):
        blocks[time_idx][time_idx] = state_covs[time_idx]

    for time_idx in range(1, n_timepoints):
        transition = transitions[time_idx - 1]
        prev_row = blocks[time_idx - 1]
        for past_idx in range(time_idx):
            prev_block = prev_row[past_idx]
            assert prev_block is not None
            block = transition @ prev_block
            blocks[time_idx][past_idx] = block
            blocks[past_idx][time_idx] = block.T

    filled_blocks: list[list[jnp.ndarray]] = []
    for row in blocks:
        filled_row: list[jnp.ndarray] = []
        for block in row:
            assert block is not None
            filled_row.append(block)
        filled_blocks.append(filled_row)

    return jnp.stack(
        [jnp.stack(row, axis=0) for row in filled_blocks],
        axis=0,
    )


def _project_response_covariance_blocks(
    response_means: jnp.ndarray,
    response_cov_blocks: jnp.ndarray,
    observation_operator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project response-trajectory covariance through the support operator."""
    emitted_means, semantic_mask = observation_operator.project_response_trajectory(response_means)
    n_timepoints, n_manifest = response_means.shape

    if observation_operator.requires_interval_summary_handling:
        response_cov_flat = _flatten_time_block_covariance(response_cov_blocks)

        def _project_flat(response_flat: jnp.ndarray) -> jnp.ndarray:
            projected, _ = observation_operator.project_response_trajectory(
                response_flat.reshape(n_timepoints, n_manifest)
            )
            return projected.reshape(-1)

        emission_jacobian = jax.jacfwd(_project_flat)(response_means.reshape(-1))
        emitted_cov_flat = emission_jacobian @ response_cov_flat @ emission_jacobian.T
        emitted_cov_blocks = _unflatten_time_block_covariance(
            emitted_cov_flat,
            n_timepoints=n_timepoints,
            n_manifest=n_manifest,
        )
    else:
        emitted_cov_blocks = response_cov_blocks

    same_covs = jnp.stack(
        [emitted_cov_blocks[time_idx, time_idx] for time_idx in range(n_timepoints)],
        axis=0,
    )
    if n_timepoints > 1:
        lag1_covs = jnp.stack(
            [emitted_cov_blocks[time_idx, time_idx - 1] for time_idx in range(1, n_timepoints)],
            axis=0,
        )
    else:
        lag1_covs = jnp.zeros((0, n_manifest, n_manifest), dtype=response_means.dtype)

    same_pair_mask = jnp.expand_dims(semantic_mask, axis=2) * jnp.expand_dims(semantic_mask, axis=1)
    same_covs = same_covs * same_pair_mask
    if n_timepoints > 1:
        lag1_pair_mask = jnp.expand_dims(semantic_mask[1:], axis=2) * jnp.expand_dims(
            semantic_mask[:-1], axis=1
        )
        lag1_covs = lag1_covs * lag1_pair_mask

    return emitted_means, same_covs, lag1_covs, semantic_mask


def _predict_observation_components(
    det: dict[str, jnp.ndarray],
    extra_params: dict,
    spec,
    times: jnp.ndarray,
    *,
    structure_runtime: SSMStructureRuntime,
    observation_support=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Predict emitted-observation mean, covariance, lagged covariance, noise scale, and mask."""
    n_l, n_m = spec.n_latent, spec.n_manifest

    drift = det.get("drift", jnp.zeros((n_l, n_l)))
    diffusion_chol = det.get("diffusion", jnp.eye(n_l))
    diffusion_cov = diffusion_chol @ diffusion_chol.T
    t0_means = det.get("t0_means", jnp.zeros(n_l))
    t0_cov = det.get("t0_cov", jnp.eye(n_l))
    manifest_cov = det.get("manifest_cov", jnp.eye(n_m))
    manifest_means = det.get("manifest_means", jnp.zeros(n_m))

    lambda_val = det.get("lambda", structure_runtime.lambda_template)

    cint = det.get("cint", jnp.zeros(n_l))
    measurement_semantics = _build_sensitivity_measurement_semantics(
        spec,
        manifest_cov=manifest_cov,
        extra_params=extra_params,
        observation_support=observation_support,
    )
    obs_kernel = measurement_semantics.obs_kernel
    observation_operator = measurement_semantics.observation_operator

    def _point_moments(x_mean: jnp.ndarray, state_cov: jnp.ndarray):
        eta_mean = lambda_val @ x_mean + manifest_means
        eta_cov = lambda_val @ state_cov @ lambda_val.T
        response_mean = obs_kernel.response_fn(eta_mean)
        _, response_jacobian = _response_latent_covariance(
            eta_mean,
            eta_cov,
            obs_kernel=obs_kernel,
        )
        response_state_loading = response_jacobian @ lambda_val
        point_noise_variance_args = _observation_noise_variance_arguments(
            eta_mean,
            response_mean,
            manifest_dists=measurement_semantics.manifest_dists,
        )
        return (
            eta_mean,
            response_mean,
            response_state_loading,
            point_noise_variance_args,
        )

    dt_array = jnp.diff(times)
    Ad, Qd, cd = discretize_system_batched(drift, diffusion_cov, cint, dt_array)

    (
        _eta_mean_0,
        response_mean_0,
        response_state_loading_0,
        point_noise_variance_args_0,
    ) = _point_moments(t0_means, t0_cov)

    def scan_fn(carry, inputs):
        x_m, P = carry
        Ad_t, Qd_t, cd_t = inputs

        x_m_next = Ad_t @ x_m + cd_t
        P_next = Ad_t @ P @ Ad_t.T + Qd_t
        (
            eta_mean_next,
            response_mean_next,
            response_state_loading_next,
            point_noise_variance_args_next,
        ) = _point_moments(x_m_next, P_next)
        return (x_m_next, P_next), (
            eta_mean_next,
            response_mean_next,
            response_state_loading_next,
            P_next,
            point_noise_variance_args_next,
        )

    (
        _,
        (
            _eta_means_rest,
            response_means_rest,
            response_state_loadings_rest,
            state_covs_rest,
            point_noise_variance_args_rest,
        ),
    ) = lax.scan(
        scan_fn,
        (t0_means, t0_cov),
        (Ad, Qd, cd),
    )

    response_means = jnp.concatenate([response_mean_0[None, :], response_means_rest], axis=0)
    response_state_loadings = jnp.concatenate(
        [response_state_loading_0[None, :, :], response_state_loadings_rest],
        axis=0,
    )
    state_covs = jnp.concatenate([t0_cov[None, :, :], state_covs_rest], axis=0)
    point_noise_variance_args = jnp.concatenate(
        [point_noise_variance_args_0[None, :], point_noise_variance_args_rest],
        axis=0,
    )

    if observation_operator.requires_interval_summary_handling:
        emitted_means, emitted_same_covs, emitted_lag1_covs, semantic_mask = (
            _support_response_covariances(
                response_means,
                response_state_loadings,
                t0_cov=t0_cov,
                transitions=Ad,
                process_covs=Qd,
                observation_operator=observation_operator,
            )
        )
    else:
        emitted_means = response_means
        emitted_same_covs, emitted_lag1_covs = _point_response_covariances(
            response_state_loadings,
            state_covs,
            Ad,
        )
        semantic_mask = jnp.ones_like(emitted_means, dtype=emitted_means.dtype)

    if observation_operator.requires_interval_summary_handling:
        point_like_mask = jnp.broadcast_to(
            observation_operator.point_like_mask(emitted_means.dtype),
            emitted_means.shape,
        )
        interval_emission_mask = jnp.maximum(semantic_mask - point_like_mask, 0.0)
        emitted_noise_variance_args = (
            point_noise_variance_args * point_like_mask + emitted_means * interval_emission_mask
        )
    else:
        emitted_noise_variance_args = point_noise_variance_args
        semantic_mask = jnp.ones_like(emitted_means, dtype=emitted_means.dtype)
    emitted_obs_noise_covs = jax.vmap(
        lambda variance_args_t: _observation_noise_covariance(
            variance_args_t,
            obs_kernel=obs_kernel,
        )
    )(emitted_noise_variance_args)
    same_pair_mask = jnp.expand_dims(semantic_mask, axis=2) * jnp.expand_dims(semantic_mask, axis=1)
    emitted_same_covs = emitted_same_covs + emitted_obs_noise_covs * same_pair_mask
    emitted_obs_noise_vars = jnp.diagonal(emitted_obs_noise_covs, axis1=1, axis2=2) * semantic_mask
    emitted_obs_noise_sd = jnp.sqrt(jnp.maximum(emitted_obs_noise_vars, NUMERICAL_EPSILON))

    return (
        emitted_means,
        emitted_same_covs,
        emitted_lag1_covs,
        emitted_obs_noise_sd,
        semantic_mask,
    )


def _flatten_lower_triangular(mats: jnp.ndarray) -> jnp.ndarray:
    """Flatten the lower triangle of a stack of symmetric matrices."""
    tri_i, tri_j = np.tril_indices(int(mats.shape[-1]))
    return mats[:, tri_i, tri_j].reshape(-1)


def _flatten_observation_moment_summary(
    means: jnp.ndarray,
    same_covs: jnp.ndarray,
    lag1_covs: jnp.ndarray,
) -> jnp.ndarray:
    """Flatten emitted-observation moments into one feature vector."""
    mean_features = means.reshape(-1)
    same_cov_features = _flatten_lower_triangular(same_covs)
    lag_cov_features = lag1_covs.reshape(-1)
    return jnp.concatenate([mean_features, same_cov_features, lag_cov_features])


def _moment_summary_row_scales(obs_noise_sd: jnp.ndarray) -> jnp.ndarray:
    """Observation-scale normalization factors aligned to the moment-summary layout."""
    mean_scales = obs_noise_sd.reshape(-1)
    same_cov_scales = _flatten_lower_triangular(
        jnp.expand_dims(obs_noise_sd, axis=2) * jnp.expand_dims(obs_noise_sd, axis=1)
    )
    if obs_noise_sd.shape[0] <= 1:
        lag_cov_scales = jnp.zeros((0,), dtype=obs_noise_sd.dtype)
    else:
        lag_cov_scales = (
            jnp.expand_dims(obs_noise_sd[1:], axis=2) * jnp.expand_dims(obs_noise_sd[:-1], axis=1)
        ).reshape(-1)
    return jnp.concatenate([mean_scales, same_cov_scales, lag_cov_scales])


def _predict_observation_moments(
    z_flat,
    unravel_fn,
    transforms,
    spec,
    times,
    *,
    structure_runtime: SSMStructureRuntime,
    observation_support=None,
    registry,
):
    """Predicted observation-space moment summary from unconstrained params.

    Runs the latent Kalman prediction equations, maps latent predictors through
    the configured observation families and interval-summary semantics, and
    returns a flat vector of emitted means, same-row covariance entries, and
    adjacent-row lagged cross-covariance entries suitable for Jacobian
    computation.
    """
    det, extra_params = _assemble_sensitivity_measurement_state(
        z_flat,
        unravel_fn,
        transforms,
        spec,
        structure_runtime=structure_runtime,
        registry=registry,
    )
    emitted_means, emitted_same_covs, emitted_lag1_covs, _emitted_obs_noise_sd, _semantic_mask = (
        _predict_observation_components(
            det,
            extra_params,
            spec,
            times,
            structure_runtime=structure_runtime,
            observation_support=observation_support,
        )
    )
    return _flatten_observation_moment_summary(
        emitted_means,
        emitted_same_covs,
        emitted_lag1_covs,
    )


@dataclass
class OutputSensitivityResult:
    """Results from output sensitivity analysis (pre-inference identifiability).

    Structural identifiability check via the Jacobian of the forward model's
    emitted-observation moment summary. A full-rank sensitivity matrix
    indicates all parameters are locally identifiable. Near-zero singular
    values indicate parameter combinations that observations cannot
    distinguish.
    """

    singular_values: list[float]  # median SVD spectrum across draws (descending)
    condition_number: float  # median max_sv / min_sv
    per_parameter: list[dict]  # [{parameter, sensitivity_norm, identifiable}]
    n_draws: int
    n_observations: int  # retained moment-feature dimension after masking
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


class OutputSensitivityUnsupportedError(ValueError):
    """Raised when the observation-space Stage 4b map is not valid for a model."""


def _observation_semantic_mask(
    spec: SSMSpec,
    times: jnp.ndarray,
    observation_support,
) -> np.ndarray | None:
    """Return the support-aware emission mask aligned to the model time grid."""
    from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
        compile_observation_operator,
    )

    observation_operator = compile_observation_operator(observation_support)
    if not observation_operator.requires_interval_summary_handling:
        return None

    _, semantic_mask = observation_operator.project_response_trajectory(
        jnp.zeros((times.shape[0], spec.n_manifest), dtype=jnp.float32)
    )
    return np.asarray(semantic_mask > 0.5)


def _build_sensitivity_output_mask(
    observations: jnp.ndarray | None,
    *,
    semantic_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return a feature mask aligned to the emitted-observation moment summary."""
    if observations is None and semantic_mask is None:
        return None

    if observations is None:
        obs_mask = np.asarray(semantic_mask, dtype=bool)
    else:
        obs_mask = ~np.isnan(np.asarray(observations))
        if semantic_mask is not None:
            obs_mask = obs_mask & np.asarray(semantic_mask, dtype=bool)
    mean_mask = obs_mask.reshape(-1)
    tri_i, tri_j = np.tril_indices(obs_mask.shape[1])
    same_cov_mask = (obs_mask[:, :, None] & obs_mask[:, None, :])[:, tri_i, tri_j].reshape(-1)
    if obs_mask.shape[0] <= 1:
        lag_cov_mask = np.zeros((0,), dtype=bool)
    else:
        lag_cov_mask = (obs_mask[1:, :, None] & obs_mask[:-1, None, :]).reshape(-1)
    return np.concatenate([mean_mask, same_cov_mask, lag_cov_mask])


def _spectral_svd_from_gram(S: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute singular values and right singular vectors via the P x P Gram matrix."""
    gram = S.T @ S
    eigvals, eigvecs = jnp.linalg.eigh(gram)
    eigvals = jnp.clip(eigvals, a_min=0.0)
    order = jnp.arange(eigvals.shape[0] - 1, -1, -1)
    singular_values = jnp.sqrt(eigvals[order])
    right_singular_vectors = eigvecs[:, order]
    return singular_values, right_singular_vectors


def _validate_output_sensitivity_supported(model: SSMModel) -> None:
    """Validate preconditions for the observation-space sensitivity map."""
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    observation_support = getattr(model, "observation_support", None)
    if observation_support is None or not observation_support.requires_interval_summary_handling:
        return

    manifest_names = _axis_names(
        model.spec.manifest_names,
        expected=model.spec.n_manifest,
        prefix="manifest",
    )
    unsupported_interval_families = {
        DistributionFamily.ORDERED_LOGISTIC,
        DistributionFamily.CATEGORICAL,
    }
    unsupported_manifests = [
        manifest_names[idx]
        for idx, (support_kind, dist) in enumerate(
            zip(observation_support.support_kinds, model.spec.manifest_dists, strict=False)
        )
        if support_kind == "interval" and dist in unsupported_interval_families
    ]
    if unsupported_manifests:
        unsupported = ", ".join(unsupported_manifests)
        raise OutputSensitivityUnsupportedError(
            "interval-summary sensitivity requires observation families with a "
            f"mean-parameter likelihood; unsupported interval manifests: {unsupported}"
        )


def _split_scalar_parameter_name(parameter: str) -> tuple[str, int]:
    """Split ``site_name[idx]`` strings into their site name and flat index."""
    match = _SCALAR_PARAMETER_INDEX_RE.fullmatch(parameter)
    if match is None:
        return parameter, 0
    return match.group("site"), int(match.group("index"))


def _axis_names(
    names: list[str] | None,
    *,
    expected: int,
    prefix: str,
) -> list[str]:
    """Return axis names with deterministic fallbacks when metadata is incomplete."""
    resolved = [str(name) for name in (names or []) if name]
    if len(resolved) >= expected:
        return resolved[:expected]
    return resolved + [f"{prefix}_{idx}" for idx in range(len(resolved), expected)]


def _binding_index_for_model(model: SSMModel) -> dict[tuple[str, int], str]:
    """Index compiler parameter bindings by sample site and flat index."""
    binding_index: dict[tuple[str, int], str] = {}
    for binding in list(getattr(model, "parameter_bindings", []) or []):
        if not isinstance(binding, dict):
            continue
        site_name = binding.get("site_name")
        flat_index = binding.get("flat_index")
        parameter = binding.get("parameter")
        if not isinstance(site_name, str) or not isinstance(flat_index, int):
            continue
        if not isinstance(parameter, str) or not parameter:
            continue
        binding_index[(site_name, flat_index)] = parameter
    return binding_index


def _fallback_interpretable_parameter_name(
    spec: SSMSpec,
    site_name: str,
    flat_index: int,
    *,
    structure_runtime: SSMStructureRuntime,
) -> str:
    """Resolve a best-effort semantic alias for one scalar sample-site element."""
    latent_names = _axis_names(spec.latent_names, expected=spec.n_latent, prefix="latent")
    manifest_names = _axis_names(spec.manifest_names, expected=spec.n_manifest, prefix="manifest")

    if site_name == "drift_diag_free" and flat_index < structure_runtime.n_drift_diag:
        latent_idx = structure_runtime.drift_diag_positions[flat_index]
        return f"rho_{latent_names[latent_idx]}"
    if site_name == "drift_offdiag_free" and flat_index < structure_runtime.n_drift_offdiag:
        effect_idx, cause_idx = structure_runtime.offdiag_positions[flat_index]
        return f"beta_{latent_names[cause_idx]}_{latent_names[effect_idx]}"
    if site_name == "diffusion_diag_free" and flat_index < structure_runtime.n_diffusion_diag:
        latent_idx = structure_runtime.diffusion_diag_positions[flat_index]
        return f"sigma_{latent_names[latent_idx]}"
    if site_name == "diffusion_lower_free" and flat_index < structure_runtime.n_diffusion_lower:
        row, col = structure_runtime.diffusion_lower_positions[flat_index]
        return f"cor_{latent_names[col]}_{latent_names[row]}"
    if site_name == "cint_free" and flat_index < structure_runtime.n_cint:
        latent_idx = structure_runtime.cint_free_positions[flat_index]
        return f"cint_{latent_names[latent_idx]}"
    if site_name == "lambda_free" and flat_index < structure_runtime.n_lambda_free:
        manifest_idx, latent_idx = structure_runtime.lambda_free_positions[flat_index]
        return f"lambda_{manifest_names[manifest_idx]}_{latent_names[latent_idx]}"
    if site_name == "manifest_means_free" and flat_index < structure_runtime.n_manifest_means:
        manifest_idx = structure_runtime.manifest_means_free_positions[flat_index]
        return f"manifest_mean_{manifest_names[manifest_idx]}"
    if site_name == "manifest_var_diag_free" and flat_index < structure_runtime.n_manifest_var_diag:
        manifest_idx = structure_runtime.manifest_var_free_positions[flat_index]
        return f"obs_sd_{manifest_names[manifest_idx]}"
    if site_name == "t0_means_free" and flat_index < structure_runtime.n_t0_means:
        latent_idx = structure_runtime.t0_means_free_positions[flat_index]
        return f"t0_mean_{latent_names[latent_idx]}"
    if site_name == "t0_var_diag_free" and flat_index < structure_runtime.n_t0_diag:
        latent_idx = structure_runtime.t0_diag_free_positions[flat_index]
        return f"t0_sd_{latent_names[latent_idx]}"
    if site_name == "t0_var_lower_free" and flat_index < structure_runtime.n_t0_correlation:
        row, col = structure_runtime.t0_correlation_positions[flat_index]
        return f"cor0_{latent_names[col]}_{latent_names[row]}"
    return site_name if flat_index == 0 else f"{site_name}[{flat_index}]"


def _interpretable_parameter_name_map(
    model: SSMModel,
    scalar_names: list[str],
) -> dict[str, str]:
    """Resolve semantic display names for all scalar Stage 4b parameters."""
    binding_index = _binding_index_for_model(model)
    structure_runtime = getattr(model, "_structure_runtime", None)
    if not isinstance(structure_runtime, SSMStructureRuntime):
        structure_runtime = SSMStructureRuntime(model.spec)

    resolved: dict[str, str] = {}
    for scalar_name in scalar_names:
        site_name, flat_index = _split_scalar_parameter_name(scalar_name)
        interpretable = binding_index.get((site_name, flat_index))
        if interpretable is None:
            interpretable = _fallback_interpretable_parameter_name(
                model.spec,
                site_name,
                flat_index,
                structure_runtime=structure_runtime,
            )
        resolved[scalar_name] = interpretable
    return resolved


def output_sensitivity_analysis(
    model: SSMModel,
    times: jnp.ndarray,
    observations: jnp.ndarray | None = None,
    n_draws: int = 8,
    seed: int = 42,
    sweep_context: Stage4bSweepContext | None = None,
) -> OutputSensitivityResult:
    """Pre-inference parametric identifiability via output sensitivity analysis.

    Computes the sensitivity matrix S[i,j] = dy_i / dtheta_j for the forward
    model's emitted-observation moment summary: means, same-row covariance
    entries, and adjacent-row lagged cross-covariance entries. The Jacobian is
    evaluated without data updates, then analyzed via SVD to detect
    structurally non-identifiable parameter directions.

    Args:
        model: SSMModel instance
        times: (T,) observation times
        n_draws: Number of prior draws for robustness (default 8)
        seed: Random seed

    Returns:
        OutputSensitivityResult with SVD spectrum and per-parameter flags
    """
    _validate_output_sensitivity_supported(model)
    rng_key = random.PRNGKey(seed)
    if sweep_context is not None:
        context = sweep_context
    else:
        cached_context = get_stage4b_sweep_context(model)
        context = Stage4bSweepContext(
            cache_key=cached_context.cache_key,
            spec=cached_context.spec,
            structure_runtime=cached_context.structure_runtime,
            site_runtime=cached_context.site_runtime,
            predict_moments_fn=cached_context.predict_moments_fn,
            jacobian_fn=jax.jit(jax.jacfwd(cached_context.predict_moments_fn, argnums=0)),
            log_lik_fn=cached_context.log_lik_fn,
            log_prior_unc_fn=cached_context.log_prior_unc_fn,
        )

    # 1. Reuse topology-dependent registry metadata and rebuild only prior values.
    P = context.flat_dim
    scalar_names = context.scalar_names
    prior_state = model.get_prior_runtime_bundle().prior_state
    semantic_mask = _observation_semantic_mask(
        context.spec,
        times,
        getattr(model, "observation_support", None),
    )

    # 3. Sample from prior (Jacobian draws + larger batch for prior std)
    prior_z, rng_key = sample_prior_unconstrained(
        rng_key,
        context.registry,
        prior_state,
        n_samples=n_draws,
    )
    prior_std_draws = min(64, max(32, n_draws * 4))
    prior_z_std, rng_key = sample_prior_unconstrained(
        rng_key,
        context.registry,
        prior_state,
        n_samples=prior_std_draws,
    )
    prior_std = jnp.std(prior_z_std, axis=0)  # (P,) per-parameter prior SD
    # Guard against degenerate priors (zero std)
    prior_std = jnp.maximum(prior_std, NUMERICAL_EPSILON)

    output_mask = _build_sensitivity_output_mask(observations, semantic_mask=semantic_mask)
    if output_mask is None:
        N_out = int(context.predict_moments_fn(prior_z[0], times).shape[0])
    else:
        N_out = int(output_mask.sum())

    # Helper to extract family-aware observation noise scales for a given parameter vector
    def _get_obs_noise_scales(z_0):
        det, extra_params = _assemble_sensitivity_measurement_state(
            z_0,
            context.unravel_fn,
            context.transforms,
            context.spec,
            structure_runtime=context.structure_runtime,
            registry=context.registry,
        )
        _projected_means, _same_covs, _lag1_covs, obs_noise_sd, _semantic = (
            _predict_observation_components(
                det,
                extra_params,
                context.spec,
                times,
                structure_runtime=context.structure_runtime,
                observation_support=getattr(model, "observation_support", None),
            )
        )
        return _moment_summary_row_scales(obs_noise_sd)

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
    skipped_nonfinite_draws = 0

    for i in range(n_draws):
        z_0 = prior_z[i]
        S = context.jacobian_fn(z_0, times)  # (N_out, P) sensitivity matrix
        if output_mask is not None:
            S = S[output_mask]
        if not bool(jnp.all(jnp.isfinite(S))):
            skipped_nonfinite_draws += 1
            continue

        # --- Raw SVD ---
        sv, V = _spectral_svd_from_gram(S)
        col_norms = jnp.linalg.norm(S, axis=0)
        if not bool(jnp.all(jnp.isfinite(sv))) or not bool(jnp.all(jnp.isfinite(col_norms))):
            skipped_nonfinite_draws += 1
            continue
        all_sv.append(sv)
        all_col_norms.append(col_norms)
        all_effective_sv.append(_per_param_effective_sv(V, sv))

        # --- Normalized SVD ---
        # S_norm[i,j] = (prior_std[j] / obs_scale[i]) * S[i,j]
        row_scales = _get_obs_noise_scales(z_0)
        if output_mask is not None:
            row_scales = row_scales[output_mask]
        row_scales = jnp.maximum(row_scales, NUMERICAL_EPSILON)
        if not bool(jnp.all(jnp.isfinite(row_scales))):
            skipped_nonfinite_draws += 1
            all_sv.pop()
            all_col_norms.pop()
            all_effective_sv.pop()
            continue
        S_norm = (prior_std[None, :] / row_scales[:, None]) * S
        sv_n, V_n = _spectral_svd_from_gram(S_norm)
        if not bool(jnp.all(jnp.isfinite(S_norm))) or not bool(jnp.all(jnp.isfinite(sv_n))):
            skipped_nonfinite_draws += 1
            all_sv.pop()
            all_col_norms.pop()
            all_effective_sv.pop()
            continue
        all_norm_effective_sv.append(_per_param_effective_sv(V_n, sv_n))

    if not all_sv:
        raise RuntimeError(
            "output sensitivity analysis produced no finite prior draws after screening"
        )
    if skipped_nonfinite_draws:
        logger.warning(
            "Output sensitivity analysis skipped %d/%d non-finite prior draws",
            skipped_nonfinite_draws,
            n_draws,
        )

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
    interpretable_names = _interpretable_parameter_name_map(model, scalar_names)
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
                "interpretable_parameter": interpretable_names[sname],
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
        n_draws=len(all_sv),
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
    prior_state = model.get_prior_runtime_bundle().prior_state

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
            # Keep the per-parameter profiler eager here. JIT-compiling a fresh
            # closure for each profiled scalar triggers expensive XLA compile
            # churn during large stage-4b sweeps.
            _j = j  # capture for closure

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
    prior_runtime = model.get_prior_runtime_bundle()
    site_runtime = prior_runtime.site_runtime
    prior_state = prior_runtime.prior_state
    backend = model.make_likelihood_backend()
    log_lik_fn, _ = _build_runtime_eval_fns_from_registry(
        model.spec,
        site_runtime.registry,
        site_runtime.unravel_fn,
        site_runtime.transforms,
        model._structure_runtime,
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
        except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as exc:
            logger.info("SBC replicate %d: simulation failed: %s", rep, exc)
            n_failed += 1
            continue

        if not jnp.all(jnp.isfinite(y_star)):
            n_failed += 1
            continue

        # d. Fit model
        rng_key, fit_key = random.split(rng_key)
        try:
            fit_result = fit(
                model, y_star, times, method=method, seed=int(fit_key[0]), **fit_kwargs
            )
        except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as exc:
            logger.info("SBC replicate %d: fit failed: %s", rep, exc)
            n_failed += 1
            continue

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
                        parts.append(jnp.asarray(unc).reshape(-1))
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

    n_attempted = n_sbc
    failure_rate = n_failed / n_attempted if n_attempted > 0 else 0.0
    if failure_rate > 0.8:
        raise RuntimeError(
            f"SBC: {n_failed}/{n_attempted} replicates failed ({failure_rate:.0%}) "
            f"— likely a model specification bug"
        )
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
