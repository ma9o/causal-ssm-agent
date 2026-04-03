"""NumPyro State-Space Model.

Bayesian State-Space Model definition using NumPyro.
This module defines the probabilistic model only — inference is in inference.py.

Supports:
- Time-series trajectories
- Any noise family (Gaussian, Poisson, Student-t, Gamma)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

if TYPE_CHECKING:
    import numpy as np

    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime

from causal_ssm_agent.distributions import (
    PriorRuntimeKind,
    get_positive_runtime_family_index,
    get_positive_runtime_kind_from_index,
    get_real_runtime_kind_from_index,
)
from causal_ssm_agent.models.likelihoods.base import CTParams, InitialStateParams, MeasurementParams
from causal_ssm_agent.models.likelihoods.observation_families import any_family_needs_level_metadata
from causal_ssm_agent.models.ssm.assembler import SSMAssembler
from causal_ssm_agent.models.ssm.constants import MIN_DT
from causal_ssm_agent.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
    stabilize_covariance_for_cholesky,
)
from causal_ssm_agent.models.ssm.parameter_names import INITIAL_STATE_CORRELATION_PRIOR_DEFAULTS
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction


@jax.custom_vjp
def _nan_safe_ll(ll):
    """Return ll if finite, else -1e30. Gradient is zeroed when ll is non-finite."""
    return jnp.where(jnp.isfinite(ll), ll, -1e30)


def _nan_safe_ll_fwd(ll):
    y = _nan_safe_ll(ll)
    return y, jnp.isfinite(ll)


def _nan_safe_ll_bwd(is_finite, g):
    return (jnp.where(is_finite, jnp.nan_to_num(g, nan=0.0), 0.0),)


_nan_safe_ll.defvjp(_nan_safe_ll_fwd, _nan_safe_ll_bwd)


@dataclass
class SSMSpec:
    """Specification for a state-space model.

    Matrix naming convention:
    - DRIFT: n_latent x n_latent continuous-time auto/cross effects
    - DIFFUSION: n_latent x n_latent process noise (Cholesky)
    - CINT: n_latent x 1 continuous intercept
    - LAMBDA: n_manifest x n_latent factor loadings
    - MANIFESTMEANS: n_manifest x 1 manifest intercepts
    - MANIFESTVAR: n_manifest x n_manifest measurement error (Cholesky)
    - T0MEANS: n_latent x 1 initial state means
    - T0VAR: n_latent x n_latent initial state variance (Cholesky)
    """

    n_latent: int
    n_manifest: int

    # Fixed or "free" specification for each matrix
    # If a matrix, use those fixed values; if "free", estimate
    drift: jnp.ndarray | Literal["free"] = "free"
    diffusion: jnp.ndarray | Literal["free", "diag"] = "free"
    cint: jnp.ndarray | Literal["free"] | None = None
    lambda_mat: jnp.ndarray | Literal["free"] = "free"
    manifest_means: jnp.ndarray | Literal["free"] | None = None
    manifest_var: jnp.ndarray | Literal["free", "diag"] = "diag"
    t0_means: jnp.ndarray | Literal["free"] = "free"
    t0_var: jnp.ndarray | Literal["free", "diag"] = "free"

    # Distribution families for observation and process noise
    diffusion_dist: DistributionFamily = DistributionFamily.GAUSSIAN
    manifest_dist: DistributionFamily = DistributionFamily.GAUSSIAN

    # Per-variable diffusion noise (overrides scalar diffusion_dist if set)
    diffusion_dists: list[DistributionFamily] | None = None

    # Per-channel observation noise (overrides scalar manifest_dist if set)
    manifest_dists: list[DistributionFamily] | None = None

    # Per-channel number of encoded levels for discrete emissions.
    # Non-discrete channels use 0.
    manifest_level_counts: list[int] | None = None

    # Link function (scalar fallback for all channels)
    manifest_link: LinkFunction = LinkFunction.IDENTITY

    # Per-channel link functions (overrides scalar manifest_link if set)
    manifest_links: list[LinkFunction] | None = None

    # Toggle first-pass (unconditional, model-level) Rao-Blackwellization
    first_pass_rb: bool = True

    # Toggle second-pass (conditional, sampler-level) Rao-Blackwellization
    second_pass_rb: bool = True

    # Parameter names for interpretability
    latent_names: list[str] | None = None
    manifest_names: list[str] | None = None

    # DAG-constrained masks (None = fully free, backward compat)
    # drift_mask: (n_latent, n_latent) bool — True where drift entry is free
    drift_mask: np.ndarray | None = None
    # lambda_mask: (n_manifest, n_latent) bool — True where loading is free to sample
    lambda_mask: np.ndarray | None = None
    # manifest_var_mask: (n_manifest,) bool — True where measurement SD is free
    # to sample on the diagonal of manifest_var. When None, manifest_var follows
    # its global mode ("diag"/"free" = all free, fixed ndarray = all fixed).
    manifest_var_mask: np.ndarray | None = None

    # t0_correlation_mask: (n_latent, n_latent) bool — True on lower-triangle
    # positions where an authored initial-state correlation parameter exists.
    t0_correlation_mask: np.ndarray | None = None

    # Time-invariant latent mask: (n_latent,) bool — True for quasi-constant latents.
    # These get near-zero drift diagonal and near-zero diffusion, so η_i(t) ≈ η_i(0).
    time_invariant_mask: np.ndarray | None = None


@dataclass
class SSMPriors:
    """Prior specifications for state-space model parameters.

    Each prior is specified as a dict with distribution parameters.
    """

    # Drift diagonal (auto-effects, typically negative for stability)
    drift_diag: dict = field(default_factory=lambda: {"mu": -0.5, "sigma": 1.0})
    # Drift off-diagonal (cross-effects)
    drift_offdiag: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 0.5})

    # Diffusion (log scale for positivity)
    diffusion_diag: dict = field(default_factory=lambda: {"sigma": 1.0})
    diffusion_offdiag: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 0.5})

    # Continuous intercept
    cint: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})

    # Factor loadings
    lambda_free: dict = field(default_factory=lambda: {"mu": 0.5, "sigma": 0.5})

    # Manifest means
    manifest_means: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 2.0})

    # Manifest variance (measurement error)
    manifest_var_diag: dict = field(default_factory=lambda: {"sigma": 1.0})

    # Observation-family extras
    obs_df: dict = field(
        default_factory=lambda: {
            "family": get_positive_runtime_family_index(PriorRuntimeKind.GAMMA),
            "concentration": 5.0,
            "rate": 1.0,
        }
    )
    obs_shape: dict = field(
        default_factory=lambda: {
            "family": get_positive_runtime_family_index(PriorRuntimeKind.GAMMA),
            "concentration": 2.0,
            "rate": 1.0,
        }
    )
    obs_r: dict = field(
        default_factory=lambda: {
            "family": get_positive_runtime_family_index(PriorRuntimeKind.GAMMA),
            "concentration": 2.0,
            "rate": 0.5,
        }
    )
    obs_concentration: dict = field(
        default_factory=lambda: {
            "family": get_positive_runtime_family_index(PriorRuntimeKind.GAMMA),
            "concentration": 5.0,
            "rate": 0.5,
        }
    )
    obs_ordered_base: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})
    obs_ordered_gaps: dict = field(default_factory=lambda: {"sigma": 1.0})
    obs_cat_intercepts: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})
    obs_cat_slopes: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})

    # Initial state
    t0_means: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 2.0})
    t0_var_diag: dict = field(default_factory=lambda: {"sigma": 2.0})
    t0_var_offdiag: dict = field(
        default_factory=lambda: dict(INITIAL_STATE_CORRELATION_PRIOR_DEFAULTS)
    )


def assemble_sampled_extra_params(
    spec: SSMSpec,
    sampled_values: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Assemble likelihood hyperparameters and derived observation metadata."""
    extra_params: dict[str, jnp.ndarray] = {}
    manifest_dists = spec.manifest_dists or [spec.manifest_dist] * spec.n_manifest
    manifest_dist_set = set(manifest_dists)

    scalar_keys = (
        "obs_df",
        "obs_shape",
        "obs_r",
        "obs_concentration",
        "proc_df",
    )
    for key in scalar_keys:
        if key in sampled_values:
            extra_params[key] = sampled_values[key]

    if spec.manifest_level_counts is None:
        return extra_params

    level_counts_list = list(spec.manifest_level_counts)
    level_counts = jnp.asarray(level_counts_list, dtype=jnp.int32)
    extra_params["obs_level_counts"] = level_counts

    max_levels = max(level_counts_list) if level_counts_list else 0
    max_cutpoints = max(max_levels - 1, 0)

    if any_family_needs_level_metadata(manifest_dist_set) and max_cutpoints <= 0:
        raise ValueError(
            "ordered_logistic/categorical requires manifest_level_counts with at least 2 levels"
        )

    if DistributionFamily.ORDERED_LOGISTIC in manifest_dist_set:
        ordered_base = sampled_values["obs_ordered_base"]
        if max_cutpoints > 1:
            ordered_gaps = sampled_values["obs_ordered_gaps"]
        else:
            ordered_gaps = jnp.zeros((spec.n_manifest, 0), dtype=ordered_base.dtype)

        raw_cutpoints = jnp.concatenate(
            [
                ordered_base[:, None],
                ordered_base[:, None] + jnp.cumsum(ordered_gaps, axis=1),
            ],
            axis=1,
        )
        cutpoint_mask = jnp.arange(max_cutpoints)[None, :] < jnp.maximum(
            level_counts[:, None] - 1, 0
        )
        cutpoint_sum = jnp.sum(jnp.where(cutpoint_mask, raw_cutpoints, 0.0), axis=1)
        cutpoint_count = jnp.maximum(level_counts - 1, 1)
        cutpoint_center = cutpoint_sum / cutpoint_count
        extra_params["obs_ordered_cutpoints"] = jnp.where(
            cutpoint_mask,
            raw_cutpoints - cutpoint_center[:, None],
            0.0,
        )

    if DistributionFamily.CATEGORICAL in manifest_dist_set:
        cat_mask = jnp.arange(max_cutpoints)[None, :] < jnp.maximum(level_counts[:, None] - 1, 0)
        extra_params["obs_cat_intercepts"] = jnp.where(
            cat_mask,
            sampled_values["obs_cat_intercepts"],
            0.0,
        )
        extra_params["obs_cat_slopes"] = jnp.where(
            cat_mask,
            sampled_values["obs_cat_slopes"],
            0.0,
        )

    return extra_params


def _make_prior_dist(prior: dict) -> dist.Distribution:
    """Build the appropriate numpyro distribution from a prior dict.

    If `lower`/`upper` bounds are present, uses TruncatedNormal to respect
    hard parameter bounds. Otherwise dispatches via serialized executable
    prior metadata or falls back to Normal.
    depending on the serialized prior semantics.

    Supports array-valued mu/sigma for per-element priors.
    """
    family = prior.get("family", 0)
    if isinstance(family, list):
        unique_families = {int(value) for value in family}
        if len(unique_families) != 1:
            raise ValueError("Mixed prior families within a single SSM field are unsupported")
        family = unique_families.pop()
    if "mu" in prior or "lower" in prior or "upper" in prior:
        if "family" in prior:
            runtime_kind = get_real_runtime_kind_from_index(int(family))
            if runtime_kind == PriorRuntimeKind.NORMAL:
                return dist.Normal(jnp.asarray(prior["mu"]), jnp.asarray(prior["sigma"]))
            if runtime_kind == PriorRuntimeKind.TRUNCATED_NORMAL:
                return dist.TruncatedNormal(
                    loc=jnp.asarray(prior["mu"]),
                    scale=jnp.asarray(prior["sigma"]),
                    low=jnp.asarray(prior["lower"]),
                    high=jnp.asarray(prior["upper"]),
                )
            if runtime_kind == PriorRuntimeKind.UNIFORM:
                return dist.Uniform(
                    low=jnp.asarray(prior["lower"]),
                    high=jnp.asarray(prior["upper"]),
                )
            raise ValueError(f"Unsupported serialized real prior runtime kind {runtime_kind!r}")
        if "lower" in prior and "upper" in prior:
            return dist.TruncatedNormal(
                loc=jnp.asarray(prior["mu"]),
                scale=jnp.asarray(prior["sigma"]),
                low=jnp.asarray(prior["lower"]),
                high=jnp.asarray(prior["upper"]),
            )
        return dist.Normal(jnp.asarray(prior["mu"]), jnp.asarray(prior["sigma"]))
    if "family" in prior:
        runtime_kind = get_positive_runtime_kind_from_index(int(family))
        if runtime_kind == PriorRuntimeKind.HALF_NORMAL:
            return dist.HalfNormal(jnp.asarray(prior["sigma"]))
        if runtime_kind == PriorRuntimeKind.GAMMA:
            return dist.Gamma(
                concentration=jnp.asarray(prior.get("concentration", 2.0)),
                rate=jnp.asarray(prior.get("rate", 1.0)),
            )
        if runtime_kind == PriorRuntimeKind.LOG_NORMAL:
            return dist.LogNormal(
                loc=jnp.asarray(prior.get("loc", 0.0)),
                scale=jnp.asarray(prior.get("sigma", 1.0)),
            )
        if runtime_kind == PriorRuntimeKind.EXPONENTIAL:
            return dist.Exponential(rate=jnp.asarray(prior.get("rate", 1.0)))
        raise ValueError(f"Unsupported serialized positive prior runtime kind {runtime_kind!r}")
    if {"concentration", "rate"} <= set(prior):
        return dist.Gamma(
            concentration=jnp.asarray(prior.get("concentration", 2.0)),
            rate=jnp.asarray(prior.get("rate", 1.0)),
        )
    return dist.HalfNormal(jnp.asarray(prior["sigma"]))


def _make_prior_batch(prior: dict, n: int) -> dist.Distribution:
    """Build a batched prior distribution with shape (n,).

    If prior already has array-valued params with length n, use directly.
    If scalar, expand to batch shape [n].
    """
    d = _make_prior_dist(prior)
    if d.batch_shape == (n,):
        return d
    if d.batch_shape == ():
        return d.expand((n,))
    raise ValueError(f"Prior batch shape {d.batch_shape} does not match expected ({n},)")


class SSMModel:
    """NumPyro state-space model definition.

    Defines the probabilistic model for Bayesian state-space models.
    Inference is handled externally by ssm.inference.fit().

    Features:
    - Continuous-time dynamics via stochastic differential equations
    - Multiple likelihood backends (Kalman, particle filter)
    """

    def __init__(
        self,
        spec: SSMSpec,
        priors: SSMPriors | None = None,
        n_particles: int = 200,
        pf_seed: int = 0,
        likelihood: Literal["particle", "kalman"] = "particle",
    ):
        """Initialize state-space model.

        Args:
            spec: Model specification
            priors: Prior distributions (uses defaults if None)
            n_particles: Number of particles for bootstrap PF
            pf_seed: Seed for fixed PF random key (deterministic for NUTS)
            likelihood: Likelihood backend - "particle" (universal, any noise family)
                or "kalman" (exact, linear Gaussian only)
        """
        self.spec = spec
        self.priors = priors or SSMPriors()
        self.n_particles = n_particles
        self.pf_key = jax.random.PRNGKey(pf_seed)
        self.likelihood = likelihood
        self._assembler = SSMAssembler(spec)
        self._artifact_cache: dict[tuple[Any, ...], Any] = {}
        self.observation_support: ObservationSupportRuntime | None = None

    def get_cached_artifact(self, cache_key: tuple[Any, ...], factory) -> Any:
        """Construct an artifact once per model instance and reuse it afterwards."""
        if cache_key not in self._artifact_cache:
            self._artifact_cache[cache_key] = factory()
        return self._artifact_cache[cache_key]

    def set_observation_support(
        self, observation_support: ObservationSupportRuntime | None
    ) -> None:
        """Attach prepared observation-support metadata and invalidate backend caches."""
        self.observation_support = observation_support
        self._artifact_cache = {
            key: value
            for key, value in self._artifact_cache.items()
            if not (isinstance(key, tuple) and key and key[0] == "backend")
        }

    def _sample_drift(self, spec: SSMSpec) -> jnp.ndarray:
        """Sample drift matrix with stability constraints."""
        n = spec.n_latent

        if isinstance(spec.drift, jnp.ndarray):
            return spec.drift

        asm = self._assembler
        n_offdiag = len(asm.offdiag_positions)

        drift_diag_pop = numpyro.sample(
            "drift_diag_pop",
            _make_prior_batch(self.priors.drift_diag, n),
        )

        if n_offdiag > 0:
            drift_offdiag_pop = jnp.asarray(
                numpyro.sample(
                    "drift_offdiag_pop",
                    _make_prior_batch(self.priors.drift_offdiag, n_offdiag),
                )
            )
        else:
            drift_offdiag_pop = jnp.array([])

        drift = asm.assemble_drift(drift_diag_pop, drift_offdiag_pop)

        # Stability guard: penalise drift matrices whose max real eigenvalue
        # approaches zero (i.e. the system is near-unstable).  Only needed
        # for multi-latent models with off-diagonal coupling.
        if n > 1 and n_offdiag > 0:
            eigvals_real = jnp.real(jnp.linalg.eigvals(drift))
            max_eig = jnp.max(eigvals_real)
            margin = 1e-2
            penalty = jnp.where(
                max_eig > -margin,
                -1e4 * jnp.maximum(max_eig + margin, 0.0),
                0.0,
            )
            numpyro.factor("drift_stability", penalty)

        numpyro.deterministic("drift", drift)
        return drift

    def _sample_diffusion(self, spec: SSMSpec) -> jnp.ndarray:
        """Sample diffusion matrix (lower Cholesky)."""
        n = spec.n_latent

        if isinstance(spec.diffusion, jnp.ndarray):
            return spec.diffusion

        diff_diag_pop = numpyro.sample(
            "diffusion_diag_pop",
            dist.HalfNormal(jnp.asarray(self.priors.diffusion_diag["sigma"])).expand((n,)),
        )

        diff_lower = None
        if spec.diffusion != "diag":
            n_lower = n * (n - 1) // 2
            if n_lower > 0:
                diff_lower = jnp.asarray(
                    numpyro.sample(
                        "diffusion_lower",
                        _make_prior_batch(self.priors.diffusion_offdiag, n_lower),
                    )
                )

        diffusion = self._assembler.assemble_diffusion(diff_diag_pop, diff_lower)

        numpyro.deterministic("diffusion", diffusion)
        return diffusion

    def _sample_cint(self, spec: SSMSpec) -> jnp.ndarray | None:
        """Sample continuous intercept."""
        if spec.cint is None:
            return None

        n = spec.n_latent

        if isinstance(spec.cint, jnp.ndarray):
            return spec.cint

        cint = numpyro.sample(
            "cint_pop",
            _make_prior_dist(self.priors.cint).expand((n,)),
        )

        numpyro.deterministic("cint", cint)
        return jnp.asarray(cint)

    def _sample_lambda(self, spec: SSMSpec) -> jnp.ndarray:
        """Sample factor loading matrix (shared across subjects).

        Three modes (determined by SSMAssembler from spec):
        1. Template+mask: sample free loadings at masked positions.
        2. Fixed: return template as-is (no sampling).
        3. Legacy: identity + extra rows filled with sampled loadings.
        """
        # Fully fixed (array with no mask): return as-is
        if isinstance(spec.lambda_mat, jnp.ndarray) and spec.lambda_mask is None:
            return spec.lambda_mat

        asm = self._assembler
        n_free = len(asm.lambda_free_positions)

        free_loadings = None
        if n_free > 0:
            free_loadings = jnp.asarray(
                numpyro.sample(
                    "lambda_free",
                    _make_prior_batch(self.priors.lambda_free, n_free),
                )
            )

        lambda_mat = asm.assemble_lambda(free_loadings)
        numpyro.deterministic("lambda", lambda_mat)
        return lambda_mat

    def _sample_manifest_params(self, spec: SSMSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample manifest means and variance (shared across subjects)."""
        n_m = spec.n_manifest

        # Means
        if spec.manifest_means is None:
            manifest_means = jnp.zeros(n_m)
        elif isinstance(spec.manifest_means, jnp.ndarray):
            manifest_means = spec.manifest_means
        else:
            manifest_means = numpyro.sample(
                "manifest_means",
                _make_prior_dist(self.priors.manifest_means).expand((n_m,)),
            )

        # Variance (Cholesky)
        if isinstance(spec.manifest_var, jnp.ndarray) and spec.manifest_var_mask is None:
            manifest_chol = spec.manifest_var
        else:
            n_free = len(self._assembler.manifest_var_free_positions)
            if n_free > 0:
                var_diag = numpyro.sample(
                    "manifest_var_diag",
                    dist.HalfNormal(self.priors.manifest_var_diag["sigma"]).expand((n_free,)),
                )
                manifest_chol = self._assembler.assemble_manifest_chol(var_diag)
            else:
                manifest_chol = self._assembler.manifest_var_template

        numpyro.deterministic("manifest_cov", manifest_chol @ manifest_chol.T)
        return jnp.asarray(manifest_means), jnp.asarray(manifest_chol)

    def _sample_t0_params(self, spec: SSMSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample initial state parameters."""
        n_l = spec.n_latent

        # Means
        if isinstance(spec.t0_means, jnp.ndarray):
            t0_means = spec.t0_means
        else:
            t0_means = numpyro.sample(
                "t0_means_pop",
                _make_prior_dist(self.priors.t0_means).expand((n_l,)),
            )

        # Variance (Cholesky)
        if isinstance(spec.t0_var, jnp.ndarray):
            t0_chol = spec.t0_var
        else:
            var_diag = numpyro.sample(
                "t0_var_diag",
                dist.HalfNormal(self.priors.t0_var_diag["sigma"]).expand((n_l,)),
            )
            t0_corr = None
            if spec.t0_var != "diag":
                n_corr = len(self._assembler.t0_correlation_positions)
                if n_corr > 0:
                    t0_corr = numpyro.sample(
                        "t0_var_lower",
                        _make_prior_dist(self.priors.t0_var_offdiag).expand((n_corr,)),
                    )
            t0_cov_raw = self._assembler.assemble_t0_cov(var_diag, t0_corr)
            t0_cov, min_eig = stabilize_covariance_for_cholesky(
                t0_cov_raw,
                min_eigenvalue=INITIAL_STATE_COV_MIN_EIGENVALUE,
            )
            numpyro.factor(
                "t0_correlation_positive_definite",
                jnp.where(
                    min_eig > INITIAL_STATE_COV_MIN_EIGENVALUE,
                    0.0,
                    -1e6 * (INITIAL_STATE_COV_MIN_EIGENVALUE - min_eig),
                ),
            )
            t0_chol = jnp.linalg.cholesky(t0_cov)

        numpyro.deterministic("t0_means", t0_means)
        numpyro.deterministic("t0_cov", t0_chol @ t0_chol.T)
        return jnp.asarray(t0_means), jnp.asarray(t0_chol)

    def make_likelihood_backend(self):
        """Construct the default likelihood backend from model configuration.

        Delegates to the standalone ``make_likelihood_backend`` factory.
        Callers that need a different backend (Laplace, Structured VI, DPF)
        construct it themselves instead of calling this.
        """
        return self.get_cached_artifact(
            ("backend", self.likelihood, self.n_particles),
            lambda: make_likelihood_backend(
                self.spec,
                self.likelihood,
                self.n_particles,
                self.pf_key,
                observation_support=self.observation_support,
            ),
        )

    def make_laplace_backend(self, n_ieks_iters: int):
        """Construct or reuse the Laplace likelihood backend for this model."""
        return self.get_cached_artifact(
            (
                "backend",
                "laplace",
                n_ieks_iters,
                id(self.observation_support),
            ),
            lambda: _build_laplace_backend(
                self.spec,
                n_ieks_iters,
                observation_support=self.observation_support,
            ),
        )

    def _sample_likelihood_extra_params(self, spec: SSMSpec) -> dict[str, jnp.ndarray]:
        """Sample likelihood hyperparameters and assemble backend-ready extras."""
        sampled_values: dict[str, jnp.ndarray] = {}
        manifest_dists = spec.manifest_dists or [spec.manifest_dist] * spec.n_manifest
        manifest_dist_set = set(manifest_dists)

        if DistributionFamily.STUDENT_T in manifest_dist_set:
            sampled_values["obs_df"] = numpyro.sample(
                "obs_df",
                _make_prior_dist(self.priors.obs_df),
            )
        if DistributionFamily.GAMMA in manifest_dist_set:
            sampled_values["obs_shape"] = numpyro.sample(
                "obs_shape",
                _make_prior_dist(self.priors.obs_shape),
            )
        if DistributionFamily.NEGATIVE_BINOMIAL in manifest_dist_set:
            sampled_values["obs_r"] = numpyro.sample(
                "obs_r",
                _make_prior_dist(self.priors.obs_r),
            )
        if DistributionFamily.BETA in manifest_dist_set:
            sampled_values["obs_concentration"] = numpyro.sample(
                "obs_concentration",
                _make_prior_dist(self.priors.obs_concentration),
            )

        if spec.manifest_level_counts is not None:
            level_counts_list = list(spec.manifest_level_counts)
            max_levels = max(level_counts_list) if level_counts_list else 0
            max_cutpoints = max(max_levels - 1, 0)

            if any_family_needs_level_metadata(manifest_dist_set) and max_cutpoints <= 0:
                raise ValueError(
                    "ordered_logistic/categorical requires manifest_level_counts with at least 2 levels"
                )

            if DistributionFamily.ORDERED_LOGISTIC in manifest_dist_set:
                sampled_values["obs_ordered_base"] = numpyro.sample(
                    "obs_ordered_base",
                    _make_prior_dist(self.priors.obs_ordered_base).expand((spec.n_manifest,)),
                )
                if max_cutpoints > 1:
                    sampled_values["obs_ordered_gaps"] = numpyro.sample(
                        "obs_ordered_gaps",
                        _make_prior_dist(self.priors.obs_ordered_gaps).expand(
                            (spec.n_manifest, max_cutpoints - 1)
                        ),
                    )

            if DistributionFamily.CATEGORICAL in manifest_dist_set:
                cat_shape = (spec.n_manifest, max_cutpoints)
                sampled_values["obs_cat_intercepts"] = numpyro.sample(
                    "obs_cat_intercepts",
                    _make_prior_dist(self.priors.obs_cat_intercepts).expand(cat_shape),
                )
                sampled_values["obs_cat_slopes"] = numpyro.sample(
                    "obs_cat_slopes",
                    _make_prior_dist(self.priors.obs_cat_slopes).expand(cat_shape),
                )

        from causal_ssm_agent.models.likelihoods.graph_analysis import has_student_t_diffusion

        if has_student_t_diffusion(spec):
            sampled_values["proc_df"] = numpyro.sample("proc_df", dist.Gamma(5.0, 1.0))

        return assemble_sampled_extra_params(spec, sampled_values)

    def model(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        likelihood_backend=None,
    ) -> None:
        """NumPyro model function.

        Args:
            observations: (N, n_manifest) observed data
            times: (N,) observation times
            likelihood_backend: Likelihood backend instance (e.g. ParticleLikelihood,
                KalmanLikelihood, LaplaceLikelihood). Required — use
                model.make_likelihood_backend() for the default.
        """
        if likelihood_backend is None:
            raise ValueError(
                "likelihood_backend is required. "
                "Use model.make_likelihood_backend() for the default."
            )

        spec = self.spec

        drift = self._sample_drift(spec)
        diffusion_chol = self._sample_diffusion(spec)
        cint = self._sample_cint(spec)
        lambda_mat = self._sample_lambda(spec)
        manifest_means, manifest_chol = self._sample_manifest_params(spec)
        t0_means, t0_chol = self._sample_t0_params(spec)

        diffusion_cov = diffusion_chol @ diffusion_chol.T
        manifest_cov = manifest_chol @ manifest_chol.T
        t0_cov = t0_chol @ t0_chol.T
        extra_params = self._sample_likelihood_extra_params(spec)

        ct_params = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=cint)
        meas_params = MeasurementParams(
            lambda_mat=lambda_mat,
            manifest_means=manifest_means,
            manifest_cov=manifest_cov,
        )

        time_intervals = jnp.diff(times, prepend=times[0])
        time_intervals = time_intervals.at[0].set(MIN_DT)

        init = InitialStateParams(mean=t0_means, cov=t0_cov)
        lnc = likelihood_backend.compute_log_likelihood(
            ct_params,
            meas_params,
            init,
            observations,
            time_intervals,
            extra_params=extra_params or None,
        )

        # lnc is (T,) cumulative log-normalizing constants from the filter.
        # lnc[-1] = total log p(y|θ).
        # diff(lnc) = per-timestep one-step-ahead predictive log p(y_t|y_{1:t-1},θ),
        # needed for proper LOO-CV on time series (innovation decomposition).
        if lnc.ndim == 0:
            total_ll = _nan_safe_ll(lnc)
            numpyro.factor("log_likelihood", total_ll)
        else:
            total_ll = _nan_safe_ll(lnc[-1])
            numpyro.factor("log_likelihood", total_ll)
            ll_per_timestep = jnp.diff(lnc, prepend=0.0)
            numpyro.deterministic("ll_per_timestep", ll_per_timestep)


def _build_laplace_backend(
    spec: SSMSpec,
    n_ieks_iters: int,
    observation_support: ObservationSupportRuntime | None = None,
):
    from causal_ssm_agent.models.likelihoods.graph_analysis import (
        get_per_channel_links,
        get_per_channel_manifest,
    )
    from causal_ssm_agent.models.ssm.laplace_em import LaplaceLikelihood

    return LaplaceLikelihood(
        n_latent=spec.n_latent,
        n_manifest=spec.n_manifest,
        manifest_dists=get_per_channel_manifest(spec),
        manifest_links=get_per_channel_links(spec),
        n_ieks_iters=n_ieks_iters,
        observation_support=observation_support,
    )


# ---------------------------------------------------------------------------
# Standalone likelihood backend factory
# ---------------------------------------------------------------------------


def make_likelihood_backend(
    spec: SSMSpec,
    likelihood: Literal["particle", "kalman"] = "particle",
    n_particles: int = 200,
    pf_key: jnp.ndarray | None = None,
    observation_support: ObservationSupportRuntime | None = None,
):
    """Construct a likelihood backend from model configuration.

    Selects between Kalman and Particle backends. When ``first_pass_rb`` is
    enabled, the shared inference-structure planner may route a decoupled
    linear-Gaussian sub-block to exact Kalman filtering while the remainder
    uses a particle filter.

    Args:
        spec: SSM specification
        likelihood: Backend type — "kalman" (exact, linear Gaussian) or
            "particle" (universal, any noise family)
        n_particles: Number of particles for bootstrap PF
        pf_key: Fixed JAX PRNG key for the particle filter
    """
    if pf_key is None:
        pf_key = jax.random.PRNGKey(0)

    from causal_ssm_agent.models.ssm.inference_structure import plan_inference_structure

    inference_structure = plan_inference_structure(
        spec,
        likelihood=likelihood,
        observation_support=observation_support,
    )

    if inference_structure.likelihood_path == "kalman":
        from causal_ssm_agent.models.likelihoods.kalman import KalmanLikelihood

        return KalmanLikelihood(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
        )

    # Resolve per-variable distributions for ParticleLikelihood
    from causal_ssm_agent.models.likelihoods.graph_analysis import (
        get_per_channel_links,
        get_per_channel_manifest,
        get_per_variable_diffusion,
    )

    per_var = get_per_variable_diffusion(spec)
    per_obs = get_per_channel_manifest(spec)
    per_links = get_per_channel_links(spec)

    if inference_structure.likelihood_path == "composed":
        from causal_ssm_agent.models.likelihoods.composed import ComposedLikelihood
        from causal_ssm_agent.models.likelihoods.kalman import KalmanLikelihood
        from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

        partition = inference_structure.first_pass_rb.partition
        if partition is None:
            raise ValueError("Composed likelihood path requires an active first-pass partition")

        n_k = len(partition.kalman_idx)
        n_obs_k = len(partition.obs_kalman_idx)
        n_p = len(partition.particle_idx)
        n_obs_p = len(partition.obs_particle_idx)

        particle_diffs = [per_var[int(i)] for i in partition.particle_idx]

        pf_manifest_dist = spec.manifest_dist
        for k in partition.obs_particle_idx:
            if per_obs[int(k)] != DistributionFamily.GAUSSIAN:
                pf_manifest_dist = per_obs[int(k)]
                break

        pf_manifest_link = spec.manifest_link
        for k in partition.obs_particle_idx:
            if per_links[int(k)] != LinkFunction.IDENTITY:
                pf_manifest_link = per_links[int(k)]
                break

        return ComposedLikelihood(
            partition=partition,
            kalman_backend=KalmanLikelihood(
                n_latent=n_k,
                n_manifest=n_obs_k,
            ),
            particle_backend=ParticleLikelihood(
                n_latent=n_p,
                n_manifest=n_obs_p,
                n_particles=n_particles,
                rng_key=pf_key,
                manifest_dist=pf_manifest_dist,
                diffusion_dist=particle_diffs,
                block_rb=spec.second_pass_rb,
                manifest_link=pf_manifest_link,
                observation_support=None,
            ),
        )

    # Fallthrough: full particle filter
    from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

    return ParticleLikelihood(
        n_latent=spec.n_latent,
        n_manifest=spec.n_manifest,
        n_particles=n_particles,
        rng_key=pf_key,
        manifest_dist=spec.manifest_dist,
        diffusion_dist=per_var,
        block_rb=False
        if observation_support is not None
        and observation_support.requires_interval_summary_handling
        else spec.second_pass_rb,
        manifest_link=spec.manifest_link,
        observation_support=observation_support,
    )
