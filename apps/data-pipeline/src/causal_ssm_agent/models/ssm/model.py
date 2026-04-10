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
import numpy as np
import numpyro
import numpyro.distributions as dist

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime

from causal_ssm_agent.artifacts.model_spec import DistributionFamily, LinkFunction
from causal_ssm_agent.distributions import (
    PriorDistributionFamily,
    get_positive_runtime_kind_from_index,
    get_real_runtime_kind_from_index,
)
from causal_ssm_agent.models.ssm.constants import MIN_DT
from causal_ssm_agent.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
    stabilize_covariance_for_cholesky,
)
from causal_ssm_agent.models.ssm.inference.backend_factory import (
    build_laplace_backend,
    make_likelihood_backend,
)
from causal_ssm_agent.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from causal_ssm_agent.models.ssm.inference.targets.observation_families import (
    any_family_needs_level_metadata,
)
from causal_ssm_agent.models.ssm.likelihood_extra_params import (
    assemble_sampled_extra_params,
)
from causal_ssm_agent.models.ssm.parameterization import (
    PriorRuntimeBundle,
    build_prior_runtime_bundle,
    build_site_prior_distribution,
)
from causal_ssm_agent.models.ssm.priors import SSMPriors
from causal_ssm_agent.models.ssm.structure_runtime import SSMStructureRuntime


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


def _sample_prior_array(site_name: str, prior: dist.Distribution) -> jnp.ndarray:
    """Sample one prior site and normalize it to a concrete JAX array."""
    return jnp.asarray(numpyro.sample(site_name, prior))


def full_drift_offdiag_mask(n_latent: int) -> np.ndarray:
    """Return the fully free off-diagonal structural support mask for a drift matrix."""
    mask = np.ones((n_latent, n_latent), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def zero_loading_mask(n_manifest: int, n_latent: int) -> np.ndarray:
    """Return the zero free-loading mask for a fixed loading template."""
    return np.zeros((n_manifest, n_latent), dtype=bool)


def full_vector_mask(n: int) -> np.ndarray:
    """Return a fully free vector support mask."""
    return np.ones(n, dtype=bool)


def zero_vector_mask(n: int) -> np.ndarray:
    """Return a zero vector support mask."""
    return np.zeros(n, dtype=bool)


def full_diagonal_mask(n: int) -> np.ndarray:
    """Return a fully free diagonal support mask."""
    return np.ones(n, dtype=bool)


def zero_diagonal_mask(n: int) -> np.ndarray:
    """Return a zero diagonal support mask."""
    return np.zeros(n, dtype=bool)


def full_cholesky_mask(n: int) -> np.ndarray:
    """Return the fully free lower-Cholesky support mask."""
    return np.tri(n, dtype=bool)


def strict_lower_triangle_mask(n: int) -> np.ndarray:
    """Return the strict lower-triangle support mask."""
    return np.tri(n, k=-1, dtype=bool)


def zero_square_mask(n: int) -> np.ndarray:
    """Return the zero square support mask."""
    return np.zeros((n, n), dtype=bool)


@dataclass
class SSMSpec:
    """Specification for a state-space model.

    Matrices:
    - drift (A): n_latent x n_latent continuous-time auto/cross effects
    - diffusion_chol (L_Q): n_latent x n_latent process noise Cholesky factor
    - cint (c): n_latent x 1 continuous intercept
    - lambda_mat (Λ): n_manifest x n_latent factor loadings
    - manifest_means (μ): n_manifest x 1 manifest intercepts
    - manifest_chol (L_R): n_manifest x n_manifest measurement-error Cholesky factor
    - t0_means (η₀): n_latent x 1 initial state means
    - t0_chol (L_0): n_latent x n_latent initial-state Cholesky factor

    Distributions:
    - diffusion_dists (Nₛ): per-latent process noise family
    - manifest_dists (Fᵢ): per-channel observation family
    - manifest_links (gᵢ): per-channel link
    - hᵢ: extra observation parameters required by Fᵢ (for example df, shape, r, concentration, cutpoints, or categorical logits)

    State:       dη(t) = (A η(t) + c) dt + L_Q dNₛ(t)
                 η(0) ~ N(t0_means, L_0 L_0ᵀ)

    Linear pred: ξᵢ(t) = (Λ η(t) + μ)ᵢ
    Mean param:  mᵢ(t) = gᵢ⁻¹(ξᵢ(t))
    Emission:    yᵢ(t) ~ Fᵢ(mᵢ(t); hᵢ, (L_R L_Rᵀ)ᵢᵢ)
    """

    n_latent: int
    n_manifest: int
    drift_diag_mask: np.ndarray
    drift_offdiag_mask: np.ndarray
    drift: jnp.ndarray
    cint_mask: np.ndarray
    cint: jnp.ndarray
    lambda_mask: np.ndarray
    lambda_mat: jnp.ndarray
    diffusion_chol_mask: np.ndarray
    diffusion_chol: jnp.ndarray
    manifest_means_mask: np.ndarray
    manifest_means: jnp.ndarray
    manifest_chol_diag_mask: np.ndarray
    manifest_chol: jnp.ndarray
    t0_means_mask: np.ndarray
    t0_means: jnp.ndarray
    t0_chol_diag_mask: np.ndarray
    t0_correlation_mask: np.ndarray
    t0_chol: jnp.ndarray
    static_state_sd_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    static_state_sds: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=jnp.float64))
    static_factor_loadings: jnp.ndarray = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float64)
    )

    # Per-variable diffusion noise families.
    diffusion_dists: list[DistributionFamily] = field(default_factory=list)

    # Per-channel observation noise families.
    manifest_dists: list[DistributionFamily] = field(default_factory=list)

    # Per-channel number of encoded levels for discrete emissions.
    # Non-discrete channels use 0.
    manifest_level_counts: list[int] | None = None

    # Per-channel link functions. When omitted, each channel uses the
    # default link for its observation family.
    manifest_links: list[LinkFunction] | None = None
    manifest_centered: list[bool] | None = None

    # Toggle first-pass (unconditional, model-level) Rao-Blackwellization
    first_pass_rb: bool = True

    # Toggle second-pass (conditional, sampler-level) Rao-Blackwellization
    second_pass_rb: bool = True

    # Parameter names for interpretability
    latent_names: list[str] | None = None
    manifest_names: list[str] | None = None
    static_factor_names: list[str] | None = None
    initialization_policy: str = "stationary"

    # drift_diag_mask: (n_latent,) bool — True where the diagonal self-dynamics
    # remain free to sample.

    # drift_offdiag_mask: (n_latent, n_latent) bool — True on off-diagonal
    # structural couplings that remain free to sample.

    # diffusion_chol_mask: (n_latent, n_latent) bool — True on lower-Cholesky
    # entries that remain free to sample.

    # cint_mask: (n_latent,) bool — True where the continuous intercept
    # remains free to sample.

    # static_state_sd_mask: (n_static_factor,) bool — True where compiled
    # baseline-factor SDs remain free to sample.

    # manifest_means_mask: (n_manifest,) bool — True where manifest
    # intercepts remain free to sample.

    # t0_means_mask: (n_latent,) bool — True where initial-state means
    # remain free to sample.

    # manifest_chol_diag_mask: (n_manifest,) bool — True where the diagonal
    # manifest-noise standard deviation remains free to sample.

    # t0_chol_diag_mask: (n_latent,) bool — True where initial-state standard
    # deviations remain free to sample.

    # t0_correlation_mask: (n_latent, n_latent) bool — True on strict lower
    # positions where initial-state correlations remain free to sample.

    # Time-invariant latent mask: (n_latent,) bool — True for quasi-constant latents.
    # These get near-zero drift diagonal and near-zero diffusion, so η_i(t) ≈ η_i(0).
    # When None, no latent is treated as quasi-constant.
    time_invariant_mask: np.ndarray | None = None

    def _coerce_drift_diag_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (self.n_latent,):
            raise ValueError(
                f"drift_diag_mask must have shape ({self.n_latent},), got {mask_array.shape}"
            )
        return mask_array

    def _coerce_drift_offdiag_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (self.n_latent, self.n_latent):
            raise ValueError(
                "drift_offdiag_mask must have shape "
                f"({self.n_latent}, {self.n_latent}), got {mask_array.shape}"
            )
        if bool(np.diag(mask_array).any()):
            raise ValueError("drift_offdiag_mask must have a zero diagonal.")
        return mask_array

    def _coerce_lambda_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (self.n_manifest, self.n_latent):
            raise ValueError(
                "lambda_mask must have shape "
                f"({self.n_manifest}, {self.n_latent}), got {mask_array.shape}"
            )
        return mask_array

    def _coerce_lambda_mat(self, lambda_mat: jnp.ndarray) -> jnp.ndarray:
        if isinstance(lambda_mat, str):
            raise ValueError(
                "lambda_mat must be an explicit loading template array. "
                "Use lambda_mask to mark free loadings."
            )
        lambda_array = jnp.asarray(lambda_mat)
        if lambda_array.shape != (self.n_manifest, self.n_latent):
            raise ValueError(
                "lambda_mat must have shape "
                f"({self.n_manifest}, {self.n_latent}), got {lambda_array.shape}"
            )
        return lambda_array

    def _coerce_vector_template(self, name: str, value: jnp.ndarray, dim: int) -> jnp.ndarray:
        if isinstance(value, str):
            raise ValueError(f"{name} must be an explicit vector template array.")
        value_array = jnp.asarray(value)
        if value_array.shape != (dim,):
            raise ValueError(f"{name} must have shape ({dim},), got {value_array.shape}")
        return value_array

    def _coerce_factor_loadings(self, value: jnp.ndarray) -> jnp.ndarray:
        value_array = jnp.asarray(value)
        if value_array.ndim != 2:
            raise ValueError("static_factor_loadings must be a rank-2 array.")
        if value_array.shape[0] not in {0, self.n_latent}:
            raise ValueError(
                "static_factor_loadings must have shape "
                f"({self.n_latent}, n_factor), got {value_array.shape}"
            )
        if value_array.shape[0] == 0 and value_array.shape[1] == 0:
            return jnp.zeros((self.n_latent, 0), dtype=jnp.float64)
        if value_array.shape[0] != self.n_latent:
            raise ValueError(
                "static_factor_loadings must have shape "
                f"({self.n_latent}, n_factor), got {value_array.shape}"
            )
        return value_array

    def _coerce_square_template(self, name: str, value: jnp.ndarray, dim: int) -> jnp.ndarray:
        if isinstance(value, str):
            raise ValueError(f"{name} must be an explicit matrix template array.")
        value_array = jnp.asarray(value)
        if value_array.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim}), got {value_array.shape}")
        return value_array

    def _coerce_diagonal_mask(self, name: str, mask: np.ndarray, dim: int) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (dim,):
            raise ValueError(f"{name} must have shape ({dim},), got {mask_array.shape}")
        return mask_array

    def _coerce_cholesky_mask(self, name: str, mask: np.ndarray, dim: int) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim}), got {mask_array.shape}")
        if bool(np.triu(mask_array, k=1).any()):
            raise ValueError(f"{name} must only mark lower-Cholesky entries.")
        return mask_array

    def _coerce_strict_lower_mask(self, name: str, mask: np.ndarray, dim: int) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim}), got {mask_array.shape}")
        if bool(np.triu(mask_array, k=0).any()):
            raise ValueError(f"{name} must only mark strict lower-triangle entries.")
        return mask_array

    def __post_init__(self) -> None:
        """Validate structural masks and canonicalize per-channel family metadata."""
        self.drift_diag_mask = self._coerce_diagonal_mask(
            "drift_diag_mask",
            self.drift_diag_mask,
            self.n_latent,
        )
        self.drift_offdiag_mask = self._coerce_drift_offdiag_mask(self.drift_offdiag_mask)
        self.drift = self._coerce_square_template("drift", self.drift, self.n_latent)
        self.cint_mask = self._coerce_diagonal_mask("cint_mask", self.cint_mask, self.n_latent)
        self.cint = self._coerce_vector_template("cint", self.cint, self.n_latent)
        self.static_factor_loadings = self._coerce_factor_loadings(self.static_factor_loadings)
        n_static_factor = self.static_factor_loadings.shape[1]
        self.static_state_sd_mask = self._coerce_diagonal_mask(
            "static_state_sd_mask",
            (
                self.static_state_sd_mask
                if np.asarray(self.static_state_sd_mask).size
                else np.zeros(n_static_factor)
            ),
            n_static_factor,
        )
        self.static_state_sds = self._coerce_vector_template(
            "static_state_sds",
            (
                self.static_state_sds
                if jnp.asarray(self.static_state_sds).size
                else jnp.zeros(n_static_factor)
            ),
            n_static_factor,
        )
        self.lambda_mask = self._coerce_lambda_mask(self.lambda_mask)
        self.lambda_mat = self._coerce_lambda_mat(self.lambda_mat)
        self.diffusion_chol_mask = self._coerce_cholesky_mask(
            "diffusion_chol_mask",
            self.diffusion_chol_mask,
            self.n_latent,
        )
        self.diffusion_chol = self._coerce_square_template(
            "diffusion_chol",
            self.diffusion_chol,
            self.n_latent,
        )
        self.manifest_means_mask = self._coerce_diagonal_mask(
            "manifest_means_mask",
            self.manifest_means_mask,
            self.n_manifest,
        )
        self.manifest_means = self._coerce_vector_template(
            "manifest_means",
            self.manifest_means,
            self.n_manifest,
        )
        self.manifest_chol_diag_mask = self._coerce_diagonal_mask(
            "manifest_chol_diag_mask",
            self.manifest_chol_diag_mask,
            self.n_manifest,
        )
        self.manifest_chol = self._coerce_square_template(
            "manifest_chol",
            self.manifest_chol,
            self.n_manifest,
        )
        self.t0_means_mask = self._coerce_diagonal_mask(
            "t0_means_mask",
            self.t0_means_mask,
            self.n_latent,
        )
        self.t0_means = self._coerce_vector_template("t0_means", self.t0_means, self.n_latent)
        self.t0_chol_diag_mask = self._coerce_diagonal_mask(
            "t0_chol_diag_mask",
            self.t0_chol_diag_mask,
            self.n_latent,
        )
        self.t0_correlation_mask = self._coerce_strict_lower_mask(
            "t0_correlation_mask",
            self.t0_correlation_mask,
            self.n_latent,
        )
        self.t0_chol = self._coerce_square_template("t0_chol", self.t0_chol, self.n_latent)

        if self.diffusion_dists:
            self.diffusion_dists = [DistributionFamily(dist) for dist in self.diffusion_dists]
        else:
            self.diffusion_dists = [DistributionFamily.GAUSSIAN] * self.n_latent
        if len(self.diffusion_dists) != self.n_latent:
            raise ValueError(
                "diffusion_dists length must match n_latent: "
                f"{len(self.diffusion_dists)} vs {self.n_latent}"
            )

        if self.manifest_dists:
            self.manifest_dists = [DistributionFamily(dist) for dist in self.manifest_dists]
        else:
            self.manifest_dists = [DistributionFamily.GAUSSIAN] * self.n_manifest
        if len(self.manifest_dists) != self.n_manifest:
            raise ValueError(
                "manifest_dists length must match n_manifest: "
                f"{len(self.manifest_dists)} vs {self.n_manifest}"
            )

        if self.manifest_links is not None:
            self.manifest_links = [LinkFunction(link) for link in self.manifest_links]
            if len(self.manifest_links) != self.n_manifest:
                raise ValueError(
                    "manifest_links length must match n_manifest: "
                    f"{len(self.manifest_links)} vs {self.n_manifest}"
                )

        if (
            self.manifest_level_counts is not None
            and len(self.manifest_level_counts) != self.n_manifest
        ):
            raise ValueError(
                "manifest_level_counts length must match n_manifest: "
                f"{len(self.manifest_level_counts)} vs {self.n_manifest}"
            )
        if self.manifest_centered is None:
            self.manifest_centered = [False] * self.n_manifest
        elif len(self.manifest_centered) != self.n_manifest:
            raise ValueError(
                "manifest_centered length must match n_manifest: "
                f"{len(self.manifest_centered)} vs {self.n_manifest}"
            )

        if self.static_factor_names is None:
            self.static_factor_names = [f"tau_{idx}" for idx in range(n_static_factor)]
        elif len(self.static_factor_names) != n_static_factor:
            raise ValueError(
                "static_factor_names length must match number of static factors: "
                f"{len(self.static_factor_names)} vs {n_static_factor}"
            )


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
            if runtime_kind == PriorDistributionFamily.NORMAL:
                return dist.Normal(jnp.asarray(prior["mu"]), jnp.asarray(prior["sigma"]))
            if runtime_kind == PriorDistributionFamily.TRUNCATED_NORMAL:
                return dist.TruncatedNormal(
                    loc=jnp.asarray(prior["mu"]),
                    scale=jnp.asarray(prior["sigma"]),
                    low=jnp.asarray(prior["lower"]),
                    high=jnp.asarray(prior["upper"]),
                )
            if runtime_kind == PriorDistributionFamily.UNIFORM:
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
        if runtime_kind == PriorDistributionFamily.HALF_NORMAL:
            return dist.HalfNormal(jnp.asarray(prior["sigma"]))
        if runtime_kind == PriorDistributionFamily.GAMMA:
            return dist.Gamma(
                concentration=jnp.asarray(prior.get("concentration", 2.0)),
                rate=jnp.asarray(prior.get("rate", 1.0)),
            )
        if runtime_kind == PriorDistributionFamily.LOG_NORMAL:
            return dist.LogNormal(
                loc=jnp.asarray(prior.get("loc", 0.0)),
                scale=jnp.asarray(prior.get("sigma", 1.0)),
            )
        if runtime_kind == PriorDistributionFamily.EXPONENTIAL:
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
        prior_runtime_bundle: PriorRuntimeBundle | None = None,
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
        self._structure_runtime = SSMStructureRuntime(spec)
        self._artifact_cache: dict[tuple[Any, ...], Any] = {}
        self.observation_support: ObservationSupportRuntime | None = None
        self.parameter_bindings: list[dict[str, Any]] = []
        self._prior_runtime_bundle = prior_runtime_bundle
        self._prior_site_index = (
            {site.name: site for site in prior_runtime_bundle.registry}
            if prior_runtime_bundle is not None
            else None
        )

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

    def get_prior_runtime_bundle(self) -> PriorRuntimeBundle:
        """Return canonical prior runtime state for this model instance."""
        if self._prior_runtime_bundle is None:
            self._prior_runtime_bundle = build_prior_runtime_bundle(self.spec, self.priors)
            self._prior_site_index = {
                site.name: site for site in self._prior_runtime_bundle.registry
            }
        return self._prior_runtime_bundle

    def _prior_distribution(self, site_name: str) -> dist.Distribution:
        """Resolve a sample-site prior from canonical runtime semantics."""
        runtime = self.get_prior_runtime_bundle()
        assert self._prior_site_index is not None
        site = self._prior_site_index.get(site_name)
        if site is None:
            raise ValueError(f"Prior runtime bundle has no site named {site_name!r}")
        return build_site_prior_distribution(site, runtime.prior_state[site_name])

    def _sample_drift(self, spec: SSMSpec) -> jnp.ndarray:
        """Sample drift matrix with stability constraints."""
        n = spec.n_latent

        structure_runtime = self._structure_runtime
        n_diag = structure_runtime.n_drift_diag
        n_offdiag = structure_runtime.n_drift_offdiag

        if n_diag == 0 and n_offdiag == 0:
            return structure_runtime.drift_template

        if n_diag > 0:
            drift_diag_free = _sample_prior_array(
                "drift_diag_free",
                self._prior_distribution("drift_diag_free"),
            )
        else:
            drift_diag_free = None

        if n_offdiag > 0:
            drift_offdiag_free = _sample_prior_array(
                "drift_offdiag_free",
                self._prior_distribution("drift_offdiag_free"),
            )
        else:
            drift_offdiag_free = None

        drift = structure_runtime.assemble_drift(drift_diag_free, drift_offdiag_free)

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

    def _sample_diffusion(self, _spec: SSMSpec) -> jnp.ndarray:
        """Sample diffusion matrix (lower Cholesky)."""
        structure_runtime = self._structure_runtime
        n_diag = structure_runtime.n_diffusion_diag
        n_lower = structure_runtime.n_diffusion_lower
        if n_diag == 0 and n_lower == 0:
            return structure_runtime.diffusion_chol_template

        diff_diag_free = None
        if n_diag > 0:
            diff_diag_free = _sample_prior_array(
                "diffusion_diag_free",
                self._prior_distribution("diffusion_diag_free"),
            )

        diff_lower_free = None
        if n_lower > 0:
            diff_lower_free = _sample_prior_array(
                "diffusion_lower_free",
                self._prior_distribution("diffusion_lower_free"),
            )

        diffusion = structure_runtime.assemble_diffusion(diff_diag_free, diff_lower_free)

        numpyro.deterministic("diffusion", diffusion)
        return diffusion

    def _sample_cint(self, _spec: SSMSpec) -> jnp.ndarray | None:
        """Sample continuous intercept."""
        n_free = self._structure_runtime.n_cint
        if n_free == 0:
            return self._structure_runtime.cint_template

        cint_free = _sample_prior_array(
            "cint_free",
            self._prior_distribution("cint_free"),
        )
        cint = self._structure_runtime.assemble_cint(cint_free)

        numpyro.deterministic("cint", cint)
        return cint

    def _sample_lambda(self, _spec: SSMSpec) -> jnp.ndarray:
        """Sample factor loading matrix for the fitted subject/model.

        Two modes (determined by SSMStructureRuntime from spec):
        1. Template+mask: sample free loadings at masked positions.
        2. Fixed: return template as-is (no sampling).
        """
        # Fully fixed (array with no free-loading positions): return as-is
        structure_runtime = self._structure_runtime
        if structure_runtime.n_lambda_free == 0:
            return structure_runtime.lambda_template
        n_free = structure_runtime.n_lambda_free

        free_loadings = None
        if n_free > 0:
            free_loadings = _sample_prior_array(
                "lambda_free",
                self._prior_distribution("lambda_free"),
            )

        lambda_mat = structure_runtime.assemble_lambda(free_loadings)
        numpyro.deterministic("lambda", lambda_mat)
        return lambda_mat

    def _sample_manifest_params(self, _spec: SSMSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample manifest means and variance for the fitted subject/model."""
        # Means
        n_means_free = self._structure_runtime.n_manifest_means
        if n_means_free == 0:
            manifest_means = self._structure_runtime.manifest_means_template
        else:
            manifest_means_free = _sample_prior_array(
                "manifest_means_free",
                self._prior_distribution("manifest_means_free"),
            )
            manifest_means = self._structure_runtime.assemble_manifest_means(manifest_means_free)

        # Variance (Cholesky)
        n_free = self._structure_runtime.n_manifest_var_diag
        if n_free == 0:
            manifest_chol = self._structure_runtime.manifest_chol_template
        else:
            var_diag = _sample_prior_array(
                "manifest_var_diag_free",
                self._prior_distribution("manifest_var_diag_free"),
            )
            manifest_chol = self._structure_runtime.assemble_manifest_chol(var_diag)

        numpyro.deterministic("manifest_cov", manifest_chol @ manifest_chol.T)
        return jnp.asarray(manifest_means), jnp.asarray(manifest_chol)

    def _sample_t0_params(self, _spec: SSMSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample initial state parameters."""
        # Means
        n_means_free = self._structure_runtime.n_t0_means
        if n_means_free == 0:
            t0_means = self._structure_runtime.t0_means_template
        else:
            t0_means_free = _sample_prior_array(
                "t0_means_free",
                self._prior_distribution("t0_means_free"),
            )
            t0_means = self._structure_runtime.assemble_t0_means(t0_means_free)

        # Variance (Cholesky)
        structure_runtime = self._structure_runtime
        n_diag = structure_runtime.n_t0_diag
        n_corr = structure_runtime.n_t0_correlation
        n_static = structure_runtime.n_static_state_sd
        if n_diag == 0 and n_corr == 0 and n_static == 0:
            t0_cov, _min_eig = stabilize_covariance_for_cholesky(
                structure_runtime.assemble_t0_cov(),
                min_eigenvalue=INITIAL_STATE_COV_MIN_EIGENVALUE,
            )
        else:
            var_diag = None
            if n_diag > 0:
                var_diag = _sample_prior_array(
                    "t0_var_diag_free",
                    self._prior_distribution("t0_var_diag_free"),
                )
            t0_corr = None
            if n_corr > 0:
                t0_corr = _sample_prior_array(
                    "t0_var_lower_free",
                    self._prior_distribution("t0_var_lower_free"),
                )
            static_state_sds = None
            if n_static > 0:
                static_state_sds = _sample_prior_array(
                    "static_state_sd_free",
                    self._prior_distribution("static_state_sd_free"),
                )
                numpyro.deterministic(
                    "static_state_sds",
                    structure_runtime.assemble_static_state_sds(static_state_sds),
                )
            t0_cov_raw = structure_runtime.assemble_t0_cov(var_diag, t0_corr, static_state_sds)
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
        numpyro.deterministic("t0_cov", t0_cov)
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
            lambda: build_laplace_backend(
                self.spec,
                n_ieks_iters,
                observation_support=self.observation_support,
            ),
        )

    def _sample_likelihood_extra_params(self, spec: SSMSpec) -> dict[str, jnp.ndarray]:
        """Sample likelihood hyperparameters and assemble backend-ready extras."""
        sampled_values: dict[str, jnp.ndarray] = {}
        manifest_dist_set = set(spec.manifest_dists)

        if DistributionFamily.STUDENT_T in manifest_dist_set:
            sampled_values["obs_df"] = _sample_prior_array(
                "obs_df",
                self._prior_distribution("obs_df"),
            )
        if DistributionFamily.GAMMA in manifest_dist_set:
            sampled_values["obs_shape"] = _sample_prior_array(
                "obs_shape",
                self._prior_distribution("obs_shape"),
            )
        if DistributionFamily.NEGATIVE_BINOMIAL in manifest_dist_set:
            sampled_values["obs_r"] = _sample_prior_array(
                "obs_r",
                self._prior_distribution("obs_r"),
            )
        if DistributionFamily.BETA in manifest_dist_set:
            sampled_values["obs_concentration"] = _sample_prior_array(
                "obs_concentration",
                self._prior_distribution("obs_concentration"),
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
                sampled_values["obs_ordered_base"] = _sample_prior_array(
                    "obs_ordered_base",
                    self._prior_distribution("obs_ordered_base"),
                )
                if max_cutpoints > 1:
                    sampled_values["obs_ordered_gaps"] = _sample_prior_array(
                        "obs_ordered_gaps",
                        self._prior_distribution("obs_ordered_gaps"),
                    )

            if DistributionFamily.CATEGORICAL in manifest_dist_set:
                sampled_values["obs_cat_intercepts"] = _sample_prior_array(
                    "obs_cat_intercepts",
                    self._prior_distribution("obs_cat_intercepts"),
                )
                sampled_values["obs_cat_slopes"] = _sample_prior_array(
                    "obs_cat_slopes",
                    self._prior_distribution("obs_cat_slopes"),
                )

        from causal_ssm_agent.models.ssm.inference.targets.graph_analysis import (
            has_student_t_diffusion,
        )

        if has_student_t_diffusion(spec):
            sampled_values["proc_df"] = _sample_prior_array(
                "proc_df",
                self._prior_distribution("proc_df"),
            )

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
