"""NumPyro State-Space Model.

Bayesian State-Space Model definition using NumPyro.
This module defines the probabilistic model only — inference is in inference.py.

Supports:
- Time-series trajectories
- Any noise family (Gaussian, Poisson, Student-t, Gamma)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.dynamics import (
        CompositeSpec,
        DiffusionBlockSpec,
        ManifestCholBlockSpec,
        SparseMatrixBlockSpec,
        SparseVectorBlockSpec,
        T0CholBlockSpec,
    )
    from nof1_causal_lab.models.ssm_observation_metadata import ObservationSupportRuntime

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
    stabilize_covariance_for_cholesky,
    symmetrize,
)
from nof1_causal_lab.models.ssm.inference.backend_factory import (
    build_laplace_backend,
)
from nof1_causal_lab.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from nof1_causal_lab.models.ssm.inference.targets.observation_families import (
    any_family_needs_level_metadata,
)
from nof1_causal_lab.models.ssm.likelihood_extra_params import (
    assemble_sampled_extra_params,
)
from nof1_causal_lab.models.ssm.parameter_layout import SSMParameterLayout
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    build_prior_runtime_bundle,
    build_site_prior_distribution,
)
from nof1_causal_lab.models.ssm.priors import SSMPriors


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

    # Structural shape metadata (required)
    n_latent: int
    n_manifest: int

    # Canonical block-spec params (required). Each block owns its
    # structural masks, template, and per-prior settings; the SSMSpec
    # itself stores no flat-field duplicates. Priors are typically
    # left None at construction time and attached at sample time from
    # the runtime PriorRuntimeBundle.
    drift_spec: CompositeSpec
    diffusion_block: DiffusionBlockSpec
    lambda_block: SparseMatrixBlockSpec
    manifest_means_block: SparseVectorBlockSpec
    manifest_chol_block: ManifestCholBlockSpec
    t0_means_block: SparseVectorBlockSpec
    t0_chol_block: T0CholBlockSpec
    input_effect_block: SparseMatrixBlockSpec
    static_state_sd_block: SparseVectorBlockSpec

    # Pure structural metadata (no sampled params).
    static_factor_loadings: jnp.ndarray = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float64)
    )
    diffusion_dists: list[DistributionFamily] = field(default_factory=list)
    manifest_dists: list[DistributionFamily] = field(default_factory=list)
    manifest_level_counts: list[int] | None = None
    manifest_links: list[LinkFunction] | None = None
    manifest_centered: list[bool] | None = None
    latent_names: list[str] | None = None
    manifest_names: list[str] | None = None
    input_names: list[str] | None = None
    input_source_indicators: list[str] | None = None
    input_scales: list[float] | None = None
    input_missing_policies: list[str] | None = None
    static_factor_names: list[str] | None = None
    initialization_policy: str = "stationary"
    observation_intercept_policy: str = "free"

    def __post_init__(self) -> None:
        """Validate block-spec shape agreement and canonicalize metadata."""
        def _shape_tuple(shape: tuple[int, ...]) -> str:
            return "(" + ", ".join(str(dim) for dim in shape) + ")"

        def _require_shape(name: str, value: Any, shape: tuple[int, ...]) -> None:
            if value is None:
                raise ValueError(f"{name} must have shape {_shape_tuple(shape)}, got None.")
            actual = np.asarray(value).shape
            if actual != shape:
                raise ValueError(
                    f"{name} must have shape {_shape_tuple(shape)}, got {actual}."
                )

        def _require_vector(name: str, value: Any, n: int) -> None:
            _require_shape(name, value, (n,))

        def _require_matrix(name: str, value: Any, rows: int, cols: int) -> None:
            _require_shape(name, value, (rows, cols))

        from nof1_causal_lab.models.ssm.dynamics import (
            StructuralDenseLinearSpec,
            StructuralInterceptSpec,
        )

        # Static factor loadings shape: (n_latent, n_factor)
        loadings = jnp.asarray(self.static_factor_loadings)
        if loadings.ndim != 2:
            raise ValueError("static_factor_loadings must be a rank-2 array.")
        if loadings.shape[0] == 0 and loadings.shape[1] == 0:
            loadings = jnp.zeros((self.n_latent, 0), dtype=jnp.float64)
        elif loadings.shape[0] != self.n_latent:
            raise ValueError(
                "static_factor_loadings must have shape "
                f"({self.n_latent}, n_factor), got {loadings.shape}"
            )
        self.static_factor_loadings = loadings
        n_static_factor = int(loadings.shape[1])

        # Cross-check block shapes against n_latent / n_manifest.
        if self.drift_spec.n_latent != self.n_latent:
            raise ValueError(
                f"drift_spec.n_latent ({self.drift_spec.n_latent}) "
                f"!= SSMSpec.n_latent ({self.n_latent})"
            )
        for component in self.drift_spec.components:
            if isinstance(component, StructuralDenseLinearSpec):
                _require_vector("drift_diag_mask", component.drift_diag_mask, self.n_latent)
                _require_matrix(
                    "drift_offdiag_mask",
                    component.drift_offdiag_mask,
                    self.n_latent,
                    self.n_latent,
                )
                _require_matrix(
                    "drift",
                    component.drift_template,
                    self.n_latent,
                    self.n_latent,
                )
            if isinstance(component, StructuralInterceptSpec):
                _require_vector("cint_mask", component.cint_mask, self.n_latent)
                _require_vector("cint", component.cint_template, self.n_latent)
        if self.diffusion_block.n_latent != self.n_latent:
            raise ValueError(
                f"diffusion_block.n_latent ({self.diffusion_block.n_latent}) "
                f"!= SSMSpec.n_latent ({self.n_latent})"
            )
        _require_matrix(
            "diffusion_chol_mask",
            self.diffusion_block.diffusion_chol_mask,
            self.n_latent,
            self.n_latent,
        )
        _require_matrix(
            "diffusion_chol",
            self.diffusion_block.diffusion_chol_template,
            self.n_latent,
            self.n_latent,
        )
        if (
            self.lambda_block.n_rows != self.n_manifest
            or self.lambda_block.n_cols != self.n_latent
        ):
            raise ValueError(
                f"lambda_block shape ({self.lambda_block.n_rows}, "
                f"{self.lambda_block.n_cols}) != "
                f"({self.n_manifest}, {self.n_latent})"
            )
        _require_matrix("lambda_mask", self.lambda_block.mask, self.n_manifest, self.n_latent)
        _require_matrix("lambda_mat", self.lambda_block.template, self.n_manifest, self.n_latent)
        if self.manifest_means_block.n != self.n_manifest:
            raise ValueError(
                f"manifest_means_block.n ({self.manifest_means_block.n}) "
                f"!= n_manifest ({self.n_manifest})"
            )
        _require_vector("manifest_means_mask", self.manifest_means_block.mask, self.n_manifest)
        _require_vector("manifest_means", self.manifest_means_block.template, self.n_manifest)
        if self.manifest_chol_block.n_manifest != self.n_manifest:
            raise ValueError(
                f"manifest_chol_block.n_manifest "
                f"({self.manifest_chol_block.n_manifest}) "
                f"!= n_manifest ({self.n_manifest})"
            )
        _require_vector(
            "manifest_chol_diag_mask",
            self.manifest_chol_block.diag_mask,
            self.n_manifest,
        )
        _require_matrix(
            "manifest_chol",
            self.manifest_chol_block.template,
            self.n_manifest,
            self.n_manifest,
        )
        if self.t0_means_block.n != self.n_latent:
            raise ValueError(
                f"t0_means_block.n ({self.t0_means_block.n}) "
                f"!= n_latent ({self.n_latent})"
            )
        _require_vector("t0_means_mask", self.t0_means_block.mask, self.n_latent)
        _require_vector("t0_means", self.t0_means_block.template, self.n_latent)
        if self.t0_chol_block.n_latent != self.n_latent:
            raise ValueError(
                f"t0_chol_block.n_latent ({self.t0_chol_block.n_latent}) "
                f"!= n_latent ({self.n_latent})"
            )
        _require_vector("t0_chol_diag_mask", self.t0_chol_block.diag_mask, self.n_latent)
        _require_matrix(
            "t0_correlation_mask",
            self.t0_chol_block.correlation_mask,
            self.n_latent,
            self.n_latent,
        )
        _require_matrix(
            "t0_chol",
            self.t0_chol_block.template,
            self.n_latent,
            self.n_latent,
        )
        if self.input_effect_block.n_rows not in {0, self.n_latent}:
            raise ValueError(
                f"input_effect_block.n_rows ({self.input_effect_block.n_rows}) "
                f"!= n_latent ({self.n_latent}) or 0"
            )
        _require_matrix(
            "input_effect_mask",
            self.input_effect_block.mask,
            self.input_effect_block.n_rows,
            self.input_effect_block.n_cols,
        )
        _require_matrix(
            "input_effect",
            self.input_effect_block.template,
            self.input_effect_block.n_rows,
            self.input_effect_block.n_cols,
        )
        if self.static_state_sd_block.n != n_static_factor:
            raise ValueError(
                f"static_state_sd_block.n ({self.static_state_sd_block.n}) "
                f"!= n_static_factor ({n_static_factor})"
            )
        _require_vector("static_state_sd_mask", self.static_state_sd_block.mask, n_static_factor)
        _require_vector("static_state_sds", self.static_state_sd_block.template, n_static_factor)

        # Resolve n_input from input_effect_block, then canonicalize names.
        n_input = self.input_effect_block.n_cols
        if self.input_names is None:
            self.input_names = [f"input_{idx}" for idx in range(n_input)]
        elif len(self.input_names) != n_input:
            raise ValueError(
                f"input_names length must match n_input: {len(self.input_names)} vs {n_input}"
            )
        if self.input_source_indicators is None:
            self.input_source_indicators = list(self.input_names)
        elif len(self.input_source_indicators) != n_input:
            raise ValueError(
                "input_source_indicators length must match n_input: "
                f"{len(self.input_source_indicators)} vs {n_input}"
            )
        if self.input_scales is None:
            self.input_scales = [1.0] * n_input
        elif len(self.input_scales) != n_input:
            raise ValueError(
                f"input_scales length must match n_input: {len(self.input_scales)} vs {n_input}"
            )
        if any(float(scale) <= 0.0 for scale in self.input_scales):
            raise ValueError("input_scales must be strictly positive")
        if self.input_missing_policies is None:
            self.input_missing_policies = ["zero"] * n_input
        elif len(self.input_missing_policies) != n_input:
            raise ValueError(
                "input_missing_policies length must match n_input: "
                f"{len(self.input_missing_policies)} vs {n_input}"
            )
        invalid_policies = sorted(
            {
                str(policy)
                for policy in self.input_missing_policies
                if policy not in {"zero", "forward_fill"}
            }
        )
        if invalid_policies:
            raise ValueError(f"Unsupported input_missing_policies: {invalid_policies}")

        # Canonicalize per-channel family + link enums.
        if self.diffusion_dists:
            self.diffusion_dists = [DistributionFamily(d) for d in self.diffusion_dists]
        else:
            self.diffusion_dists = [DistributionFamily.GAUSSIAN] * self.n_latent
        if len(self.diffusion_dists) != self.n_latent:
            raise ValueError(
                "diffusion_dists length must match n_latent: "
                f"{len(self.diffusion_dists)} vs {self.n_latent}"
            )
        if self.manifest_dists:
            self.manifest_dists = [DistributionFamily(d) for d in self.manifest_dists]
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

    def structural_drift_components(self):
        """Return the structural dense-linear drift components.

        The NumPyro ``SSMModel.model`` path is the dense-linear SSM path.
        Nonlinear composite drifts use the composite inference drivers and
        should not be coerced into dense-linear sample sites.
        """
        from nof1_causal_lab.models.ssm.dynamics import (
            StructuralDenseLinearSpec,
            StructuralInterceptSpec,
        )

        components = self.drift_spec.components
        if (
            len(components) >= 2
            and isinstance(components[0], StructuralDenseLinearSpec)
            and isinstance(components[1], StructuralInterceptSpec)
        ):
            return components[0], components[1]
        raise TypeError(
            "Dense-linear SSM assembly requires drift_spec components "
            "(StructuralDenseLinearSpec, StructuralInterceptSpec). "
            "Use the composite inference path for nonlinear drift specs."
        )

    def assemble_drift(
        self,
        base_decay_free: jnp.ndarray | None = None,
        offdiag_free: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        drift_component, _ = self.structural_drift_components()
        return drift_component.assemble_drift(base_decay_free, offdiag_free)

    def assemble_cint(self, cint_free: jnp.ndarray | None = None) -> jnp.ndarray:
        _, cint_component = self.structural_drift_components()
        return cint_component.assemble_cint(cint_free)

    def assemble_diffusion(
        self,
        diag_free: jnp.ndarray | None = None,
        lower_free: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        return self.diffusion_block.assemble(diag_free, lower_free)

    def assemble_input_effect(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.input_effect_block.assemble(free)

    def assemble_lambda(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.lambda_block.assemble(free)

    def assemble_manifest_means(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.manifest_means_block.assemble(free)

    def assemble_manifest_chol(self, free_diag: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.manifest_chol_block.assemble(free_diag)

    def assemble_t0_means(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.t0_means_block.assemble(free)

    def assemble_static_state_sds(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.static_state_sd_block.assemble(free)

    def assemble_t0_cov(
        self,
        t0_diag_free: jnp.ndarray | None = None,
        t0_correlation_free: jnp.ndarray | None = None,
        static_state_sd_free: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        cov = self.t0_chol_block.assemble_cov(t0_diag_free, t0_correlation_free)
        factor_sds = self.assemble_static_state_sds(static_state_sd_free)
        if factor_sds.size:
            loadings = jnp.asarray(self.static_factor_loadings)
            cov = cov + loadings @ jnp.diag(factor_sds**2) @ loadings.T
        return symmetrize(cov)


def _make_prior_dist(prior: dict) -> dist.Distribution:
    """Build a numpyro distribution from a legacy SSMPriors dict-config.

    Thin delegation to :func:`materialize_prior` in
    ``dynamics/config.py`` — the single materialiser that accepts both
    the legacy flat format used here and the nested ``params`` format
    used by the composite spec components.
    """
    from nof1_causal_lab.models.ssm.dynamics.config import materialize_prior

    return materialize_prior(prior)


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
    - IEKS/Laplace marginal likelihood backend
    """

    def __init__(
        self,
        spec: SSMSpec,
        priors: SSMPriors | None = None,
        prior_runtime_bundle: PriorRuntimeBundle | None = None,
    ):
        """Initialize state-space model.

        Args:
            spec: Model specification
            priors: Prior distributions (uses defaults if None)
        """
        self.spec = spec
        self.priors = priors or SSMPriors()
        self._parameter_layout = SSMParameterLayout.from_spec(spec)
        self._artifact_cache: dict[tuple[Any, ...], Any] = {}
        self.observation_support: ObservationSupportRuntime | None = None
        self.transition_inputs: jnp.ndarray | None = None
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

    def set_transition_inputs(self, transition_inputs: jnp.ndarray | None) -> None:
        """Attach prepared known-input trajectories aligned to transition intervals."""
        self.transition_inputs = transition_inputs

    @property
    def vector_field(self):
        """Unified drift representation as a :class:`CompositeVectorField`.

        After :class:`SSMSpec` auto-builds ``drift_spec`` in
        ``__post_init__``, every spec has a populated ``drift_spec`` — the
        linear path becomes a 2-component composite
        (``StructuralDenseLinear`` + ``StructuralIntercept``), the composite
        path stays whatever the user / Stage 4 declared. The compiled
        vector field is what downstream consumers
        (``compute_steady_state``, ``simulate``,
        ``check_jacobian_stability``, the per-step linearisation in the
        auxiliary Kalman MH, …) all consume uniformly.
        """

        def _build():
            from nof1_causal_lab.models.ssm.dynamics import compile_composite

            return compile_composite(self.spec.drift_spec).vector_field

        return self.get_cached_artifact(("vector_field",), _build)

    @property
    def parameter_layout(self) -> SSMParameterLayout:
        """Return the derived parameter layout for this model."""
        return self._parameter_layout

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

    def _sample_drift(self) -> jnp.ndarray:
        """Sample drift matrix via the structural dense-linear component.

        Reads the structural component directly from ``spec.drift_spec``;
        attaches runtime priors via ``dataclasses.replace``.
        """
        layout = self._parameter_layout
        if layout.n_drift_base_decay == 0 and layout.n_drift_offdiag == 0:
            return self.spec.assemble_drift()

        drift_component, _ = self.spec.structural_drift_components()
        component_spec = replace(
            drift_component,
            base_decay_prior=(
                self._prior_distribution("drift_base_decay_free")
                if layout.n_drift_base_decay > 0
                else None
            ),
            offdiag_prior=(
                self._prior_distribution("drift_offdiag_free")
                if layout.n_drift_offdiag > 0
                else None
            ),
        )
        return component_spec.sample_params(prefix="")["drift"]

    def _sample_diffusion(self, _spec: SSMSpec) -> jnp.ndarray:
        """Sample diffusion matrix (lower Cholesky) via the diffusion block."""
        layout = self._parameter_layout
        if layout.n_diffusion_diag == 0 and layout.n_diffusion_lower == 0:
            return self.spec.assemble_diffusion()

        block = replace(
            self.spec.diffusion_block,
            diag_prior=(
                self._prior_distribution("diffusion_diag_free")
                if layout.n_diffusion_diag > 0
                else None
            ),
            lower_prior=(
                self._prior_distribution("diffusion_lower_free")
                if layout.n_diffusion_lower > 0
                else None
            ),
        )
        return block.sample_params(prefix="")["diffusion"]

    def _sample_cint(self, _spec: SSMSpec) -> jnp.ndarray | None:
        """Sample continuous intercept via the structural intercept component."""
        layout = self._parameter_layout
        if layout.n_cint == 0:
            return self.spec.assemble_cint()

        _, cint_component = self.spec.structural_drift_components()
        component_spec = replace(
            cint_component,
            cint_prior=self._prior_distribution("cint_free"),
        )
        return component_spec.sample_params(prefix="")["cint"]

    def _sample_input_effect(self, _spec: SSMSpec) -> jnp.ndarray:
        """Sample known-input transition effects via the input-effect block."""
        layout = self._parameter_layout
        if layout.n_input_effect == 0:
            return self.spec.assemble_input_effect()

        block = replace(
            self.spec.input_effect_block,
            prior=self._prior_distribution("input_effect_free"),
        )
        return block.sample_params(prefix="")["input_effect"]

    def _sample_lambda(self, _spec: SSMSpec) -> jnp.ndarray:
        """Sample factor loading matrix via the lambda block."""
        layout = self._parameter_layout
        if layout.n_lambda_free == 0:
            return self.spec.assemble_lambda()

        block = replace(
            self.spec.lambda_block,
            prior=self._prior_distribution("lambda_free"),
        )
        return block.sample_params(prefix="")["lambda"]

    def _sample_manifest_params(self, _spec: SSMSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample manifest means + Cholesky and emit the composed manifest_cov."""
        layout = self._parameter_layout

        # Means
        if layout.n_manifest_means == 0:
            manifest_means = self.spec.assemble_manifest_means()
        else:
            means_block = replace(
                self.spec.manifest_means_block,
                prior=self._prior_distribution("manifest_means_free"),
            )
            manifest_means = means_block.sample_params(prefix="")["manifest_means"]

        # Variance (diagonal Cholesky)
        if layout.n_manifest_var_diag == 0:
            manifest_chol = self.spec.assemble_manifest_chol()
        else:
            chol_block = replace(
                self.spec.manifest_chol_block,
                diag_prior=self._prior_distribution("manifest_var_diag_free"),
            )
            manifest_chol = chol_block.sample_params(prefix="")["manifest_chol"]

        numpyro.deterministic("manifest_cov", manifest_chol @ manifest_chol.T)
        return jnp.asarray(manifest_means), jnp.asarray(manifest_chol)

    def _sample_t0_params(self, _spec: SSMSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample initial state parameters.

        Means delegate to the t0_means block. The covariance assembly
        (free diagonal SDs + free correlation entries + static factor SDs
        combined via factor loadings) stays inline because the composed
        structure is genuinely multi-block — block-spec just owns the
        masks/templates; assembly is at the model layer.
        """
        # Means via block-spec. The block emits the ``t0_means``
        # deterministic when sampled; for the no-free case we emit it
        # explicitly below so the site is always present (legacy
        # contract).
        layout = self._parameter_layout
        if layout.n_t0_means == 0:
            t0_means = self.spec.assemble_t0_means()
            numpyro.deterministic("t0_means", t0_means)
        else:
            means_block = replace(
                self.spec.t0_means_block,
                prior=self._prior_distribution("t0_means_free"),
            )
            t0_means = means_block.sample_params(prefix="")["t0_means"]

        # Variance (Cholesky)
        n_diag = layout.n_t0_diag
        n_corr = layout.n_t0_correlation
        n_static = layout.n_static_state_sd
        if n_diag == 0 and n_corr == 0 and n_static == 0:
            t0_cov, _min_eig = stabilize_covariance_for_cholesky(
                self.spec.assemble_t0_cov(),
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
                    self.spec.assemble_static_state_sds(static_state_sds),
                )
            t0_cov_raw = self.spec.assemble_t0_cov(var_diag, t0_corr, static_state_sds)
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

        # ``t0_means`` deterministic is emitted either inside the
        # SparseVectorBlockSpec (sampled branch) or by the explicit
        # ``numpyro.deterministic`` call in the no-free branch above —
        # always exactly once.
        numpyro.deterministic("t0_cov", t0_cov)
        return jnp.asarray(t0_means), jnp.asarray(t0_chol)

    def make_likelihood_backend(self):
        """Construct or reuse the default Laplace likelihood backend."""
        return self.make_laplace_backend(n_ieks_iters=6)

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

        from nof1_causal_lab.models.ssm.inference.targets.spec_metadata import (
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
            likelihood_backend: Laplace likelihood backend instance. Required —
                use model.make_likelihood_backend() for the default.
        """
        if likelihood_backend is None:
            raise ValueError(
                "likelihood_backend is required. "
                "Use model.make_likelihood_backend() for the default."
            )

        spec = self.spec

        drift = self._sample_drift()
        diffusion_chol = self._sample_diffusion(spec)
        cint = self._sample_cint(spec)
        input_effect = self._sample_input_effect(spec)
        lambda_mat = self._sample_lambda(spec)
        manifest_means, manifest_chol = self._sample_manifest_params(spec)
        t0_means, t0_chol = self._sample_t0_params(spec)

        diffusion_cov = diffusion_chol @ diffusion_chol.T
        manifest_cov = manifest_chol @ manifest_chol.T
        t0_cov = t0_chol @ t0_chol.T
        extra_params = self._sample_likelihood_extra_params(spec)

        ct_params = CTParams(
            drift=drift,
            diffusion_cov=diffusion_cov,
            cint=cint,
            input_effect=input_effect,
        )
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
            transition_inputs=self.transition_inputs,
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
