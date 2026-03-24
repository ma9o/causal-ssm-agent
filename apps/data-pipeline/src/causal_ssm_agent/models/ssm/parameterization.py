"""Canonical site registry and compile-stable prior evaluation.

Provides:
- SiteDescriptor: metadata for each sample site, derived deterministically from SSMSpec
- build_site_registry: enumerate all sample sites without model tracing
- build_prior_runtime_state: create fixed-structure JAX pytree from SSMPriors
- log_prior_unconstrained: pure-JAX prior log-density with vectorized family dispatch
- sample_prior_unconstrained: pure-JAX prior sampling

The site registry is the single authority for "what sample sites exist."
It replaces the three overlapping conventions:
- parameter_bindings (semantic parameter → site mapping)
- _discover_sites (trace-driven site enumeration)
- _assemble_deterministics (hardcoded site name checks)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.distributions import (
    PriorRuntimeKind,
    get_positive_runtime_kind_from_index,
    get_real_runtime_kind_from_index,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.assembler import SSMAssembler
    from causal_ssm_agent.models.ssm.model import SSMPriors, SSMSpec


# ---------------------------------------------------------------------------
# Support classes
# ---------------------------------------------------------------------------


class SupportClass(Enum):
    """Runtime support class for a sample site.

    Determines the unconstrained ↔ constrained transform and the set of
    valid prior families.  Static per topology — changing it requires
    recompilation (expected and rare).
    """

    REAL = "real"
    POSITIVE = "positive"


class TransformKind(Enum):
    """Explicit unconstrained -> constrained transform metadata."""

    IDENTITY = "identity"
    EXP = "exp"


class SiteKind(Enum):
    """Semantic role for each sample site."""

    DRIFT_DIAG = "drift_diag"
    DRIFT_OFFDIAG = "drift_offdiag"
    DIFFUSION_DIAG = "diffusion_diag"
    DIFFUSION_LOWER = "diffusion_lower"
    CINT = "cint"
    LOADING = "loading"
    MANIFEST_MEANS = "manifest_means"
    MANIFEST_VAR_DIAG = "manifest_var_diag"
    T0_MEANS = "t0_means"
    T0_VAR_DIAG = "t0_var_diag"
    OBS_DF = "obs_df"
    OBS_SHAPE = "obs_shape"
    OBS_R = "obs_r"
    OBS_CONCENTRATION = "obs_concentration"
    OBS_ORDERED_BASE = "obs_ordered_base"
    OBS_ORDERED_GAPS = "obs_ordered_gaps"
    OBS_CAT_INTERCEPTS = "obs_cat_intercepts"
    OBS_CAT_SLOPES = "obs_cat_slopes"
    PROC_DF = "proc_df"


# ---------------------------------------------------------------------------
# Site descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteDescriptor:
    """Metadata for a single sample site, derived from SSMSpec.

    Attributes:
        name: NumPyro sample site name (e.g. ``"drift_diag_pop"``).
        shape: Array shape of the sampled value.
        support: Support class determining transform and valid families.
        assembly_group: Which matrix group this site contributes to
            (``"drift"``, ``"diffusion"``, ``"cint"``, ``"lambda"``,
            ``"manifest"``, ``"t0"``, ``"likelihood"``).
        site_kind: Semantic site role within the assembly group.
        transform_kind: Explicit transform metadata for runtime consumers.
        deterministic_name: Output name assembled from this site, if any.
        fixed_spec_field: Spec field to broadcast when this site is absent
            because the parameter is fixed.
        priors_field: ``SSMPriors`` field name when this site is directly
            controlled by user priors.
    """

    name: str
    shape: tuple[int, ...]
    support: SupportClass
    assembly_group: str
    site_kind: SiteKind
    transform_kind: TransformKind
    deterministic_name: str | None = None
    fixed_spec_field: str | None = None
    priors_field: str | None = None
    runtime_prior_key: str | None = None
    is_runtime_prior_controlled: bool = True


def _site_size(shape: tuple[int, ...]) -> int:
    """Number of scalar elements in a site shape."""
    size = 1
    for d in shape:
        size *= d
    return size


@dataclass
class SiteRuntimeBundle:
    """Reusable topology-only runtime components derived from site metadata."""

    registry: list[SiteDescriptor]
    transforms: dict[str, dist.transforms.Transform]
    flat_dim: int
    unravel_fn: Any

    def constrain(self, z: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Map one unconstrained parameter vector to constrained site values."""
        unconstrained = self.unravel_fn(z)
        return {name: self.transforms[name](unconstrained[name]) for name in unconstrained}

    def constrain_batched(self, z_samples: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Map a batch of unconstrained draws to constrained site samples."""
        if self.flat_dim == 0:
            return {}

        unconstrained = jax.vmap(self.unravel_fn)(z_samples)
        return {
            site.name: jax.vmap(self.transforms[site.name])(unconstrained[site.name])
            for site in self.registry
        }

    @property
    def param_names(self) -> list[str]:
        """Ordered list of sample site names."""
        return [site.name for site in self.registry]

    @property
    def site_shapes(self) -> dict[str, tuple[int, ...]]:
        """Map from site name to array shape."""
        return {site.name: site.shape for site in self.registry}

    @property
    def scalar_names(self) -> list[str]:
        """Flat list of per-element names (e.g. ``drift_diag_pop[0]``)."""
        names: list[str] = []
        for site in self.registry:
            size = _site_size(site.shape)
            if size == 1:
                names.append(site.name)
            else:
                for k in range(size):
                    names.append(f"{site.name}[{k}]")
        return names

    @property
    def param_index(self) -> dict[str, tuple[int, int]]:
        """Map from site name to ``(offset, size)`` in the flat vector."""
        index: dict[str, tuple[int, int]] = {}
        offset = 0
        for site in self.registry:
            size = _site_size(site.shape)
            index[site.name] = (offset, size)
            offset += size
        return index


@dataclass
class PriorRuntimeBundle:
    """Reusable runtime components derived from compiled prior semantics."""

    site_runtime: SiteRuntimeBundle
    prior_state: PriorRuntimeState

    @property
    def registry(self) -> list[SiteDescriptor]:
        return self.site_runtime.registry

    @property
    def transforms(self) -> dict[str, dist.transforms.Transform]:
        return self.site_runtime.transforms

    @property
    def flat_dim(self) -> int:
        return self.site_runtime.flat_dim

    @property
    def unravel_fn(self) -> Any:
        return self.site_runtime.unravel_fn

    def constrain(self, z: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return self.site_runtime.constrain(z)

    def constrain_batched(self, z_samples: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return self.site_runtime.constrain_batched(z_samples)


def _site(
    name: str,
    shape: tuple[int, ...],
    support: SupportClass,
    assembly_group: str,
    site_kind: SiteKind,
    *,
    deterministic_name: str | None = None,
    fixed_spec_field: str | None = None,
    priors_field: str | None = None,
    runtime_prior_key: str | None = None,
) -> SiteDescriptor:
    """Construct a site descriptor with consistent metadata defaults."""
    transform_kind = TransformKind.IDENTITY if support == SupportClass.REAL else TransformKind.EXP
    runtime_key = runtime_prior_key or name
    return SiteDescriptor(
        name=name,
        shape=shape,
        support=support,
        assembly_group=assembly_group,
        site_kind=site_kind,
        transform_kind=transform_kind,
        deterministic_name=deterministic_name,
        fixed_spec_field=fixed_spec_field,
        priors_field=priors_field,
        runtime_prior_key=runtime_key,
        is_runtime_prior_controlled=True,
    )


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def build_site_registry(
    spec: SSMSpec,
    assembler: SSMAssembler | None = None,
) -> list[SiteDescriptor]:
    """Enumerate all sample sites deterministically from *spec*.

    No model tracing needed.  The returned list is sorted by site name
    (matching JAX pytree dict-key ordering used by ``ravel_pytree``).
    """
    from causal_ssm_agent.models.ssm.assembler import SSMAssembler as Asm
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    if assembler is None:
        assembler = Asm(spec)

    sites: list[SiteDescriptor] = []
    n_l, n_m = spec.n_latent, spec.n_manifest

    # -- Core parameter sites (mirroring SSMModel._sample_* methods) --------

    if not isinstance(spec.drift, jnp.ndarray):
        sites.append(
            _site(
                "drift_diag_pop",
                (n_l,),
                SupportClass.REAL,
                "drift",
                SiteKind.DRIFT_DIAG,
                deterministic_name="drift",
                fixed_spec_field="drift",
                priors_field="drift_diag",
            )
        )
        n_offdiag = len(assembler.offdiag_positions)
        if n_offdiag > 0:
            sites.append(
                _site(
                    "drift_offdiag_pop",
                    (n_offdiag,),
                    SupportClass.REAL,
                    "drift",
                    SiteKind.DRIFT_OFFDIAG,
                    deterministic_name="drift",
                    fixed_spec_field="drift",
                    priors_field="drift_offdiag",
                )
            )

    if not isinstance(spec.diffusion, jnp.ndarray):
        sites.append(
            _site(
                "diffusion_diag_pop",
                (n_l,),
                SupportClass.POSITIVE,
                "diffusion",
                SiteKind.DIFFUSION_DIAG,
                deterministic_name="diffusion",
                fixed_spec_field="diffusion",
                priors_field="diffusion_diag",
            )
        )
        if spec.diffusion != "diag":
            n_lower = n_l * (n_l - 1) // 2
            if n_lower > 0:
                sites.append(
                    _site(
                        "diffusion_lower",
                        (n_lower,),
                        SupportClass.REAL,
                        "diffusion",
                        SiteKind.DIFFUSION_LOWER,
                        deterministic_name="diffusion",
                        fixed_spec_field="diffusion",
                        priors_field="diffusion_offdiag",
                    )
                )

    if spec.cint is not None and not isinstance(spec.cint, jnp.ndarray):
        sites.append(
            _site(
                "cint_pop",
                (n_l,),
                SupportClass.REAL,
                "cint",
                SiteKind.CINT,
                deterministic_name="cint",
                fixed_spec_field="cint",
                priors_field="cint",
            )
        )

    if not (isinstance(spec.lambda_mat, jnp.ndarray) and spec.lambda_mask is None):
        n_free = len(assembler.lambda_free_positions)
        if n_free > 0:
            sites.append(
                _site(
                    "lambda_free",
                    (n_free,),
                    SupportClass.REAL,
                    "lambda",
                    SiteKind.LOADING,
                    deterministic_name="lambda",
                    fixed_spec_field="lambda_mat",
                    priors_field="lambda_free",
                )
            )

    if spec.manifest_means is not None and not isinstance(spec.manifest_means, jnp.ndarray):
        sites.append(
            _site(
                "manifest_means",
                (n_m,),
                SupportClass.REAL,
                "manifest",
                SiteKind.MANIFEST_MEANS,
                deterministic_name="manifest_means",
                fixed_spec_field="manifest_means",
                priors_field="manifest_means",
            )
        )

    if not isinstance(spec.manifest_var, jnp.ndarray):
        sites.append(
            _site(
                "manifest_var_diag",
                (n_m,),
                SupportClass.POSITIVE,
                "manifest",
                SiteKind.MANIFEST_VAR_DIAG,
                deterministic_name="manifest_cov",
                fixed_spec_field="manifest_var",
                priors_field="manifest_var_diag",
            )
        )

    if not isinstance(spec.t0_means, jnp.ndarray):
        sites.append(
            _site(
                "t0_means_pop",
                (n_l,),
                SupportClass.REAL,
                "t0",
                SiteKind.T0_MEANS,
                deterministic_name="t0_means",
                fixed_spec_field="t0_means",
                priors_field="t0_means",
            )
        )

    if not isinstance(spec.t0_var, jnp.ndarray):
        sites.append(
            _site(
                "t0_var_diag",
                (n_l,),
                SupportClass.POSITIVE,
                "t0",
                SiteKind.T0_VAR_DIAG,
                deterministic_name="t0_cov",
                fixed_spec_field="t0_var",
                priors_field="t0_var_diag",
            )
        )

    # -- Likelihood extra-parameter sites -----------------------------------

    manifest_dists = spec.manifest_dists or [spec.manifest_dist] * n_m
    manifest_dist_set = set(manifest_dists)

    if DistributionFamily.STUDENT_T in manifest_dist_set:
        sites.append(_site("obs_df", (), SupportClass.POSITIVE, "likelihood", SiteKind.OBS_DF))
    if DistributionFamily.GAMMA in manifest_dist_set:
        sites.append(
            _site("obs_shape", (), SupportClass.POSITIVE, "likelihood", SiteKind.OBS_SHAPE)
        )
    if DistributionFamily.NEGATIVE_BINOMIAL in manifest_dist_set:
        sites.append(_site("obs_r", (), SupportClass.POSITIVE, "likelihood", SiteKind.OBS_R))
    if DistributionFamily.BETA in manifest_dist_set:
        sites.append(
            _site(
                "obs_concentration",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.OBS_CONCENTRATION,
            )
        )

    if spec.manifest_level_counts is not None:
        level_counts_list = list(spec.manifest_level_counts)
        max_levels = max(level_counts_list) if level_counts_list else 0
        max_cutpoints = max(max_levels - 1, 0)

        if DistributionFamily.ORDERED_LOGISTIC in manifest_dist_set and max_cutpoints > 0:
            sites.append(
                _site(
                    "obs_ordered_base",
                    (n_m,),
                    SupportClass.REAL,
                    "likelihood",
                    SiteKind.OBS_ORDERED_BASE,
                )
            )
            if max_cutpoints > 1:
                sites.append(
                    _site(
                        "obs_ordered_gaps",
                        (n_m, max_cutpoints - 1),
                        SupportClass.POSITIVE,
                        "likelihood",
                        SiteKind.OBS_ORDERED_GAPS,
                    )
                )

        if DistributionFamily.CATEGORICAL in manifest_dist_set and max_cutpoints > 0:
            cat_shape = (n_m, max_cutpoints)
            sites.append(
                _site(
                    "obs_cat_intercepts",
                    cat_shape,
                    SupportClass.REAL,
                    "likelihood",
                    SiteKind.OBS_CAT_INTERCEPTS,
                )
            )
            sites.append(
                _site(
                    "obs_cat_slopes",
                    cat_shape,
                    SupportClass.REAL,
                    "likelihood",
                    SiteKind.OBS_CAT_SLOPES,
                )
            )

    from causal_ssm_agent.models.likelihoods.graph_analysis import has_student_t_diffusion

    if has_student_t_diffusion(spec):
        sites.append(_site("proc_df", (), SupportClass.POSITIVE, "likelihood", SiteKind.PROC_DF))

    # Sort by name to match JAX pytree dict-key ordering.
    sites.sort(key=lambda s: s.name)
    return sites


# ---------------------------------------------------------------------------
# Transforms (static per topology — no recompilation on prior change)
# ---------------------------------------------------------------------------


def build_transforms(
    registry: list[SiteDescriptor],
) -> dict[str, dist.transforms.Transform]:
    """Build constrained ↔ unconstrained transforms from the registry."""
    transforms: dict[str, dist.transforms.Transform] = {}
    for site in registry:
        if site.transform_kind == TransformKind.IDENTITY:
            transforms[site.name] = dist.transforms.IdentityTransform()
        elif site.transform_kind == TransformKind.EXP:
            transforms[site.name] = dist.transforms.ExpTransform()
    return transforms


def build_unravel_fn(
    registry: list[SiteDescriptor],
):
    """Build ``ravel_pytree``-compatible unravel from the registry.

    Returns ``(flat_dim, unravel_fn)`` where *flat_dim* is the total
    number of unconstrained scalar parameters.
    """
    example = {site.name: jnp.zeros(site.shape) for site in registry}
    flat, unravel_fn = ravel_pytree(example)
    return flat.shape[0], unravel_fn


def _build_site_runtime_bundle_from_registry(
    registry: list[SiteDescriptor],
) -> SiteRuntimeBundle:
    """Build reusable topology-only runtime components from a site registry."""
    transforms = build_transforms(registry)
    flat_dim, unravel_fn = build_unravel_fn(registry)
    return SiteRuntimeBundle(
        registry=registry,
        transforms=transforms,
        flat_dim=flat_dim,
        unravel_fn=unravel_fn,
    )


def build_site_runtime_bundle(
    spec: SSMSpec,
    assembler: SSMAssembler | None = None,
) -> SiteRuntimeBundle:
    """Build reusable topology-only runtime components from ``spec``."""
    registry = build_site_registry(spec, assembler)
    return _build_site_runtime_bundle_from_registry(registry)


def group_sites_by_assembly_role(
    registry: list[SiteDescriptor],
) -> dict[str, list[SiteDescriptor]]:
    """Group site descriptors by assembly role."""
    grouped: dict[str, list[SiteDescriptor]] = {}
    for site in registry:
        grouped.setdefault(site.assembly_group, []).append(site)
    return grouped


def select_site_samples(
    samples: dict[str, jnp.ndarray],
    registry: list[SiteDescriptor],
    *,
    assembly_group: str | None = None,
) -> dict[str, jnp.ndarray]:
    """Select sampled site values using registry metadata instead of name lists."""
    selected: dict[str, jnp.ndarray] = {}
    for site in registry:
        if assembly_group is not None and site.assembly_group != assembly_group:
            continue
        if site.name in samples:
            selected[site.name] = samples[site.name]
    return selected


def _resolve_num_draws(
    samples: dict[str, jnp.ndarray],
    n_draws: int | None,
) -> int:
    if n_draws is not None:
        return n_draws
    if samples:
        return int(next(iter(samples.values())).shape[0])
    raise ValueError("n_draws is required when assembling deterministic values without samples")


def _broadcast_fixed(
    value: jnp.ndarray,
    n_draws: int,
) -> jnp.ndarray:
    return jnp.broadcast_to(value, (n_draws, *value.shape))


def _assemble_diag_to_cov(
    site: SiteDescriptor | None,
    samples: dict[str, jnp.ndarray],
    fixed_chol: jnp.ndarray | None,
    n_draws: int,
    dim: int,
) -> jnp.ndarray | None:
    """Convert diagonal variance samples to full covariance, or broadcast fixed Cholesky."""
    if site is not None and site.name in samples:
        return jax.vmap(lambda d: jnp.diag(d**2))(samples[site.name])
    if isinstance(fixed_chol, jnp.ndarray):
        fixed_cov = fixed_chol @ fixed_chol.T
        return jnp.broadcast_to(fixed_cov, (n_draws, dim, dim))
    return None


def assemble_deterministics_from_registry(
    samples: dict[str, jnp.ndarray],
    spec: SSMSpec,
    registry: list[SiteDescriptor],
    *,
    assembler: SSMAssembler | None = None,
    n_draws: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Assemble deterministic matrices using registry metadata as authority."""
    from causal_ssm_agent.models.ssm.assembler import SSMAssembler as Asm

    n_draws = _resolve_num_draws(samples, n_draws)
    if assembler is None:
        assembler = Asm(spec)

    by_kind = {site.site_kind: site for site in registry}
    det: dict[str, jnp.ndarray] = {}
    n_l, n_m = spec.n_latent, spec.n_manifest

    drift_diag_site = by_kind.get(SiteKind.DRIFT_DIAG)
    drift_offdiag_site = by_kind.get(SiteKind.DRIFT_OFFDIAG)
    if drift_diag_site is not None and drift_diag_site.name in samples:
        offdiag = (
            samples[drift_offdiag_site.name]
            if drift_offdiag_site is not None and drift_offdiag_site.name in samples
            else jnp.zeros((n_draws, max(len(assembler.offdiag_positions), 0)))
        )
        det["drift"] = jax.vmap(assembler.assemble_drift)(samples[drift_diag_site.name], offdiag)
    elif isinstance(spec.drift, jnp.ndarray):
        det["drift"] = _broadcast_fixed(spec.drift, n_draws)

    diffusion_diag_site = by_kind.get(SiteKind.DIFFUSION_DIAG)
    diffusion_lower_site = by_kind.get(SiteKind.DIFFUSION_LOWER)
    if diffusion_diag_site is not None and diffusion_diag_site.name in samples:
        if diffusion_lower_site is not None and diffusion_lower_site.name in samples:
            det["diffusion"] = jax.vmap(assembler.assemble_diffusion)(
                samples[diffusion_diag_site.name],
                samples[diffusion_lower_site.name],
            )
        else:
            det["diffusion"] = jax.vmap(assembler.assemble_diffusion)(
                samples[diffusion_diag_site.name]
            )
    elif isinstance(spec.diffusion, jnp.ndarray):
        det["diffusion"] = _broadcast_fixed(spec.diffusion, n_draws)

    cint_site = by_kind.get(SiteKind.CINT)
    if cint_site is not None and cint_site.name in samples:
        det["cint"] = samples[cint_site.name]
    elif isinstance(spec.cint, jnp.ndarray):
        det["cint"] = _broadcast_fixed(spec.cint, n_draws)

    loading_site = by_kind.get(SiteKind.LOADING)
    if (
        loading_site is not None
        and loading_site.name in samples
        and len(assembler.lambda_free_positions) > 0
    ):
        det["lambda"] = jax.vmap(assembler.assemble_lambda)(samples[loading_site.name])
    else:
        det["lambda"] = jnp.broadcast_to(assembler.lambda_template, (n_draws, n_m, n_l))

    manifest_means_site = by_kind.get(SiteKind.MANIFEST_MEANS)
    if manifest_means_site is not None and manifest_means_site.name in samples:
        det["manifest_means"] = samples[manifest_means_site.name]
    elif isinstance(spec.manifest_means, jnp.ndarray):
        det["manifest_means"] = _broadcast_fixed(spec.manifest_means, n_draws)

    manifest_cov = _assemble_diag_to_cov(
        by_kind.get(SiteKind.MANIFEST_VAR_DIAG), samples, spec.manifest_var, n_draws, n_m
    )
    if manifest_cov is not None:
        det["manifest_cov"] = manifest_cov

    t0_means_site = by_kind.get(SiteKind.T0_MEANS)
    if t0_means_site is not None and t0_means_site.name in samples:
        det["t0_means"] = samples[t0_means_site.name]
    elif isinstance(spec.t0_means, jnp.ndarray):
        det["t0_means"] = _broadcast_fixed(spec.t0_means, n_draws)

    t0_cov = _assemble_diag_to_cov(
        by_kind.get(SiteKind.T0_VAR_DIAG), samples, spec.t0_var, n_draws, n_l
    )
    if t0_cov is not None:
        det["t0_cov"] = t0_cov

    return det


def assemble_extra_params_from_registry(
    spec: SSMSpec,
    samples: dict[str, jnp.ndarray],
    registry: list[SiteDescriptor],
) -> dict[str, jnp.ndarray]:
    """Assemble likelihood extra parameters using registry metadata as authority."""
    from causal_ssm_agent.models.ssm.model import assemble_sampled_extra_params

    return assemble_sampled_extra_params(
        spec,
        select_site_samples(samples, registry, assembly_group="likelihood"),
    )


# ---------------------------------------------------------------------------
# Pure-JAX prior log-probability helpers
# ---------------------------------------------------------------------------

_LOG_2PI = jnp.log(2.0 * jnp.pi)


def _normal_log_prob_terms(x, loc, scale):
    """Element-wise Normal(loc, scale) log-density."""
    return -0.5 * _LOG_2PI - jnp.log(scale) - 0.5 * ((x - loc) / scale) ** 2


def _normal_log_prob(x, loc, scale):
    """Total Normal(loc, scale) log-density."""
    return jnp.sum(_normal_log_prob_terms(x, loc, scale))


def _truncated_normal_log_prob_terms(x, loc, scale, low, high):
    """Element-wise TruncatedNormal(loc, scale, low, high) log-density."""
    return dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high).log_prob(x)


def _uniform_log_prob_terms(x, low, high):
    """Element-wise Uniform(low, high) log-density."""
    width = high - low
    in_support = (x >= low) & (x <= high) & (width > 0)
    log_density = -jnp.log(jnp.maximum(width, 1e-30))
    return jnp.where(in_support, log_density, -jnp.inf)


def _half_normal_log_prob_terms(x, scale):
    """Element-wise HalfNormal(scale) log-density for x > 0."""
    return jnp.log(2.0) - 0.5 * _LOG_2PI - jnp.log(scale) - 0.5 * (x / scale) ** 2


def _half_normal_log_prob(x, scale):
    """Total HalfNormal(scale) log-density."""
    return jnp.sum(_half_normal_log_prob_terms(x, scale))


def _log_normal_log_prob_terms(x, loc, scale):
    """Element-wise LogNormal(loc, scale) log-density."""
    log_x = jnp.log(x)
    return -jnp.log(x) - 0.5 * _LOG_2PI - jnp.log(scale) - 0.5 * ((log_x - loc) / scale) ** 2


def _gamma_log_prob_terms(x, concentration, rate):
    """Element-wise Gamma(concentration, rate) log-density."""
    return (
        concentration * jnp.log(rate)
        - jax.lax.lgamma(concentration)
        + (concentration - 1.0) * jnp.log(x)
        - rate * x
    )


def _gamma_log_prob(x, concentration, rate):
    """Total Gamma(concentration, rate) log-density."""
    return jnp.sum(_gamma_log_prob_terms(x, concentration, rate))


def _exponential_log_prob_terms(x, rate):
    """Element-wise Exponential(rate) log-density."""
    return jnp.log(rate) - rate * x


def _real_log_prob(x, family_idx, loc, scale, low, high):
    """Log density for a REAL-support site with family dispatch.

    Families:
        0 — Normal(loc, scale)
        1 — TruncatedNormal(loc, scale, low, high)
        2 — Uniform(low, high)
    """
    families = jnp.broadcast_to(jnp.asarray(family_idx, dtype=jnp.int32), jnp.shape(x))
    normal_terms = _normal_log_prob_terms(x, loc, scale)
    truncated_terms = _truncated_normal_log_prob_terms(x, loc, scale, low, high)
    uniform_terms = _uniform_log_prob_terms(x, low, high)
    return jnp.sum(
        jnp.where(families == 0, normal_terms, 0.0)
        + jnp.where(families == 1, truncated_terms, 0.0)
        + jnp.where(families == 2, uniform_terms, 0.0)
    )


def _positive_log_prob(x, family_idx, loc, scale, concentration, rate):
    """Log density for a POSITIVE-support site with family dispatch.

    Families:
        0 — HalfNormal(scale)
        1 — Gamma(concentration, rate)
        2 — LogNormal(loc, scale)
        3 — Exponential(rate)
    """
    families = jnp.broadcast_to(jnp.asarray(family_idx, dtype=jnp.int32), jnp.shape(x))
    half_normal_terms = _half_normal_log_prob_terms(x, scale)
    gamma_terms = _gamma_log_prob_terms(x, concentration, rate)
    log_normal_terms = _log_normal_log_prob_terms(x, loc, scale)
    exponential_terms = _exponential_log_prob_terms(x, rate)
    return jnp.sum(
        jnp.where(families == 0, half_normal_terms, 0.0)
        + jnp.where(families == 1, gamma_terms, 0.0)
        + jnp.where(families == 2, log_normal_terms, 0.0)
        + jnp.where(families == 3, exponential_terms, 0.0)
    )


# ---------------------------------------------------------------------------
# Composite prior log-density
# ---------------------------------------------------------------------------

# Type alias: the prior state is a plain nested dict (valid JAX pytree).
# Structure: {site_name: {family: int32, loc: array, scale: array, ...}}
PriorRuntimeState = dict[str, dict[str, jnp.ndarray]]


def log_prior_unconstrained(
    z: jnp.ndarray,
    unravel_fn,
    registry: list[SiteDescriptor],
    prior_state: PriorRuntimeState,
) -> jnp.ndarray:
    """Log prior density in unconstrained space.

    ``prior_state`` is a fixed-structure JAX pytree whose *values* can
    change between calls without triggering JAX recompilation.

    Args:
        z: Flat unconstrained parameter vector ``(D,)``.
        unravel_fn: Converts *z* to ``dict[site_name, array]``.
        registry: Site descriptors (static per topology).
        prior_state: Per-site prior parameters (dynamic JAX pytree).

    Returns:
        Scalar log density ``log p_unc(z) = Σ_i [log p(T_i(z_i)) + log|J_i|]``.
    """
    unc = unravel_fn(z)
    lp = jnp.array(0.0)

    for site in registry:
        z_site = unc[site.name]
        params = prior_state[site.name]

        if site.support == SupportClass.REAL:
            # Constrained == unconstrained for REAL support
            low = params.get("low", jnp.full_like(params["loc"], -1e6))
            high = params.get("high", jnp.full_like(params["loc"], 1e6))
            lp = lp + _real_log_prob(
                z_site,
                params["family"],
                params["loc"],
                params["scale"],
                low,
                high,
            )
            # log|det J| = 0 for identity transform

        elif site.support == SupportClass.POSITIVE:
            x_site = jnp.exp(z_site)
            lp = lp + _positive_log_prob(
                x_site,
                params["family"],
                params["loc"],
                params["scale"],
                params["concentration"],
                params["rate"],
            )
            # log|det J| = sum(z) for exp transform
            lp = lp + jnp.sum(z_site)

    return lp


# ---------------------------------------------------------------------------
# Prior sampling
# ---------------------------------------------------------------------------


def sample_prior_unconstrained(
    rng_key: jnp.ndarray,
    registry: list[SiteDescriptor],
    prior_state: PriorRuntimeState,
    n_samples: int = 200,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample from the prior in unconstrained space.

    Returns ``(samples, rng_key)`` where *samples* has shape
    ``(n_samples, D)``.

    This is a Python-level loop (not JIT'd) — intended for initialization,
    not inner-loop hot paths.
    """
    if not registry:
        return jnp.zeros((n_samples, 0), dtype=jnp.float32), rng_key

    all_samples = []
    for _ in range(n_samples):
        parts = []
        for site in registry:
            rng_key, sk = random.split(rng_key)
            params = prior_state[site.name]

            if site.support == SupportClass.REAL:
                shape = site.shape if site.shape else ()
                family = jnp.broadcast_to(
                    jnp.asarray(params["family"], dtype=jnp.int32),
                    shape if shape else (),
                )
                if not jnp.all((family == 0) | (family == 1) | (family == 2)):
                    raise ValueError(f"Unknown REAL family index in site {site.name}")

                sk_normal, sk_trunc, sk_uniform = random.split(sk, 3)
                normal_sample = params["loc"] + params["scale"] * random.normal(
                    sk_normal, shape=shape
                )
                truncated_sample = dist.TruncatedNormal(
                    loc=params["loc"],
                    scale=params["scale"],
                    low=params.get("low", jnp.full_like(params["loc"], -1e6)),
                    high=params.get("high", jnp.full_like(params["loc"], 1e6)),
                ).sample(sk_trunc)
                uniform_sample = random.uniform(
                    sk_uniform,
                    shape=shape,
                    minval=params.get("low", jnp.full_like(params["loc"], 0.0)),
                    maxval=params.get("high", jnp.full_like(params["loc"], 1.0)),
                )
                x = jnp.where(
                    family == 0,
                    normal_sample,
                    jnp.where(family == 1, truncated_sample, uniform_sample),
                )
                parts.append(x.reshape(-1))

            elif site.support == SupportClass.POSITIVE:
                shape = site.shape if site.shape else ()
                family = jnp.broadcast_to(
                    jnp.asarray(params["family"], dtype=jnp.int32),
                    shape if shape else (),
                )
                if not jnp.all((family == 0) | (family == 1) | (family == 2) | (family == 3)):
                    raise ValueError(f"Unknown POSITIVE family index in site {site.name}")

                sk_half, sk_gamma, sk_log_normal, sk_exp = random.split(sk, 4)
                half_normal_sample = jnp.abs(params["scale"] * random.normal(sk_half, shape=shape))
                gamma_sample = random.gamma(sk_gamma, params["concentration"], shape=shape) / params[
                    "rate"
                ]
                log_normal_sample = jnp.exp(
                    params["loc"] + params["scale"] * random.normal(sk_log_normal, shape=shape)
                )
                exponential_sample = random.exponential(sk_exp, shape=shape) / params["rate"]
                x = jnp.where(
                    family == 0,
                    half_normal_sample,
                    jnp.where(
                        family == 1,
                        gamma_sample,
                        jnp.where(family == 2, log_normal_sample, exponential_sample),
                    ),
                )
                # Unconstrained = log(x)
                parts.append(jnp.log(jnp.maximum(x, 1e-30)).reshape(-1))

        if parts:
            all_samples.append(jnp.concatenate(parts))
        else:
            all_samples.append(jnp.zeros((0,), dtype=jnp.float32))

    return jnp.stack(all_samples), rng_key


# ---------------------------------------------------------------------------
# Prior state construction
# ---------------------------------------------------------------------------

# Hardcoded defaults for likelihood extra-parameter sites.
# These sites have no SSMPriors field — their priors are baked into
# SSMModel._sample_likelihood_extra_params.
_LIKELIHOOD_EXTRA_DEFAULTS: dict[str, dict] = {
    "obs_df": {"family": 1, "concentration": 5.0, "rate": 1.0},
    "obs_shape": {"family": 1, "concentration": 2.0, "rate": 1.0},
    "obs_r": {"family": 1, "concentration": 2.0, "rate": 0.5},
    "obs_concentration": {"family": 1, "concentration": 5.0, "rate": 0.5},
    "obs_ordered_base": {"family": 0, "loc": 0.0, "scale": 1.0},
    "obs_ordered_gaps": {"family": 0, "scale": 1.0},  # HalfNormal
    "obs_cat_intercepts": {"family": 0, "loc": 0.0, "scale": 1.0},
    "obs_cat_slopes": {"family": 0, "loc": 0.0, "scale": 1.0},
    "proc_df": {"family": 1, "concentration": 5.0, "rate": 1.0},
}


def _make_positive_params(
    shape: tuple[int, ...],
    *,
    family: int | list[int] = 0,
    loc: float = 0.0,
    scale: float = 1.0,
    concentration: float = 1.0,
    rate: float = 1.0,
) -> dict[str, jnp.ndarray]:
    """Build canonical param dict for a POSITIVE-support site."""
    s = shape if shape else ()
    return {
        "family": jnp.array(family, dtype=jnp.int32),
        "loc": jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float32), s),
        "scale": jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), s),
        "concentration": jnp.broadcast_to(jnp.asarray(concentration, dtype=jnp.float32), s),
        "rate": jnp.broadcast_to(jnp.asarray(rate, dtype=jnp.float32), s),
    }


def _make_real_params(
    shape: tuple[int, ...],
    *,
    family: int | list[int] = 0,
    loc: float = 0.0,
    scale: float = 1.0,
    low: float | list[float] | None = None,
    high: float | list[float] | None = None,
) -> dict[str, jnp.ndarray]:
    """Build canonical param dict for a REAL-support site."""
    s = shape if shape else ()
    params = {
        "family": jnp.array(family, dtype=jnp.int32),
        "loc": jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float32), s),
        "scale": jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), s),
    }
    if low is not None and high is not None:
        params["low"] = jnp.broadcast_to(jnp.asarray(low, dtype=jnp.float32), s)
        params["high"] = jnp.broadcast_to(jnp.asarray(high, dtype=jnp.float32), s)
    return params


def build_prior_runtime_state(
    registry: list[SiteDescriptor],
    priors: SSMPriors | None = None,
) -> PriorRuntimeState:
    """Build a ``PriorRuntimeState`` from the registry and optional SSMPriors.

    The returned dict has fixed structure per topology — only leaf values
    change when priors change.
    """
    from causal_ssm_agent.models.ssm.model import SSMPriors as SSMPriorsClass

    if priors is None:
        priors = SSMPriorsClass()

    state: PriorRuntimeState = {}

    for site in registry:
        priors_field = site.priors_field

        if priors_field is not None:
            # Core parameter site — read from SSMPriors
            prior_dict = getattr(priors, priors_field)
            state[site.name] = _params_from_prior_dict(site, prior_dict)

        elif site.name in _LIKELIHOOD_EXTRA_DEFAULTS:
            # Likelihood extra site — use hardcoded defaults
            defaults = _LIKELIHOOD_EXTRA_DEFAULTS[site.name]
            if site.support == SupportClass.POSITIVE:
                state[site.name] = _make_positive_params(
                    site.shape,
                    family=defaults.get("family", 0),
                    scale=defaults.get("scale", 1.0),
                    concentration=defaults.get("concentration", 1.0),
                    rate=defaults.get("rate", 1.0),
                )
            else:
                state[site.name] = _make_real_params(
                    site.shape,
                    family=defaults.get("family", 0),
                    loc=defaults.get("loc", 0.0),
                    scale=defaults.get("scale", 1.0),
                )

        else:
            # Fallback: sensible defaults
            if site.support == SupportClass.POSITIVE:
                state[site.name] = _make_positive_params(site.shape)
            else:
                state[site.name] = _make_real_params(site.shape)

    return state


def _params_from_prior_dict(
    site: SiteDescriptor,
    prior_dict: dict,
) -> dict[str, jnp.ndarray]:
    """Convert an SSMPriors prior dict to canonical params for a site."""
    if site.support == SupportClass.REAL:
        has_bounds = "lower" in prior_dict and "upper" in prior_dict
        return _make_real_params(
            site.shape,
            family=prior_dict.get("family", 1 if has_bounds else 0),
            loc=prior_dict.get("mu", 0.0),
            scale=prior_dict.get("sigma", 1.0),
            low=prior_dict.get("lower"),
            high=prior_dict.get("upper"),
        )
    elif site.support == SupportClass.POSITIVE:
        return _make_positive_params(
            site.shape,
            family=prior_dict.get("family", 0),
            loc=prior_dict.get("loc", 0.0),
            scale=prior_dict.get("sigma", 1.0),
            concentration=prior_dict.get("concentration", 1.0),
            rate=prior_dict.get("rate", 1.0),
        )
    raise ValueError(f"Unknown support class: {site.support}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_registry_matches_trace(
    registry: list[SiteDescriptor],
    site_info: dict,
) -> None:
    """Assert that registry site names and shapes match a traced site_info.

    Raises ``AssertionError`` with a detailed message on mismatch.
    """
    registry_names = {s.name for s in registry}
    trace_names = set(site_info.keys())

    missing = registry_names - trace_names
    extra = trace_names - registry_names

    errors = []
    if missing:
        errors.append(f"Registry has sites not found in trace: {sorted(missing)}")
    if extra:
        errors.append(f"Trace has sites not found in registry: {sorted(extra)}")

    for site in registry:
        if site.name in site_info:
            traced_shape = site_info[site.name]["shape"]
            if site.shape != traced_shape:
                errors.append(
                    f"Shape mismatch for {site.name}: registry={site.shape}, trace={traced_shape}"
                )

    if errors:
        raise AssertionError("Registry/trace mismatch:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# Serialization / deserialization for compiled artifacts
# ---------------------------------------------------------------------------


def serialize_site_registry(registry: list[SiteDescriptor]) -> list[dict]:
    """Serialize site registry for JSON storage inside ``_compiled_ssm``."""
    return [
        {
            "name": s.name,
            "shape": list(s.shape),
            "support": s.support.value,
            "assembly_group": s.assembly_group,
            "site_kind": s.site_kind.value,
            "transform_kind": s.transform_kind.value,
            "deterministic_name": s.deterministic_name,
            "fixed_spec_field": s.fixed_spec_field,
            "priors_field": s.priors_field,
            "runtime_prior_key": s.runtime_prior_key,
            "is_runtime_prior_controlled": s.is_runtime_prior_controlled,
        }
        for s in registry
    ]


def deserialize_site_registry(payload: list[dict]) -> list[SiteDescriptor]:
    """Restore site registry from serialized form."""
    return [
        SiteDescriptor(
            name=d["name"],
            shape=tuple(d["shape"]),
            support=SupportClass(d["support"]),
            assembly_group=d["assembly_group"],
            site_kind=SiteKind(d["site_kind"]),
            transform_kind=TransformKind(d["transform_kind"]),
            deterministic_name=d.get("deterministic_name"),
            fixed_spec_field=d.get("fixed_spec_field"),
            priors_field=d.get("priors_field"),
            runtime_prior_key=d.get("runtime_prior_key"),
            is_runtime_prior_controlled=d.get("is_runtime_prior_controlled", True),
        )
        for d in payload
    ]


def serialize_prior_runtime_state(state: PriorRuntimeState) -> dict:
    """Serialize prior runtime state for JSON storage."""
    import numpy as np

    result = {}
    for name, params in state.items():
        result[name] = {}
        for k, v in params.items():
            if hasattr(v, "tolist"):
                result[name][k] = np.asarray(v).tolist()
            else:
                result[name][k] = v
    return result


def deserialize_prior_runtime_state(
    payload: dict,
    registry: list[SiteDescriptor],
) -> PriorRuntimeState:
    """Restore prior runtime state from serialized form.

    Uses the registry to determine correct dtypes (int32 for family,
    float32 for all others).
    """
    state: PriorRuntimeState = {}
    for site in registry:
        raw = payload[site.name]
        params: dict[str, jnp.ndarray] = {}
        for k, v in raw.items():
            if k == "family":
                params[k] = jnp.array(v, dtype=jnp.int32)
            else:
                params[k] = jnp.asarray(v, dtype=jnp.float32)
        state[site.name] = params
    return state


def compile_prior_semantics(
    spec: SSMSpec,
    priors: SSMPriors | None = None,
) -> dict:
    """Build the ``compiled_prior_semantics`` block for a compiled artifact.

    This is the single cross-stage source of truth for prior/runtime
    semantics.  Downstream readers (``make_builder_from_compiled_artifact``,
    Stage 4b, prior predictive) should use this as the only supported
    serialized prior/runtime representation.
    """
    bundle = build_prior_runtime_bundle(spec, priors)
    return {
        "schema_version": 4,
        "site_registry": serialize_site_registry(bundle.registry),
        "prior_state": serialize_prior_runtime_state(bundle.prior_state),
    }


def build_prior_runtime_bundle(
    spec: SSMSpec,
    priors: SSMPriors | None = None,
) -> PriorRuntimeBundle:
    """Build reusable runtime components directly from ``spec`` and ``priors``."""
    site_runtime = build_site_runtime_bundle(spec)
    prior_state = build_prior_runtime_state(site_runtime.registry, priors)
    return PriorRuntimeBundle(
        site_runtime=site_runtime,
        prior_state=prior_state,
    )


def load_prior_runtime_bundle(
    compiled_prior_semantics: dict,
) -> PriorRuntimeBundle:
    """Restore reusable runtime components from ``compiled_prior_semantics``."""
    schema_version = compiled_prior_semantics.get("schema_version")
    if schema_version != 4:
        raise ValueError(
            f"Unsupported compiled_prior_semantics schema_version {schema_version!r}; expected 4."
        )

    registry = deserialize_site_registry(compiled_prior_semantics["site_registry"])
    prior_state = deserialize_prior_runtime_state(compiled_prior_semantics["prior_state"], registry)
    site_runtime = _build_site_runtime_bundle_from_registry(registry)
    return PriorRuntimeBundle(
        site_runtime=site_runtime,
        prior_state=prior_state,
    )


# ---------------------------------------------------------------------------
# SSMPriors reconstruction from compiled prior semantics
# ---------------------------------------------------------------------------


def _to_prior_value(arr: jnp.ndarray):
    """Convert a JAX array to an SSMPriors-compatible value.

    - 0-d arrays → Python float
    - 1-d arrays where all elements are equal → Python float (scalar prior)
    - 1-d arrays with varying elements → Python float list
    """
    import numpy as np

    arr = np.asarray(arr)
    if arr.ndim == 0:
        return float(arr)
    values = arr.ravel()
    if values.size == 1 or np.all(values == values[0]):
        return float(values[0])
    return [float(v) for v in values]


def reconstruct_ssm_priors(
    registry: list[SiteDescriptor],
    prior_state: PriorRuntimeState,
) -> SSMPriors:
    """Reconstruct an ``SSMPriors`` from the compiled prior semantics.

    This is the inverse of ``build_prior_runtime_state``: it maps canonical
    prior parameters back to the ``SSMPriors`` dict-based format that
    ``SSMModel.__init__`` expects.

    Likelihood extra sites (obs_df, proc_df, etc.) have no SSMPriors field
    and are silently skipped.
    """
    import numpy as np

    from causal_ssm_agent.models.ssm.model import SSMPriors as SSMPriorsClass

    kwargs: dict[str, dict] = {}

    for site in registry:
        priors_field = site.priors_field
        if priors_field is None:
            continue  # likelihood extra, not in SSMPriors

        params = prior_state[site.name]

        if site.support == SupportClass.REAL:
            family_values = np.asarray(params["family"], dtype=int).ravel()
            family = int(family_values[0]) if family_values.size else 0
            runtime_kind = get_real_runtime_kind_from_index(family)
            prior_kwargs = {
                "mu": _to_prior_value(params["loc"]),
                "sigma": _to_prior_value(params["scale"]),
            }
            if "low" in params and "high" in params:
                prior_kwargs["lower"] = _to_prior_value(params["low"])
                prior_kwargs["upper"] = _to_prior_value(params["high"])
            if runtime_kind != PriorRuntimeKind.NORMAL:
                prior_kwargs["family"] = family
            kwargs[priors_field] = prior_kwargs

        elif site.support == SupportClass.POSITIVE:
            family_values = np.asarray(params["family"], dtype=int).ravel()
            family = int(family_values[0]) if family_values.size else 0
            runtime_kind = get_positive_runtime_kind_from_index(family)
            if runtime_kind == PriorRuntimeKind.HALF_NORMAL:
                kwargs[priors_field] = {"sigma": _to_prior_value(params["scale"])}
            elif runtime_kind == PriorRuntimeKind.GAMMA:
                kwargs[priors_field] = {
                    "family": family,
                    "concentration": _to_prior_value(params["concentration"]),
                    "rate": _to_prior_value(params["rate"]),
                }
            elif runtime_kind == PriorRuntimeKind.LOG_NORMAL:
                kwargs[priors_field] = {
                    "family": family,
                    "loc": _to_prior_value(params["loc"]),
                    "sigma": _to_prior_value(params["scale"]),
                }
            elif runtime_kind == PriorRuntimeKind.EXPONENTIAL:
                kwargs[priors_field] = {
                    "family": family,
                    "rate": _to_prior_value(params["rate"]),
                }
            else:
                raise ValueError(f"Unknown POSITIVE family index {family}")

    return SSMPriorsClass(**kwargs)
