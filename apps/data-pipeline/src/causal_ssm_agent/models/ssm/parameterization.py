"""Canonical site registry and compile-stable prior evaluation.

Provides:
- SiteDescriptor: metadata for each sample site, derived deterministically from SSMSpec
- build_site_registry: enumerate all sample sites without model tracing
- build_prior_runtime_state: create fixed-structure JAX pytree from SSMPriors
- log_prior_unconstrained: pure-JAX prior log-density with switch pattern
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


@dataclass
class PriorRuntimeBundle:
    """Reusable runtime components derived from compiled prior semantics."""

    registry: list[SiteDescriptor]
    prior_state: PriorRuntimeState
    transforms: dict[str, dist.transforms.Transform]
    flat_dim: int
    unravel_fn: Any


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

    if spec.diffusion_dist == DistributionFamily.STUDENT_T:
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


def group_sites_by_assembly_role(
    registry: list[SiteDescriptor],
) -> dict[str, list[SiteDescriptor]]:
    """Group site descriptors by assembly role."""
    grouped: dict[str, list[SiteDescriptor]] = {}
    for site in registry:
        grouped.setdefault(site.assembly_group, []).append(site)
    return grouped


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


# ---------------------------------------------------------------------------
# Pure-JAX prior log-probability helpers
# ---------------------------------------------------------------------------

_LOG_2PI = jnp.log(2.0 * jnp.pi)


def _normal_log_prob(x, loc, scale):
    """Sum of element-wise Normal(loc, scale) log-density."""
    return jnp.sum(-0.5 * _LOG_2PI - jnp.log(scale) - 0.5 * ((x - loc) / scale) ** 2)


def _half_normal_log_prob(x, scale):
    """Sum of element-wise HalfNormal(scale) log-density for x > 0."""
    return jnp.sum(jnp.log(2.0) - 0.5 * _LOG_2PI - jnp.log(scale) - 0.5 * (x / scale) ** 2)


def _gamma_log_prob(x, concentration, rate):
    """Sum of element-wise Gamma(concentration, rate) log-density."""
    return jnp.sum(
        concentration * jnp.log(rate)
        - jax.lax.lgamma(concentration)
        + (concentration - 1.0) * jnp.log(x)
        - rate * x
    )


def _real_log_prob(x, family_idx, loc, scale):
    """Log density for a REAL-support site with family dispatch.

    Families:
        0 — Normal(loc, scale)
    """
    branches = [
        lambda loc, scale: _normal_log_prob(x, loc, scale),
    ]
    return jax.lax.switch(family_idx, branches, loc, scale)


def _positive_log_prob(x, family_idx, scale, concentration, rate):
    """Log density for a POSITIVE-support site with family dispatch.

    Families:
        0 — HalfNormal(scale)
        1 — Gamma(concentration, rate)
    """
    branches = [
        lambda s, _c, _r: _half_normal_log_prob(x, s),
        lambda _s, c, r: _gamma_log_prob(x, c, r),
    ]
    return jax.lax.switch(family_idx, branches, scale, concentration, rate)


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
            lp = lp + _real_log_prob(z_site, params["family"], params["loc"], params["scale"])
            # log|det J| = 0 for identity transform

        elif site.support == SupportClass.POSITIVE:
            x_site = jnp.exp(z_site)
            lp = lp + _positive_log_prob(
                x_site,
                params["family"],
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
            family = int(params["family"])

            if site.support == SupportClass.REAL:
                shape = site.shape if site.shape else ()
                x = params["loc"] + params["scale"] * random.normal(sk, shape=shape)
                parts.append(x.reshape(-1))

            elif site.support == SupportClass.POSITIVE:
                shape = site.shape if site.shape else ()
                if family == 0:  # HalfNormal
                    x = jnp.abs(params["scale"] * random.normal(sk, shape=shape))
                elif family == 1:  # Gamma
                    x = random.gamma(sk, params["concentration"], shape=shape) / params["rate"]
                else:
                    raise ValueError(f"Unknown POSITIVE family index {family}")
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
    family: int = 0,
    scale: float = 1.0,
    concentration: float = 1.0,
    rate: float = 1.0,
) -> dict[str, jnp.ndarray]:
    """Build canonical param dict for a POSITIVE-support site."""
    s = shape if shape else ()
    return {
        "family": jnp.array(family, dtype=jnp.int32),
        "scale": jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), s),
        "concentration": jnp.broadcast_to(jnp.asarray(concentration, dtype=jnp.float32), s),
        "rate": jnp.broadcast_to(jnp.asarray(rate, dtype=jnp.float32), s),
    }


def _make_real_params(
    shape: tuple[int, ...],
    *,
    family: int = 0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> dict[str, jnp.ndarray]:
    """Build canonical param dict for a REAL-support site."""
    s = shape if shape else ()
    return {
        "family": jnp.array(family, dtype=jnp.int32),
        "loc": jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float32), s),
        "scale": jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), s),
    }


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
        return _make_real_params(
            site.shape,
            loc=prior_dict.get("mu", 0.0),
            scale=prior_dict.get("sigma", 1.0),
        )
    elif site.support == SupportClass.POSITIVE:
        return _make_positive_params(
            site.shape,
            family=0,
            scale=prior_dict.get("sigma", 1.0),
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
        "schema_version": 2,
        "site_registry": serialize_site_registry(bundle.registry),
        "prior_state": serialize_prior_runtime_state(bundle.prior_state),
    }


def build_prior_runtime_bundle(
    spec: SSMSpec,
    priors: SSMPriors | None = None,
) -> PriorRuntimeBundle:
    """Build reusable runtime components directly from ``spec`` and ``priors``."""
    registry = build_site_registry(spec)
    prior_state = build_prior_runtime_state(registry, priors)
    transforms = build_transforms(registry)
    flat_dim, unravel_fn = build_unravel_fn(registry)
    return PriorRuntimeBundle(
        registry=registry,
        prior_state=prior_state,
        transforms=transforms,
        flat_dim=flat_dim,
        unravel_fn=unravel_fn,
    )


def load_prior_runtime_bundle(
    compiled_prior_semantics: dict,
) -> PriorRuntimeBundle:
    """Restore reusable runtime components from ``compiled_prior_semantics``."""
    schema_version = compiled_prior_semantics.get("schema_version")
    if schema_version != 2:
        raise ValueError(
            f"Unsupported compiled_prior_semantics schema_version {schema_version!r}; expected 2."
        )

    registry = deserialize_site_registry(compiled_prior_semantics["site_registry"])
    prior_state = deserialize_prior_runtime_state(compiled_prior_semantics["prior_state"], registry)
    transforms = build_transforms(registry)
    flat_dim, unravel_fn = build_unravel_fn(registry)
    return PriorRuntimeBundle(
        registry=registry,
        prior_state=prior_state,
        transforms=transforms,
        flat_dim=flat_dim,
        unravel_fn=unravel_fn,
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
    from causal_ssm_agent.models.ssm.model import SSMPriors as SSMPriorsClass

    kwargs: dict[str, dict] = {}

    for site in registry:
        priors_field = site.priors_field
        if priors_field is None:
            continue  # likelihood extra, not in SSMPriors

        params = prior_state[site.name]

        if site.support == SupportClass.REAL:
            kwargs[priors_field] = {
                "mu": _to_prior_value(params["loc"]),
                "sigma": _to_prior_value(params["scale"]),
            }

        elif site.support == SupportClass.POSITIVE:
            kwargs[priors_field] = {"sigma": _to_prior_value(params["scale"])}

    return SSMPriorsClass(**kwargs)
