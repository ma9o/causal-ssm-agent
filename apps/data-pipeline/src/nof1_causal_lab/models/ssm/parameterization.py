"""Canonical site registry and compile-stable prior evaluation.

Provides:
- SiteDescriptor: metadata for each sample site, derived deterministically from SSMSpec
- build_site_registry: enumerate all sample sites without model tracing
- build_prior_runtime_state: create fixed-structure JAX pytree from PriorRegistry
- sample_prior_unconstrained: pure-JAX prior sampling

The site registry is the single authority for "what sample sites exist."
It replaces the three overlapping conventions:
- parameter_bindings (semantic parameter → site mapping)
- _discover_sites (trace-driven site enumeration)
- _assemble_deterministics (hardcoded site name checks)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree

from nof1_causal_lab.distributions import (
    PriorDistributionFamily,
    get_positive_runtime_family_index,
    get_positive_runtime_kind_from_index,
    get_real_runtime_family_index,
    get_real_runtime_kind_from_index,
)
from nof1_causal_lab.models.ssm.compile.contracts import (
    CompiledPriorSemantics,
    CompiledPriorSite,
    SerializedSiteDescriptor,
)
from nof1_causal_lab.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
    stabilize_covariance_for_cholesky,
)
from nof1_causal_lab.models.ssm.structure.sites import (
    SiteDescriptor,
    SiteKind,
    SupportClass,
    TransformKind,
)
from nof1_causal_lab.models.ssm.structure.sites import (
    make_site as _site,
)
from nof1_causal_lab.models.ssm.structure.sites import (
    site_size as _site_size,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.execution.contracts import LikelihoodExtraParams
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.priors import PriorRegistry, PriorSpec


@dataclass
class SiteRuntimeBundle:
    """Reusable topology-only runtime components derived from site metadata."""

    registry: list[SiteDescriptor]
    transforms: dict[str, dist.transforms.Transform]
    flat_dim: int
    unravel_fn: Any

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
        """Flat list of per-element names (e.g. ``vf_0_decay[0]``)."""
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


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def build_site_registry(spec: SSMSpec) -> list[SiteDescriptor]:
    """Enumerate all sample sites deterministically from *spec*.

    Core model sites come from ``spec.iter_sample_sites()`` (each dynamics
    component and structure block owns its own descriptor). Likelihood-extra sites
    (`obs_df`, `obs_shape`, cutpoints, etc.) depend on
    ``spec.manifest_dists`` and are generated locally because they are
    not block-owned.

    The returned list is sorted by site name to match JAX pytree
    dict-key ordering used by ``ravel_pytree``.
    """
    from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily

    sites: list[SiteDescriptor] = list(spec.iter_sample_sites())
    n_m = spec.n_manifest

    # -- Likelihood extra-parameter sites -----------------------------------

    manifest_dist_set = set(spec.manifest_dists)

    if DistributionFamily.STUDENT_T in manifest_dist_set:
        sites.append(
            _site(
                "obs_df",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.OBS_DF,
                priors_field="obs_df",
            )
        )
    if DistributionFamily.GAMMA in manifest_dist_set:
        sites.append(
            _site(
                "obs_shape",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.OBS_SHAPE,
                priors_field="obs_shape",
            )
        )
    if DistributionFamily.NEGATIVE_BINOMIAL in manifest_dist_set:
        sites.append(
            _site(
                "obs_r",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.OBS_R,
                priors_field="obs_r",
            )
        )
    if DistributionFamily.BETA in manifest_dist_set:
        sites.append(
            _site(
                "obs_concentration",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.OBS_CONCENTRATION,
                priors_field="obs_concentration",
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
                    priors_field="obs_ordered_base",
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
                        priors_field="obs_ordered_gaps",
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
                    priors_field="obs_cat_intercepts",
                )
            )
            sites.append(
                _site(
                    "obs_cat_slopes",
                    cat_shape,
                    SupportClass.REAL,
                    "likelihood",
                    SiteKind.OBS_CAT_SLOPES,
                    priors_field="obs_cat_slopes",
                )
            )

    from nof1_causal_lab.models.ssm.spec_metadata import has_student_t_diffusion

    if has_student_t_diffusion(spec):
        sites.append(
            _site(
                "proc_df",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.PROC_DF,
                priors_field="proc_df",
            )
        )

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
        elif site.transform_kind == TransformKind.CORRELATION:
            transforms[site.name] = dist.transforms.ComposeTransform(
                [
                    dist.transforms.SigmoidTransform(),
                    dist.transforms.AffineTransform(loc=-1.0, scale=2.0),
                ]
            )
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


def build_site_runtime_bundle(spec: SSMSpec) -> SiteRuntimeBundle:
    """Build reusable topology-only runtime components from ``spec``."""
    registry = build_site_registry(spec)
    return _build_site_runtime_bundle_from_registry(registry)


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


def _maybe_vmap_assemble(
    assemble_fn,
    *site_samples,
    n_draws: int,
) -> jnp.ndarray:
    """Vmap ``assemble_fn`` over the present samples; broadcast the
    template otherwise.

    Each entry of ``site_samples`` is either a ``(n_draws, …)`` array
    (sampled) or ``None`` (fixed). ``assemble_fn`` is called with the
    same positional shape, threading ``None`` through for absent sites.
    """
    if all(s is None for s in site_samples):
        fixed = assemble_fn()
        return jnp.broadcast_to(fixed, (n_draws, *fixed.shape))

    present_indices = [i for i, s in enumerate(site_samples) if s is not None]
    present_arrays = tuple(site_samples[i] for i in present_indices)

    def _one(*present):
        full = list(site_samples)
        for idx, value in zip(present_indices, present, strict=True):
            full[idx] = value
        return assemble_fn(*full)

    return jax.vmap(_one)(*present_arrays)


def _compose_t0_cov_batched(
    spec: SSMSpec,
    diag_samples: jnp.ndarray | None,
    corr_samples: jnp.ndarray | None,
    static_state_samples: jnp.ndarray | None,
    n_draws: int,
) -> jnp.ndarray:
    """Vmap t0_cov composition (latent chol + static-factor contribution).

    Always emits a stabilized covariance regardless of which inputs are
    fixed. One vmap, one composition function — no per-input powerset.
    """

    def _one(diag_values=None, corr_values=None, static_values=None):
        cov = spec.t0_chol_block.assemble_cov(diag_values, corr_values)
        static_sds = spec.static_state_sd_block.assemble(static_values)
        if static_sds.size:
            loadings = jnp.asarray(spec.static_factor_loadings)
            cov = cov + loadings @ jnp.diag(static_sds**2) @ loadings.T
        cov = 0.5 * (cov + cov.T)
        stable_cov, _ = stabilize_covariance_for_cholesky(
            cov, min_eigenvalue=INITIAL_STATE_COV_MIN_EIGENVALUE
        )
        return stable_cov

    return _maybe_vmap_assemble(
        _one,
        diag_samples,
        corr_samples,
        static_state_samples,
        n_draws=n_draws,
    )


def assemble_deterministics_from_registry(
    samples: dict[str, jnp.ndarray],
    spec: SSMSpec,
    *,
    n_draws: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Assemble deterministic matrices from per-site posterior samples.

    Each structure block produces its deterministic via
    ``_maybe_vmap_assemble``; missing samples (fixed blocks) broadcast
    the template. ``manifest_cov`` and ``t0_cov`` are multi-block
    composition steps.
    """
    n_draws = _resolve_num_draws(samples, n_draws)
    det: dict[str, jnp.ndarray] = {}

    # Diffusion Cholesky
    det["diffusion"] = _maybe_vmap_assemble(
        spec.diffusion_block.assemble,
        samples.get("diffusion_diag_free"),
        samples.get("diffusion_lower_free"),
        n_draws=n_draws,
    )

    # Input-effect matrix
    det["input_effect"] = _maybe_vmap_assemble(
        spec.input_effect_block.assemble,
        samples.get("input_effect_free"),
        n_draws=n_draws,
    )

    # Static-factor SDs
    det["static_state_sds"] = _maybe_vmap_assemble(
        spec.static_state_sd_block.assemble,
        samples.get("static_state_sd_free"),
        n_draws=n_draws,
    )

    # Loading matrix
    det["lambda"] = _maybe_vmap_assemble(
        spec.lambda_block.assemble,
        samples.get("lambda_free"),
        n_draws=n_draws,
    )

    # Manifest means
    det["manifest_means"] = _maybe_vmap_assemble(
        spec.manifest_means_block.assemble,
        samples.get("manifest_means_free"),
        n_draws=n_draws,
    )

    # Manifest covariance: Cholesky from block then chol @ cholᵀ
    manifest_chol_batched = _maybe_vmap_assemble(
        spec.manifest_chol_block.assemble,
        samples.get("manifest_var_diag_free"),
        n_draws=n_draws,
    )
    det["manifest_cov"] = jax.vmap(lambda chol: chol @ chol.T)(manifest_chol_batched)

    # t0 means
    det["t0_means"] = _maybe_vmap_assemble(
        spec.t0_means_block.assemble,
        samples.get("t0_means_free"),
        n_draws=n_draws,
    )

    # t0 covariance: latent chol + static-factor contribution, stabilized
    det["t0_cov"] = _compose_t0_cov_batched(
        spec,
        samples.get("t0_var_diag_free"),
        samples.get("t0_var_lower_free"),
        samples.get("static_state_sd_free"),
        n_draws,
    )

    return det


def assemble_extra_params_from_registry(
    spec: SSMSpec,
    samples: dict[str, jnp.ndarray],
    registry: list[SiteDescriptor],
) -> LikelihoodExtraParams:
    """Assemble likelihood extra parameters using registry metadata as authority."""
    from nof1_causal_lab.models.ssm.likelihood_extra_params import assemble_sampled_extra_params

    return assemble_sampled_extra_params(
        spec,
        select_site_samples(samples, registry, assembly_group="likelihood"),
    )


# Type alias: the prior state is a plain nested dict (valid JAX pytree).
# Structure: {site_name: {family: int64, loc: array, scale: array, ...}}
PriorRuntimeState = dict[str, dict[str, jnp.ndarray]]


def _correlation_inverse(x: jnp.ndarray) -> jnp.ndarray:
    """Map correlation values in ``(-1, 1)`` to the unconstrained real line."""
    x_safe = jnp.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)
    p = 0.5 * (x_safe + 1.0)
    return jnp.log(p) - jnp.log1p(-p)


# ---------------------------------------------------------------------------
# Prior sampling
# ---------------------------------------------------------------------------


def _stable_site_key(rng_key: jnp.ndarray, site_name: str) -> jnp.ndarray:
    """Derive a site stream that is unchanged by registry insertion or reordering."""
    digest = hashlib.sha256(site_name.encode()).digest()
    first = int.from_bytes(digest[:4], "little")
    second = int.from_bytes(digest[4:8], "little")
    return random.fold_in(random.fold_in(rng_key, first), second)


def sample_prior_unconstrained(
    rng_key: jnp.ndarray,
    registry: list[SiteDescriptor],
    prior_state: PriorRuntimeState,
    n_samples: int = 200,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample from the prior in unconstrained space.

    Returns ``(samples, rng_key)`` where *samples* has shape
    ``(n_samples, D)``.

    Uses ``build_site_prior_distribution`` as the single authority for
    family → distribution mapping, then applies the inverse of each
    site's unconstrained ↔ constrained transform.

    This is a Python-level loop (not JIT'd) — intended for initialization,
    not inner-loop hot paths.
    """
    if not registry:
        return jnp.zeros((n_samples, 0), dtype=jnp.float32), rng_key

    all_samples = []
    for sample_index in range(n_samples):
        sample_key = random.fold_in(rng_key, sample_index)
        parts = []
        for site in registry:
            sk = _stable_site_key(sample_key, site.name)
            d = build_site_prior_distribution(site, prior_state[site.name])
            x = d.sample(sk)

            if site.support == SupportClass.POSITIVE:
                parts.append(jnp.log(jnp.maximum(x, 1e-30)).reshape(-1))
            elif site.support == SupportClass.CORRELATION:
                parts.append(_correlation_inverse(x).reshape(-1))
            else:
                parts.append(x.reshape(-1))

        if parts:
            all_samples.append(jnp.concatenate(parts))
        else:
            all_samples.append(jnp.zeros((0,), dtype=jnp.float32))

    return jnp.stack(all_samples), random.fold_in(rng_key, n_samples)


# ---------------------------------------------------------------------------
# Prior state construction
# ---------------------------------------------------------------------------

_DEFAULT_REAL_LOW = -1e6
_DEFAULT_REAL_HIGH = 1e6


def _make_positive_params(
    shape: tuple[int, ...],
    *,
    family: int | list[int] = 0,
    loc: float = 0.0,
    scale: float = 1.0,
    concentration: float = 1.0,
    rate: float = 1.0,
    value: float = 1.0,
) -> dict[str, jnp.ndarray]:
    """Build canonical param dict for a POSITIVE-support site."""
    s = shape or ()
    return {
        "family": jnp.array(family, dtype=jnp.int32),
        "loc": jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float32), s),
        "scale": jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), s),
        "concentration": jnp.broadcast_to(jnp.asarray(concentration, dtype=jnp.float32), s),
        "rate": jnp.broadcast_to(jnp.asarray(rate, dtype=jnp.float32), s),
        "value": jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float32), s),
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
    s = shape or ()
    low_value = _DEFAULT_REAL_LOW if low is None else low
    high_value = _DEFAULT_REAL_HIGH if high is None else high
    return {
        "family": jnp.array(family, dtype=jnp.int32),
        "loc": jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float32), s),
        "scale": jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), s),
        "low": jnp.broadcast_to(jnp.asarray(low_value, dtype=jnp.float32), s),
        "high": jnp.broadcast_to(jnp.asarray(high_value, dtype=jnp.float32), s),
    }


def build_prior_runtime_state(
    registry: list[SiteDescriptor],
    priors: PriorRegistry | None = None,
) -> PriorRuntimeState:
    """Build a ``PriorRuntimeState`` from the registry and optional PriorRegistry.

    The returned dict has fixed structure per topology — only leaf values
    change when priors change.
    """
    from nof1_causal_lab.models.ssm.priors import default_prior_for_descriptor

    state: PriorRuntimeState = {}

    for site in registry:
        prior = (priors.get(site.name) if priors is not None else None) or (
            default_prior_for_descriptor(site)
        )
        state[site.name] = _params_from_prior_spec(site, prior)

    return state


def _params_from_prior_spec(
    site: SiteDescriptor,
    prior: PriorSpec,
) -> dict[str, jnp.ndarray]:
    """Convert a canonical prior spec to runtime prior-state params."""
    params = prior.params
    if site.support in {SupportClass.REAL, SupportClass.CORRELATION}:
        if prior.family not in {
            PriorDistributionFamily.NORMAL,
            PriorDistributionFamily.TRUNCATED_NORMAL,
            PriorDistributionFamily.UNIFORM,
        }:
            raise ValueError(
                f"Prior family {prior.family.value!r} is incompatible with "
                f"{site.support.value} site {site.name!r}"
            )
        has_bounds = "lower" in params and "upper" in params
        return _make_real_params(
            site.shape,
            family=get_real_runtime_family_index(
                PriorDistributionFamily.TRUNCATED_NORMAL
                if prior.family == PriorDistributionFamily.NORMAL
                and (has_bounds or site.support == SupportClass.CORRELATION)
                else prior.family
            ),
            loc=params.get("mu", 0.0),
            scale=params.get("sigma", 1.0),
            low=params.get("lower", -1.0 if site.support == SupportClass.CORRELATION else None),
            high=params.get("upper", 1.0 if site.support == SupportClass.CORRELATION else None),
        )
    if site.support == SupportClass.POSITIVE:
        if prior.family not in {
            PriorDistributionFamily.HALF_NORMAL,
            PriorDistributionFamily.GAMMA,
            PriorDistributionFamily.LOG_NORMAL,
            PriorDistributionFamily.EXPONENTIAL,
            PriorDistributionFamily.DELTA,
        }:
            raise ValueError(
                f"Prior family {prior.family.value!r} is incompatible with "
                f"positive site {site.name!r}"
            )
        return _make_positive_params(
            site.shape,
            family=get_positive_runtime_family_index(prior.family),
            loc=params.get("mu", params.get("loc", 0.0)),
            scale=params.get("sigma", 1.0),
            concentration=params.get("concentration", 1.0),
            rate=params.get("rate", 1.0),
            value=params.get("value", 1.0),
        )
    raise ValueError(f"Unknown support class: {site.support}")


# ---------------------------------------------------------------------------
# Serialization / deserialization for compiled artifacts
# ---------------------------------------------------------------------------


def serialize_site_registry(registry: list[SiteDescriptor]) -> list[SerializedSiteDescriptor]:
    """Serialize site registry for JSON storage inside ``_compiled_ssm``."""
    return [
        SerializedSiteDescriptor(
            name=s.name,
            shape=list(s.shape),
            support=s.support,
            assembly_group=s.assembly_group,
            site_kind=s.site_kind,
            transform_kind=s.transform_kind,
            deterministic_name=s.deterministic_name,
            fixed_spec_field=s.fixed_spec_field,
            priors_field=s.priors_field,
            runtime_prior_key=s.runtime_prior_key,
            is_runtime_prior_controlled=s.is_runtime_prior_controlled,
        )
        for s in registry
    ]


def deserialize_site_registry(payload: list[SerializedSiteDescriptor]) -> list[SiteDescriptor]:
    """Restore site registry from serialized form."""
    return [
        SiteDescriptor(
            name=d.name,
            shape=tuple(d.shape),
            support=d.support,
            assembly_group=d.assembly_group,
            site_kind=d.site_kind,
            transform_kind=d.transform_kind,
            deterministic_name=d.deterministic_name,
            fixed_spec_field=d.fixed_spec_field,
            priors_field=d.priors_field,
            runtime_prior_key=d.runtime_prior_key,
            is_runtime_prior_controlled=d.is_runtime_prior_controlled,
        )
        for d in payload
    ]


def serialize_prior_runtime_state(state: PriorRuntimeState) -> dict[str, CompiledPriorSite]:
    """Serialize prior runtime state for JSON storage."""
    import numpy as np

    result: dict[str, CompiledPriorSite] = {}
    for name, params in state.items():
        serialized_params = {}
        for k, v in params.items():
            if hasattr(v, "tolist"):
                serialized_params[k] = np.asarray(v).tolist()
            else:
                serialized_params[k] = v
        result[name] = CompiledPriorSite.model_validate(serialized_params)
    return result


def deserialize_prior_runtime_state(
    payload: dict[str, CompiledPriorSite],
    registry: list[SiteDescriptor],
) -> PriorRuntimeState:
    """Restore prior runtime state from serialized form.

    Uses the registry to determine correct dtypes (int32 for family,
    float32 for all others).
    """
    state: PriorRuntimeState = {}
    for site in registry:
        raw = payload[site.name].model_dump(exclude_none=True)
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
    priors: PriorRegistry | None = None,
) -> CompiledPriorSemantics:
    """Build the ``compiled_prior_semantics`` block for a compiled artifact.

    This is the single cross-stage source of truth for prior/runtime
    semantics.  Downstream readers (compiled-artifact model construction,
    pre-fit diagnostics, prior predictive) should use this as the only
    supported serialized prior/runtime representation.
    """
    bundle = build_prior_runtime_bundle(spec, priors)
    return CompiledPriorSemantics(
        schema_version=5,
        site_registry=serialize_site_registry(bundle.site_runtime.registry),
        prior_state=serialize_prior_runtime_state(bundle.prior_state),
    )


def build_prior_runtime_bundle(
    spec: SSMSpec,
    priors: PriorRegistry | None = None,
) -> PriorRuntimeBundle:
    """Build reusable runtime components directly from ``spec`` and ``priors``."""
    site_runtime = build_site_runtime_bundle(spec)
    prior_state = build_prior_runtime_state(site_runtime.registry, priors)
    return PriorRuntimeBundle(
        site_runtime=site_runtime,
        prior_state=prior_state,
    )


def load_prior_runtime_bundle(
    compiled_prior_semantics: CompiledPriorSemantics,
) -> PriorRuntimeBundle:
    """Restore reusable runtime components from ``compiled_prior_semantics``."""
    registry = deserialize_site_registry(compiled_prior_semantics.site_registry)
    prior_state = deserialize_prior_runtime_state(compiled_prior_semantics.prior_state, registry)
    site_runtime = _build_site_runtime_bundle_from_registry(registry)
    return PriorRuntimeBundle(
        site_runtime=site_runtime,
        prior_state=prior_state,
    )


# ---------------------------------------------------------------------------
# Runtime prior distributions from canonical prior semantics
# ---------------------------------------------------------------------------


def build_site_prior_distribution(
    site: SiteDescriptor,
    params: dict[str, jnp.ndarray],
) -> dist.Distribution:
    """Build a NumPyro distribution directly from canonical prior-state params."""
    import numpy as np

    family_values = np.asarray(params["family"], dtype=int).ravel()
    family = int(family_values[0]) if family_values.size else 0
    if family_values.size and not np.all(family_values == family):
        decode = (
            get_real_runtime_kind_from_index
            if site.support in {SupportClass.REAL, SupportClass.CORRELATION}
            else get_positive_runtime_kind_from_index
        )
        names = sorted({decode(int(v)).value for v in set(family_values.tolist())})
        raise ValueError(
            f"Mixed prior families within site {site.name!r} are unsupported "
            f"(found: {', '.join(names)}). The site pools this parameter across "
            "ALL admitted constructs — author the family already in use."
        )

    if site.support in {SupportClass.REAL, SupportClass.CORRELATION}:
        runtime_kind = get_real_runtime_kind_from_index(family)
        if runtime_kind == PriorDistributionFamily.NORMAL:
            return dist.Normal(loc=params["loc"], scale=params["scale"])
        if runtime_kind == PriorDistributionFamily.TRUNCATED_NORMAL:
            return dist.TruncatedNormal(
                loc=params["loc"],
                scale=params["scale"],
                low=params.get("low"),
                high=params.get("high"),
            )
        if runtime_kind == PriorDistributionFamily.UNIFORM:
            low = params.get("low")
            high = params.get("high")
            if low is None or high is None:
                raise ValueError(f"Uniform prior site {site.name!r} is missing low/high bounds")
            return dist.Uniform(
                low=low,
                high=high,
            )
        raise ValueError(
            f"Unsupported canonical real prior runtime kind {runtime_kind!r} for site {site.name!r}"
        )

    if site.support == SupportClass.POSITIVE:
        runtime_kind = get_positive_runtime_kind_from_index(family)
        if runtime_kind == PriorDistributionFamily.HALF_NORMAL:
            return dist.HalfNormal(scale=params["scale"])
        if runtime_kind == PriorDistributionFamily.GAMMA:
            return dist.Gamma(
                concentration=params["concentration"],
                rate=params["rate"],
            )
        if runtime_kind == PriorDistributionFamily.LOG_NORMAL:
            return dist.LogNormal(
                loc=params["loc"],
                scale=params["scale"],
            )
        if runtime_kind == PriorDistributionFamily.EXPONENTIAL:
            return dist.Exponential(rate=params["rate"])
        if runtime_kind == PriorDistributionFamily.DELTA:
            return dist.Delta(params["value"])
        raise ValueError(
            f"Unsupported canonical positive prior runtime kind {runtime_kind!r} "
            f"for site {site.name!r}"
        )

    raise ValueError(f"Unsupported support class {site.support!r} for site {site.name!r}")
