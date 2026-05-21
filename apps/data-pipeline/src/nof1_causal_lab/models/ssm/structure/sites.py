"""Canonical sample-site descriptor and support enums.

Lives in ``structure`` so block specs can return them from
``iter_sites()`` without circular imports with ``parameterization.py``.

The descriptor is the canonical identity of a sample site: name, shape,
support class, semantic kind, assembly grouping, and the various prior
binding keys used by compile-time and runtime layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum


class SupportClass(Enum):
    """Runtime support class for a sample site."""

    REAL = "real"
    POSITIVE = "positive"
    CORRELATION = "correlation"


class TransformKind(Enum):
    """Unconstrained -> constrained transform metadata."""

    IDENTITY = "identity"
    EXP = "exp"
    CORRELATION = "correlation"


class SiteKind(Enum):
    """Semantic role for each sample site."""

    DYNAMICS_DECAY = "dynamics_decay"
    DYNAMICS_CINT = "dynamics_cint"
    DYNAMICS_WEIGHT = "dynamics_weight"
    HILL_EMAX = "hill_emax"
    HILL_EC50 = "hill_ec50"
    HILL_N = "hill_n"
    DIFFUSION_DIAG = "diffusion_diag"
    DIFFUSION_LOWER = "diffusion_lower"
    INPUT_EFFECT = "input_effect"
    STATIC_STATE_SD = "static_state_sd"
    LOADING = "loading"
    MANIFEST_MEANS = "manifest_means"
    MANIFEST_VAR_DIAG = "manifest_var_diag"
    T0_MEANS = "t0_means"
    T0_VAR_DIAG = "t0_var_diag"
    T0_VAR_LOWER = "t0_var_lower"
    OBS_DF = "obs_df"
    OBS_SHAPE = "obs_shape"
    OBS_R = "obs_r"
    OBS_CONCENTRATION = "obs_concentration"
    OBS_ORDERED_BASE = "obs_ordered_base"
    OBS_ORDERED_GAPS = "obs_ordered_gaps"
    OBS_CAT_INTERCEPTS = "obs_cat_intercepts"
    OBS_CAT_SLOPES = "obs_cat_slopes"
    PROC_DF = "proc_df"


class PriorAuthoringTransform(StrEnum):
    """How an authored semantic prior is transformed before site attachment."""

    IDENTITY = "identity"
    POSITIVE_IDENTITY = "positive_identity"
    DT_PERSISTENCE_TO_CT_DECAY = "dt_persistence_to_ct_decay"
    DT_EFFECT_TO_CT_RATE = "dt_effect_to_ct_rate"
    INITIAL_STATE_CORRELATION = "initial_state_correlation"
    SITE_WIDE = "site_wide"


@dataclass(frozen=True)
class SiteDescriptor:
    """Metadata for a single sample site.

    ``positions`` is the free-entry index list owned by the originating
    block: ``list[int]`` for vector-shaped sites, ``list[tuple[int, int]]``
    for matrix-shaped sites. Compile-time consumers use it to translate
    flat-vector parameter indices back to structural ``(i, j)`` positions.
    """

    name: str
    shape: tuple[int, ...]
    support: SupportClass
    assembly_group: str
    site_kind: SiteKind
    transform_kind: TransformKind
    positions: tuple = ()
    deterministic_name: str | None = None
    fixed_spec_field: str | None = None
    priors_field: str | None = None
    runtime_prior_key: str | None = None
    is_runtime_prior_controlled: bool = True


@dataclass(frozen=True)
class SemanticBinding:
    """One semantic model parameter bound to one runtime sample-site scalar."""

    parameter_name: str
    site_name: str
    flat_index: int
    site_kind: SiteKind
    transform: PriorAuthoringTransform = PriorAuthoringTransform.IDENTITY
    prior_field: str | None = None
    construct_names: tuple[str, ...] = ()
    indicator_names: tuple[str, ...] = ()
    component_index: int | None = None
    effect_idx: int | None = None
    cause_idx: int | None = None


def transform_kind_for(support: SupportClass) -> TransformKind:
    if support == SupportClass.REAL:
        return TransformKind.IDENTITY
    if support == SupportClass.POSITIVE:
        return TransformKind.EXP
    return TransformKind.CORRELATION


def make_site(
    name: str,
    shape: tuple[int, ...],
    support: SupportClass,
    assembly_group: str,
    site_kind: SiteKind,
    *,
    positions: tuple = (),
    deterministic_name: str | None = None,
    fixed_spec_field: str | None = None,
    priors_field: str | None = None,
    runtime_prior_key: str | None = None,
) -> SiteDescriptor:
    """Construct a SiteDescriptor with transform_kind derived from support."""
    return SiteDescriptor(
        name=name,
        shape=shape,
        support=support,
        assembly_group=assembly_group,
        site_kind=site_kind,
        transform_kind=transform_kind_for(support),
        positions=positions,
        deterministic_name=deterministic_name,
        fixed_spec_field=fixed_spec_field,
        priors_field=priors_field,
        runtime_prior_key=runtime_prior_key or name,
        is_runtime_prior_controlled=True,
    )


def site_size(shape: tuple[int, ...]) -> int:
    """Number of scalar elements in a site shape."""
    size = 1
    for d in shape:
        size *= d
    return size
