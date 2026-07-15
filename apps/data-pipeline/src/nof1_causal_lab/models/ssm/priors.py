"""Canonical prior registry for SSM sample sites."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor

from nof1_causal_lab.distributions import (
    PriorDistributionFamily,
    get_positive_runtime_kind_from_index,
    get_real_runtime_kind_from_index,
)
from nof1_causal_lab.models.ssm.parameter_names import INITIAL_STATE_CORRELATION_PRIOR_DEFAULTS


def _scalar_family_index(value: Any) -> int:
    values = np.asarray(value, dtype=int).ravel()
    if values.size == 0:
        raise ValueError("Prior family index payload is empty")
    family = int(values[0])
    if not np.all(values == family):
        unique, counts = np.unique(values, return_counts=True)
        breakdown = {int(k): int(v) for k, v in zip(unique, counts, strict=True)}
        raise ValueError(
            "Mixed prior families within one sample site are unsupported "
            f"(family-index counts: {breakdown}). The site pools this parameter "
            "across ALL admitted constructs, so a newly authored prior must use "
            "the same distribution family the site's existing entries already "
            "use — match the family authored by earlier constructs."
        )
    return family


@dataclass(frozen=True)
class PriorSpec:
    """Prior family and parameters for one runtime sample site."""

    family: PriorDistributionFamily
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", PriorDistributionFamily(self.family))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True)
class PriorRegistry:
    """Mapping from runtime sample-site name to canonical prior spec."""

    priors_by_site: Mapping[str, PriorSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "priors_by_site", MappingProxyType(dict(self.priors_by_site)))

    def get(self, site_name: str) -> PriorSpec | None:
        """Return the prior for *site_name* when one is registered."""
        return self.priors_by_site.get(site_name)


DEFAULT_PRIOR_SPECS_BY_FIELD: dict[str, PriorSpec] = {
    "dynamics_decay": PriorSpec(
        PriorDistributionFamily.GAMMA,
        {"concentration": 2.0, "rate": 4.0},
    ),
    "dynamics_cint": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 1.0},
    ),
    "dynamics_potential_center": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 1.0},
    ),
    "dynamics_potential_quartic": PriorSpec(
        PriorDistributionFamily.HALF_NORMAL,
        {"sigma": 0.5},
    ),
    "linear_edge_weight": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 0.5},
    ),
    "multiplicative_weight": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 1.0},
    ),
    "hill_emax": PriorSpec(
        PriorDistributionFamily.LOG_NORMAL,
        {"mu": 0.0, "sigma": 1.0},
    ),
    "hill_ec50": PriorSpec(
        PriorDistributionFamily.LOG_NORMAL,
        {"mu": 0.0, "sigma": 1.0},
    ),
    "hill_n": PriorSpec(
        PriorDistributionFamily.TRUNCATED_NORMAL,
        {"mu": 2.0, "sigma": 0.5, "lower": 1.0, "upper": 4.0},
    ),
    "diffusion_diag": PriorSpec(
        PriorDistributionFamily.HALF_NORMAL,
        {"sigma": 1.0},
    ),
    "diffusion_offdiag": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 0.5},
    ),
    "input_effect": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 0.5},
    ),
    "static_state_sd": PriorSpec(
        PriorDistributionFamily.HALF_NORMAL,
        {"sigma": 1.0},
    ),
    "lambda_free": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.5, "sigma": 0.5},
    ),
    "manifest_means": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 2.0},
    ),
    "manifest_var_diag": PriorSpec(
        PriorDistributionFamily.HALF_NORMAL,
        {"sigma": 1.0},
    ),
    "obs_df": PriorSpec(
        PriorDistributionFamily.GAMMA,
        {"concentration": 5.0, "rate": 1.0},
    ),
    "obs_shape": PriorSpec(
        PriorDistributionFamily.GAMMA,
        {"concentration": 2.0, "rate": 1.0},
    ),
    "obs_r": PriorSpec(
        PriorDistributionFamily.GAMMA,
        {"concentration": 2.0, "rate": 0.5},
    ),
    "obs_concentration": PriorSpec(
        PriorDistributionFamily.GAMMA,
        {"concentration": 5.0, "rate": 0.5},
    ),
    "obs_ordered_base": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 1.0},
    ),
    "obs_ordered_gaps": PriorSpec(
        PriorDistributionFamily.HALF_NORMAL,
        {"sigma": 1.0},
    ),
    "obs_cat_intercepts": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 1.0},
    ),
    "obs_cat_slopes": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 1.0},
    ),
    "proc_df": PriorSpec(
        PriorDistributionFamily.GAMMA,
        {"concentration": 5.0, "rate": 1.0},
    ),
    "t0_means": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 2.0},
    ),
    "t0_var_diag": PriorSpec(
        PriorDistributionFamily.HALF_NORMAL,
        {"sigma": 2.0},
    ),
    "t0_var_offdiag": PriorSpec(
        PriorDistributionFamily.TRUNCATED_NORMAL,
        dict(INITIAL_STATE_CORRELATION_PRIOR_DEFAULTS),
    ),
}


def default_prior_for_descriptor(site: SiteDescriptor) -> PriorSpec:
    """Return the default prior for a descriptor-owned sample site."""
    if site.priors_field is None:
        raise KeyError(
            f"Sample site {site.name!r} is missing priors_field; cannot resolve a default prior."
        )
    try:
        return DEFAULT_PRIOR_SPECS_BY_FIELD[site.priors_field]
    except KeyError as exc:
        raise KeyError(
            f"No default prior registered for prior field {site.priors_field!r} "
            f"on sample site {site.name!r}"
        ) from exc


def default_prior_registry_for_sites(
    sites: list[SiteDescriptor] | tuple[SiteDescriptor, ...],
) -> PriorRegistry:
    """Build a default prior registry keyed by descriptor name for the given active sites."""
    return PriorRegistry({site.name: default_prior_for_descriptor(site) for site in sites})


def prior_spec_from_normalized_params(
    normalized: Mapping[str, Any],
    *,
    support: str,
) -> PriorSpec:
    """Convert normalized compiler params into a canonical prior.

    ``support`` is one of ``"real"``, ``"positive"``, or ``"correlation"``.
    The normalized shape is an internal compiler format; this function is
    the boundary that removes runtime family indexes from canonical prior
    objects.
    """
    if support == "positive":
        family = (
            get_positive_runtime_kind_from_index(_scalar_family_index(normalized["family"]))
            if "family" in normalized
            else PriorDistributionFamily.HALF_NORMAL
        )
        params: dict[str, Any] = {}
        if family == PriorDistributionFamily.HALF_NORMAL:
            params["sigma"] = normalized.get("sigma", 1.0)
        elif family == PriorDistributionFamily.GAMMA:
            params["concentration"] = normalized.get("concentration", 2.0)
            params["rate"] = normalized.get("rate", 1.0)
        elif family == PriorDistributionFamily.LOG_NORMAL:
            params["mu"] = normalized.get("mu", normalized.get("loc", 0.0))
            params["sigma"] = normalized.get("sigma", 1.0)
        elif family == PriorDistributionFamily.EXPONENTIAL:
            params["rate"] = normalized.get("rate", 1.0)
        elif family == PriorDistributionFamily.DELTA:
            params["value"] = normalized.get("value", 1.0)
        else:
            raise ValueError(f"Unsupported positive-support prior family {family!r}")
        return PriorSpec(family, params)

    if support in {"real", "correlation"}:
        has_bounds = "lower" in normalized or "upper" in normalized
        family = (
            get_real_runtime_kind_from_index(_scalar_family_index(normalized["family"]))
            if "family" in normalized
            else (
                PriorDistributionFamily.TRUNCATED_NORMAL
                if has_bounds or support == "correlation"
                else PriorDistributionFamily.NORMAL
            )
        )
        params = {}
        if family == PriorDistributionFamily.NORMAL:
            params["mu"] = normalized.get("mu", 0.0)
            params["sigma"] = normalized.get("sigma", 1.0)
        elif family == PriorDistributionFamily.TRUNCATED_NORMAL:
            params["mu"] = normalized.get("mu", 0.0)
            params["sigma"] = normalized.get("sigma", 1.0)
            params["lower"] = normalized.get("lower", -1.0 if support == "correlation" else -1e6)
            params["upper"] = normalized.get("upper", 1.0 if support == "correlation" else 1e6)
        elif family == PriorDistributionFamily.UNIFORM:
            params["lower"] = normalized.get("lower", -1.0 if support == "correlation" else -1e6)
            params["upper"] = normalized.get("upper", 1.0 if support == "correlation" else 1e6)
        else:
            raise ValueError(f"Unsupported real-support prior family {family!r}")
        return PriorSpec(family, params)

    raise ValueError(f"Unsupported prior support {support!r}")


def prior_spec_to_normalized_params(
    prior: PriorSpec,
) -> dict[str, Any]:
    """Convert a canonical prior into compiler-normalized parameter names."""
    params = dict(prior.params)
    if prior.family == PriorDistributionFamily.LOG_NORMAL and "mu" in params:
        params["loc"] = params.pop("mu")
    return params
