"""Canonical prior registry for SSM sample sites."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist

if TYPE_CHECKING:
    from collections.abc import Mapping

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
        raise ValueError("Mixed prior families within one sample site are unsupported")
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


SITE_NAME_FOR_PRIOR_FIELD: dict[str, str] = {
    "drift_base_decay": "drift_base_decay_free",
    "drift_offdiag": "drift_offdiag_free",
    "diffusion_diag": "diffusion_diag_free",
    "diffusion_offdiag": "diffusion_lower_free",
    "input_effect": "input_effect_free",
    "cint": "cint_free",
    "static_state_sd": "static_state_sd_free",
    "lambda_free": "lambda_free",
    "manifest_means": "manifest_means_free",
    "manifest_var_diag": "manifest_var_diag_free",
    "t0_means": "t0_means_free",
    "t0_var_diag": "t0_var_diag_free",
    "t0_var_offdiag": "t0_var_lower_free",
    "obs_df": "obs_df",
    "obs_shape": "obs_shape",
    "obs_r": "obs_r",
    "obs_concentration": "obs_concentration",
    "obs_ordered_base": "obs_ordered_base",
    "obs_ordered_gaps": "obs_ordered_gaps",
    "obs_cat_intercepts": "obs_cat_intercepts",
    "obs_cat_slopes": "obs_cat_slopes",
    "proc_df": "proc_df",
}

DEFAULT_PRIOR_SPECS_BY_SITE: dict[str, PriorSpec] = {
    "drift_base_decay_free": PriorSpec(
        PriorDistributionFamily.GAMMA,
        {"concentration": 2.0, "rate": 4.0},
    ),
    "drift_offdiag_free": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 0.5},
    ),
    "diffusion_diag_free": PriorSpec(
        PriorDistributionFamily.HALF_NORMAL,
        {"sigma": 1.0},
    ),
    "diffusion_lower_free": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 0.5},
    ),
    "cint_free": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 1.0},
    ),
    "input_effect_free": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 0.5},
    ),
    "static_state_sd_free": PriorSpec(
        PriorDistributionFamily.HALF_NORMAL,
        {"sigma": 1.0},
    ),
    "lambda_free": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.5, "sigma": 0.5},
    ),
    "manifest_means_free": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 2.0},
    ),
    "manifest_var_diag_free": PriorSpec(
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
    "t0_means_free": PriorSpec(
        PriorDistributionFamily.NORMAL,
        {"mu": 0.0, "sigma": 2.0},
    ),
    "t0_var_diag_free": PriorSpec(
        PriorDistributionFamily.HALF_NORMAL,
        {"sigma": 2.0},
    ),
    "t0_var_lower_free": PriorSpec(
        PriorDistributionFamily.TRUNCATED_NORMAL,
        dict(INITIAL_STATE_CORRELATION_PRIOR_DEFAULTS),
    ),
}


def default_prior_registry() -> PriorRegistry:
    """Return the compiler-owned default prior registry."""
    return PriorRegistry(dict(DEFAULT_PRIOR_SPECS_BY_SITE))


def default_prior_for_site(site_name: str) -> PriorSpec:
    """Return the compiler-owned default prior for a sample site."""
    try:
        return DEFAULT_PRIOR_SPECS_BY_SITE[site_name]
    except KeyError as exc:
        raise KeyError(f"No default prior registered for sample site {site_name!r}") from exc


def _broadcast_prior_param(value: Any, shape: tuple[int, ...]):
    if not shape:
        return jnp.asarray(value)
    return jnp.broadcast_to(jnp.asarray(value), shape)


def materialize_prior_distribution(prior_cfg: Mapping[str, Any]) -> ndist.Distribution:
    """Materialize a NumPyro distribution from canonical prior config."""
    if "params" not in prior_cfg:
        raise ValueError("Prior config must use the canonical {'family', 'params'} shape.")
    family = PriorDistributionFamily(prior_cfg.get("family", PriorDistributionFamily.NORMAL))
    params = prior_cfg.get("params", {})
    shape = tuple(prior_cfg.get("shape", ()))

    def _bcast(value: Any):
        return _broadcast_prior_param(value, shape)

    if family is PriorDistributionFamily.NORMAL:
        return ndist.Normal(_bcast(params.get("mu", 0.0)), _bcast(params.get("sigma", 1.0)))
    if family is PriorDistributionFamily.HALF_NORMAL:
        return ndist.HalfNormal(_bcast(params.get("sigma", 1.0)))
    if family is PriorDistributionFamily.TRUNCATED_NORMAL:
        return ndist.TruncatedNormal(
            loc=_bcast(params.get("mu", 0.0)),
            scale=_bcast(params.get("sigma", 1.0)),
            low=params.get("lower", -float("inf")),
            high=params.get("upper", float("inf")),
        )
    if family is PriorDistributionFamily.LOG_NORMAL:
        return ndist.LogNormal(_bcast(params.get("mu", 0.0)), _bcast(params.get("sigma", 1.0)))
    if family is PriorDistributionFamily.GAMMA:
        return ndist.Gamma(
            _bcast(params.get("concentration", 2.0)),
            _bcast(params.get("rate", 1.0)),
        )
    if family is PriorDistributionFamily.EXPONENTIAL:
        return ndist.Exponential(_bcast(params.get("rate", 1.0)))
    if family is PriorDistributionFamily.BETA:
        return ndist.Beta(_bcast(params.get("alpha", 2.0)), _bcast(params.get("beta", 2.0)))
    if family is PriorDistributionFamily.UNIFORM:
        return ndist.Uniform(_bcast(params.get("lower", 0.0)), _bcast(params.get("upper", 1.0)))
    if family is PriorDistributionFamily.DELTA:
        return ndist.Delta(_bcast(params.get("value", 0.0)))
    raise ValueError(f"Unsupported prior family for SSM: {family.value!r}")


def resolve_prior_distribution(prior: Any) -> ndist.Distribution | None:
    """Resolve a runtime prior from a distribution object or canonical config."""
    if prior is None:
        return None
    if isinstance(prior, dict):
        return materialize_prior_distribution(prior)
    return prior


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
