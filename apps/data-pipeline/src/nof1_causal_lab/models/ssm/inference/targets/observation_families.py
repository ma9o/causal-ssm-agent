"""ObservationFamilySpec registry — single source of truth for per-family dispatch.

Replaces ~11 if/elif chains across emissions.py, kernels.py, and ssm_builder.py
with a flat dict keyed by DistributionFamily.

Each entry fully describes one observation family's behavior at every dispatch site:
- Emission log-prob factories (emissions.py)
- Score/weight factories for IEKS (emissions.py)
- Variance and response function factories (kernels.py)
- Grad/hess strategy tag (kernels.py)
- Support validation (ssm_builder.py)
- Discrete level hydration (ssm_builder.py)
- Posterior predictive sampling branches (posterior_predictive.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core

import causal_ssm_agent.models.ssm.inference.targets.emissions as emission_math
from causal_ssm_agent.artifacts.model_spec import (
    VALID_LINKS_FOR_DISTRIBUTION,
    DistributionFamily,
    LinkFunction,
)
from causal_ssm_agent.models.ssm.inference.targets.base import NUMERICAL_EPSILON

from .emissions import (
    categorical_probabilities,
    ordered_logistic_probabilities,
)
from .observation_kernel_helpers import (
    _make_discrete_response_categorical,
    _make_discrete_response_ordered_logistic,
    _make_discrete_variance_from_moments,
    _make_variance_bernoulli,
    _make_variance_beta,
    _make_variance_gamma,
    _make_variance_identity,
    _make_variance_negative_binomial,
    _make_variance_poisson,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class ObservationFamilySpec:
    """Registry entry for observation family behavior across all dispatch sites."""

    default_link: LinkFunction
    """Default link used when callers omit an explicit link."""

    # --- ssm_builder.py concerns ---
    validate_support: Callable[[np.ndarray], np.ndarray]
    """values -> invalid_mask (bool array). Empty mask means no support constraint."""
    support_description: str
    """Human-readable constraint for error messages."""
    hydrate_levels: Callable[[np.ndarray], int | None]
    """column_values -> level_count or None if not a discrete family."""
    needs_level_metadata: bool
    """Whether this family requires hydrated manifest_level_counts."""

    # --- emissions.py concerns ---
    emission_fns: dict[str, Callable]
    """link_str -> factory(extra_params) -> emission_log_prob_fn."""
    score_weight_fns: dict[str, Callable]
    """link_str -> factory(extra_params) -> score_weight_fn | None."""

    # --- kernels.py concerns ---
    make_variance_fn: Callable
    """(extra_params, manifest_cov) -> variance_fn."""
    grad_hess_strategy: str
    """One of 'gaussian', 'student_t', or 'glm'."""
    make_response_fn: Callable | None
    """(extra_params) -> response_fn, or None to use _RESPONSE_FNS[link]."""
    posterior_predictive_fns: dict[str, Callable]
    """link_str -> posterior predictive branch used by lax.switch."""


# ---------------------------------------------------------------------------
# Support validators (ssm_builder.py)
# ---------------------------------------------------------------------------


def _positive_param(params: dict, key: str, default: float) -> float:
    """Extract a parameter that must be strictly positive, raising early on violation."""
    val = params.get(key, default)
    # Likelihood kernels are built inside JAX-transformed code paths during
    # Laplace/SMC evaluation, where positive sites arrive as tracers rather
    # than concrete Python scalars. Upstream parameterization already enforces
    # positivity for sampled sites, so only perform eager validation when the
    # value is concrete.
    if isinstance(val, jax_core.Tracer):
        return val
    if np.any(np.asarray(val) <= 0):
        raise ValueError(f"{key} must be positive, got {val}")
    return val


def _no_constraint(values: np.ndarray) -> np.ndarray:
    return np.zeros(values.shape, dtype=bool)


def _positive_only(values: np.ndarray) -> np.ndarray:
    return values <= 0.0


def _unit_interval(values: np.ndarray) -> np.ndarray:
    return (values <= 0.0) | (values >= 1.0)


def _nonneg_integer(values: np.ndarray) -> np.ndarray:
    rounded = np.rint(values)
    return (values < 0.0) | (~np.isclose(values, rounded, atol=1e-6))


def _binary(values: np.ndarray) -> np.ndarray:
    return ~np.isin(values, [0.0, 1.0])


def _no_levels(_values: np.ndarray) -> int | None:
    return None


def _infer_contiguous_levels(values: np.ndarray) -> int | None:
    if values.size == 0:
        raise ValueError("discrete emission has no observed data")

    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-6):
        raise ValueError("data are not integer-encoded")

    unique_levels = sorted({int(v) for v in rounded.tolist()})
    if unique_levels[0] != 0 or unique_levels != list(range(unique_levels[-1] + 1)):
        raise ValueError(f"encoded levels are not contiguous from 0: {unique_levels}")
    if len(unique_levels) < 2:
        raise ValueError(f"only {len(unique_levels)} level(s) are present")
    return len(unique_levels)


def _coerce_distribution_family(
    dist: DistributionFamily | str,
) -> DistributionFamily:
    if isinstance(dist, DistributionFamily):
        return dist
    try:
        return DistributionFamily(dist)
    except ValueError as exc:
        raise ValueError(f"Unknown distribution family: {dist!r}") from exc


def _link_key(link: LinkFunction | str) -> str:
    return link.value if isinstance(link, LinkFunction) else str(link)


def _ordered_links(
    dist: DistributionFamily,
    default_link: LinkFunction,
) -> tuple[LinkFunction, ...]:
    remaining = sorted(
        (link for link in VALID_LINKS_FOR_DISTRIBUTION[dist] if link != default_link),
        key=lambda link: link.value,
    )
    return (default_link, *remaining)


def _build_link_dispatch_map(
    dist: DistributionFamily,
    default_link: LinkFunction,
    default_fn: Callable,
    *,
    overrides: dict[LinkFunction | str, Callable] | None = None,
    include_default_key: bool,
) -> dict[str, Callable]:
    mapping: dict[str, Callable] = {}
    if include_default_key:
        mapping["default"] = default_fn
    for link in _ordered_links(dist, default_link):
        mapping[link.value] = default_fn
    for link, fn in (overrides or {}).items():
        mapping[_link_key(link)] = fn
    return mapping


# ---------------------------------------------------------------------------
# Emission-fn factories  (link_str -> factory(extra_params) -> callable)
# ---------------------------------------------------------------------------


def _emission_factory_gaussian(_params: dict):
    return emission_math.emission_log_prob_gaussian


def _emission_factory_poisson(_params: dict):
    return emission_math.emission_log_prob_poisson


def _emission_factory_student_t(params: dict):
    df = _positive_param(params, "obs_df", 5.0)
    return lambda y, z, H, d, R, m: emission_math.emission_log_prob_student_t(y, z, H, d, R, m, df)


def _emission_factory_gamma_log(params: dict):
    shape = _positive_param(params, "obs_shape", 1.0)
    return lambda y, z, H, d, R, m: emission_math.emission_log_prob_gamma(y, z, H, d, R, m, shape)


def _emission_factory_gamma_inverse(params: dict):
    shape = _positive_param(params, "obs_shape", 1.0)
    return lambda y, z, H, d, R, m: emission_math.emission_log_prob_gamma_inverse(
        y, z, H, d, R, m, shape
    )


def _emission_factory_bernoulli_logit(_params: dict):
    return emission_math.emission_log_prob_bernoulli


def _emission_factory_bernoulli_probit(_params: dict):
    return emission_math.emission_log_prob_bernoulli_probit


def _emission_factory_negbin(params: dict):
    r = _positive_param(params, "obs_r", 5.0)
    return lambda y, z, H, d, R, m: emission_math.emission_log_prob_negative_binomial(
        y, z, H, d, R, m, r
    )


def _emission_factory_beta_logit(params: dict):
    conc = _positive_param(params, "obs_concentration", 10.0)
    return lambda y, z, H, d, R, m: emission_math.emission_log_prob_beta(y, z, H, d, R, m, conc)


def _emission_factory_beta_probit(params: dict):
    conc = _positive_param(params, "obs_concentration", 10.0)
    return lambda y, z, H, d, R, m: emission_math.emission_log_prob_beta_probit(
        y, z, H, d, R, m, conc
    )


def _emission_factory_ordered_logistic(params: dict):
    level_counts, cutpoints = emission_math.get_ordered_logistic_extra_params(params)
    return lambda y, z, H, d, R, m: emission_math.emission_log_prob_ordered_logistic(
        y, z, H, d, R, m, cutpoints, level_counts
    )


def _emission_factory_categorical(params: dict):
    level_counts, intercepts, slopes = emission_math.get_categorical_extra_params(params)
    return lambda y, z, H, d, R, m: emission_math.emission_log_prob_categorical(
        y, z, H, d, R, m, intercepts, slopes, level_counts
    )


# ---------------------------------------------------------------------------
# Score/weight factories  (link_str -> factory(extra_params) -> callable|None)
# ---------------------------------------------------------------------------


def _sw_factory_none(_params: dict):
    return None


def _sw_factory_poisson(_params: dict):
    return emission_math._score_weight_poisson


def _sw_factory_bernoulli_logit(_params: dict):
    return emission_math._score_weight_bernoulli_logit


def _sw_factory_bernoulli_probit(_params: dict):
    return emission_math._score_weight_bernoulli_probit


def _sw_factory_beta_logit(params: dict):
    conc = _positive_param(params, "obs_concentration", 10.0)
    return lambda y, eta, m: emission_math._score_weight_beta_logit(y, eta, m, conc)


def _sw_factory_beta_probit(params: dict):
    conc = _positive_param(params, "obs_concentration", 10.0)
    return lambda y, eta, m: emission_math._score_weight_beta_probit(y, eta, m, conc)


def _sw_factory_gamma_log(params: dict):
    shape = _positive_param(params, "obs_shape", 1.0)
    return lambda y, eta, m: emission_math._score_weight_gamma_log(y, eta, m, shape)


def _sw_factory_gamma_inverse(params: dict):
    shape = _positive_param(params, "obs_shape", 1.0)
    return lambda y, eta, m: emission_math._score_weight_gamma_inverse(y, eta, m, shape)


def _sw_factory_negbin(params: dict):
    r = _positive_param(params, "obs_r", 5.0)
    return lambda y, eta, m: emission_math._score_weight_negative_binomial(y, eta, m, r)


def _sw_factory_ordered_logistic(params: dict):
    level_counts, cutpoints = emission_math.get_ordered_logistic_extra_params(params)
    return lambda y, eta, m: emission_math._score_weight_ordered_logistic(
        y, eta, m, cutpoints, level_counts
    )


def _sw_factory_categorical(params: dict):
    level_counts, intercepts, slopes = emission_math.get_categorical_extra_params(params)
    return lambda y, eta, m: emission_math._score_weight_categorical(
        y, eta, m, intercepts, slopes, level_counts
    )


# ---------------------------------------------------------------------------
# Variance-fn factories  (extra_params, manifest_cov) -> variance_fn
# ---------------------------------------------------------------------------


def _variance_factory_gaussian_like(_params: dict, manifest_cov):
    if manifest_cov is not None:
        return _make_variance_identity(manifest_cov)

    def _lazy_error(_mean):
        raise RuntimeError(
            "variance_fn for gaussian/student_t requires manifest_cov; "
            "pass it to build_observation_kernel()"
        )

    return _lazy_error


def _variance_factory_poisson(_params: dict, _manifest_cov):
    return _make_variance_poisson()


def _variance_factory_negbin(params: dict, _manifest_cov):
    r = _positive_param(params, "obs_r", 5.0)
    return _make_variance_negative_binomial(r)


def _variance_factory_gamma(params: dict, _manifest_cov):
    shape = _positive_param(params, "obs_shape", 1.0)
    return _make_variance_gamma(shape)


def _variance_factory_bernoulli(_params: dict, _manifest_cov):
    return _make_variance_bernoulli()


def _variance_factory_beta(params: dict, _manifest_cov):
    conc = _positive_param(params, "obs_concentration", 10.0)
    return _make_variance_beta(conc)


def _variance_factory_ordered_logistic(params: dict, _manifest_cov):
    level_counts, cutpoints = emission_math.get_ordered_logistic_extra_params(params)
    return _make_discrete_variance_from_moments(
        lambda eta: emission_math.ordered_logistic_moments(eta, cutpoints, level_counts)
    )


def _variance_factory_categorical(params: dict, _manifest_cov):
    level_counts, intercepts, slopes = emission_math.get_categorical_extra_params(params)
    return _make_discrete_variance_from_moments(
        lambda eta: emission_math.categorical_moments(eta, intercepts, slopes, level_counts)
    )


# ---------------------------------------------------------------------------
# Response-fn factories  (extra_params) -> response_fn  (or None → use link)
# ---------------------------------------------------------------------------


def _response_factory_ordered_logistic(params: dict):
    level_counts, cutpoints = emission_math.get_ordered_logistic_extra_params(params)
    return _make_discrete_response_ordered_logistic(cutpoints, level_counts)


def _response_factory_categorical(params: dict):
    level_counts, intercepts, slopes = emission_math.get_categorical_extra_params(params)
    return _make_discrete_response_categorical(intercepts, slopes, level_counts)


def _sample_discrete_from_probs(key: jax.Array, probs: jnp.ndarray) -> jnp.ndarray:
    return jax.random.categorical(
        key,
        jnp.log(jnp.maximum(probs, NUMERICAL_EPSILON)),
        axis=-1,
    ).astype(jnp.float64)


def _ppc_gaussian(
    loc, key, std, _df, _shape, _r, _phi, _level_count, _cutpoints, _cat_intercepts, _cat_slopes
):
    return loc + std * jax.random.normal(key, ())


def _ppc_student_t(
    loc, key, std, df, _shape, _r, _phi, _level_count, _cutpoints, _cat_intercepts, _cat_slopes
):
    key_num, key_den = jax.random.split(key)
    z = jax.random.normal(key_num, ())
    chi2 = 2.0 * jax.random.gamma(key_den, df / 2.0)
    t_val = z * jnp.sqrt(df / jnp.maximum(chi2, NUMERICAL_EPSILON))
    return loc + std * t_val


def _ppc_poisson(
    loc, key, _std, _df, _shape, _r, _phi, _level_count, _cutpoints, _cat_intercepts, _cat_slopes
):
    rate = jnp.exp(loc)
    return jax.random.poisson(key, rate).astype(jnp.float64)


def _ppc_gamma_log(
    loc, key, _std, _df, shape, _r, _phi, _level_count, _cutpoints, _cat_intercepts, _cat_slopes
):
    mean = jnp.exp(loc)
    scale = jnp.maximum(mean / jnp.maximum(shape, 1e-8), 1e-8)
    return jax.random.gamma(key, shape) * scale


def _ppc_gamma_inverse(
    loc, key, _std, _df, shape, _r, _phi, _level_count, _cutpoints, _cat_intercepts, _cat_slopes
):
    valid_loc = jnp.isfinite(loc) & (loc > 0.0)
    safe_loc = jnp.where(valid_loc, loc, 1.0)
    mean = 1.0 / safe_loc
    scale = jnp.maximum(mean / jnp.maximum(shape, 1e-8), 1e-8)
    draw = jax.random.gamma(key, shape) * scale
    return jnp.where(valid_loc, draw, jnp.nan)


def _ppc_bernoulli_logit(
    loc, key, _std, _df, _shape, _r, _phi, _level_count, _cutpoints, _cat_intercepts, _cat_slopes
):
    return jax.random.bernoulli(key, jax.nn.sigmoid(loc)).astype(jnp.float64)


def _ppc_bernoulli_probit(
    loc, key, _std, _df, _shape, _r, _phi, _level_count, _cutpoints, _cat_intercepts, _cat_slopes
):
    return jax.random.bernoulli(key, jax.scipy.stats.norm.cdf(loc)).astype(jnp.float64)


def _ppc_negative_binomial(
    loc, key, _std, _df, _shape, r, _phi, _level_count, _cutpoints, _cat_intercepts, _cat_slopes
):
    mu = jnp.exp(loc)
    key_gamma, key_poisson = jax.random.split(key)
    gamma_draw = jax.random.gamma(key_gamma, r) * mu / jnp.maximum(r, 1e-8)
    return jax.random.poisson(
        key_poisson,
        jnp.maximum(gamma_draw, NUMERICAL_EPSILON),
    ).astype(jnp.float64)


def _ppc_beta_logit(
    loc, key, _std, _df, _shape, _r, phi, _level_count, _cutpoints, _cat_intercepts, _cat_slopes
):
    mean = jax.nn.sigmoid(loc)
    alpha = jnp.maximum(mean * phi, 1e-4)
    beta_param = jnp.maximum((1.0 - mean) * phi, 1e-4)
    key_alpha, key_beta = jax.random.split(key)
    gamma_alpha = jax.random.gamma(key_alpha, alpha)
    gamma_beta = jax.random.gamma(key_beta, beta_param)
    return gamma_alpha / jnp.maximum(gamma_alpha + gamma_beta, NUMERICAL_EPSILON)


def _ppc_beta_probit(
    loc, key, _std, _df, _shape, _r, phi, _level_count, _cutpoints, _cat_intercepts, _cat_slopes
):
    mean = jax.scipy.stats.norm.cdf(loc)
    alpha = jnp.maximum(mean * phi, 1e-4)
    beta_param = jnp.maximum((1.0 - mean) * phi, 1e-4)
    key_alpha, key_beta = jax.random.split(key)
    gamma_alpha = jax.random.gamma(key_alpha, alpha)
    gamma_beta = jax.random.gamma(key_beta, beta_param)
    return gamma_alpha / jnp.maximum(gamma_alpha + gamma_beta, NUMERICAL_EPSILON)


def _ppc_ordered_logistic(
    loc,
    key,
    _std,
    _df,
    _shape,
    _r,
    _phi,
    level_count,
    cutpoints,
    _cat_intercepts,
    _cat_slopes,
):
    probs = ordered_logistic_probabilities(
        jnp.asarray([loc]),
        cutpoints[None, :],
        jnp.asarray([level_count], dtype=jnp.int64),
    )[0]
    return _sample_discrete_from_probs(key, probs)


def _ppc_categorical(
    loc,
    key,
    _std,
    _df,
    _shape,
    _r,
    _phi,
    level_count,
    _cutpoints,
    cat_intercepts,
    cat_slopes,
):
    probs = categorical_probabilities(
        jnp.asarray([loc]),
        cat_intercepts[None, :],
        cat_slopes[None, :],
        jnp.asarray([level_count], dtype=jnp.int64),
    )[0]
    return _sample_discrete_from_probs(key, probs)


# ===========================================================================
# The registry
# ===========================================================================

FAMILY_REGISTRY: dict[DistributionFamily, ObservationFamilySpec] = {
    # ---- Gaussian ----
    DistributionFamily.GAUSSIAN: ObservationFamilySpec(
        default_link=LinkFunction.IDENTITY,
        validate_support=_no_constraint,
        support_description="",
        hydrate_levels=_no_levels,
        needs_level_metadata=False,
        emission_fns=_build_link_dispatch_map(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            _emission_factory_gaussian,
            include_default_key=True,
        ),
        score_weight_fns=_build_link_dispatch_map(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            _sw_factory_none,
            include_default_key=True,
        ),
        make_variance_fn=_variance_factory_gaussian_like,
        grad_hess_strategy="gaussian",
        make_response_fn=None,
        posterior_predictive_fns=_build_link_dispatch_map(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            _ppc_gaussian,
            include_default_key=False,
        ),
    ),
    # ---- Student-t ----
    DistributionFamily.STUDENT_T: ObservationFamilySpec(
        default_link=LinkFunction.IDENTITY,
        validate_support=_no_constraint,
        support_description="",
        hydrate_levels=_no_levels,
        needs_level_metadata=False,
        emission_fns=_build_link_dispatch_map(
            DistributionFamily.STUDENT_T,
            LinkFunction.IDENTITY,
            _emission_factory_student_t,
            include_default_key=True,
        ),
        score_weight_fns=_build_link_dispatch_map(
            DistributionFamily.STUDENT_T,
            LinkFunction.IDENTITY,
            _sw_factory_none,
            include_default_key=True,
        ),
        make_variance_fn=_variance_factory_gaussian_like,
        grad_hess_strategy="student_t",
        make_response_fn=None,
        posterior_predictive_fns=_build_link_dispatch_map(
            DistributionFamily.STUDENT_T,
            LinkFunction.IDENTITY,
            _ppc_student_t,
            include_default_key=False,
        ),
    ),
    # ---- Poisson ----
    DistributionFamily.POISSON: ObservationFamilySpec(
        default_link=LinkFunction.LOG,
        validate_support=_nonneg_integer,
        support_description="poisson requires non-negative integer counts",
        hydrate_levels=_no_levels,
        needs_level_metadata=False,
        emission_fns=_build_link_dispatch_map(
            DistributionFamily.POISSON,
            LinkFunction.LOG,
            _emission_factory_poisson,
            include_default_key=True,
        ),
        score_weight_fns=_build_link_dispatch_map(
            DistributionFamily.POISSON,
            LinkFunction.LOG,
            _sw_factory_poisson,
            include_default_key=True,
        ),
        make_variance_fn=_variance_factory_poisson,
        grad_hess_strategy="glm",
        make_response_fn=None,
        posterior_predictive_fns=_build_link_dispatch_map(
            DistributionFamily.POISSON,
            LinkFunction.LOG,
            _ppc_poisson,
            include_default_key=False,
        ),
    ),
    # ---- Gamma ----
    DistributionFamily.GAMMA: ObservationFamilySpec(
        default_link=LinkFunction.LOG,
        validate_support=_positive_only,
        support_description="gamma requires y > 0",
        hydrate_levels=_no_levels,
        needs_level_metadata=False,
        emission_fns=_build_link_dispatch_map(
            DistributionFamily.GAMMA,
            LinkFunction.LOG,
            _emission_factory_gamma_log,
            overrides={LinkFunction.INVERSE: _emission_factory_gamma_inverse},
            include_default_key=True,
        ),
        score_weight_fns=_build_link_dispatch_map(
            DistributionFamily.GAMMA,
            LinkFunction.LOG,
            _sw_factory_gamma_log,
            overrides={LinkFunction.INVERSE: _sw_factory_gamma_inverse},
            include_default_key=True,
        ),
        make_variance_fn=_variance_factory_gamma,
        grad_hess_strategy="glm",
        make_response_fn=None,
        posterior_predictive_fns=_build_link_dispatch_map(
            DistributionFamily.GAMMA,
            LinkFunction.LOG,
            _ppc_gamma_log,
            overrides={LinkFunction.INVERSE: _ppc_gamma_inverse},
            include_default_key=False,
        ),
    ),
    # ---- Bernoulli ----
    DistributionFamily.BERNOULLI: ObservationFamilySpec(
        default_link=LinkFunction.LOGIT,
        validate_support=_binary,
        support_description="bernoulli requires binary values in {0, 1}",
        hydrate_levels=_no_levels,
        needs_level_metadata=False,
        emission_fns=_build_link_dispatch_map(
            DistributionFamily.BERNOULLI,
            LinkFunction.LOGIT,
            _emission_factory_bernoulli_logit,
            overrides={LinkFunction.PROBIT: _emission_factory_bernoulli_probit},
            include_default_key=True,
        ),
        score_weight_fns=_build_link_dispatch_map(
            DistributionFamily.BERNOULLI,
            LinkFunction.LOGIT,
            _sw_factory_bernoulli_logit,
            overrides={LinkFunction.PROBIT: _sw_factory_bernoulli_probit},
            include_default_key=True,
        ),
        make_variance_fn=_variance_factory_bernoulli,
        grad_hess_strategy="glm",
        make_response_fn=None,
        posterior_predictive_fns=_build_link_dispatch_map(
            DistributionFamily.BERNOULLI,
            LinkFunction.LOGIT,
            _ppc_bernoulli_logit,
            overrides={LinkFunction.PROBIT: _ppc_bernoulli_probit},
            include_default_key=False,
        ),
    ),
    # ---- Negative Binomial ----
    DistributionFamily.NEGATIVE_BINOMIAL: ObservationFamilySpec(
        default_link=LinkFunction.LOG,
        validate_support=_nonneg_integer,
        support_description="negative_binomial requires non-negative integer counts",
        hydrate_levels=_no_levels,
        needs_level_metadata=False,
        emission_fns=_build_link_dispatch_map(
            DistributionFamily.NEGATIVE_BINOMIAL,
            LinkFunction.LOG,
            _emission_factory_negbin,
            include_default_key=True,
        ),
        score_weight_fns=_build_link_dispatch_map(
            DistributionFamily.NEGATIVE_BINOMIAL,
            LinkFunction.LOG,
            _sw_factory_negbin,
            include_default_key=True,
        ),
        make_variance_fn=_variance_factory_negbin,
        grad_hess_strategy="glm",
        make_response_fn=None,
        posterior_predictive_fns=_build_link_dispatch_map(
            DistributionFamily.NEGATIVE_BINOMIAL,
            LinkFunction.LOG,
            _ppc_negative_binomial,
            include_default_key=False,
        ),
    ),
    # ---- Beta ----
    DistributionFamily.BETA: ObservationFamilySpec(
        default_link=LinkFunction.LOGIT,
        validate_support=_unit_interval,
        support_description="beta requires 0 < y < 1",
        hydrate_levels=_no_levels,
        needs_level_metadata=False,
        emission_fns=_build_link_dispatch_map(
            DistributionFamily.BETA,
            LinkFunction.LOGIT,
            _emission_factory_beta_logit,
            overrides={LinkFunction.PROBIT: _emission_factory_beta_probit},
            include_default_key=True,
        ),
        score_weight_fns=_build_link_dispatch_map(
            DistributionFamily.BETA,
            LinkFunction.LOGIT,
            _sw_factory_beta_logit,
            overrides={LinkFunction.PROBIT: _sw_factory_beta_probit},
            include_default_key=True,
        ),
        make_variance_fn=_variance_factory_beta,
        grad_hess_strategy="glm",
        make_response_fn=None,
        posterior_predictive_fns=_build_link_dispatch_map(
            DistributionFamily.BETA,
            LinkFunction.LOGIT,
            _ppc_beta_logit,
            overrides={LinkFunction.PROBIT: _ppc_beta_probit},
            include_default_key=False,
        ),
    ),
    # ---- Ordered Logistic ----
    DistributionFamily.ORDERED_LOGISTIC: ObservationFamilySpec(
        default_link=LinkFunction.CUMULATIVE_LOGIT,
        validate_support=_nonneg_integer,
        support_description="ordered_logistic requires non-negative integer-encoded levels",
        hydrate_levels=_infer_contiguous_levels,
        needs_level_metadata=True,
        emission_fns=_build_link_dispatch_map(
            DistributionFamily.ORDERED_LOGISTIC,
            LinkFunction.CUMULATIVE_LOGIT,
            _emission_factory_ordered_logistic,
            include_default_key=True,
        ),
        score_weight_fns=_build_link_dispatch_map(
            DistributionFamily.ORDERED_LOGISTIC,
            LinkFunction.CUMULATIVE_LOGIT,
            _sw_factory_ordered_logistic,
            include_default_key=True,
        ),
        make_variance_fn=_variance_factory_ordered_logistic,
        grad_hess_strategy="glm",
        make_response_fn=_response_factory_ordered_logistic,
        posterior_predictive_fns=_build_link_dispatch_map(
            DistributionFamily.ORDERED_LOGISTIC,
            LinkFunction.CUMULATIVE_LOGIT,
            _ppc_ordered_logistic,
            include_default_key=False,
        ),
    ),
    # ---- Categorical ----
    DistributionFamily.CATEGORICAL: ObservationFamilySpec(
        default_link=LinkFunction.SOFTMAX,
        validate_support=_nonneg_integer,
        support_description="categorical requires non-negative integer-encoded levels",
        hydrate_levels=_infer_contiguous_levels,
        needs_level_metadata=True,
        emission_fns=_build_link_dispatch_map(
            DistributionFamily.CATEGORICAL,
            LinkFunction.SOFTMAX,
            _emission_factory_categorical,
            include_default_key=True,
        ),
        score_weight_fns=_build_link_dispatch_map(
            DistributionFamily.CATEGORICAL,
            LinkFunction.SOFTMAX,
            _sw_factory_categorical,
            include_default_key=True,
        ),
        make_variance_fn=_variance_factory_categorical,
        grad_hess_strategy="glm",
        make_response_fn=_response_factory_categorical,
        posterior_predictive_fns=_build_link_dispatch_map(
            DistributionFamily.CATEGORICAL,
            LinkFunction.SOFTMAX,
            _ppc_categorical,
            include_default_key=False,
        ),
    ),
}


def _validate_registry_links() -> None:
    for dist, spec in FAMILY_REGISTRY.items():
        expected = {link.value for link in VALID_LINKS_FOR_DISTRIBUTION[dist]}
        if spec.default_link.value not in expected:
            raise ValueError(
                f"ObservationFamilySpec for {dist.value} has default link "
                f"{spec.default_link.value!r} not in expected {sorted(expected)}"
            )
        emission_keys = {key for key in spec.emission_fns if key != "default"}
        if emission_keys != expected:
            raise ValueError(
                f"ObservationFamilySpec for {dist.value} has emission links {sorted(emission_keys)} "
                f"but expected {sorted(expected)}"
            )

        score_weight_keys = {key for key in spec.score_weight_fns if key != "default"}
        if spec.grad_hess_strategy == "glm" and score_weight_keys != expected:
            raise ValueError(
                f"ObservationFamilySpec for {dist.value} has score-weight links "
                f"{sorted(score_weight_keys)} but expected {sorted(expected)}"
            )

        posterior_predictive_keys = set(spec.posterior_predictive_fns)
        if posterior_predictive_keys != expected:
            raise ValueError(
                f"ObservationFamilySpec for {dist.value} has posterior predictive links "
                f"{sorted(posterior_predictive_keys)} but expected {sorted(expected)}"
            )


_validate_registry_links()


POSTERIOR_PREDICTIVE_SWITCH_ORDER: tuple[tuple[DistributionFamily, LinkFunction], ...] = tuple(
    (dist, LinkFunction(link))
    for dist, spec in FAMILY_REGISTRY.items()
    for link in spec.posterior_predictive_fns
)

POSTERIOR_PREDICTIVE_SWITCH_BRANCHES = tuple(
    FAMILY_REGISTRY[dist].posterior_predictive_fns[link.value]
    for dist, link in POSTERIOR_PREDICTIVE_SWITCH_ORDER
)

_POSTERIOR_PREDICTIVE_SWITCH_INDEX: dict[tuple[DistributionFamily, LinkFunction], int] = {
    key: idx for idx, key in enumerate(POSTERIOR_PREDICTIVE_SWITCH_ORDER)
}


def _coerce_link_function(
    link: LinkFunction | str | None,
) -> LinkFunction | None:
    if link is None or isinstance(link, LinkFunction):
        return link
    try:
        return LinkFunction(link)
    except ValueError as exc:
        raise ValueError(f"Unknown link function: {link!r}") from exc


def supported_distribution_families() -> frozenset[DistributionFamily]:
    """Return the set of supported observation families."""
    return frozenset(FAMILY_REGISTRY)


def get_family_spec(
    dist: DistributionFamily | str,
) -> ObservationFamilySpec | None:
    """Look up an observation-family spec, accepting enums or serialized strings."""
    family = _coerce_distribution_family(dist)
    return FAMILY_REGISTRY.get(family)


def get_posterior_predictive_switch_index(
    dist: DistributionFamily | str,
    *,
    link: LinkFunction | str | None = None,
) -> int:
    """Resolve the lax.switch branch index for posterior predictive sampling."""
    family = _coerce_distribution_family(dist)
    spec = FAMILY_REGISTRY[family]
    default_index = _POSTERIOR_PREDICTIVE_SWITCH_INDEX[(family, spec.default_link)]
    link_fn = _coerce_link_function(link)
    if link_fn is None:
        return default_index
    return _POSTERIOR_PREDICTIVE_SWITCH_INDEX.get((family, link_fn), default_index)


def any_family_needs_level_metadata(
    dists: list[DistributionFamily] | set[DistributionFamily] | set[str],
) -> bool:
    """Return True when any requested family requires hydrated level counts."""
    return any(
        spec.needs_level_metadata for dist in dists if (spec := get_family_spec(dist)) is not None
    )


def resolve_manifest_families_and_links(
    manifest_dists: Sequence[DistributionFamily | str],
    *,
    manifest_links: Sequence[LinkFunction | str | None] | None = None,
) -> tuple[list[DistributionFamily], list[LinkFunction]]:
    """Resolve per-channel families and links, filling in family defaults when omitted."""
    dists = [DistributionFamily(dist) for dist in manifest_dists]
    if manifest_links is not None and len(manifest_links) != len(dists):
        raise ValueError(
            "manifest_links length must match manifest_dists: "
            f"{len(manifest_links)} vs {len(dists)}"
        )

    effective_links = manifest_links if manifest_links is not None else [None] * len(dists)
    links: list[LinkFunction] = []
    for dist, link in zip(dists, effective_links, strict=False):
        link_fn = _coerce_link_function(link)
        if link_fn is None:
            links.append(FAMILY_REGISTRY[dist].default_link)
        else:
            links.append(link_fn)
    return dists, links
