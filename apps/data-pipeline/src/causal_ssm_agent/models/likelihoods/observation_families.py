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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ObservationFamilySpec:
    """Registry entry for observation family behavior across all dispatch sites."""

    # --- ssm_builder.py concerns ---
    validate_support: Callable[[np.ndarray], np.ndarray]
    """values -> invalid_mask (bool array). Empty mask means no support constraint."""
    support_description: str
    """Human-readable constraint for error messages."""
    hydrate_levels: Callable[[np.ndarray], int | None]
    """column_values -> level_count or None if not a discrete family."""
    requires_integer_encoding: bool
    """Whether this family needs integer-encoded observations."""

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


# ---------------------------------------------------------------------------
# Support validators (ssm_builder.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Lazy imports — avoid circular deps (called at registry-build time only)
# ---------------------------------------------------------------------------


def _get_emissions():
    from causal_ssm_agent.models.likelihoods import emissions as em

    return em


def _get_kernels():
    from causal_ssm_agent.models.likelihoods import kernels as kr

    return kr


# ---------------------------------------------------------------------------
# Emission-fn factories  (link_str -> factory(extra_params) -> callable)
# ---------------------------------------------------------------------------


def _emission_factory_gaussian(_params: dict):
    return _get_emissions().emission_log_prob_gaussian


def _emission_factory_poisson(_params: dict):
    return _get_emissions().emission_log_prob_poisson


def _emission_factory_student_t(params: dict):
    em = _get_emissions()
    df = params.get("obs_df", 5.0)
    return lambda y, z, H, d, R, m: em.emission_log_prob_student_t(y, z, H, d, R, m, df)


def _emission_factory_gamma_log(params: dict):
    em = _get_emissions()
    shape = params.get("obs_shape", 1.0)
    return lambda y, z, H, d, R, m: em.emission_log_prob_gamma(y, z, H, d, R, m, shape)


def _emission_factory_gamma_inverse(params: dict):
    em = _get_emissions()
    shape = params.get("obs_shape", 1.0)
    return lambda y, z, H, d, R, m: em.emission_log_prob_gamma_inverse(y, z, H, d, R, m, shape)


def _emission_factory_bernoulli_logit(_params: dict):
    return _get_emissions().emission_log_prob_bernoulli


def _emission_factory_bernoulli_probit(_params: dict):
    return _get_emissions().emission_log_prob_bernoulli_probit


def _emission_factory_negbin(params: dict):
    em = _get_emissions()
    r = params.get("obs_r", 5.0)
    return lambda y, z, H, d, R, m: em.emission_log_prob_negative_binomial(y, z, H, d, R, m, r)


def _emission_factory_beta_logit(params: dict):
    em = _get_emissions()
    conc = params.get("obs_concentration", 10.0)
    return lambda y, z, H, d, R, m: em.emission_log_prob_beta(y, z, H, d, R, m, conc)


def _emission_factory_beta_probit(params: dict):
    em = _get_emissions()
    conc = params.get("obs_concentration", 10.0)
    return lambda y, z, H, d, R, m: em.emission_log_prob_beta_probit(y, z, H, d, R, m, conc)


def _emission_factory_ordered_logistic(params: dict):
    em = _get_emissions()
    level_counts, cutpoints = em.get_ordered_logistic_extra_params(params)
    return lambda y, z, H, d, R, m: em.emission_log_prob_ordered_logistic(
        y, z, H, d, R, m, cutpoints, level_counts
    )


def _emission_factory_categorical(params: dict):
    em = _get_emissions()
    level_counts, intercepts, slopes = em.get_categorical_extra_params(params)
    return lambda y, z, H, d, R, m: em.emission_log_prob_categorical(
        y, z, H, d, R, m, intercepts, slopes, level_counts
    )


# ---------------------------------------------------------------------------
# Score/weight factories  (link_str -> factory(extra_params) -> callable|None)
# ---------------------------------------------------------------------------


def _sw_factory_none(_params: dict):
    return None


def _sw_factory_poisson(_params: dict):
    return _get_emissions()._score_weight_poisson


def _sw_factory_bernoulli_logit(_params: dict):
    return _get_emissions()._score_weight_bernoulli_logit


def _sw_factory_bernoulli_probit(_params: dict):
    return _get_emissions()._score_weight_bernoulli_probit


def _sw_factory_beta_logit(params: dict):
    em = _get_emissions()
    conc = params.get("obs_concentration", 10.0)
    return lambda y, eta, m: em._score_weight_beta_logit(y, eta, m, conc)


def _sw_factory_beta_probit(params: dict):
    em = _get_emissions()
    conc = params.get("obs_concentration", 10.0)
    return lambda y, eta, m: em._score_weight_beta_probit(y, eta, m, conc)


def _sw_factory_gamma_log(params: dict):
    em = _get_emissions()
    shape = params.get("obs_shape", 1.0)
    return lambda y, eta, m: em._score_weight_gamma_log(y, eta, m, shape)


def _sw_factory_gamma_inverse(params: dict):
    em = _get_emissions()
    shape = params.get("obs_shape", 1.0)
    return lambda y, eta, m: em._score_weight_gamma_inverse(y, eta, m, shape)


def _sw_factory_negbin(params: dict):
    em = _get_emissions()
    r = params.get("obs_r", 5.0)
    return lambda y, eta, m: em._score_weight_negative_binomial(y, eta, m, r)


def _sw_factory_ordered_logistic(params: dict):
    em = _get_emissions()
    level_counts, cutpoints = em.get_ordered_logistic_extra_params(params)
    return lambda y, eta, m: em._score_weight_ordered_logistic(y, eta, m, cutpoints, level_counts)


def _sw_factory_categorical(params: dict):
    em = _get_emissions()
    level_counts, intercepts, slopes = em.get_categorical_extra_params(params)
    return lambda y, eta, m: em._score_weight_categorical(
        y, eta, m, intercepts, slopes, level_counts
    )


# ---------------------------------------------------------------------------
# Variance-fn factories  (extra_params, manifest_cov) -> variance_fn
# ---------------------------------------------------------------------------


def _variance_factory_gaussian_like(_params: dict, manifest_cov):
    kr = _get_kernels()
    if manifest_cov is not None:
        return kr._make_variance_identity(manifest_cov)

    def _lazy_error(_mean):
        raise RuntimeError(
            "variance_fn for gaussian/student_t requires manifest_cov; "
            "pass it to build_observation_kernel()"
        )

    return _lazy_error


def _variance_factory_poisson(_params: dict, _manifest_cov):
    return _get_kernels()._make_variance_poisson()


def _variance_factory_negbin(params: dict, _manifest_cov):
    r = params.get("obs_r", 5.0)
    return _get_kernels()._make_variance_negative_binomial(r)


def _variance_factory_gamma(params: dict, _manifest_cov):
    shape = params.get("obs_shape", 1.0)
    return _get_kernels()._make_variance_gamma(shape)


def _variance_factory_bernoulli(_params: dict, _manifest_cov):
    return _get_kernels()._make_variance_bernoulli()


def _variance_factory_beta(params: dict, _manifest_cov):
    conc = params.get("obs_concentration", 10.0)
    return _get_kernels()._make_variance_beta(conc)


def _variance_factory_ordered_logistic(params: dict, _manifest_cov):
    em = _get_emissions()
    kr = _get_kernels()
    level_counts, cutpoints = em.get_ordered_logistic_extra_params(params)
    return kr._make_discrete_variance_from_moments(
        lambda eta: em.ordered_logistic_moments(eta, cutpoints, level_counts)
    )


def _variance_factory_categorical(params: dict, _manifest_cov):
    em = _get_emissions()
    kr = _get_kernels()
    level_counts, intercepts, slopes = em.get_categorical_extra_params(params)
    return kr._make_discrete_variance_from_moments(
        lambda eta: em.categorical_moments(eta, intercepts, slopes, level_counts)
    )


# ---------------------------------------------------------------------------
# Response-fn factories  (extra_params) -> response_fn  (or None → use link)
# ---------------------------------------------------------------------------


def _response_factory_ordered_logistic(params: dict):
    em = _get_emissions()
    kr = _get_kernels()
    level_counts, cutpoints = em.get_ordered_logistic_extra_params(params)
    return kr._make_discrete_response_ordered_logistic(cutpoints, level_counts)


def _response_factory_categorical(params: dict):
    em = _get_emissions()
    kr = _get_kernels()
    level_counts, intercepts, slopes = em.get_categorical_extra_params(params)
    return kr._make_discrete_response_categorical(intercepts, slopes, level_counts)


# ===========================================================================
# The registry
# ===========================================================================

FAMILY_REGISTRY: dict[DistributionFamily, ObservationFamilySpec] = {
    # ---- Gaussian ----
    DistributionFamily.GAUSSIAN: ObservationFamilySpec(
        validate_support=_no_constraint,
        support_description="",
        hydrate_levels=_no_levels,
        requires_integer_encoding=False,
        emission_fns={
            "default": _emission_factory_gaussian,
        },
        score_weight_fns={
            "default": _sw_factory_none,
        },
        make_variance_fn=_variance_factory_gaussian_like,
        grad_hess_strategy="gaussian",
        make_response_fn=None,
    ),
    # ---- Student-t ----
    DistributionFamily.STUDENT_T: ObservationFamilySpec(
        validate_support=_no_constraint,
        support_description="",
        hydrate_levels=_no_levels,
        requires_integer_encoding=False,
        emission_fns={
            "default": _emission_factory_student_t,
        },
        score_weight_fns={
            "default": _sw_factory_none,
        },
        make_variance_fn=_variance_factory_gaussian_like,
        grad_hess_strategy="student_t",
        make_response_fn=None,
    ),
    # ---- Poisson ----
    DistributionFamily.POISSON: ObservationFamilySpec(
        validate_support=_nonneg_integer,
        support_description="poisson requires non-negative integer counts",
        hydrate_levels=_no_levels,
        requires_integer_encoding=False,
        emission_fns={
            "default": _emission_factory_poisson,
        },
        score_weight_fns={
            "default": _sw_factory_poisson,
        },
        make_variance_fn=_variance_factory_poisson,
        grad_hess_strategy="glm",
        make_response_fn=None,
    ),
    # ---- Gamma ----
    DistributionFamily.GAMMA: ObservationFamilySpec(
        validate_support=_positive_only,
        support_description="gamma requires y > 0",
        hydrate_levels=_no_levels,
        requires_integer_encoding=False,
        emission_fns={
            "default": _emission_factory_gamma_log,
            "log": _emission_factory_gamma_log,
            "inverse": _emission_factory_gamma_inverse,
        },
        score_weight_fns={
            "default": _sw_factory_gamma_log,
            "log": _sw_factory_gamma_log,
            "inverse": _sw_factory_gamma_inverse,
        },
        make_variance_fn=_variance_factory_gamma,
        grad_hess_strategy="glm",
        make_response_fn=None,
    ),
    # ---- Bernoulli ----
    DistributionFamily.BERNOULLI: ObservationFamilySpec(
        validate_support=_binary,
        support_description="bernoulli requires binary values in {0, 1}",
        hydrate_levels=_no_levels,
        requires_integer_encoding=False,
        emission_fns={
            "default": _emission_factory_bernoulli_logit,
            "logit": _emission_factory_bernoulli_logit,
            "probit": _emission_factory_bernoulli_probit,
        },
        score_weight_fns={
            "default": _sw_factory_bernoulli_logit,
            "logit": _sw_factory_bernoulli_logit,
            "probit": _sw_factory_bernoulli_probit,
        },
        make_variance_fn=_variance_factory_bernoulli,
        grad_hess_strategy="glm",
        make_response_fn=None,
    ),
    # ---- Negative Binomial ----
    DistributionFamily.NEGATIVE_BINOMIAL: ObservationFamilySpec(
        validate_support=_nonneg_integer,
        support_description="negative_binomial requires non-negative integer counts",
        hydrate_levels=_no_levels,
        requires_integer_encoding=False,
        emission_fns={
            "default": _emission_factory_negbin,
        },
        score_weight_fns={
            "default": _sw_factory_negbin,
        },
        make_variance_fn=_variance_factory_negbin,
        grad_hess_strategy="glm",
        make_response_fn=None,
    ),
    # ---- Beta ----
    DistributionFamily.BETA: ObservationFamilySpec(
        validate_support=_unit_interval,
        support_description="beta requires 0 < y < 1",
        hydrate_levels=_no_levels,
        requires_integer_encoding=False,
        emission_fns={
            "default": _emission_factory_beta_logit,
            "logit": _emission_factory_beta_logit,
            "probit": _emission_factory_beta_probit,
        },
        score_weight_fns={
            "default": _sw_factory_beta_logit,
            "logit": _sw_factory_beta_logit,
            "probit": _sw_factory_beta_probit,
        },
        make_variance_fn=_variance_factory_beta,
        grad_hess_strategy="glm",
        make_response_fn=None,
    ),
    # ---- Ordered Logistic ----
    DistributionFamily.ORDERED_LOGISTIC: ObservationFamilySpec(
        validate_support=_nonneg_integer,
        support_description="ordered_logistic requires non-negative integer-encoded levels",
        hydrate_levels=_no_levels,  # hydration handled by _hydrate_discrete_manifest_metadata
        requires_integer_encoding=True,
        emission_fns={
            "default": _emission_factory_ordered_logistic,
        },
        score_weight_fns={
            "default": _sw_factory_ordered_logistic,
        },
        make_variance_fn=_variance_factory_ordered_logistic,
        grad_hess_strategy="glm",
        make_response_fn=_response_factory_ordered_logistic,
    ),
    # ---- Categorical ----
    DistributionFamily.CATEGORICAL: ObservationFamilySpec(
        validate_support=_nonneg_integer,
        support_description="categorical requires non-negative integer-encoded levels",
        hydrate_levels=_no_levels,  # hydration handled by _hydrate_discrete_manifest_metadata
        requires_integer_encoding=True,
        emission_fns={
            "default": _emission_factory_categorical,
        },
        score_weight_fns={
            "default": _sw_factory_categorical,
        },
        make_variance_fn=_variance_factory_categorical,
        grad_hess_strategy="glm",
        make_response_fn=_response_factory_categorical,
    ),
}
