"""Kernel layer: pre-resolved callables for SSM inference.

Separates the specification domain (SSMSpec: serializable enums for web UI)
from the inference domain (kernels: bound JAX callables). Kernels are built
once per likelihood evaluation from spec enums + sampled hyperparameters,
then passed to all backend internals.

ObservationKernel: p(y_t | x_t) — emission log-prob, inverse link, EKF variance.
TransitionKernel: p(x_t | x_{t-1}) — process noise sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.stats as jstats

from causal_ssm_agent.models.likelihoods.emissions import get_emission_fn
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# Kernel dataclasses
# =============================================================================


@dataclass(frozen=True)
class ObservationKernel:
    """Pre-resolved observation model for SSM inference.

    Built once from DistributionFamily + LinkFunction + sampled hyperparameters.
    Consumed by all inference backends (RBPF, block RBPF, bootstrap PF,
    Laplace-EM, structured VI, DPF).

    Attributes:
        emission_fn: Log-probability (y, z, H, d, R, mask) -> scalar.
        response_fn: Inverse link, maps linear predictor to mean (elementwise).
        variance_fn: Maps predicted mean to (n_m, n_m) pseudo-covariance for
            EKF linearization. Diagonal for GLM families; full manifest_cov
            for Gaussian/Student-t. Not used when is_gaussian is True (callers
            take the exact Kalman path), but available for uniformity.
        is_gaussian: If True, callers should use exact Kalman update/marginal
            instead of EKF linearization + quadrature.
    """

    emission_fn: Callable
    response_fn: Callable
    variance_fn: Callable
    is_gaussian: bool


@dataclass(frozen=True)
class TransitionKernel:
    """Pre-resolved process noise model for SSM transition dynamics.

    Callers compute the deterministic mean (Ad @ x + cd) and Cholesky of Qd,
    then call sample_noise_fn to generate the stochastic perturbation.

    Attributes:
        sample_noise_fn: (key, chol_Q) -> (n,) noise vector (Cholesky applied).
        is_gaussian: Whether dynamics are Gaussian (enables Rao-Blackwellization).
    """

    sample_noise_fn: Callable
    is_gaussian: bool


# =============================================================================
# Response functions (inverse links)
# =============================================================================


def _response_identity(eta: jnp.ndarray) -> jnp.ndarray:
    return eta


def _response_exp(eta: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(eta)


def _response_sigmoid(eta: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(eta)


def _response_probit(eta: jnp.ndarray) -> jnp.ndarray:
    return jstats.norm.cdf(eta)


def _response_inverse(eta: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / jnp.clip(eta, 1e-6)


_RESPONSE_FNS: dict[LinkFunction, Callable] = {
    LinkFunction.IDENTITY: _response_identity,
    LinkFunction.LOG: _response_exp,
    LinkFunction.LOGIT: _response_sigmoid,
    LinkFunction.PROBIT: _response_probit,
    LinkFunction.INVERSE: _response_inverse,
}


# =============================================================================
# Variance functions (EKF linearization pseudo-covariance)
# =============================================================================


def _make_variance_poisson() -> Callable:
    """Poisson: Var(Y) = lambda = mean.

    Clamps mean away from zero to prevent singular EKF pseudo-covariance.
    """

    def variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        return jnp.diag(jnp.maximum(mean, 1e-8))

    return variance_fn


def _make_variance_negative_binomial(r: float) -> Callable:
    """NegBin: Var(Y) = mu + mu^2/r.

    Clamps mean away from zero to prevent singular EKF pseudo-covariance.
    """

    def variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        mu = jnp.maximum(mean, 1e-8)
        return jnp.diag(mu + mu**2 / (r + 1e-8))

    return variance_fn


def _make_variance_gamma(shape: float) -> Callable:
    """Gamma: Var(Y) = mean^2 / shape.

    Clamps mean away from zero to prevent singular EKF pseudo-covariance.
    """

    def variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        mu = jnp.maximum(mean, 1e-8)
        return jnp.diag(mu**2 / (shape + 1e-8))

    return variance_fn


def _make_variance_bernoulli() -> Callable:
    """Bernoulli: Var(Y) = p(1-p).

    Clamps p away from 0/1 boundaries to prevent singular EKF pseudo-covariance.
    """

    def variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        p = jnp.clip(mean, 1e-7, 1.0 - 1e-7)
        return jnp.diag(p * (1.0 - p))

    return variance_fn


def _make_variance_beta(concentration: float) -> Callable:
    """Beta: Var(Y) = p(1-p) / (phi + 1).

    Clamps p away from 0/1 boundaries to prevent singular EKF pseudo-covariance.
    """

    def variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        p = jnp.clip(mean, 1e-7, 1.0 - 1e-7)
        return jnp.diag(p * (1.0 - p) / (concentration + 1.0))

    return variance_fn


def _make_variance_identity(manifest_cov: jnp.ndarray) -> Callable:
    """Gaussian/Student-t: pseudo-R = measurement covariance (constant)."""

    def variance_fn(_mean: jnp.ndarray) -> jnp.ndarray:
        return manifest_cov

    return variance_fn


# =============================================================================
# ObservationKernel factory
# =============================================================================


def build_observation_kernel(
    dist: DistributionFamily,
    link: LinkFunction,
    extra_params: dict | None = None,
    manifest_cov: jnp.ndarray | None = None,
) -> ObservationKernel:
    """Build an ObservationKernel from spec enums + sampled hyperparameters.

    This is the single resolution point: enums and runtime parameters go in,
    pre-bound callables come out. Called once per likelihood evaluation.

    Args:
        dist: Distribution family enum.
        link: Link function enum.
        extra_params: Sampled hyperparameters (obs_df, obs_shape, obs_r,
            obs_concentration, etc.).
        manifest_cov: Measurement noise covariance matrix. Required for
            Gaussian/Student-t (used as EKF pseudo-covariance). Ignored
            for GLM families.
    """
    extra_params = extra_params or {}

    # Emission log-prob (delegates to existing canonical functions)
    emission_fn = get_emission_fn(dist, extra_params, link=link)

    # Response function (inverse link)
    response_fn = _RESPONSE_FNS.get(link)
    if response_fn is None:
        raise ValueError(
            f"No response function for link={link!r}. Supported: {list(_RESPONSE_FNS.keys())}"
        )

    # Variance function + is_gaussian flag
    is_gaussian = dist == DistributionFamily.GAUSSIAN

    if dist in (DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T):
        if manifest_cov is not None:
            variance_fn = _make_variance_identity(manifest_cov)
        else:
            # Backends that skip EKF linearization (structured VI, DPF) don't
            # call variance_fn, so a lazy error is appropriate.
            def variance_fn(_mean: jnp.ndarray) -> jnp.ndarray:
                raise RuntimeError(
                    f"variance_fn for {dist} requires manifest_cov; "
                    f"pass it to build_observation_kernel()"
                )

    elif dist == DistributionFamily.POISSON:
        variance_fn = _make_variance_poisson()
    elif dist == DistributionFamily.NEGATIVE_BINOMIAL:
        r = extra_params.get("obs_r", 5.0)
        variance_fn = _make_variance_negative_binomial(r)
    elif dist == DistributionFamily.GAMMA:
        shape = extra_params.get("obs_shape", 1.0)
        variance_fn = _make_variance_gamma(shape)
    elif dist == DistributionFamily.BERNOULLI:
        variance_fn = _make_variance_bernoulli()
    elif dist == DistributionFamily.BETA:
        conc = extra_params.get("obs_concentration", 10.0)
        variance_fn = _make_variance_beta(conc)
    else:
        raise ValueError(
            f"No variance function for dist={dist!r}. "
            f"Supported: gaussian, student_t, poisson, negative_binomial, "
            f"gamma, bernoulli, beta."
        )

    return ObservationKernel(
        emission_fn=emission_fn,
        response_fn=response_fn,
        variance_fn=variance_fn,
        is_gaussian=is_gaussian,
    )


# =============================================================================
# TransitionKernel factory
# =============================================================================


def _make_gaussian_noise() -> Callable:
    def sample_noise(key: jax.Array, chol_Q: jnp.ndarray) -> jnp.ndarray:
        n = chol_Q.shape[0]
        return chol_Q @ random.normal(key, (n,))

    return sample_noise


def _make_student_t_noise(df: float) -> Callable:
    def sample_noise(key: jax.Array, chol_Q: jnp.ndarray) -> jnp.ndarray:
        n = chol_Q.shape[0]
        df_safe = jnp.maximum(df, 2.1)
        key_z, key_chi2 = random.split(key)
        z = random.normal(key_z, (n,))
        chi2 = jnp.maximum(random.gamma(key_chi2, df_safe / 2.0) * 2.0, 1e-8)
        scale = jnp.sqrt((df_safe - 2.0) / chi2)
        return chol_Q @ (z * scale)

    return sample_noise


def build_transition_kernel(
    dist: DistributionFamily,
    extra_params: dict | None = None,
) -> TransitionKernel:
    """Build a TransitionKernel from spec enum + sampled hyperparameters.

    Args:
        dist: Diffusion distribution family enum.
        extra_params: Sampled hyperparameters (proc_df, etc.).
    """
    extra_params = extra_params or {}

    if dist == DistributionFamily.GAUSSIAN:
        return TransitionKernel(
            sample_noise_fn=_make_gaussian_noise(),
            is_gaussian=True,
        )
    elif dist == DistributionFamily.STUDENT_T:
        df = extra_params.get("proc_df", 5.0)
        return TransitionKernel(
            sample_noise_fn=_make_student_t_noise(df),
            is_gaussian=False,
        )
    else:
        raise ValueError(
            f"No transition kernel for diffusion_dist={dist!r}. Supported: gaussian, student_t."
        )
