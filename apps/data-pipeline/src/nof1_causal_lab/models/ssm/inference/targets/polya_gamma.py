"""Polya-Gamma observation conditioning for exact affine-logit measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction

PolyaGammaSampler = Literal["truncated_sum", "devroye", "devroye_integer"]
SUPPORTED_POLYA_GAMMA_SAMPLERS: tuple[PolyaGammaSampler, ...] = (
    "truncated_sum",
    "devroye",
    "devroye_integer",
)
_DEVROYE_TRUNC = 0.64


class PolyaGammaAuxiliaryState(NamedTuple):
    """Per-observation PG auxiliary variables and sufficient stats."""

    omega: jnp.ndarray
    active_mask: jnp.ndarray
    kappa: jnp.ndarray
    shape: jnp.ndarray
    linear_offset: jnp.ndarray
    observed_values: jnp.ndarray
    gamma_base_terms: jnp.ndarray


@dataclass(frozen=True)
class PolyaGammaObservationPlan:
    """Static channel plan compiled from manifest families and links."""

    channel_mask: jnp.ndarray
    bernoulli_channel_mask: jnp.ndarray
    negative_binomial_channel_mask: jnp.ndarray
    num_terms: int
    sampler: PolyaGammaSampler
    enabled: bool
    consumes_all_channels: bool
    max_integer_shape: int | None = None


def normalize_polya_gamma_sampler(sampler: str) -> PolyaGammaSampler:
    """Normalize and validate the PG sampler backend."""
    normalized = str(sampler).strip().lower()
    if normalized not in SUPPORTED_POLYA_GAMMA_SAMPLERS:
        raise ValueError(
            f"Unsupported polya_gamma_sampler {sampler!r}. "
            f"Supported: {', '.join(repr(name) for name in SUPPORTED_POLYA_GAMMA_SAMPLERS)}."
        )
    return cast("PolyaGammaSampler", normalized)


def build_polya_gamma_observation_plan(
    manifest_dists: list[DistributionFamily],
    manifest_links: list[LinkFunction],
    *,
    num_terms: int,
    sampler: str = "truncated_sum",
    enabled: bool = True,
    max_integer_shape: int | None = None,
) -> PolyaGammaObservationPlan:
    """Identify manifest channels with exact PG-conditionable likelihoods."""
    if num_terms < 1:
        raise ValueError(f"polya_gamma_num_terms must be >= 1, got {num_terms}.")
    normalized_sampler = normalize_polya_gamma_sampler(sampler)
    if len(manifest_dists) != len(manifest_links):
        raise ValueError(
            "manifest_dists and manifest_links must have the same length for PG planning: "
            f"{len(manifest_dists)} vs {len(manifest_links)}."
        )
    if enabled:
        bernoulli_channel_mask = jnp.asarray(
            [
                dist == DistributionFamily.BERNOULLI and link == LinkFunction.LOGIT
                for dist, link in zip(manifest_dists, manifest_links, strict=True)
            ],
            dtype=bool,
        )
        negative_binomial_channel_mask = jnp.asarray(
            [
                dist == DistributionFamily.NEGATIVE_BINOMIAL and link == LinkFunction.LOG
                for dist, link in zip(manifest_dists, manifest_links, strict=True)
            ],
            dtype=bool,
        )
    else:
        bernoulli_channel_mask = jnp.zeros((len(manifest_dists),), dtype=bool)
        negative_binomial_channel_mask = jnp.zeros((len(manifest_dists),), dtype=bool)
    has_negative_binomial_channels = bool(np.any(np.asarray(negative_binomial_channel_mask)))
    if normalized_sampler == "devroye" and has_negative_binomial_channels:
        raise ValueError(
            "polya_gamma_sampler='devroye' currently supports only Bernoulli-logit PG(1, eta) "
            "channels. Use polya_gamma_sampler='truncated_sum' for negative-binomial log-link "
            "channels."
        )
    integer_shape_max = None
    if normalized_sampler == "devroye_integer":
        if has_negative_binomial_channels and max_integer_shape is None:
            raise ValueError(
                "polya_gamma_sampler='devroye_integer' for negative-binomial log-link channels "
                "requires max_integer_shape from validated integer counts and fixed integer obs_r."
            )
        integer_shape_max = int(max_integer_shape) if max_integer_shape is not None else 1
        if integer_shape_max < 1:
            raise ValueError(
                "max_integer_shape must be >= 1 for polya_gamma_sampler='devroye_integer', "
                f"got {integer_shape_max}."
            )
    elif max_integer_shape is not None:
        raise ValueError(
            "max_integer_shape is only valid with polya_gamma_sampler='devroye_integer'."
        )
    channel_mask = bernoulli_channel_mask | negative_binomial_channel_mask
    host_mask = np.asarray(channel_mask)
    return PolyaGammaObservationPlan(
        channel_mask=channel_mask,
        bernoulli_channel_mask=bernoulli_channel_mask,
        negative_binomial_channel_mask=negative_binomial_channel_mask,
        num_terms=int(num_terms),
        sampler=normalized_sampler,
        enabled=bool(np.any(host_mask)),
        consumes_all_channels=bool(np.all(host_mask)) if host_mask.size else False,
        max_integer_shape=integer_shape_max,
    )


def active_polya_gamma_mask(
    plan: PolyaGammaObservationPlan,
    observations: jnp.ndarray,
) -> jnp.ndarray:
    """Return the observed cells consumed by the PG augmentation."""
    return (~jnp.isnan(observations)) & plan.channel_mask[None, :]


def mask_polya_gamma_observations(
    plan: PolyaGammaObservationPlan,
    observations: jnp.ndarray,
) -> jnp.ndarray:
    """Remove PG-consumed cells from the residual observation likelihood surface."""
    active = active_polya_gamma_mask(plan, observations)
    return jnp.where(active, jnp.nan, observations)


def _linear_predictor_trajectory(context, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
    if context.H_rows is not None:
        assert context.d_rows is not None
        return jax.vmap(lambda state_t, H_t, d_t: H_t @ state_t + d_t)(
            latent_trajectory,
            context.H_rows,
            context.d_rows,
        )
    return latent_trajectory @ context.H.T + context.d_meas


def _linear_predictor_at(context, latent_state: jnp.ndarray, time_idx: jnp.ndarray) -> jnp.ndarray:
    if context.H_rows is not None:
        assert context.d_rows is not None
        H_t = context.H_rows[time_idx]
        d_t = context.d_rows[time_idx]
    else:
        H_t = context.H
        d_t = context.d_meas
    return H_t @ latent_state + d_t


def expected_pg1(eta: jnp.ndarray) -> jnp.ndarray:
    """Return E[PG(1, eta)], with the eta=0 limit handled explicitly."""
    abs_eta = jnp.abs(eta)
    safe_eta = jnp.maximum(abs_eta, 1e-8)
    mean = jnp.tanh(0.5 * safe_eta) / (2.0 * safe_eta)
    return jnp.where(abs_eta < 1e-6, 0.25, mean)


def expected_polya_gamma(shape: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    """Return E[PG(shape, eta)]."""
    return jnp.asarray(shape, dtype=eta.dtype) * expected_pg1(eta)


def sample_polya_gamma_truncated_sum(
    key: jax.Array,
    shape: jnp.ndarray,
    eta: jnp.ndarray,
    *,
    num_terms: int,
) -> jnp.ndarray:
    """Sample the standard finite-sum approximation to PG(shape, eta)."""
    eta = jnp.asarray(eta)
    shape = jnp.broadcast_to(jnp.asarray(shape, dtype=eta.dtype), eta.shape)
    term_idx = jnp.arange(num_terms, dtype=eta.dtype) + 0.5
    tilt = eta[..., None] / (2.0 * jnp.pi)
    denom = term_idx * term_idx + tilt * tilt
    gamma_shape = jnp.broadcast_to(shape[..., None], (*eta.shape, num_terms))
    gamma_draws = jax.random.gamma(key, gamma_shape)
    return jnp.sum(gamma_draws / denom, axis=-1) / (2.0 * jnp.pi * jnp.pi)


def _truncated_sum_base_denominators(*, num_terms: int, dtype) -> jnp.ndarray:
    term_idx = jnp.arange(num_terms, dtype=dtype) + 0.5
    return 2.0 * jnp.pi * jnp.pi * term_idx * term_idx


def _truncated_sum_conditional_mean_terms(
    shape: jnp.ndarray,
    eta: jnp.ndarray,
    *,
    num_terms: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    eta = jnp.asarray(eta)
    shape = jnp.broadcast_to(jnp.asarray(shape, dtype=eta.dtype), eta.shape)
    base_denom = _truncated_sum_base_denominators(num_terms=num_terms, dtype=eta.dtype)
    rate = 1.0 + eta[..., None] * eta[..., None] / (2.0 * base_denom)
    terms = shape[..., None] / rate
    omega = jnp.sum(terms / base_denom, axis=-1)
    return omega, terms


def sample_polya_gamma_truncated_sum_base_terms(
    key: jax.Array,
    shape: jnp.ndarray,
    eta: jnp.ndarray,
    *,
    num_terms: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample zero-tilt Gamma-series terms and their PG sum under finite truncation.

    The stored terms are the auxiliary variables under the PG(b, 0) series.  Given
    eta, their conditional law is Gamma(b, rate=1 + eta^2 / (2 d_k)), which keeps
    the latent target Gaussian while exposing the b-dependent density for
    parameter updates.

    TODO: Replace this finite-sum NB path with a JAX-native Windle/Polson/Scott
    hybrid PG sampler and matching log-density terms when arbitrary-shape PG
    becomes a bottleneck.
    """
    eta = jnp.asarray(eta)
    shape = jnp.broadcast_to(jnp.asarray(shape, dtype=eta.dtype), eta.shape)
    base_denom = _truncated_sum_base_denominators(num_terms=num_terms, dtype=eta.dtype)
    rate = 1.0 + eta[..., None] * eta[..., None] / (2.0 * base_denom)
    gamma_shape = jnp.broadcast_to(shape[..., None], (*eta.shape, num_terms))
    unit_rate_terms = jax.random.gamma(key, gamma_shape)
    terms = unit_rate_terms / rate
    omega = jnp.sum(terms / base_denom, axis=-1)
    return omega, terms


def _devroye_a(n: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    n_float = n.astype(x.dtype)
    k = (n_float + 0.5) * jnp.pi
    large_x = k * jnp.exp(-0.5 * k * k * x)
    small_x = (2.0 / (jnp.pi * x)) ** 1.5 * k * jnp.exp(-2.0 * k * k / x)
    return jnp.where(x > _DEVROYE_TRUNC, large_x, small_x)


def _devroye_mass_texpon(z: jnp.ndarray) -> jnp.ndarray:
    trunc = jnp.asarray(_DEVROYE_TRUNC, dtype=z.dtype)
    rate = 0.125 * jnp.pi * jnp.pi + 0.5 * z * z
    sqrt_inv_trunc = jax.lax.rsqrt(trunc)
    b = sqrt_inv_trunc * (trunc * z - 1.0)
    a = -sqrt_inv_trunc * (trunc * z + 1.0)
    x0 = jnp.log(rate) + rate * trunc
    xb = x0 - z + jsp_special.log_ndtr(b)
    xa = x0 + z + jsp_special.log_ndtr(a)
    q_div_p = (4.0 / jnp.pi) * (jnp.exp(xb) + jnp.exp(xa))
    return 1.0 / (1.0 + q_div_p)


def _rtigauss_large_mu(key: jax.Array, z: jnp.ndarray) -> jnp.ndarray:
    trunc = jnp.asarray(_DEVROYE_TRUNC, dtype=z.dtype)

    def _cond(carry):
        _key, _x, accepted = carry
        return ~accepted

    def _body(carry):
        key, _x, _accepted = carry
        key, e1_key, e2_key, uniform_key = jax.random.split(key, 4)
        e1 = jax.random.exponential(e1_key, dtype=z.dtype)
        e2 = jax.random.exponential(e2_key, dtype=z.dtype)
        x = trunc / (1.0 + trunc * e1) ** 2
        squeeze_accept = e1 * e1 <= 2.0 * e2 / trunc
        tilt_accept = jax.random.uniform(uniform_key, dtype=z.dtype) <= jnp.exp(-0.5 * z * z * x)
        return key, x, squeeze_accept & tilt_accept

    _, x, _ = jax.lax.while_loop(
        _cond,
        _body,
        (key, trunc, jnp.asarray(False)),
    )
    return x


def _rtigauss_small_mu(key: jax.Array, z: jnp.ndarray) -> jnp.ndarray:
    trunc = jnp.asarray(_DEVROYE_TRUNC, dtype=z.dtype)
    mu = 1.0 / z

    def _cond(carry):
        _key, x = carry
        return x > trunc

    def _body(carry):
        key, _x = carry
        key, normal_key, uniform_key = jax.random.split(key, 3)
        normal = jax.random.normal(normal_key, dtype=z.dtype)
        y = normal * normal
        x = mu + 0.5 * mu * mu * y - 0.5 * mu * jnp.sqrt(4.0 * mu * y + mu * mu * y * y)
        flip = jax.random.uniform(uniform_key, dtype=z.dtype) > mu / (mu + x)
        x = jnp.where(flip, mu * mu / x, x)
        return key, x

    _, x = jax.lax.while_loop(_cond, _body, (key, trunc + 1.0))
    return x


def _rtigauss(key: jax.Array, z: jnp.ndarray) -> jnp.ndarray:
    trunc = jnp.asarray(_DEVROYE_TRUNC, dtype=z.dtype)
    return jax.lax.cond(
        1.0 / jnp.maximum(z, jnp.asarray(1e-12, dtype=z.dtype)) > trunc,
        lambda k: _rtigauss_large_mu(k, z),
        lambda k: _rtigauss_small_mu(k, z),
        key,
    )


def _devroye_accept(key: jax.Array, x: jnp.ndarray) -> jnp.ndarray:
    key, uniform_key = jax.random.split(key)
    a0 = _devroye_a(jnp.asarray(0), x)
    y = jax.random.uniform(uniform_key, dtype=x.dtype) * a0

    def _cond(carry):
        _n, _series_sum, _done, _accepted = carry
        return ~_done

    def _body(carry):
        n, series_sum, _done, _accepted = carry
        n_next = n + jnp.asarray(1, dtype=n.dtype)
        term = _devroye_a(n_next, x)
        odd = (n_next % jnp.asarray(2, dtype=n.dtype)) == 1
        next_sum = jnp.where(odd, series_sum - term, series_sum + term)
        accept_now = odd & (y <= next_sum)
        reject_now = (~odd) & (y > next_sum)
        done = accept_now | reject_now
        return n_next, next_sum, done, accept_now

    _, _, _, accepted = jax.lax.while_loop(
        _cond,
        _body,
        (jnp.asarray(0), a0, jnp.asarray(False), jnp.asarray(False)),
    )
    return accepted


def _sample_pg1_devroye_scalar(key: jax.Array, eta: jnp.ndarray) -> jnp.ndarray:
    z = 0.5 * jnp.abs(eta)
    mass = _devroye_mass_texpon(z)
    rate = 0.125 * jnp.pi * jnp.pi + 0.5 * z * z
    trunc = jnp.asarray(_DEVROYE_TRUNC, dtype=eta.dtype)

    def _cond(carry):
        _key, _x, accepted = carry
        return ~accepted

    def _body(carry):
        key, _x, _accepted = carry
        key, proposal_key, accept_key = jax.random.split(key, 3)
        proposal_key, uniform_key, exp_key, igauss_key = jax.random.split(proposal_key, 4)
        use_exponential = jax.random.uniform(uniform_key, dtype=eta.dtype) < mass
        x_exponential = trunc + jax.random.exponential(exp_key, dtype=eta.dtype) / rate
        x_inverse_gaussian = _rtigauss(igauss_key, z)
        x = jnp.where(use_exponential, x_exponential, x_inverse_gaussian)
        accepted = _devroye_accept(accept_key, x)
        return key, x, accepted

    _, x, _ = jax.lax.while_loop(
        _cond,
        _body,
        (key, trunc, jnp.asarray(False)),
    )
    return 0.25 * x


def sample_pg1_devroye(key: jax.Array, eta: jnp.ndarray) -> jnp.ndarray:
    """Sample PG(1, eta) with Devroye's exact accept-reject sampler."""
    eta = jnp.asarray(eta)
    flat_eta = jnp.reshape(eta, (-1,))
    keys = jax.random.split(key, flat_eta.shape[0])
    flat_samples = jax.vmap(_sample_pg1_devroye_scalar)(keys, flat_eta)
    return jnp.reshape(flat_samples, eta.shape)


def sample_polya_gamma_integer_shape_devroye(
    key: jax.Array,
    shape: jnp.ndarray,
    eta: jnp.ndarray,
    *,
    max_shape: int,
) -> jnp.ndarray:
    """Sample PG(n, eta) exactly for validated positive integer n by summing PG(1, eta)."""
    if max_shape < 1:
        raise ValueError(f"max_shape must be >= 1 for integer-shape PG sampling, got {max_shape}.")
    eta = jnp.asarray(eta)
    shape_int = jnp.rint(jnp.broadcast_to(jnp.asarray(shape), eta.shape)).astype(jnp.int32)
    keys = jax.random.split(key, int(max_shape))
    draws = jax.vmap(lambda draw_key: sample_pg1_devroye(draw_key, eta))(keys)
    arange_shape = (int(max_shape),) + (1,) * eta.ndim
    include_draw = (
        jnp.reshape(jnp.arange(int(max_shape), dtype=jnp.int32), arange_shape) < shape_int
    )
    return jnp.sum(jnp.where(include_draw, draws, 0.0), axis=0)


def sample_polya_gamma(
    key: jax.Array,
    shape: jnp.ndarray,
    eta: jnp.ndarray,
    *,
    sampler: PolyaGammaSampler,
    num_terms: int,
    max_integer_shape: int | None = None,
) -> jnp.ndarray:
    """Sample PG(shape, eta) with the selected backend."""
    if sampler == "truncated_sum":
        return sample_polya_gamma_truncated_sum(key, shape, eta, num_terms=num_terms)
    if sampler == "devroye":
        return sample_pg1_devroye(key, eta)
    if sampler == "devroye_integer":
        if max_integer_shape is None:
            raise ValueError("max_integer_shape is required for devroye_integer PG sampling.")
        return sample_polya_gamma_integer_shape_devroye(
            key,
            shape,
            eta,
            max_shape=max_integer_shape,
        )
    raise ValueError(f"Unsupported Polya-Gamma sampler {sampler!r}.")


def _channel_param(
    context,
    key: str,
    *,
    default: float,
    n_channels: int,
    dtype,
) -> jnp.ndarray:
    extra_params = getattr(context, "extra_params", None) or {}
    value = extra_params.get(key, default)
    array = jnp.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return jnp.broadcast_to(array, (n_channels,))
    return jnp.broadcast_to(array, (n_channels,))


def polya_gamma_sufficient_statistics(
    plan: PolyaGammaObservationPlan,
    context,
    observations: jnp.ndarray,
    dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return PG shape, kappa, and affine predictor offset for each observation cell."""
    clean_observations = jnp.nan_to_num(observations, nan=0.0).astype(dtype)
    n_channels = int(observations.shape[-1])
    bernoulli_mask = plan.bernoulli_channel_mask[None, :]
    negative_binomial_mask = plan.negative_binomial_channel_mask[None, :]

    bernoulli_shape = jnp.ones_like(clean_observations)
    bernoulli_kappa = clean_observations - 0.5
    bernoulli_offset = jnp.zeros_like(clean_observations)

    r = _channel_param(context, "obs_r", default=5.0, n_channels=n_channels, dtype=dtype)
    r_rows = r[None, :]
    negative_binomial_shape = clean_observations + r_rows
    negative_binomial_kappa = 0.5 * (clean_observations - r_rows)
    negative_binomial_offset = -jnp.log(r_rows)

    shape = jnp.where(
        negative_binomial_mask,
        negative_binomial_shape,
        jnp.where(bernoulli_mask, bernoulli_shape, jnp.ones_like(clean_observations)),
    )
    kappa = jnp.where(
        negative_binomial_mask,
        negative_binomial_kappa,
        jnp.where(bernoulli_mask, bernoulli_kappa, 0.0),
    )
    linear_offset = jnp.where(
        negative_binomial_mask,
        negative_binomial_offset,
        jnp.where(bernoulli_mask, bernoulli_offset, 0.0),
    )
    return shape.astype(dtype), kappa.astype(dtype), linear_offset.astype(dtype)


def _empty_gamma_base_terms(reference: jnp.ndarray, *, num_terms: int) -> jnp.ndarray:
    return jnp.zeros((*reference.shape, int(num_terms)), dtype=reference.dtype)


def negative_binomial_finite_sum_base_log_terms(
    observations: jnp.ndarray,
    obs_r: jnp.ndarray,
    gamma_base_terms: jnp.ndarray,
    active_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Return NB PG finite-sum terms that depend on the dispersion parameter."""
    dtype = observations.dtype
    active = active_mask.astype(bool)
    y = jnp.nan_to_num(observations, nan=0.0).astype(dtype)
    r = jnp.broadcast_to(jnp.asarray(obs_r, dtype=dtype), y.shape)
    safe_r = jnp.maximum(r, jnp.asarray(1e-8, dtype=dtype))
    shape = y + safe_r
    combinatorial = jax.lax.lgamma(shape) - jax.lax.lgamma(safe_r) - jax.lax.lgamma(y + 1.0)
    nb_terms = combinatorial - shape * jnp.log(jnp.asarray(2.0, dtype=dtype))
    if gamma_base_terms.shape[-1] > 0:
        safe_shape = jnp.where(active, shape, jnp.ones_like(shape))
        safe_terms = jnp.where(
            active[..., None],
            jnp.maximum(gamma_base_terms.astype(dtype), jnp.asarray(1e-30, dtype=dtype)),
            jnp.ones_like(gamma_base_terms, dtype=dtype),
        )
        gamma_terms = (
            (safe_shape[..., None] - 1.0) * jnp.log(safe_terms)
            - safe_terms
            - jax.lax.lgamma(safe_shape[..., None])
        )
        nb_terms = nb_terms + jnp.sum(gamma_terms, axis=-1)
    return jnp.where(active, nb_terms, 0.0)


def polya_gamma_gaussian_logpdf_correction(
    kappa: jnp.ndarray,
    omega: jnp.ndarray,
    active_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Convert Gaussian pseudo-observation logpdf terms back to PG quadratics."""
    dtype = omega.dtype
    omega_safe = jnp.maximum(omega, jnp.asarray(1e-8, dtype=dtype))
    correction = (
        0.5 * kappa * kappa / omega_safe
        - 0.5 * jnp.log(omega_safe)
        + 0.5 * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=dtype))
    )
    return jnp.where(active_mask.astype(bool), correction, 0.0)


def _current_sufficient_statistics(
    plan: PolyaGammaObservationPlan,
    state: PolyaGammaAuxiliaryState,
    context,
    dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return polya_gamma_sufficient_statistics(
        plan,
        context,
        state.observed_values,
        dtype,
    )


def _negative_binomial_auxiliary_log_terms(
    plan: PolyaGammaObservationPlan,
    state: PolyaGammaAuxiliaryState,
    context,
    dtype,
) -> jnp.ndarray:
    n_channels = int(state.observed_values.shape[-1])
    obs_r = _channel_param(context, "obs_r", default=5.0, n_channels=n_channels, dtype=dtype)
    active_nb = (
        (state.active_mask > 0.0)
        & plan.negative_binomial_channel_mask[None, :]
        & plan.channel_mask[None, :]
    )
    return negative_binomial_finite_sum_base_log_terms(
        state.observed_values.astype(dtype),
        obs_r[None, :],
        state.gamma_base_terms.astype(dtype),
        active_nb,
    )


def initialize_polya_gamma_auxiliary_state(
    plan: PolyaGammaObservationPlan,
    context,
    latent_trajectory: jnp.ndarray,
    observations: jnp.ndarray,
) -> PolyaGammaAuxiliaryState:
    """Build a deterministic valid initial PG state from conditional means."""
    active = active_polya_gamma_mask(plan, observations)
    clean_observations = jnp.nan_to_num(observations, nan=0.0).astype(latent_trajectory.dtype)
    shape, kappa, linear_offset = polya_gamma_sufficient_statistics(
        plan,
        context,
        observations,
        latent_trajectory.dtype,
    )
    eta = _linear_predictor_trajectory(context, latent_trajectory)
    psi = eta + linear_offset
    if plan.sampler == "truncated_sum":
        omega, gamma_base_terms = _truncated_sum_conditional_mean_terms(
            shape,
            psi,
            num_terms=plan.num_terms,
        )
    else:
        omega = expected_polya_gamma(shape, psi)
        gamma_base_terms = _empty_gamma_base_terms(omega, num_terms=0)
    omega = jnp.where(active, omega, 0.0)
    gamma_base_terms = jnp.where(active[..., None], gamma_base_terms, 0.0)
    return PolyaGammaAuxiliaryState(
        omega=omega.astype(latent_trajectory.dtype),
        active_mask=active.astype(latent_trajectory.dtype),
        kappa=kappa.astype(latent_trajectory.dtype),
        shape=shape.astype(latent_trajectory.dtype),
        linear_offset=linear_offset.astype(latent_trajectory.dtype),
        observed_values=clean_observations.astype(latent_trajectory.dtype),
        gamma_base_terms=gamma_base_terms.astype(latent_trajectory.dtype),
    )


def refresh_polya_gamma_auxiliary_state(
    key: jax.Array,
    plan: PolyaGammaObservationPlan,
    context,
    latent_trajectory: jnp.ndarray,
    observations: jnp.ndarray,
) -> PolyaGammaAuxiliaryState:
    """Refresh omega once from the current trajectory and parameters."""
    active = active_polya_gamma_mask(plan, observations)
    clean_observations = jnp.nan_to_num(observations, nan=0.0).astype(latent_trajectory.dtype)
    shape, kappa, linear_offset = polya_gamma_sufficient_statistics(
        plan,
        context,
        observations,
        latent_trajectory.dtype,
    )
    eta = _linear_predictor_trajectory(context, latent_trajectory)
    psi = eta + linear_offset
    if plan.sampler == "truncated_sum":
        omega, gamma_base_terms = sample_polya_gamma_truncated_sum_base_terms(
            key,
            shape,
            psi,
            num_terms=plan.num_terms,
        )
    else:
        omega = sample_polya_gamma(
            key,
            shape,
            psi,
            sampler=plan.sampler,
            num_terms=plan.num_terms,
            max_integer_shape=plan.max_integer_shape,
        )
        gamma_base_terms = _empty_gamma_base_terms(omega, num_terms=0)
    omega = jnp.where(active, omega, 0.0)
    gamma_base_terms = jnp.where(active[..., None], gamma_base_terms, 0.0)
    return PolyaGammaAuxiliaryState(
        omega=omega.astype(latent_trajectory.dtype),
        active_mask=active.astype(latent_trajectory.dtype),
        kappa=kappa.astype(latent_trajectory.dtype),
        shape=shape.astype(latent_trajectory.dtype),
        linear_offset=linear_offset.astype(latent_trajectory.dtype),
        observed_values=clean_observations.astype(latent_trajectory.dtype),
        gamma_base_terms=gamma_base_terms.astype(latent_trajectory.dtype),
    )


def polya_gamma_quadratic_log_prob(
    plan: PolyaGammaObservationPlan,
    state: PolyaGammaAuxiliaryState,
    context,
    latent_trajectory: jnp.ndarray,
) -> jnp.ndarray:
    """Return the eta-dependent PG joint log-kernel."""
    eta = _linear_predictor_trajectory(context, latent_trajectory)
    _shape, kappa, linear_offset = _current_sufficient_statistics(
        plan,
        state,
        context,
        latent_trajectory.dtype,
    )
    psi = eta + linear_offset
    active = state.active_mask * plan.channel_mask[None, :].astype(latent_trajectory.dtype)
    terms = active * (kappa * psi - 0.5 * state.omega * psi * psi)
    terms = terms + _negative_binomial_auxiliary_log_terms(
        plan,
        state,
        context,
        latent_trajectory.dtype,
    )
    return jnp.asarray(jnp.sum(terms), dtype=latent_trajectory.dtype)


def polya_gamma_quadratic_log_probs(
    plan: PolyaGammaObservationPlan,
    state: PolyaGammaAuxiliaryState,
    context,
    latent_trajectory: jnp.ndarray,
) -> jnp.ndarray:
    """Return per-time eta-dependent PG joint log-kernels."""
    eta = _linear_predictor_trajectory(context, latent_trajectory)
    _shape, kappa, linear_offset = _current_sufficient_statistics(
        plan,
        state,
        context,
        latent_trajectory.dtype,
    )
    psi = eta + linear_offset
    active = state.active_mask * plan.channel_mask[None, :].astype(latent_trajectory.dtype)
    terms = active * (kappa * psi - 0.5 * state.omega * psi * psi)
    terms = terms + _negative_binomial_auxiliary_log_terms(
        plan,
        state,
        context,
        latent_trajectory.dtype,
    )
    return jnp.asarray(jnp.sum(terms, axis=-1), dtype=latent_trajectory.dtype)


def polya_gamma_increment_log_prob(
    plan: PolyaGammaObservationPlan,
    state: PolyaGammaAuxiliaryState,
    context,
    latent_state: jnp.ndarray,
    time_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Return the PG log-kernel contribution for one model row."""
    eta_t = _linear_predictor_at(context, latent_state, time_idx)
    _shape, kappa, linear_offset = _current_sufficient_statistics(
        plan,
        state,
        context,
        latent_state.dtype,
    )
    psi_t = eta_t + linear_offset[time_idx]
    mask_t = state.active_mask[time_idx] * plan.channel_mask.astype(latent_state.dtype)
    kappa_t = kappa[time_idx]
    omega_t = state.omega[time_idx]
    terms = mask_t * (kappa_t * psi_t - 0.5 * omega_t * psi_t * psi_t)
    terms = (
        terms
        + _negative_binomial_auxiliary_log_terms(
            plan,
            state,
            context,
            latent_state.dtype,
        )[time_idx]
    )
    return jnp.asarray(jnp.sum(terms), dtype=latent_state.dtype)
