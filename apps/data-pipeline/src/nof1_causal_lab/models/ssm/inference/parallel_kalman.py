"""Cuthbert-backed Kalman filter and RTS sampler wrappers for aux-Gibbs.

``aux_gibbs`` needs a compact interface that works with covariance matrices
and returns filtered covariances/log-likelihood increments. Cuthbert's public
Kalman API works in square-root form and returns cumulative log normalizers.
This module is the adapter between those two interfaces.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from cuthbert.filtering import filter as cuthbert_filter
from cuthbert.gaussian.kalman import build_filter as build_cuthbert_kalman_filter
from cuthbertlib.kalman import sampling as cuthbert_sampling

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter

_DEFAULT_JITTER = 1e-6


class ParallelFilterState(NamedTuple):
    """Per-step moments returned by :func:`filter_lgssm`."""

    pred_mean: jnp.ndarray
    pred_cov: jnp.ndarray
    filt_mean: jnp.ndarray
    filt_cov: jnp.ndarray
    loglik: jnp.ndarray


class AuxiliaryFilterState(NamedTuple):
    """Filtered moments for the identity-observation auxiliary LGSSM."""

    filt_mean: jnp.ndarray
    filt_cov: jnp.ndarray
    loglik: jnp.ndarray


def _get_init_params(model_inputs):
    return model_inputs["m0"], model_inputs["chol_P0"]


def _get_dynamics_params(model_inputs):
    return model_inputs["F"], model_inputs["b"], model_inputs["chol_Q"]


def _get_observation_params(model_inputs):
    return model_inputs["H"], model_inputs["c"], model_inputs["chol_R"], model_inputs["y"]


def _cholesky_from_covariance(cov: jnp.ndarray, *, jitter: float) -> jnp.ndarray:
    return jnp.linalg.cholesky(symmetrize_with_jitter(cov, jitter=jitter))


def _cholesky_stack(covs: jnp.ndarray, *, jitter: float) -> jnp.ndarray:
    return jax.vmap(lambda cov: _cholesky_from_covariance(cov, jitter=jitter))(covs)


def _covariances_from_cholesky(chols: jnp.ndarray) -> jnp.ndarray:
    return symmetrize(chols @ jnp.swapaxes(chols, -1, -2))


def _prepend_init_slot(steps: jnp.ndarray) -> jnp.ndarray:
    head = jnp.zeros((1, *steps.shape[1:]), dtype=steps.dtype)
    return jnp.concatenate([head, steps], axis=0)


def _coerce_time_vectors(
    name: str,
    values: jnp.ndarray,
    *,
    T: int,
    dim: int,
    dtype,
) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=dtype)
    if values.ndim == 0:
        if dim != 1:
            raise ValueError(f"{name} must have shape (T, D); scalar is only valid for D=1.")
        return jnp.broadcast_to(values, (T, 1))
    if values.ndim == 1:
        if dim != 1:
            raise ValueError(f"{name} must have shape (T, D) when D={dim}.")
        if values.shape != (T,):
            raise ValueError(f"{name} must have shape (T,) for D=1; got {values.shape}.")
        return values[:, None]
    if values.shape != (T, dim):
        raise ValueError(f"{name} must have shape {(T, dim)}; got {values.shape}.")
    return values


def _build_cuthbert_model_inputs(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
    Hs: jnp.ndarray,
    Rs: jnp.ndarray,
    cs: jnp.ndarray,
    ys: jnp.ndarray,
    *,
    jitter: float,
) -> dict[str, jnp.ndarray]:
    dtype = jnp.result_type(init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys)
    init_mean = jnp.asarray(init_mean, dtype=dtype)
    init_cov = jnp.asarray(init_cov, dtype=dtype)
    Fs = jnp.asarray(Fs, dtype=dtype)
    Qs = jnp.asarray(Qs, dtype=dtype)
    bs = jnp.asarray(bs, dtype=dtype)
    Hs = jnp.asarray(Hs, dtype=dtype)
    Rs = jnp.asarray(Rs, dtype=dtype)
    cs = jnp.asarray(cs, dtype=dtype)
    ys = jnp.asarray(ys, dtype=dtype)

    T = ys.shape[0]
    state_dim = init_mean.shape[0]
    obs_dim = ys.shape[-1]
    bs = _coerce_time_vectors("bs", bs, T=T, dim=state_dim, dtype=dtype)
    cs = _coerce_time_vectors("cs", cs, T=T, dim=obs_dim, dtype=dtype)
    chol_P0 = _cholesky_from_covariance(init_cov, jitter=jitter)
    chol_Qs = _cholesky_stack(Qs, jitter=jitter)
    chol_Rs = _cholesky_stack(Rs, jitter=jitter)

    return {
        "m0": jnp.broadcast_to(init_mean, (T + 1, state_dim)),
        "chol_P0": jnp.broadcast_to(chol_P0, (T + 1, state_dim, state_dim)),
        "F": _prepend_init_slot(Fs),
        "b": _prepend_init_slot(bs),
        "chol_Q": _prepend_init_slot(chol_Qs),
        "H": _prepend_init_slot(Hs),
        "c": _prepend_init_slot(cs),
        "chol_R": _prepend_init_slot(chol_Rs),
        "y": _prepend_init_slot(ys),
    }


def _run_cuthbert_filter(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
    Hs: jnp.ndarray,
    Rs: jnp.ndarray,
    cs: jnp.ndarray,
    ys: jnp.ndarray,
    *,
    jitter: float,
    parallel: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model_inputs = _build_cuthbert_model_inputs(
        init_mean,
        init_cov,
        Fs,
        Qs,
        bs,
        Hs,
        Rs,
        cs,
        ys,
        jitter=jitter,
    )
    filter_obj = build_cuthbert_kalman_filter(
        get_init_params=_get_init_params,
        get_dynamics_params=_get_dynamics_params,
        get_observation_params=_get_observation_params,
    )
    states = cuthbert_filter(filter_obj, model_inputs, parallel=parallel)
    filt_means = states.mean[1:]
    filt_covs = _covariances_from_cholesky(states.chol_cov[1:])
    cumulative_loglik = states.log_normalizing_constant[1:]
    loglik = jnp.diff(
        cumulative_loglik,
        prepend=jnp.zeros((1,), dtype=cumulative_loglik.dtype),
    )
    return filt_means, filt_covs, loglik


def _prediction_moments(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
    filt_means: jnp.ndarray,
    filt_covs: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    init_pred_mean = Fs[0] @ init_mean + bs[0]
    init_pred_cov = symmetrize_with_jitter(Fs[0] @ init_cov @ Fs[0].T + Qs[0], jitter=jitter)
    if Fs.shape[0] == 1:
        return init_pred_mean[None, ...], init_pred_cov[None, ...]

    pred_mean_tail = jax.vmap(lambda F, m, b: F @ m + b)(
        Fs[1:],
        filt_means[:-1],
        bs[1:],
    )
    pred_cov_tail = jax.vmap(
        lambda F, P, Q: symmetrize_with_jitter(F @ P @ F.T + Q, jitter=jitter)
    )(Fs[1:], filt_covs[:-1], Qs[1:])
    pred_means = jnp.concatenate([init_pred_mean[None, ...], pred_mean_tail], axis=0)
    pred_covs = jnp.concatenate([init_pred_cov[None, ...], pred_cov_tail], axis=0)
    return pred_means, pred_covs


def filter_lgssm(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
    Hs: jnp.ndarray,
    Rs: jnp.ndarray,
    cs: jnp.ndarray,
    ys: jnp.ndarray,
    *,
    jitter: float = _DEFAULT_JITTER,
    propagate_first: bool = True,
    parallel: bool = True,
) -> ParallelFilterState:
    """Kalman filter for a general LGSSM using cuthbert's square-root filter."""
    if not propagate_first:
        raise NotImplementedError("filter_lgssm requires propagate_first=True.")

    filt_means, filt_covs, loglik = _run_cuthbert_filter(
        init_mean,
        init_cov,
        Fs,
        Qs,
        bs,
        Hs,
        Rs,
        cs,
        ys,
        jitter=jitter,
        parallel=parallel,
    )
    pred_means, pred_covs = _prediction_moments(
        init_mean,
        init_cov,
        Fs,
        Qs,
        bs,
        filt_means,
        filt_covs,
        jitter=jitter,
    )
    return ParallelFilterState(
        pred_mean=pred_means,
        pred_cov=pred_covs,
        filt_mean=filt_means,
        filt_cov=filt_covs,
        loglik=loglik,
    )


def _identity_observation_inputs(
    pseudo_observations: jnp.ndarray,
    aux_variance: jnp.ndarray,
    *,
    dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    T = pseudo_observations.shape[0]
    state_dim = pseudo_observations.shape[-1]
    Hs = jnp.broadcast_to(jnp.eye(state_dim, dtype=dtype), (T, state_dim, state_dim))
    aux_variance_arr = jnp.asarray(aux_variance, dtype=dtype)
    if aux_variance_arr.ndim == 0:
        variance = jnp.broadcast_to(aux_variance_arr, (T,))
    elif aux_variance_arr.shape == (T,):
        variance = aux_variance_arr
    else:
        raise ValueError(
            f"aux_variance must be scalar or shape (T,); got {aux_variance_arr.shape} with T={T}."
        )
    Rs = variance[:, None, None] * jnp.eye(state_dim, dtype=dtype)[None, :, :]
    cs = jnp.zeros((T, state_dim), dtype=dtype)
    return Hs, Rs, cs


def aux_filter_lgssm(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
    pseudo_observations: jnp.ndarray,
    aux_variance: jnp.ndarray,
    *,
    jitter: float = _DEFAULT_JITTER,
    parallel: bool = True,
) -> ParallelFilterState:
    """Kalman filter for the identity-observation auxiliary LGSSM."""
    dtype = jnp.result_type(init_mean, init_cov, Fs, Qs, bs, pseudo_observations, aux_variance)
    pseudo_observations = jnp.asarray(pseudo_observations, dtype=dtype)
    Hs, Rs, cs = _identity_observation_inputs(
        pseudo_observations,
        aux_variance,
        dtype=dtype,
    )
    return filter_lgssm(
        jnp.asarray(init_mean, dtype=dtype),
        jnp.asarray(init_cov, dtype=dtype),
        jnp.asarray(Fs, dtype=dtype),
        jnp.asarray(Qs, dtype=dtype),
        jnp.asarray(bs, dtype=dtype),
        Hs,
        Rs,
        cs,
        pseudo_observations,
        jitter=jitter,
        parallel=parallel,
    )


def aux_filter_lgssm_lightweight(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
    pseudo_observations: jnp.ndarray,
    aux_variance: jnp.ndarray,
    *,
    jitter: float = _DEFAULT_JITTER,
    parallel: bool = True,
) -> AuxiliaryFilterState:
    """Auxiliary cuthbert filter returning only filtered moments and log-likelihood."""
    dtype = jnp.result_type(init_mean, init_cov, Fs, Qs, bs, pseudo_observations, aux_variance)
    pseudo_observations = jnp.asarray(pseudo_observations, dtype=dtype)
    Hs, Rs, cs = _identity_observation_inputs(
        pseudo_observations,
        aux_variance,
        dtype=dtype,
    )
    filt_means, filt_covs, loglik = _run_cuthbert_filter(
        jnp.asarray(init_mean, dtype=dtype),
        jnp.asarray(init_cov, dtype=dtype),
        jnp.asarray(Fs, dtype=dtype),
        jnp.asarray(Qs, dtype=dtype),
        jnp.asarray(bs, dtype=dtype),
        Hs,
        Rs,
        cs,
        pseudo_observations,
        jitter=jitter,
        parallel=parallel,
    )
    return AuxiliaryFilterState(
        filt_mean=filt_means,
        filt_cov=filt_covs,
        loglik=loglik,
    )


def sample_lgssm_trajectory(
    key: jnp.ndarray,
    filt_means: jnp.ndarray,
    filt_covs: jnp.ndarray,
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
    *,
    jitter: float = _DEFAULT_JITTER,
    parallel: bool = True,
) -> jnp.ndarray:
    """Sample a trajectory from the LGSSM smoothing distribution via cuthbert."""
    del parallel

    chol_filt_covs = _cholesky_stack(filt_covs, jitter=jitter)
    chol_Qs = _cholesky_stack(Qs, jitter=jitter)
    elems = cuthbert_sampling.sqrt_associative_params(
        key,
        filt_means,
        chol_filt_covs,
        Fs,
        bs,
        chol_Qs,
        shape=(),
    )
    _gains, samples = jax.lax.associative_scan(
        jax.vmap(cuthbert_sampling.sampling_operator),
        elems,
        reverse=True,
    )
    return samples


aux_sample_lgssm_trajectory = sample_lgssm_trajectory


__all__ = [
    "AuxiliaryFilterState",
    "ParallelFilterState",
    "filter_lgssm",
    "aux_filter_lgssm",
    "aux_filter_lgssm_lightweight",
    "sample_lgssm_trajectory",
    "aux_sample_lgssm_trajectory",
]
