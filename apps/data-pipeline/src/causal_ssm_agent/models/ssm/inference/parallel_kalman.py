"""Parallel-in-time associative Kalman filter and RTS sampler for LGSSMs.

Implements the O(log T) prefix-sum filter of Särkkä & García-Fernández (2021)
and the associative RTS sampling operator described in Corenflos & Särkkä
(2025, Appendix A.2). Both primitives run ``jax.lax.associative_scan`` on
per-step elements built in parallel, so a full filter/smoother costs
O(log T) sequential steps on parallel hardware while still matching the
sequential Kalman filter to numerical precision.

The module exposes two layers:

* :func:`filter_lgssm` and :func:`sample_lgssm_trajectory` accept general
  per-step dynamics ``(F_t, Q_t, b_t)``, per-step observation
  ``(H_t, R_t, c_t, y_t)`` and run the parallel scan. They are generic
  enough to be reused from any LGSSM code path (point-in-time, augmented
  interval summaries, block-diagonal decomposable states).
* :func:`aux_filter_lgssm` / :func:`aux_sample_lgssm_trajectory` specialise
  the generic version for auxiliary LGSSMs with identity observations
  ``y_t = x_t + eps`` and ``eps ~ N(0, (delta/2) I)``. ``aux_gibbs`` uses
  this both for the eq-8 reparametrised proposal (raw pseudo-observations
  ``u_t``) and for the eq-10/11 non-reparametrised proposal (shifted
  pseudo-observations ``u_t + (delta/2) v_t``). Keeping the specialisation on
  top of the generic core avoids duplicating the associative operators while
  still letting the aux sampler skip a few matrix multiplies when ``H = I``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from causal_ssm_agent.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter

_DEFAULT_JITTER = 1e-6


class ParallelFilterState(NamedTuple):
    """Per-step moments returned by :func:`filter_lgssm`."""

    pred_mean: jnp.ndarray
    pred_cov: jnp.ndarray
    filt_mean: jnp.ndarray
    filt_cov: jnp.ndarray
    loglik: jnp.ndarray


def _filter_op(elem1, elem2):
    return _filter_op_one(*elem1, *elem2)


def _filter_op_one(A1, b1, C1, eta1, J1, A2, b2, C2, eta2, J2):
    """Corenflos/Särkkä filter associative operator in ``(A, b, C, eta, J)``."""
    state_dim = b1.shape[0]
    eye = jnp.eye(state_dim, dtype=b1.dtype)
    ip_cj = eye + C1 @ J2
    ip_jc = eye + J2 @ C1
    a_ip_cj_inv = jnp.linalg.solve(ip_cj.T, A2.T).T
    a_ip_jc_inv = jnp.linalg.solve(ip_jc.T, A1).T

    A = a_ip_cj_inv @ A1
    b = a_ip_cj_inv @ (b1 + C1 @ eta2) + b2
    C = a_ip_cj_inv @ C1 @ A2.T + C2
    eta = a_ip_jc_inv @ (eta2 - J2 @ b1) + eta1
    J = a_ip_jc_inv @ J2 @ A1 + J1
    return A, b, symmetrize(C), eta, symmetrize(J)


def _filter_init_one(
    F: jnp.ndarray,
    Q: jnp.ndarray,
    b: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    c: jnp.ndarray,
    y: jnp.ndarray,
    m: jnp.ndarray,
    P: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Associative-filter initial element for a generic observation ``H m + c``."""
    m_pred = F @ m + b
    P_pred = symmetrize_with_jitter(F @ P @ F.T + Q, jitter=jitter)
    S = symmetrize_with_jitter(H @ P_pred @ H.T + R, jitter=jitter)
    chol_S = jnp.linalg.cholesky(S)
    S_inv_H = jla.cho_solve((chol_S, True), H)
    K = P_pred @ S_inv_H.T
    A = F - K @ H @ F
    innov_m = y - (H @ m_pred + c)
    b_std = m_pred + K @ innov_m
    C = P_pred - K @ S @ K.T
    temp = (S_inv_H @ F).T
    innov_b = y - (H @ b + c)
    eta = temp @ innov_b
    J = temp @ H @ F
    return A, b_std, symmetrize(C), eta, symmetrize(J)


def _filter_init_identity_obs(
    F: jnp.ndarray,
    Q: jnp.ndarray,
    b: jnp.ndarray,
    y: jnp.ndarray,
    m: jnp.ndarray,
    P: jnp.ndarray,
    R: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Associative-filter initial element specialised to ``H = I, c = 0``."""
    m_pred = F @ m + b
    P_pred = symmetrize_with_jitter(F @ P @ F.T + Q, jitter=jitter)
    S = symmetrize_with_jitter(P_pred + R, jitter=jitter)
    chol_S = jnp.linalg.cholesky(S)
    gain = jla.cho_solve((chol_S, True), P_pred).T
    A = F - gain @ F
    b_std = m_pred + gain @ (y - m_pred)
    C = P_pred - gain @ S @ gain.T
    temp = jla.cho_solve((chol_S, True), F).T
    eta = temp @ (y - b)
    J = temp @ F
    return A, b_std, symmetrize(C), eta, symmetrize(J)


def _kalman_predict(
    m: jnp.ndarray, P: jnp.ndarray, F: jnp.ndarray, Q: jnp.ndarray, c: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return F @ m + c, symmetrize(F @ P @ F.T + Q)


def _kalman_update(
    m_pred: jnp.ndarray,
    P_pred: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    c: jnp.ndarray,
    y: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    innov = y - (H @ m_pred + c)
    S = symmetrize_with_jitter(H @ P_pred @ H.T + R, jitter=jitter)
    chol_S = jnp.linalg.cholesky(S)
    gain = jla.cho_solve((chol_S, True), H @ P_pred).T
    m_upd = m_pred + gain @ innov
    P_upd = symmetrize(P_pred - gain @ S @ gain.T)
    whitened = jla.solve_triangular(chol_S, innov, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_S)))
    dim = innov.shape[-1]
    loglik = -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + whitened @ whitened)
    return m_upd, P_upd, loglik


def _kalman_update_identity_obs(
    m_pred: jnp.ndarray,
    P_pred: jnp.ndarray,
    R: jnp.ndarray,
    y: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    innov = y - m_pred
    S = symmetrize_with_jitter(P_pred + R, jitter=jitter)
    chol_S = jnp.linalg.cholesky(S)
    gain = jla.cho_solve((chol_S, True), P_pred).T
    m_upd = m_pred + gain @ innov
    P_upd = symmetrize(P_pred - gain @ S @ gain.T)
    whitened = jla.solve_triangular(chol_S, innov, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_S)))
    dim = innov.shape[-1]
    loglik = -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + whitened @ whitened)
    return m_upd, P_upd, loglik


def _sequential_filter_lgssm(
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
) -> ParallelFilterState:
    """Plain O(T) sequential Kalman filter used when ``parallel=False``.

    Same predict/update primitives as the parallel path so the two agree to
    numerical precision — the only difference is that this version streams
    through ``jax.lax.scan`` instead of an associative scan, which trades the
    log-depth parallelism for a lighter per-step constant.
    """
    init_pred_mean = Fs[0] @ init_mean + bs[0]
    init_pred_cov = symmetrize_with_jitter(Fs[0] @ init_cov @ Fs[0].T + Qs[0], jitter=jitter)
    filt_mean_0, filt_cov_0, loglik_0 = _kalman_update(
        init_pred_mean, init_pred_cov, Hs[0], Rs[0], cs[0], ys[0], jitter=jitter
    )

    def _step(carry, inputs):
        mean_prev, cov_prev = carry
        F_t, Q_t, b_t, H_t, R_t, c_t, y_t = inputs
        m_pred, P_pred = _kalman_predict(mean_prev, cov_prev, F_t, Q_t, b_t)
        P_pred = symmetrize_with_jitter(P_pred, jitter=jitter)
        m_upd, P_upd, ll = _kalman_update(m_pred, P_pred, H_t, R_t, c_t, y_t, jitter=jitter)
        return (m_upd, P_upd), (m_pred, P_pred, m_upd, P_upd, ll)

    (_fm, _fc), history = jax.lax.scan(
        _step,
        (filt_mean_0, filt_cov_0),
        (Fs[1:], Qs[1:], bs[1:], Hs[1:], Rs[1:], cs[1:], ys[1:]),
    )
    pred_mean_tail, pred_cov_tail, filt_mean_tail, filt_cov_tail, loglik_tail = history
    pred_means = jnp.concatenate([init_pred_mean[None, ...], pred_mean_tail], axis=0)
    pred_covs = jnp.concatenate([init_pred_cov[None, ...], pred_cov_tail], axis=0)
    filt_means = jnp.concatenate([filt_mean_0[None, ...], filt_mean_tail], axis=0)
    filt_covs = jnp.concatenate([filt_cov_0[None, ...], filt_cov_tail], axis=0)
    loglik = jnp.concatenate([loglik_0[None, ...], loglik_tail], axis=0)
    return ParallelFilterState(
        pred_mean=pred_means,
        pred_cov=pred_covs,
        filt_mean=filt_means,
        filt_cov=filt_covs,
        loglik=loglik,
    )


def _sequential_aux_filter_lgssm(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
    pseudo_observations: jnp.ndarray,
    R_aux_per_t: jnp.ndarray,
    *,
    jitter: float,
) -> ParallelFilterState:
    """O(T) sequential Kalman filter for the identity-observation auxiliary LGSSM.

    ``R_aux_per_t`` has shape ``(T, D, D)`` so callers can specify per-time-step
    auxiliary observation noise (e.g. for the δ_t stabilisation strategies).
    """
    init_pred_mean = Fs[0] @ init_mean + bs[0]
    init_pred_cov = symmetrize_with_jitter(Fs[0] @ init_cov @ Fs[0].T + Qs[0], jitter=jitter)
    filt_mean_0, filt_cov_0, loglik_0 = _kalman_update_identity_obs(
        init_pred_mean, init_pred_cov, R_aux_per_t[0], pseudo_observations[0], jitter=jitter
    )

    def _step(carry, inputs):
        mean_prev, cov_prev = carry
        F_t, Q_t, b_t, y_t, R_t = inputs
        m_pred, P_pred = _kalman_predict(mean_prev, cov_prev, F_t, Q_t, b_t)
        P_pred = symmetrize_with_jitter(P_pred, jitter=jitter)
        m_upd, P_upd, ll = _kalman_update_identity_obs(m_pred, P_pred, R_t, y_t, jitter=jitter)
        return (m_upd, P_upd), (m_pred, P_pred, m_upd, P_upd, ll)

    (_fm, _fc), history = jax.lax.scan(
        _step,
        (filt_mean_0, filt_cov_0),
        (Fs[1:], Qs[1:], bs[1:], pseudo_observations[1:], R_aux_per_t[1:]),
    )
    pred_mean_tail, pred_cov_tail, filt_mean_tail, filt_cov_tail, loglik_tail = history
    pred_means = jnp.concatenate([init_pred_mean[None, ...], pred_mean_tail], axis=0)
    pred_covs = jnp.concatenate([init_pred_cov[None, ...], pred_cov_tail], axis=0)
    filt_means = jnp.concatenate([filt_mean_0[None, ...], filt_mean_tail], axis=0)
    filt_covs = jnp.concatenate([filt_cov_0[None, ...], filt_cov_tail], axis=0)
    loglik = jnp.concatenate([loglik_0[None, ...], loglik_tail], axis=0)
    return ParallelFilterState(
        pred_mean=pred_means,
        pred_cov=pred_covs,
        filt_mean=filt_means,
        filt_cov=filt_covs,
        loglik=loglik,
    )


def _sequential_sample_lgssm_trajectory(
    key: jnp.ndarray,
    filt_means: jnp.ndarray,
    filt_covs: jnp.ndarray,
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
    *,
    jitter: float,
) -> jnp.ndarray:
    """Plain backward RTS sampler used when ``parallel=False``.

    Indexing follows :func:`sample_lgssm_trajectory`: ``Fs/Qs/bs`` describe
    transitions from time ``t-1`` to ``t`` and have length ``T - 1``.
    """
    T = filt_means.shape[0]
    dtype = filt_means.dtype
    keys = random.split(key, T)

    chol_last = jnp.linalg.cholesky(symmetrize_with_jitter(filt_covs[-1], jitter=jitter))
    last_sample = filt_means[-1] + chol_last @ random.normal(
        keys[-1], (filt_means.shape[-1],), dtype=dtype
    )

    if T == 1:
        return last_sample[None, ...]

    def _backward(x_next, inputs):
        key_t, m_t, P_t, F_next, Q_next, b_next = inputs
        m_pred_next = F_next @ m_t + b_next
        P_pred_next = symmetrize_with_jitter(F_next @ P_t @ F_next.T + Q_next, jitter=jitter)
        gain = jla.solve(P_pred_next, F_next @ P_t, assume_a="pos").T
        smooth_mean = m_t + gain @ (x_next - m_pred_next)
        smooth_cov = symmetrize_with_jitter(P_t - gain @ P_pred_next @ gain.T, jitter=jitter)
        chol = jnp.linalg.cholesky(smooth_cov)
        x_t = smooth_mean + chol @ random.normal(key_t, x_next.shape, dtype=dtype)
        return x_t, x_t

    _, x_rev = jax.lax.scan(
        _backward,
        last_sample,
        (
            keys[:-1][::-1],
            filt_means[:-1][::-1],
            filt_covs[:-1][::-1],
            Fs[::-1],
            Qs[::-1],
            bs[::-1],
        ),
    )
    return jnp.concatenate([x_rev[::-1], last_sample[None, ...]], axis=0)


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
    """Kalman filter for a general LGSSM.

    When ``parallel=True`` this runs the Corenflos/Särkkä O(log T)
    associative-scan filter; when ``parallel=False`` it falls back to the
    plain O(T) sequential Kalman filter using ``jax.lax.scan``.

    The trajectory convention follows :mod:`trajectory_mcmc.auxiliary_kalman`:

    * ``Fs[t], Qs[t], bs[t]`` describe the transition ``x_{t-1} -> x_t``
      (with ``propagate_first=True`` the first step propagates the prior).
    * ``Hs[t], Rs[t], cs[t], ys[t]`` describe the observation at time ``t``.

    With ``propagate_first=True`` the returned ``pred_mean[0]`` and
    ``pred_cov[0]`` are ``F0 m0 + b0`` and ``F0 P0 F0.T + Q0`` respectively —
    matching the initial one-step-ahead prior used by ``_initial_latent_moments``
    in :mod:`trajectory_mcmc.auxiliary_kalman`.
    """
    if not parallel:
        if not propagate_first:
            raise NotImplementedError(
                "Sequential filter_lgssm currently only supports "
                "propagate_first=True (matches auxiliary_kalman)."
            )
        return _sequential_filter_lgssm(
            init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys, jitter=jitter
        )
    T = ys.shape[0]
    state_dim = init_mean.shape[0]
    dtype = init_mean.dtype

    if propagate_first:
        init_pred_mean = Fs[0] @ init_mean + bs[0]
        init_pred_cov = symmetrize_with_jitter(Fs[0] @ init_cov @ Fs[0].T + Qs[0], jitter=jitter)
    else:
        init_pred_mean = init_mean
        init_pred_cov = symmetrize_with_jitter(init_cov, jitter=jitter)

    filt_mean_0, filt_cov_0, loglik_0 = _kalman_update(
        init_pred_mean, init_pred_cov, Hs[0], Rs[0], cs[0], ys[0], jitter=jitter
    )

    if T == 1:
        return ParallelFilterState(
            pred_mean=init_pred_mean[None, ...],
            pred_cov=init_pred_cov[None, ...],
            filt_mean=filt_mean_0[None, ...],
            filt_cov=filt_cov_0[None, ...],
            loglik=loglik_0[None, ...],
        )

    ms = jnp.concatenate(
        [filt_mean_0[None, ...], jnp.zeros((T - 2, state_dim), dtype=dtype)], axis=0
    )
    Ps = jnp.concatenate(
        [
            filt_cov_0[None, ...],
            jnp.zeros((T - 2, state_dim, state_dim), dtype=dtype),
        ],
        axis=0,
    )
    init_elems = jax.vmap(
        lambda F, Q, b, H, R, c, y, m, P: _filter_init_one(F, Q, b, H, R, c, y, m, P, jitter=jitter)
    )(Fs[1:], Qs[1:], bs[1:], Hs[1:], Rs[1:], cs[1:], ys[1:], ms, Ps)
    _ops, filt_tail, filt_cov_tail, _eta, _J = jax.lax.associative_scan(
        jax.vmap(_filter_op), init_elems
    )
    filt_means = jnp.concatenate([filt_mean_0[None, ...], filt_tail], axis=0)
    filt_covs = jnp.concatenate([filt_cov_0[None, ...], filt_cov_tail], axis=0)
    pred_mean_tail, pred_cov_tail = jax.vmap(_kalman_predict)(
        filt_means[:-1], filt_covs[:-1], Fs[1:], Qs[1:], bs[1:]
    )
    pred_means = jnp.concatenate([init_pred_mean[None, ...], pred_mean_tail], axis=0)
    pred_covs = jnp.concatenate([init_pred_cov[None, ...], pred_cov_tail], axis=0)

    def _tail_loglik(m_pred, P_pred, H, R, c, y):
        return _kalman_update(m_pred, P_pred, H, R, c, y, jitter=jitter)[2]

    loglik_tail = jax.vmap(_tail_loglik)(
        pred_means[1:], pred_covs[1:], Hs[1:], Rs[1:], cs[1:], ys[1:]
    )
    loglik = jnp.concatenate([loglik_0[None, ...], loglik_tail], axis=0)
    return ParallelFilterState(
        pred_mean=pred_means,
        pred_cov=pred_covs,
        filt_mean=filt_means,
        filt_cov=filt_covs,
        loglik=loglik,
    )


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
    """Kalman filter for the identity-observation auxiliary LGSSM ``y_t = x_t + eps_t``.

    ``aux_variance`` is the isotropic variance of ``eps_t`` (i.e. ``delta/2``).
    It can be a Python/JAX scalar for a uniform-in-time δ, or a ``(T,)`` array
    for the per-time-step δ_t stabilisation strategies described in
    Corenflos & Särkkä §4.4.

    ``init_mean`` / ``init_cov`` are the raw prior moments; the first-step
    propagation matches ``_initial_latent_moments`` so the result is drop-in
    compatible with the existing auxiliary-Kalman kernel.

    When ``parallel=True`` this dispatches the Corenflos/Särkkä associative
    scan; when ``parallel=False`` it uses a plain sequential ``lax.scan``
    Kalman filter. Both paths share identical predict/update primitives so
    the filtering means/covariances agree to numerical precision.
    """
    state_dim = init_mean.shape[0]
    dtype = init_mean.dtype
    eye = jnp.eye(state_dim, dtype=dtype)
    T = pseudo_observations.shape[0]
    aux_variance_arr = jnp.asarray(aux_variance, dtype=dtype)
    if aux_variance_arr.ndim == 0:
        aux_variance_per_t = jnp.broadcast_to(aux_variance_arr, (T,))
    else:
        if aux_variance_arr.shape != (T,):
            raise ValueError(
                f"aux_variance must be scalar or shape (T,); got {aux_variance_arr.shape} with T={T}."
            )
        aux_variance_per_t = aux_variance_arr
    R_aux_per_t = aux_variance_per_t[:, None, None] * eye[None, :, :]
    if not parallel:
        return _sequential_aux_filter_lgssm(
            init_mean,
            init_cov,
            Fs,
            Qs,
            bs,
            pseudo_observations,
            R_aux_per_t,
            jitter=jitter,
        )

    init_pred_mean = Fs[0] @ init_mean + bs[0]
    init_pred_cov = symmetrize_with_jitter(Fs[0] @ init_cov @ Fs[0].T + Qs[0], jitter=jitter)

    filt_mean_0, filt_cov_0, loglik_0 = _kalman_update_identity_obs(
        init_pred_mean,
        init_pred_cov,
        R_aux_per_t[0],
        pseudo_observations[0],
        jitter=jitter,
    )

    if T == 1:
        return ParallelFilterState(
            pred_mean=init_pred_mean[None, ...],
            pred_cov=init_pred_cov[None, ...],
            filt_mean=filt_mean_0[None, ...],
            filt_cov=filt_cov_0[None, ...],
            loglik=loglik_0[None, ...],
        )

    ms = jnp.concatenate(
        [filt_mean_0[None, ...], jnp.zeros((T - 2, state_dim), dtype=dtype)], axis=0
    )
    Ps = jnp.concatenate(
        [
            filt_cov_0[None, ...],
            jnp.zeros((T - 2, state_dim, state_dim), dtype=dtype),
        ],
        axis=0,
    )
    init_elems = jax.vmap(
        lambda F, Q, b, y, m, P, R: _filter_init_identity_obs(F, Q, b, y, m, P, R, jitter=jitter)
    )(Fs[1:], Qs[1:], bs[1:], pseudo_observations[1:], ms, Ps, R_aux_per_t[1:])
    _ops, filt_tail, filt_cov_tail, _eta, _J = jax.lax.associative_scan(
        jax.vmap(_filter_op), init_elems
    )
    filt_means = jnp.concatenate([filt_mean_0[None, ...], filt_tail], axis=0)
    filt_covs = jnp.concatenate([filt_cov_0[None, ...], filt_cov_tail], axis=0)
    pred_mean_tail, pred_cov_tail = jax.vmap(_kalman_predict)(
        filt_means[:-1], filt_covs[:-1], Fs[1:], Qs[1:], bs[1:]
    )
    pred_means = jnp.concatenate([init_pred_mean[None, ...], pred_mean_tail], axis=0)
    pred_covs = jnp.concatenate([init_pred_cov[None, ...], pred_cov_tail], axis=0)

    def _tail_loglik(m_pred, P_pred, y, R):
        return _kalman_update_identity_obs(m_pred, P_pred, R, y, jitter=jitter)[2]

    loglik_tail = jax.vmap(_tail_loglik)(
        pred_means[1:], pred_covs[1:], pseudo_observations[1:], R_aux_per_t[1:]
    )
    loglik = jnp.concatenate([loglik_0[None, ...], loglik_tail], axis=0)
    return ParallelFilterState(
        pred_mean=pred_means,
        pred_cov=pred_covs,
        filt_mean=filt_means,
        filt_cov=filt_covs,
        loglik=loglik,
    )


def _sampling_op(elem1, elem2):
    return _sampling_op_one(*elem1, *elem2)


def _sampling_op_one(
    gain1: jnp.ndarray,
    increment1: jnp.ndarray,
    gain2: jnp.ndarray,
    increment2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return gain2 @ gain1, gain2 @ increment1 + increment2


def _sampling_init_one(
    F: jnp.ndarray,
    Q: jnp.ndarray,
    b: jnp.ndarray,
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    epsilon: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    S = symmetrize_with_jitter(F @ cov @ F.T + Q, jitter=jitter)
    chol_S = jnp.linalg.cholesky(S)
    gain = jla.cho_solve((chol_S, True), F @ cov.T).T
    increment_cov = symmetrize_with_jitter(cov - gain @ S @ gain.T, jitter=jitter)
    chol = jnp.linalg.cholesky(increment_cov)
    increment_mean = mean - gain @ (F @ mean + b)
    increment = increment_mean + chol @ epsilon
    return gain, increment


def _sample_last_step(
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    epsilon: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    chol = jnp.linalg.cholesky(symmetrize_with_jitter(cov, jitter=jitter))
    last_sample = mean + chol @ epsilon
    gain = jnp.zeros_like(cov)
    return gain, last_sample


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
    """RTS sampler given filter moments.

    ``Fs[t], Qs[t], bs[t]`` encode the transition from time ``t-1`` to time
    ``t`` and are expected to start at index ``1`` (i.e. ``Fs`` has shape
    ``(T-1, D, D)``). This matches how ``_sample_auxiliary_trajectory`` slices
    its context.

    ``parallel=True`` uses the O(log T) associative-scan RTS sampler;
    ``parallel=False`` uses a plain backward ``lax.scan``. Both paths draw
    samples from the same smoothing posterior and consume exactly ``T``
    random draws; only their runtime profiles differ.
    """
    if not parallel:
        return _sequential_sample_lgssm_trajectory(
            key, filt_means, filt_covs, Fs, Qs, bs, jitter=jitter
        )
    T = filt_means.shape[0]
    dtype = filt_means.dtype
    epsilons = random.normal(key, filt_means.shape, dtype=dtype)
    if T == 1:
        last_gain, last_increment = _sample_last_step(
            filt_means[0], filt_covs[0], epsilons[0], jitter=jitter
        )
        gains = last_gain[None, ...]
        increments = last_increment[None, ...]
    else:
        gains_head, incr_head = jax.vmap(
            lambda F, Q, b, m, P, eps: _sampling_init_one(F, Q, b, m, P, eps, jitter=jitter)
        )(Fs, Qs, bs, filt_means[:-1], filt_covs[:-1], epsilons[:-1])
        last_gain, last_increment = _sample_last_step(
            filt_means[-1], filt_covs[-1], epsilons[-1], jitter=jitter
        )
        gains = jnp.concatenate([gains_head, last_gain[None, ...]], axis=0)
        increments = jnp.concatenate([incr_head, last_increment[None, ...]], axis=0)

    _gains, samples = jax.lax.associative_scan(
        jax.vmap(_sampling_op),
        (gains, increments),
        reverse=True,
    )
    return samples


aux_sample_lgssm_trajectory = sample_lgssm_trajectory


__all__ = [
    "ParallelFilterState",
    "filter_lgssm",
    "aux_filter_lgssm",
    "sample_lgssm_trajectory",
    "aux_sample_lgssm_trajectory",
]
