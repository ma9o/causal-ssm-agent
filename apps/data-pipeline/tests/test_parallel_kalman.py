"""Correctness tests for the parallel-in-time Kalman filter/sampler.

These tests verify that the associative-scan parallel filter in
:mod:`causal_ssm_agent.models.ssm.inference.parallel_kalman` matches a plain
sequential Kalman filter to numerical precision on the three observation
paths exercised by ``aux_gibbs``:

1. ``test_filter_matches_sequential_point_in_time``: simple LGSSM with point
   observations at every time step.
2. ``test_filter_matches_sequential_interval_summary``: augmented-state
   LGSSM produced by ``build_linear_summary_augmented_system``, where the
   state contains latent coordinates plus running accumulators used by
   interval summaries.
3. ``test_filter_matches_sequential_block_diagonal``: LGSSM whose transition
   matrix is block-diagonal across independent subsystems — this stresses
   the associative operator on matrices with many exact zero blocks.

The sampler is tested by checking that the posterior-predictive moments of
many parallel samples agree with the analytical RTS smoothing posterior.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
import pytest

from causal_ssm_agent.models.ssm.covariance_utils import symmetrize_with_jitter
from causal_ssm_agent.models.ssm.inference.parallel_kalman import (
    aux_filter_lgssm,
    filter_lgssm,
    sample_lgssm_trajectory,
)

_JITTER = 1e-6


def _seq_kalman_predict(m, P, F, Q, b):
    return F @ m + b, 0.5 * (F @ P @ F.T + Q + (F @ P @ F.T + Q).T)


def _seq_kalman_update(m_pred, P_pred, H, R, c, y):
    innov = y - (H @ m_pred + c)
    S = symmetrize_with_jitter(H @ P_pred @ H.T + R, jitter=_JITTER)
    chol = jnp.linalg.cholesky(S)
    gain = jla.cho_solve((chol, True), H @ P_pred).T
    m_upd = m_pred + gain @ innov
    P_upd = P_pred - gain @ S @ gain.T
    P_upd = 0.5 * (P_upd + P_upd.T)
    whitened = jla.solve_triangular(chol, innov, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    dim = innov.shape[-1]
    loglik = -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + whitened @ whitened)
    return m_upd, P_upd, loglik


def _sequential_filter(init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys):
    T = ys.shape[0]
    m_pred0, P_pred0 = _seq_kalman_predict(init_mean, init_cov, Fs[0], Qs[0], bs[0])
    P_pred0 = symmetrize_with_jitter(P_pred0, jitter=_JITTER)
    m_f, P_f, ll0 = _seq_kalman_update(m_pred0, P_pred0, Hs[0], Rs[0], cs[0], ys[0])
    pred_means = [m_pred0]
    pred_covs = [P_pred0]
    filt_means = [m_f]
    filt_covs = [P_f]
    lls = [ll0]
    for t in range(1, T):
        m_p, P_p = _seq_kalman_predict(filt_means[-1], filt_covs[-1], Fs[t], Qs[t], bs[t])
        P_p = symmetrize_with_jitter(P_p, jitter=_JITTER)
        m_u, P_u, ll = _seq_kalman_update(m_p, P_p, Hs[t], Rs[t], cs[t], ys[t])
        pred_means.append(m_p)
        pred_covs.append(P_p)
        filt_means.append(m_u)
        filt_covs.append(P_u)
        lls.append(ll)
    return (
        jnp.stack(pred_means),
        jnp.stack(pred_covs),
        jnp.stack(filt_means),
        jnp.stack(filt_covs),
        jnp.stack(lls),
    )


def _sequential_rts_smooth(filt_means, filt_covs, Fs, Qs, bs):
    """Reference RTS smoother returning the full posterior mean/cov path."""
    T = filt_means.shape[0]
    smooth_means = [None] * T
    smooth_covs = [None] * T
    smooth_means[-1] = filt_means[-1]
    smooth_covs[-1] = filt_covs[-1]
    for t in range(T - 2, -1, -1):
        m_pred = Fs[t + 1 - 1] @ filt_means[t] + bs[t + 1 - 1]  # Fs/bs indexed 0..T-2
        P_pred = symmetrize_with_jitter(
            Fs[t + 1 - 1] @ filt_covs[t] @ Fs[t + 1 - 1].T + Qs[t + 1 - 1],
            jitter=_JITTER,
        )
        gain = jla.solve(P_pred, Fs[t + 1 - 1] @ filt_covs[t], assume_a="pos").T
        sm = filt_means[t] + gain @ (smooth_means[t + 1] - m_pred)
        sc = filt_covs[t] + gain @ (smooth_covs[t + 1] - P_pred) @ gain.T
        sc = 0.5 * (sc + sc.T)
        smooth_means[t] = sm
        smooth_covs[t] = sc
    return jnp.stack(smooth_means), jnp.stack(smooth_covs)


def _make_random_lgssm(key, T, D, Dy, dtype=jnp.float64):
    """Build a stable per-step LGSSM with well-conditioned dynamics."""
    keys = random.split(key, 10)
    F_base = jnp.eye(D, dtype=dtype) + 0.05 * random.normal(keys[0], (D, D), dtype=dtype)
    F_eig = 0.9 * F_base / jnp.linalg.norm(F_base, ord=2)
    Fs = jnp.broadcast_to(F_eig, (T, D, D))
    Fs = Fs + 0.005 * random.normal(keys[1], (T, D, D), dtype=dtype)
    Q_base = 0.05 * jnp.eye(D, dtype=dtype)
    Qs = jnp.broadcast_to(Q_base, (T, D, D))
    bs = 0.01 * random.normal(keys[2], (T, D), dtype=dtype)
    Hs = jnp.broadcast_to(random.normal(keys[3], (Dy, D), dtype=dtype), (T, Dy, D))
    Rs = jnp.broadcast_to(0.1 * jnp.eye(Dy, dtype=dtype), (T, Dy, Dy))
    cs = 0.01 * random.normal(keys[4], (T, Dy), dtype=dtype)
    ys = random.normal(keys[5], (T, Dy), dtype=dtype)
    init_mean = 0.05 * random.normal(keys[6], (D,), dtype=dtype)
    init_cov = 0.2 * jnp.eye(D, dtype=dtype)
    return init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys


@pytest.fixture(autouse=True)
def _enable_x64():
    """Flip on float64 so the parallel-vs-sequential comparison is tight."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_filter_matches_sequential_point_in_time():
    T, D, Dy = 32, 4, 3
    (init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys) = _make_random_lgssm(
        random.PRNGKey(0), T, D, Dy
    )
    parallel = filter_lgssm(init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys)
    sequential = filter_lgssm(init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys, parallel=False)
    seq_pm, seq_pc, seq_fm, seq_fc, seq_ll = _sequential_filter(
        init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys
    )
    np.testing.assert_allclose(parallel.filt_mean, seq_fm, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(parallel.filt_cov, seq_fc, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(parallel.pred_mean, seq_pm, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(parallel.pred_cov, seq_pc, atol=1e-5, rtol=1e-5)
    # loglik accumulates jitter-induced mismatches across T steps; relax slightly.
    np.testing.assert_allclose(parallel.loglik, seq_ll, atol=1e-3, rtol=1e-4)
    # Sequential fallback shares predict/update primitives with the reference
    # implementation so it should match essentially bit-for-bit.
    np.testing.assert_allclose(sequential.filt_mean, seq_fm, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(sequential.filt_cov, seq_fc, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(sequential.pred_mean, seq_pm, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(sequential.pred_cov, seq_pc, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(sequential.loglik, seq_ll, atol=1e-10, rtol=1e-10)


def test_filter_matches_sequential_interval_summary():
    """Augmented-state LGSSM produced by ``build_linear_summary_augmented_system``."""
    import sys

    from causal_ssm_agent.models.ssm.constants import MIN_DT
    from causal_ssm_agent.models.ssm.inference.targets.laplace.shared import (
        _build_linear_summary_accumulator_plan,
    )
    from causal_ssm_agent.models.ssm.inference.targets.linear_summary_augmentation import (
        build_linear_summary_augmented_system,
    )
    from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
        get_support_kind_codes,
    )

    sys.path.insert(
        0,
        "/Users/ma9o/Desktop/causal-ssm-agent/trees/main/apps/data-pipeline/scripts",
    )
    from run_nuts_mixed_support_recovery import _make_mixed_support_recovery_data

    from causal_ssm_agent.artifacts import LinkFunction

    data = _make_mixed_support_recovery_data(n_time=12)
    spec = data["spec"]
    observation_support = data["observation_support"]
    manifest_links = spec.manifest_links or [LinkFunction.IDENTITY] * spec.n_manifest
    plan = _build_linear_summary_accumulator_plan(
        observation_support,
        spec.manifest_dists,
        manifest_links,
    )
    support_kind_codes = get_support_kind_codes(observation_support)
    times = data["times"]
    time_intervals = jnp.diff(times, prepend=times[0]).at[0].set(MIN_DT)

    drift = -0.3 * jnp.eye(spec.n_latent, dtype=jnp.float64)
    diffusion_cov = 0.04 * jnp.eye(spec.n_latent, dtype=jnp.float64)
    cint = jnp.zeros(spec.n_latent, dtype=jnp.float64)
    H = jnp.asarray(spec.lambda_mat, dtype=jnp.float64)
    d = jnp.zeros(spec.n_manifest, dtype=jnp.float64)
    init_mean = jnp.zeros(spec.n_latent, dtype=jnp.float64)
    init_cov = 0.1 * jnp.eye(spec.n_latent, dtype=jnp.float64)

    (
        Ad_aug,
        Qd_aug,
        cd_aug,
        init_mean_aug,
        init_cov_aug,
        _H_rows,
        _d_rows,
    ) = build_linear_summary_augmented_system(
        plan=plan,
        time_intervals=time_intervals,
        drift=drift,
        diffusion_cov=diffusion_cov,
        cint=cint,
        H=H,
        d=d,
        init_mean=init_mean,
        init_cov=init_cov,
        support_kind_codes=support_kind_codes,
    )
    aug_dim = Ad_aug.shape[1]
    T = int(Ad_aug.shape[0])

    # Pseudo-observations: ``H = I``, ``R = (delta/2) I`` — exactly the
    # auxiliary-Kalman proposal used by aux_gibbs for the augmented state.
    delta = 0.4
    u = 0.3 * random.normal(random.PRNGKey(1), (T, aug_dim), dtype=jnp.float64)

    state = aux_filter_lgssm(
        init_mean=init_mean_aug,
        init_cov=init_cov_aug,
        Fs=Ad_aug,
        Qs=Qd_aug,
        bs=cd_aug,
        pseudo_observations=u,
        aux_variance=0.5 * delta,
    )

    Hs = jnp.broadcast_to(jnp.eye(aug_dim, dtype=jnp.float64), (T, aug_dim, aug_dim))
    Rs = jnp.broadcast_to(0.5 * delta * jnp.eye(aug_dim, dtype=jnp.float64), (T, aug_dim, aug_dim))
    cs = jnp.zeros((T, aug_dim), dtype=jnp.float64)
    seq_pm, seq_pc, seq_fm, seq_fc, seq_ll = _sequential_filter(
        init_mean_aug, init_cov_aug, Ad_aug, Qd_aug, cd_aug, Hs, Rs, cs, u
    )
    np.testing.assert_allclose(state.filt_mean, seq_fm, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(state.filt_cov, seq_fc, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(state.pred_mean, seq_pm, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(state.pred_cov, seq_pc, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(state.loglik, seq_ll, atol=1e-2, rtol=1e-3)


def test_filter_matches_sequential_block_diagonal():
    """Block-diagonal transition (two independent subsystems) stress test.

    This mirrors the ``intervals block diag`` path — subsystems whose
    dynamics are independent produce ``F`` matrices with exact zero
    off-diagonal blocks, and we rely on the associative operator to stay
    numerically stable despite the many zeros.
    """
    from scipy.linalg import block_diag

    T, Db, nblocks = 24, 3, 3
    D = Db * nblocks
    Dy = Db  # observe first block only
    key = random.PRNGKey(7)

    def _block(subkey):
        F = jnp.eye(Db, dtype=jnp.float64) + 0.05 * random.normal(
            subkey, (Db, Db), dtype=jnp.float64
        )
        return 0.9 * F / jnp.linalg.norm(F, ord=2)

    block_Fs = [_block(random.fold_in(key, i)) for i in range(nblocks)]
    F_big = jnp.asarray(block_diag(*[np.asarray(Fb) for Fb in block_Fs]))
    Fs = jnp.broadcast_to(F_big, (T, D, D))
    Qs = jnp.broadcast_to(0.03 * jnp.eye(D, dtype=jnp.float64), (T, D, D))
    bs = jnp.zeros((T, D), dtype=jnp.float64)

    H = jnp.concatenate(
        [
            jnp.eye(Db, dtype=jnp.float64),
            jnp.zeros((Db, D - Db), dtype=jnp.float64),
        ],
        axis=1,
    )
    Hs = jnp.broadcast_to(H, (T, Dy, D))
    Rs = jnp.broadcast_to(0.15 * jnp.eye(Dy, dtype=jnp.float64), (T, Dy, Dy))
    cs = jnp.zeros((T, Dy), dtype=jnp.float64)
    ys = random.normal(random.fold_in(key, 100), (T, Dy), dtype=jnp.float64)

    init_mean = jnp.zeros(D, dtype=jnp.float64)
    init_cov = 0.2 * jnp.eye(D, dtype=jnp.float64)

    parallel = filter_lgssm(init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys)
    _seq_pm, _seq_pc, seq_fm, seq_fc, seq_ll = _sequential_filter(
        init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys
    )
    np.testing.assert_allclose(parallel.filt_mean, seq_fm, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(parallel.filt_cov, seq_fc, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(parallel.loglik, seq_ll, atol=1e-5, rtol=1e-5)


def test_sampler_parallel_equals_sequential_moments():
    """Monte-Carlo moments from parallel and sequential RTS samplers agree."""
    T, D, Dy = 16, 3, 2
    init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys = _make_random_lgssm(
        random.PRNGKey(5), T, D, Dy
    )
    filt = filter_lgssm(init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys)
    n_samples = 3000
    keys = random.split(random.PRNGKey(9), n_samples)

    def _parallel_draw(k):
        return sample_lgssm_trajectory(
            k, filt.filt_mean, filt.filt_cov, Fs[1:], Qs[1:], bs[1:], parallel=True
        )

    def _sequential_draw(k):
        return sample_lgssm_trajectory(
            k, filt.filt_mean, filt.filt_cov, Fs[1:], Qs[1:], bs[1:], parallel=False
        )

    par_samples = jax.vmap(_parallel_draw)(keys)
    seq_samples = jax.vmap(_sequential_draw)(keys)
    np.testing.assert_allclose(
        jnp.mean(par_samples, axis=0), jnp.mean(seq_samples, axis=0), atol=0.05
    )
    par_cov = jnp.cov(par_samples.reshape(n_samples, -1), rowvar=False)
    seq_cov = jnp.cov(seq_samples.reshape(n_samples, -1), rowvar=False)
    np.testing.assert_allclose(par_cov, seq_cov, atol=0.05)


def test_sampler_moments_match_rts_smoother():
    """Many parallel samples reproduce the analytical RTS posterior moments."""
    T, D, Dy = 16, 3, 2
    init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys = _make_random_lgssm(
        random.PRNGKey(3), T, D, Dy
    )
    parallel = filter_lgssm(init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, ys)
    smooth_mean, smooth_cov = _sequential_rts_smooth(
        parallel.filt_mean, parallel.filt_cov, Fs[1:], Qs[1:], bs[1:]
    )

    n_samples = 4000
    keys = random.split(random.PRNGKey(77), n_samples)

    def _draw(k):
        return sample_lgssm_trajectory(
            k, parallel.filt_mean, parallel.filt_cov, Fs[1:], Qs[1:], bs[1:]
        )

    samples = jax.vmap(_draw)(keys)
    empirical_mean = jnp.mean(samples, axis=0)
    centred = samples - empirical_mean[None, ...]
    empirical_cov = jnp.einsum("nti,ntj->tij", centred, centred) / n_samples

    np.testing.assert_allclose(empirical_mean, smooth_mean, atol=0.04, rtol=0.0)
    np.testing.assert_allclose(empirical_cov, smooth_cov, atol=0.07, rtol=0.0)


def test_aux_filter_matches_sequential_with_auxiliary_variance():
    T, D = 20, 5
    init_mean, init_cov, Fs, Qs, bs, *_ = _make_random_lgssm(random.PRNGKey(11), T, D, D)
    delta = 0.5
    u = 0.4 * random.normal(random.PRNGKey(21), (T, D), dtype=jnp.float64)

    state = aux_filter_lgssm(
        init_mean=init_mean,
        init_cov=init_cov,
        Fs=Fs,
        Qs=Qs,
        bs=bs,
        pseudo_observations=u,
        aux_variance=0.5 * delta,
    )
    Hs = jnp.broadcast_to(jnp.eye(D, dtype=jnp.float64), (T, D, D))
    Rs = jnp.broadcast_to(0.5 * delta * jnp.eye(D, dtype=jnp.float64), (T, D, D))
    cs = jnp.zeros((T, D), dtype=jnp.float64)
    seq_pm, seq_pc, seq_fm, seq_fc, seq_ll = _sequential_filter(
        init_mean, init_cov, Fs, Qs, bs, Hs, Rs, cs, u
    )
    np.testing.assert_allclose(state.filt_mean, seq_fm, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(state.filt_cov, seq_fc, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(state.pred_mean, seq_pm, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(state.pred_cov, seq_pc, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(state.loglik, seq_ll, atol=1e-5, rtol=1e-5)
