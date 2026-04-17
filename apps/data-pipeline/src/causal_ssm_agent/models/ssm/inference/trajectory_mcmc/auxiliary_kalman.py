"""Auxiliary-Kalman latent MH (reparametrised, eq 8) and MALA parameter kernel.

Implements Corenflos & Sarkka (2025, Sec 2.2, eq 8) with the reparametrised
augmentation ``u ~ N(x + (delta/2) grad_x f(x; theta), (delta/2) I)``. Under
this augmentation the LGSSM proposal for x is independent of the current
trajectory, which collapses the forward and reverse smoothing filters into a
single pass. The parameter block stays as a MALA update on the complete
log-posterior at fixed x. Both kernels report their own accept signal so the
scale adaptation in ``run_aux_gibbs`` can tune each scale against its own
target.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.artifacts import LinkFunction
from causal_ssm_agent.models.ssm.constants import MIN_DT
from causal_ssm_agent.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference.shared import _trace_public_sites
from causal_ssm_agent.models.ssm.inference.targets.graph_analysis import has_student_t_diffusion
from causal_ssm_agent.models.ssm.inference.targets.kernels import compile_measurement_semantics
from causal_ssm_agent.models.ssm.inference.targets.laplace.shared import (
    GaussianTrajectoryPriorTerms,
    _build_gaussian_trajectory_prior_terms,
    _build_linear_summary_accumulator_plan,
    _predictive_latent_init,
    _trajectory_prior_log_prob_from_terms,
)
from causal_ssm_agent.models.ssm.inference.targets.linear_summary_augmentation import (
    build_linear_summary_augmented_system,
    row_observation_log_prob,
)
from causal_ssm_agent.models.ssm.inference.targets.rao_blackwell import (
    _kalman_predict,
    _kalman_update_gaussian,
)
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    get_support_kind_codes,
    trajectory_observation_log_prob,
)
from causal_ssm_agent.models.ssm.inference.utils import (
    _assemble_likelihood_inputs,
    _build_original_sample_resolver,
    _discover_sites,
    _DummyLikelihoodBackend,
)
from causal_ssm_agent.models.ssm.parameterization import build_site_registry

# Match laplace/shared.py's default jitter so the auxiliary-Kalman proposal and
# the target trajectory log-prob agree on the covariance being evaluated.
_AUX_JITTER = 1e-6


class _LatentContext(NamedTuple):
    Ad: jnp.ndarray
    Qd: jnp.ndarray
    cd: jnp.ndarray
    init_mean: jnp.ndarray
    init_cov: jnp.ndarray
    H: jnp.ndarray
    d_meas: jnp.ndarray
    R: jnp.ndarray
    extra_params: dict[str, jnp.ndarray] | None
    H_rows: jnp.ndarray | None
    d_rows: jnp.ndarray | None


def _gaussian_log_prob_isotropic(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    variance: jnp.ndarray,
) -> jnp.ndarray:
    diff = jnp.reshape(value - mean, (-1,))
    dim = diff.shape[0]
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi * variance) + jnp.sum(diff * diff) / variance)


def _select_tree(accepted: jnp.ndarray, proposal_tree, current_tree):
    """Elementwise ``where`` over a pytree, tolerating ``None`` leaves."""
    return jax.tree_util.tree_map(
        lambda proposal, current: (
            None if proposal is None else jnp.where(accepted, proposal, current)
        ),
        proposal_tree,
        current_tree,
        is_leaf=lambda leaf: leaf is None,
    )


def _initial_latent_moments(context: _LatentContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    init_pred_mean = context.Ad[0] @ context.init_mean + context.cd[0]
    init_pred_cov = symmetrize_with_jitter(
        context.Ad[0] @ context.init_cov @ context.Ad[0].T + context.Qd[0],
        jitter=_AUX_JITTER,
    )
    return init_pred_mean, init_pred_cov


def _parallel_filtering_op(elem1, elem2):
    return _parallel_filtering_op_one(*elem1, *elem2)


def _parallel_filtering_op_one(A1, b1, C1, eta1, J1, A2, b2, C2, eta2, J2):
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


def _parallel_filtering_init_one(
    F: jnp.ndarray,
    Q: jnp.ndarray,
    b: jnp.ndarray,
    y: jnp.ndarray,
    m: jnp.ndarray,
    P: jnp.ndarray,
    R: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    m_pred = F @ m + b
    P_pred = symmetrize_with_jitter(F @ P @ F.T + Q, jitter=_AUX_JITTER)
    S = symmetrize_with_jitter(P_pred + R, jitter=_AUX_JITTER)
    chol_S = jnp.linalg.cholesky(S)
    gain = jla.cho_solve((chol_S, True), P_pred).T
    A = F - gain @ F
    b_std = m_pred + gain @ (y - m_pred)
    C = P_pred - gain @ S @ gain.T
    temp = jla.cho_solve((chol_S, True), F).T
    eta = temp @ (y - b)
    J = temp @ F
    return A, b_std, symmetrize(C), eta, symmetrize(J)


def _parallel_filtering_init(
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
    ys: jnp.ndarray,
    m0: jnp.ndarray,
    P0: jnp.ndarray,
    R: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_steps = int(bs.shape[0])
    if n_steps == 0:
        zeros_m = jnp.zeros((0, *m0.shape), dtype=m0.dtype)
        zeros_P = jnp.zeros((0, *P0.shape), dtype=P0.dtype)
        return zeros_P, zeros_m, zeros_P, zeros_m, zeros_P

    ms = jnp.concatenate(
        [
            m0[None, ...],
            jnp.zeros((n_steps - 1, *m0.shape), dtype=m0.dtype),
        ],
        axis=0,
    )
    Ps = jnp.concatenate(
        [
            P0[None, ...],
            jnp.zeros((n_steps - 1, *P0.shape), dtype=P0.dtype),
        ],
        axis=0,
    )
    return jax.vmap(
        _parallel_filtering_init_one,
        in_axes=(0, 0, 0, 0, 0, 0, None),
    )(Fs, Qs, bs, ys, ms, Ps, R)


def _auxiliary_loglik_one(
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray,
    observation: jnp.ndarray,
    R_aux: jnp.ndarray,
) -> jnp.ndarray:
    S = symmetrize_with_jitter(pred_cov + R_aux, jitter=_AUX_JITTER)
    chol_S = jnp.linalg.cholesky(S)
    diff = observation - pred_mean
    whitened = jla.solve_triangular(chol_S, diff, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_S)))
    dim = diff.shape[-1]
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + whitened @ whitened)


def _parallel_sampling_op(elem1, elem2):
    return _parallel_sampling_op_one(*elem1, *elem2)


def _parallel_sampling_op_one(
    gain1: jnp.ndarray,
    increment1: jnp.ndarray,
    gain2: jnp.ndarray,
    increment2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return gain2 @ gain1, gain2 @ increment1 + increment2


def _parallel_sampling_init_one(
    F: jnp.ndarray,
    Q: jnp.ndarray,
    b: jnp.ndarray,
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    epsilon: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    S = symmetrize_with_jitter(F @ cov @ F.T + Q, jitter=_AUX_JITTER)
    chol_S = jnp.linalg.cholesky(S)
    gain = jla.cho_solve((chol_S, True), F @ cov.T).T
    increment_cov = symmetrize_with_jitter(cov - gain @ S @ gain.T, jitter=_AUX_JITTER)
    chol = jnp.linalg.cholesky(increment_cov)
    increment_mean = mean - gain @ (F @ mean + b)
    increment = increment_mean + chol @ epsilon
    return gain, increment


def _parallel_sample_last_step(
    mean: jnp.ndarray,
    cov: jnp.ndarray,
    epsilon: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    chol = jnp.linalg.cholesky(symmetrize_with_jitter(cov, jitter=_AUX_JITTER))
    last_sample = mean + chol @ epsilon
    gain = jnp.zeros_like(cov)
    return gain, last_sample


def _parallel_sampling_init(
    key: jnp.ndarray,
    means: jnp.ndarray,
    covs: jnp.ndarray,
    Fs: jnp.ndarray,
    Qs: jnp.ndarray,
    bs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    epsilons = random.normal(key, means.shape, dtype=means.dtype)
    if means.shape[0] == 1:
        last_gain, last_increment = _parallel_sample_last_step(means[0], covs[0], epsilons[0])
        return last_gain[None, ...], last_increment[None, ...]

    gains, increments = jax.vmap(_parallel_sampling_init_one)(
        Fs,
        Qs,
        bs,
        means[:-1],
        covs[:-1],
        epsilons[:-1],
    )
    last_gain, last_increment = _parallel_sample_last_step(means[-1], covs[-1], epsilons[-1])
    return (
        jnp.concatenate([gains, last_gain[None, ...]], axis=0),
        jnp.concatenate([increments, last_increment[None, ...]], axis=0),
    )


def _filter_auxiliary_lgssm(
    context: _LatentContext,
    pseudo_observations: jnp.ndarray,
    delta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    state_dim = int(context.Ad.shape[-1])
    eye = jnp.eye(state_dim, dtype=context.Ad.dtype)
    R_aux = 0.5 * delta * eye
    zero = jnp.zeros((state_dim,), dtype=context.Ad.dtype)
    obs_mask = jnp.ones((state_dim,), dtype=bool)
    init_mean, init_cov = _initial_latent_moments(context)

    filt_mean_0, filt_cov_0, loglik_0 = _kalman_update_gaussian(
        init_mean,
        init_cov,
        eye,
        R_aux,
        zero,
        pseudo_observations[0],
        obs_mask,
    )

    if pseudo_observations.shape[0] == 1:
        return (
            init_mean[None, ...],
            init_cov[None, ...],
            filt_mean_0[None, ...],
            filt_cov_0[None, ...],
            loglik_0[None, ...],
        )

    init_elems = _parallel_filtering_init(
        context.Ad[1:],
        context.Qd[1:],
        context.cd[1:],
        pseudo_observations[1:],
        filt_mean_0,
        filt_cov_0,
        R_aux,
    )
    _ops, filt_tail, filt_cov_tail, _eta, _J = jax.lax.associative_scan(
        jax.vmap(_parallel_filtering_op),
        init_elems,
    )
    filt_means = jnp.concatenate([filt_mean_0[None, ...], filt_tail], axis=0)
    filt_covs = jnp.concatenate([filt_cov_0[None, ...], filt_cov_tail], axis=0)
    pred_mean_tail, pred_cov_tail = jax.vmap(_kalman_predict)(
        filt_means[:-1],
        filt_covs[:-1],
        context.Ad[1:],
        context.Qd[1:],
        context.cd[1:],
    )
    pred_means = jnp.concatenate([init_mean[None, ...], pred_mean_tail], axis=0)
    pred_covs = jnp.concatenate([init_cov[None, ...], pred_cov_tail], axis=0)
    loglik_tail = jax.vmap(_auxiliary_loglik_one, in_axes=(0, 0, 0, None))(
        pred_means[1:],
        pred_covs[1:],
        pseudo_observations[1:],
        R_aux,
    )
    loglik = jnp.concatenate([loglik_0[None, ...], loglik_tail], axis=0)
    return pred_means, pred_covs, filt_means, filt_covs, loglik


def _sample_auxiliary_trajectory(
    key: jnp.ndarray,
    context: _LatentContext,
    *,
    filt_means: jnp.ndarray,
    filt_covs: jnp.ndarray,
) -> jnp.ndarray:
    gains, increments = _parallel_sampling_init(
        key,
        filt_means,
        filt_covs,
        context.Ad[1:],
        context.Qd[1:],
        context.cd[1:],
    )
    _gains, samples = jax.lax.associative_scan(
        jax.vmap(_parallel_sampling_op),
        (gains, increments),
        reverse=True,
    )
    return samples


def _auxiliary_posterior_log_prob(
    latent_trajectory: jnp.ndarray,
    context: _LatentContext,
    pseudo_observations: jnp.ndarray,
    *,
    delta: jnp.ndarray,
    log_evidence: jnp.ndarray,
    prior_terms: GaussianTrajectoryPriorTerms,
) -> jnp.ndarray:
    prior_lp = _trajectory_prior_log_prob_from_terms(
        latent_trajectory,
        context.Ad,
        context.cd,
        prior_terms,
    )
    pseudo_lp = _gaussian_log_prob_isotropic(
        pseudo_observations,
        latent_trajectory,
        0.5 * delta,
    )
    return prior_lp + pseudo_lp - log_evidence


def build_auxiliary_kalman_bundle(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    trace_key: jnp.ndarray,
    reparam,
) -> dict[str, Any]:
    """Assemble all static helpers needed by the auxiliary Kalman method."""
    if has_student_t_diffusion(model.spec):
        raise ValueError(
            "aux_gibbs with latent_kernel='kalman' currently requires Gaussian latent diffusion for every state."
        )

    site_info = _discover_sites(
        model,
        observations,
        times,
        trace_key,
        _DummyLikelihoodBackend(),
        reparam=reparam,
    )
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)

    transforms = {name: info["transform"] for name, info in site_info.items()}
    distributions = {name: info["distribution"] for name, info in site_info.items()}
    sample_resolver = _build_original_sample_resolver(
        site_info,
        model=model,
        observations=observations,
        times=times,
        reparam=reparam,
    )
    if sample_resolver is None:
        raise ValueError(
            "aux_gibbs with latent_kernel='kalman' only supports no reparameterization or AutoReparam with fixed centering."
        )

    runtime_registry = build_site_registry(model.spec, model.structure_runtime)
    obs_mask = ~jnp.isnan(observations)
    time_intervals = jnp.diff(times, prepend=times[0]).at[0].set(MIN_DT)
    observation_support = getattr(model, "observation_support", None)
    manifest_links = model.spec.manifest_links or [
        LinkFunction.IDENTITY for _ in range(model.spec.n_manifest)
    ]
    support_kind_codes = (
        get_support_kind_codes(observation_support)
        if observation_support is not None
        else jnp.zeros((model.spec.n_manifest,), dtype=jnp.int64)
    )
    linear_summary_plan = _build_linear_summary_accumulator_plan(
        observation_support,
        model.spec.manifest_dists,
        manifest_links,
    )
    use_linear_summary_augmentation = (
        observation_support is not None and observation_support.requires_interval_summary_handling
    )
    if use_linear_summary_augmentation and linear_summary_plan is None:
        raise ValueError(
            "aux_gibbs with latent_kernel='kalman' only supports linear interval summaries "
            "(mean/sum with supported Gaussian or Student-t identity-link measurements)."
        )
    public_sites = _trace_public_sites(
        functools.partial(model.model, likelihood_backend=_DummyLikelihoodBackend()),
        observations,
        times,
    )

    def _constrain(z: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        unconstrained = unravel_fn(z)
        constrained = {name: transforms[name](unconstrained[name]) for name in unconstrained}
        return constrained, unconstrained

    def log_prior_unc_fn(z: jnp.ndarray) -> jnp.ndarray:
        constrained, unconstrained = _constrain(z)
        log_prior = jnp.array(0.0, dtype=observations.dtype)
        log_jacobian = jnp.array(0.0, dtype=observations.dtype)
        for name in unconstrained:
            log_prior = log_prior + jnp.sum(distributions[name].log_prob(constrained[name]))
            log_jacobian = log_jacobian + jnp.sum(
                transforms[name].log_abs_det_jacobian(unconstrained[name], constrained[name])
            )
        return log_prior + log_jacobian

    def latent_context_fn(z: jnp.ndarray) -> _LatentContext:
        constrained, _ = _constrain(z)
        original_samples = sample_resolver(constrained)
        ct_params, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
            structure_runtime=model.structure_runtime,
        )
        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift,
            ct_params.diffusion_cov,
            ct_params.cint,
            time_intervals,
        )
        cd_scan = (
            jnp.zeros((Ad.shape[0], Ad.shape[1]), dtype=Ad.dtype) if cd is None else jnp.asarray(cd)
        )
        H_rows = None
        d_rows = None
        init_mean = initial_state.mean
        init_cov = initial_state.cov
        H = measurement_params.lambda_mat
        d_meas = measurement_params.manifest_means
        if use_linear_summary_augmentation:
            (
                Ad,
                Qd,
                cd_scan,
                init_mean,
                init_cov,
                H_rows,
                d_rows,
            ) = build_linear_summary_augmented_system(
                plan=linear_summary_plan,
                time_intervals=time_intervals,
                drift=ct_params.drift,
                diffusion_cov=ct_params.diffusion_cov,
                cint=ct_params.cint
                if ct_params.cint is not None
                else jnp.zeros((ct_params.drift.shape[0],), dtype=ct_params.drift.dtype),
                H=measurement_params.lambda_mat,
                d=measurement_params.manifest_means,
                init_mean=initial_state.mean,
                init_cov=initial_state.cov,
                support_kind_codes=support_kind_codes,
            )
        return _LatentContext(
            Ad=Ad,
            Qd=Qd,
            cd=cd_scan,
            init_mean=init_mean,
            init_cov=init_cov,
            H=H,
            d_meas=d_meas,
            R=measurement_params.manifest_cov,
            extra_params=extra_params,
            H_rows=H_rows,
            d_rows=d_rows,
        )

    def _measurement_semantics_from_context(context: _LatentContext):
        return compile_measurement_semantics(
            model.spec.manifest_dists,
            manifest_cov=context.R,
            extra_params=context.extra_params,
            manifest_links=manifest_links,
            observation_support=None if use_linear_summary_augmentation else observation_support,
        )

    def observation_log_prob_from_context_fn(
        context: _LatentContext,
        latent_trajectory: jnp.ndarray,
    ) -> jnp.ndarray:
        measurement_semantics = _measurement_semantics_from_context(context)
        if use_linear_summary_augmentation:
            assert context.H_rows is not None
            assert context.d_rows is not None
            obs_lp = row_observation_log_prob(
                latent_trajectory,
                observations,
                obs_mask,
                context.H_rows,
                context.d_rows,
                context.R,
                measurement_semantics.obs_kernel,
            )
        else:
            obs_lp = trajectory_observation_log_prob(
                latent_trajectory,
                observations,
                obs_mask,
                context.H,
                context.d_meas,
                context.R,
                measurement_semantics.obs_kernel,
                measurement_semantics.mean_log_prob_fn,
                observation_support,
            )
        return jnp.asarray(obs_lp, dtype=latent_trajectory.dtype)

    def observation_log_prob_fn(z: jnp.ndarray, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        context = latent_context_fn(z)
        return observation_log_prob_from_context_fn(context, latent_trajectory)

    observation_grad_fn = jax.grad(observation_log_prob_fn, argnums=1)
    observation_grad_from_context_fn = jax.grad(observation_log_prob_from_context_fn, argnums=1)

    def _prior_terms_from_context(context: _LatentContext) -> GaussianTrajectoryPriorTerms:
        return _build_gaussian_trajectory_prior_terms(
            context.Ad,
            context.Qd,
            context.cd,
            context.init_mean,
            context.init_cov,
            jitter=_AUX_JITTER,
        )

    def trajectory_log_prob_from_context_fn(
        context: _LatentContext,
        latent_trajectory: jnp.ndarray,
        prior_terms: GaussianTrajectoryPriorTerms | None = None,
    ) -> jnp.ndarray:
        if prior_terms is None:
            prior_terms = _prior_terms_from_context(context)
        prior_lp = _trajectory_prior_log_prob_from_terms(
            latent_trajectory,
            context.Ad,
            context.cd,
            prior_terms,
        )
        total = prior_lp + observation_log_prob_from_context_fn(context, latent_trajectory)
        return jnp.asarray(total, dtype=latent_trajectory.dtype)

    def trajectory_log_prob_fn(z: jnp.ndarray, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        context = latent_context_fn(z)
        return trajectory_log_prob_from_context_fn(context, latent_trajectory)

    def complete_log_posterior_from_context_fn(
        z: jnp.ndarray,
        context: _LatentContext,
        latent_trajectory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        trajectory_lp = trajectory_log_prob_from_context_fn(context, latent_trajectory)
        complete_lp = log_prior_unc_fn(z) + trajectory_lp
        return complete_lp, trajectory_lp

    def complete_log_posterior_with_aux_fn(
        z: jnp.ndarray,
        latent_trajectory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, _LatentContext]]:
        context = latent_context_fn(z)
        complete_lp, trajectory_lp = complete_log_posterior_from_context_fn(
            z,
            context,
            latent_trajectory,
        )
        return complete_lp, (trajectory_lp, context)

    def initial_latent_from_context_fn(context: _LatentContext) -> jnp.ndarray:
        return _predictive_latent_init(context.Ad, context.cd, context.init_mean)

    def initial_latent_fn(z: jnp.ndarray) -> jnp.ndarray:
        context = latent_context_fn(z)
        return initial_latent_from_context_fn(context)

    return {
        "dim": int(flat_example.shape[0]),
        "flat_example": flat_example,
        "site_info": site_info,
        "unravel_fn": unravel_fn,
        "public_sites": public_sites,
        "log_prior_unc_fn": log_prior_unc_fn,
        "latent_context_fn": latent_context_fn,
        "observation_log_prob_fn": observation_log_prob_fn,
        "observation_log_prob_from_context_fn": observation_log_prob_from_context_fn,
        "observation_grad_fn": observation_grad_fn,
        "observation_grad_from_context_fn": observation_grad_from_context_fn,
        "trajectory_log_prob_fn": trajectory_log_prob_fn,
        "trajectory_log_prob_from_context_fn": trajectory_log_prob_from_context_fn,
        "prior_terms_from_context_fn": _prior_terms_from_context,
        "complete_log_posterior_from_context_fn": complete_log_posterior_from_context_fn,
        "complete_log_posterior_with_aux_fn": complete_log_posterior_with_aux_fn,
        "initial_latent_fn": initial_latent_fn,
        "initial_latent_from_context_fn": initial_latent_from_context_fn,
        "project_latent_trajectory_fn": (
            (lambda latent_trajectory: latent_trajectory[:, : model.spec.n_latent])
            if use_linear_summary_augmentation
            else (lambda latent_trajectory: latent_trajectory)
        ),
    }


def build_auxiliary_kalman_latent_kernel(
    bundle: dict[str, Any],
    *,
    delta: float,
    target_accept: float,
) -> dict[str, Any]:
    """Auxiliary-Kalman latent MH under the eq-8 reparametrised augmentation.

    One iteration at ``(x, theta)`` with scale ``delta``:

    1. ``grad_x_curr = grad_x log g(y | x, theta)``.
    2. Draw ``u = x + (delta/2) grad_x_curr + sqrt(delta/2) eps`` — exact
       conditional of the eq-8 augmented target ``pi(x, theta, u)``, always
       accepted.
    3. Filter the LGSSM with pseudo-observations ``u`` (no gradient term) and
       smoothing-sample ``x*``. Under eq (8) the LGSSM proposal is independent
       of the current ``x``, so forward and reverse proposals evaluate
       ``q(x* | u, theta)`` and ``q(x | u, theta)`` under the *same* filter.
    4. Accept/reject ``x*`` with
       ``log pi(x*, theta) - log pi(x, theta)``
       ``+ log N(u; x* + (delta/2) grad_x_star, delta/2)``
       ``- log N(u; x + (delta/2) grad_x_curr, delta/2)``
       ``+ log q(x | u, theta) - log q(x* | u, theta)``.
    """

    def _latent_mh_step(state, key: jnp.ndarray):
        aux_key, sample_key, accept_key = random.split(key, 3)
        x_curr = state.latent_trajectory
        context = state.latent_context
        latent_dtype = x_curr.dtype
        traj_dtype = state.trajectory_log_prob.dtype
        complete_dtype = state.complete_log_posterior.dtype
        delta_val = state.latent_delta
        half_delta = 0.5 * delta_val

        prior_terms = bundle["prior_terms_from_context_fn"](context)
        traj_curr = jnp.asarray(
            bundle["trajectory_log_prob_from_context_fn"](
                context, x_curr, prior_terms
            ),
            dtype=traj_dtype,
        )
        log_prior_z = jnp.asarray(
            bundle["log_prior_unc_fn"](state.position), dtype=complete_dtype
        )

        # (1) grad_x log g at current x.
        grad_curr = jnp.asarray(
            bundle["observation_grad_from_context_fn"](context, x_curr),
            dtype=latent_dtype,
        )

        # (2) Sample u from eq-8 conditional.
        u = (
            x_curr
            + half_delta * grad_curr
            + jnp.sqrt(half_delta)
            * random.normal(aux_key, x_curr.shape, dtype=latent_dtype)
        )

        # (3) Single filter pass at current theta with pseudo-obs = u.
        _pm, _pc, filt_means, filt_covs, loglik = _filter_auxiliary_lgssm(
            context, u, delta_val
        )
        x_prop = jnp.asarray(
            _sample_auxiliary_trajectory(
                sample_key, context, filt_means=filt_means, filt_covs=filt_covs
            ),
            dtype=latent_dtype,
        )
        log_evidence = jnp.sum(loglik)

        # (4) Smoothing densities q(x*|u, theta) and q(x|u, theta) — same LGSSM.
        q_fwd = jnp.asarray(
            _auxiliary_posterior_log_prob(
                x_prop,
                context,
                u,
                delta=delta_val,
                log_evidence=log_evidence,
                prior_terms=prior_terms,
            ),
            dtype=traj_dtype,
        )
        q_rev = jnp.asarray(
            _auxiliary_posterior_log_prob(
                x_curr,
                context,
                u,
                delta=delta_val,
                log_evidence=log_evidence,
                prior_terms=prior_terms,
            ),
            dtype=traj_dtype,
        )

        # (5) grad_x log g at proposed x (for the eq-8 aux factor at x*).
        grad_prop = jnp.asarray(
            bundle["observation_grad_from_context_fn"](context, x_prop),
            dtype=latent_dtype,
        )
        traj_prop = jnp.asarray(
            bundle["trajectory_log_prob_from_context_fn"](
                context, x_prop, prior_terms
            ),
            dtype=traj_dtype,
        )

        # (6) MH log acceptance.
        log_alpha = traj_prop - traj_curr
        log_alpha = log_alpha + _gaussian_log_prob_isotropic(
            u, x_prop + half_delta * grad_prop, half_delta
        )
        log_alpha = log_alpha - _gaussian_log_prob_isotropic(
            u, x_curr + half_delta * grad_curr, half_delta
        )
        log_alpha = log_alpha + q_rev - q_fwd

        accept_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
        accept = random.bernoulli(accept_key, accept_prob)
        next_traj = jnp.asarray(
            jnp.where(accept, x_prop, x_curr), dtype=latent_dtype
        )
        next_traj_lp = jnp.asarray(
            jnp.where(accept, traj_prop, traj_curr), dtype=traj_dtype
        )
        next_complete = log_prior_z + next_traj_lp.astype(complete_dtype)
        next_state = state._replace(
            latent_trajectory=next_traj,
            trajectory_log_prob=next_traj_lp,
            complete_log_posterior=next_complete,
        )
        return next_state, {"accepted": accept.astype(state.position.dtype)}

    return {
        "name": "kalman",
        "scale_field": "latent_delta",
        "initial_scale": delta,
        "target_accept": target_accept,
        "step_fn": _latent_mh_step,
    }


def build_mala_parameter_kernel(
    bundle: dict[str, Any],
    *,
    step_size: float,
    target_accept: float,
) -> dict[str, Any]:
    """MALA update on the complete log-posterior at fixed latent trajectory."""

    complete_value_and_grad = jax.value_and_grad(
        bundle["complete_log_posterior_with_aux_fn"],
        argnums=0,
        has_aux=True,
    )

    def _parameter_mala_step(state, key: jnp.ndarray):
        if bundle["dim"] == 0:
            return state, {
                "accepted": jnp.asarray(1.0, dtype=state.latent_trajectory.dtype)
            }

        proposal_key, accept_key = random.split(key)
        (complete_curr, (traj_curr, _curr_context)), grad_curr = (
            complete_value_and_grad(state.position, state.latent_trajectory)
        )
        h = state.param_step_size
        mean_fwd = state.position + 0.5 * (h ** 2) * grad_curr
        proposal = mean_fwd + h * random.normal(
            proposal_key, state.position.shape, dtype=state.position.dtype
        )
        (complete_prop, (traj_prop, context_prop)), grad_prop = (
            complete_value_and_grad(proposal, state.latent_trajectory)
        )
        mean_rev = proposal + 0.5 * (h ** 2) * grad_prop
        log_alpha = complete_prop - complete_curr
        log_alpha = log_alpha + _gaussian_log_prob_isotropic(
            state.position, mean_rev, h ** 2
        )
        log_alpha = log_alpha - _gaussian_log_prob_isotropic(
            proposal, mean_fwd, h ** 2
        )
        accept_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
        accept = random.bernoulli(accept_key, accept_prob)
        next_context = _select_tree(accept, context_prop, state.latent_context)
        next_state = state._replace(
            position=jnp.where(accept, proposal, state.position),
            latent_context=next_context,
            trajectory_log_prob=jnp.where(accept, traj_prop, traj_curr),
            complete_log_posterior=jnp.where(accept, complete_prop, complete_curr),
        )
        return next_state, {"accepted": accept.astype(state.position.dtype)}

    return {
        "name": "mala",
        "scale_field": "param_step_size",
        "initial_scale": step_size,
        "target_accept": target_accept,
        "step_fn": _parameter_mala_step,
    }
