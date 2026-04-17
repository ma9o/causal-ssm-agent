"""Auxiliary Kalman trajectory updates with conditional parameter MALA."""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.models.ssm.constants import MIN_DT
from causal_ssm_agent.models.ssm.covariance_utils import symmetrize_with_jitter
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference.shared import _trace_public_sites
from causal_ssm_agent.models.ssm.inference.targets.graph_analysis import has_student_t_diffusion
from causal_ssm_agent.models.ssm.inference.targets.kernels import compile_measurement_semantics
from causal_ssm_agent.models.ssm.inference.targets.laplace.shared import (
    _build_gaussian_trajectory_prior_terms,
    _predictive_latent_init,
    _trajectory_prior_log_prob_from_terms,
)
from causal_ssm_agent.models.ssm.inference.targets.rao_blackwell import (
    _kalman_predict,
    _kalman_update_gaussian,
)
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    trajectory_observation_log_prob,
)
from causal_ssm_agent.models.ssm.inference.utils import (
    _assemble_likelihood_inputs,
    _build_original_sample_resolver,
    _discover_sites,
    _DummyLikelihoodBackend,
)
from causal_ssm_agent.models.ssm.parameterization import build_site_registry

_SAFE_LOG_FLOOR = -1e30


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


def _gaussian_log_prob_isotropic(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    variance: jnp.ndarray,
) -> jnp.ndarray:
    diff = jnp.reshape(value - mean, (-1,))
    dim = diff.shape[0]
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi * variance) + jnp.sum(diff * diff) / variance)


def _sample_gaussian(
    key: jnp.ndarray,
    mean: jnp.ndarray,
    cov: jnp.ndarray,
) -> jnp.ndarray:
    chol = jnp.linalg.cholesky(symmetrize_with_jitter(cov))
    noise = random.normal(key, mean.shape, dtype=mean.dtype)
    return mean + chol @ noise


def _initial_latent_moments(context: _LatentContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    init_pred_mean = context.Ad[0] @ context.init_mean + context.cd[0]
    init_pred_cov = symmetrize_with_jitter(
        context.Ad[0] @ context.init_cov @ context.Ad[0].T + context.Qd[0]
    )
    return init_pred_mean, init_pred_cov


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

    def _step(carry, inputs):
        mean_prev, cov_prev = carry
        F_t, Q_t, c_t, y_t = inputs
        pred_mean_t, pred_cov_t = _kalman_predict(mean_prev, cov_prev, F_t, Q_t, c_t)
        filt_mean_t, filt_cov_t, loglik_t = _kalman_update_gaussian(
            pred_mean_t,
            pred_cov_t,
            eye,
            R_aux,
            zero,
            y_t,
            obs_mask,
        )
        return (filt_mean_t, filt_cov_t), (
            pred_mean_t,
            pred_cov_t,
            filt_mean_t,
            filt_cov_t,
            loglik_t,
        )

    (_, _), history = jax.lax.scan(
        _step,
        (filt_mean_0, filt_cov_0),
        (
            context.Ad[1:],
            context.Qd[1:],
            context.cd[1:],
            pseudo_observations[1:],
        ),
    )
    pred_means = jnp.concatenate([init_mean[None, ...], history[0]], axis=0)
    pred_covs = jnp.concatenate([init_cov[None, ...], history[1]], axis=0)
    filt_means = jnp.concatenate([filt_mean_0[None, ...], history[2]], axis=0)
    filt_covs = jnp.concatenate([filt_cov_0[None, ...], history[3]], axis=0)
    loglik = jnp.concatenate([loglik_0[None, ...], history[4]], axis=0)
    return pred_means, pred_covs, filt_means, filt_covs, loglik


def _sample_auxiliary_trajectory(
    key: jnp.ndarray,
    context: _LatentContext,
    *,
    filt_means: jnp.ndarray,
    filt_covs: jnp.ndarray,
    pred_means: jnp.ndarray,
    pred_covs: jnp.ndarray,
) -> jnp.ndarray:
    n_time = int(filt_means.shape[0])
    keys = random.split(key, n_time)
    last_sample = _sample_gaussian(keys[-1], filt_means[-1], filt_covs[-1])

    if n_time == 1:
        return last_sample[None, ...]

    def _backward_step(x_next, inputs):
        key_t, filt_mean_t, filt_cov_t, pred_mean_next, pred_cov_next, F_next = inputs
        gain = jla.solve(pred_cov_next, F_next @ filt_cov_t, assume_a="pos").T
        smooth_mean = filt_mean_t + gain @ (x_next - pred_mean_next)
        smooth_cov = symmetrize_with_jitter(filt_cov_t - gain @ pred_cov_next @ gain.T)
        x_t = _sample_gaussian(key_t, smooth_mean, smooth_cov)
        return x_t, x_t

    _, x_rev = jax.lax.scan(
        _backward_step,
        last_sample,
        (
            keys[:-1][::-1],
            filt_means[:-1][::-1],
            filt_covs[:-1][::-1],
            pred_means[1:][::-1],
            pred_covs[1:][::-1],
            context.Ad[1:][::-1],
        ),
    )
    return jnp.concatenate([x_rev[::-1], last_sample[None, ...]], axis=0)


def _auxiliary_posterior_log_prob(
    latent_trajectory: jnp.ndarray,
    context: _LatentContext,
    pseudo_observations: jnp.ndarray,
    *,
    delta: jnp.ndarray,
    log_evidence: jnp.ndarray,
) -> jnp.ndarray:
    prior_terms = _build_gaussian_trajectory_prior_terms(
        context.Ad,
        context.Qd,
        context.cd,
        context.init_mean,
        context.init_cov,
    )
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
        return _LatentContext(
            Ad=Ad,
            Qd=Qd,
            cd=cd_scan,
            init_mean=initial_state.mean,
            init_cov=initial_state.cov,
            H=measurement_params.lambda_mat,
            d_meas=measurement_params.manifest_means,
            R=measurement_params.manifest_cov,
            extra_params=extra_params,
        )

    def observation_log_prob_fn(z: jnp.ndarray, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        context = latent_context_fn(z)
        measurement_semantics = compile_measurement_semantics(
            model.spec.manifest_dists,
            manifest_cov=context.R,
            extra_params=context.extra_params,
            manifest_links=model.spec.manifest_links,
            observation_support=observation_support,
        )
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
        return jnp.where(jnp.isfinite(obs_lp), obs_lp, jnp.asarray(_SAFE_LOG_FLOOR, obs_lp.dtype))

    observation_grad_fn = jax.grad(observation_log_prob_fn, argnums=1)

    def trajectory_log_prob_fn(z: jnp.ndarray, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        context = latent_context_fn(z)
        prior_terms = _build_gaussian_trajectory_prior_terms(
            context.Ad,
            context.Qd,
            context.cd,
            context.init_mean,
            context.init_cov,
        )
        prior_lp = _trajectory_prior_log_prob_from_terms(
            latent_trajectory,
            context.Ad,
            context.cd,
            prior_terms,
        )
        total = prior_lp + observation_log_prob_fn(z, latent_trajectory)
        return jnp.where(jnp.isfinite(total), total, jnp.asarray(_SAFE_LOG_FLOOR, total.dtype))

    def complete_log_posterior_with_aux_fn(
        z: jnp.ndarray,
        latent_trajectory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        trajectory_lp = trajectory_log_prob_fn(z, latent_trajectory)
        complete_lp = log_prior_unc_fn(z) + trajectory_lp
        safe_complete = jnp.where(
            jnp.isfinite(complete_lp),
            complete_lp,
            jnp.asarray(_SAFE_LOG_FLOOR, complete_lp.dtype),
        )
        return safe_complete, trajectory_lp

    def initial_latent_fn(z: jnp.ndarray) -> jnp.ndarray:
        context = latent_context_fn(z)
        return _predictive_latent_init(context.Ad, context.cd, context.init_mean)

    return {
        "dim": int(flat_example.shape[0]),
        "flat_example": flat_example,
        "site_info": site_info,
        "unravel_fn": unravel_fn,
        "public_sites": public_sites,
        "log_prior_unc_fn": log_prior_unc_fn,
        "latent_context_fn": latent_context_fn,
        "observation_log_prob_fn": observation_log_prob_fn,
        "observation_grad_fn": observation_grad_fn,
        "trajectory_log_prob_fn": trajectory_log_prob_fn,
        "complete_log_posterior_with_aux_fn": complete_log_posterior_with_aux_fn,
        "initial_latent_fn": initial_latent_fn,
    }


def build_auxiliary_kalman_latent_kernel(
    bundle: dict[str, Any],
    *,
    delta: float,
    target_accept: float,
) -> dict[str, Any]:
    """Build the auxiliary Kalman latent-path update kernel."""

    def _latent_mh_step(state, key: jnp.ndarray):
        aux_key, sample_key, accept_key = random.split(key, 3)
        latent_shape = state.latent_trajectory.shape
        u = state.latent_trajectory + jnp.sqrt(0.5 * state.latent_delta) * random.normal(
            aux_key,
            latent_shape,
            dtype=state.latent_trajectory.dtype,
        )
        context = bundle["latent_context_fn"](state.position)
        grad_curr = bundle["observation_grad_fn"](state.position, state.latent_trajectory)
        pseudo_obs_fwd = u + 0.5 * state.latent_delta * grad_curr
        pred_means, pred_covs, filt_means, filt_covs, loglik_fwd = _filter_auxiliary_lgssm(
            context,
            pseudo_obs_fwd,
            state.latent_delta,
        )
        latent_prop = _sample_auxiliary_trajectory(
            sample_key,
            context,
            filt_means=filt_means,
            filt_covs=filt_covs,
            pred_means=pred_means,
            pred_covs=pred_covs,
        )
        q_fwd = _auxiliary_posterior_log_prob(
            latent_prop,
            context,
            pseudo_obs_fwd,
            delta=state.latent_delta,
            log_evidence=jnp.sum(loglik_fwd),
        )
        traj_prop = bundle["trajectory_log_prob_fn"](state.position, latent_prop)
        grad_prop = bundle["observation_grad_fn"](state.position, latent_prop)
        pseudo_obs_rev = u + 0.5 * state.latent_delta * grad_prop
        _pred_rev, _cov_rev, _filt_rev, _filt_cov_rev, loglik_rev = _filter_auxiliary_lgssm(
            context,
            pseudo_obs_rev,
            state.latent_delta,
        )
        q_rev = _auxiliary_posterior_log_prob(
            state.latent_trajectory,
            context,
            pseudo_obs_rev,
            delta=state.latent_delta,
            log_evidence=jnp.sum(loglik_rev),
        )
        log_alpha = traj_prop - state.trajectory_log_prob
        log_alpha = log_alpha + _gaussian_log_prob_isotropic(
            u,
            latent_prop,
            0.5 * state.latent_delta,
        )
        log_alpha = log_alpha - _gaussian_log_prob_isotropic(
            u,
            state.latent_trajectory,
            0.5 * state.latent_delta,
        )
        log_alpha = log_alpha + q_rev - q_fwd
        finite = (
            jnp.isfinite(log_alpha)
            & jnp.isfinite(traj_prop)
            & jnp.isfinite(q_fwd)
            & jnp.isfinite(q_rev)
        )
        accept_prob = jnp.where(finite, jnp.exp(jnp.minimum(log_alpha, 0.0)), 0.0)
        accept = random.bernoulli(accept_key, accept_prob)
        next_traj = jnp.where(accept, latent_prop, state.latent_trajectory)
        next_traj_lp = jnp.where(accept, traj_prop, state.trajectory_log_prob)
        next_complete = state.complete_log_posterior + (next_traj_lp - state.trajectory_log_prob)
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
