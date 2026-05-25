"""Latent trajectory initializers for complete-data SSM MCMC."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import jax.scipy.special as jsp_special

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.auxiliary_kalman import (
    AUX_JITTER,
    _initial_latent_moments,
)


class ParticleSmootherInitResult(NamedTuple):
    """FFBSi particle-smoother latent initialization output."""

    trajectories: jnp.ndarray
    log_normalizer: jnp.ndarray
    min_ess: jnp.ndarray
    trajectory_log_prob: jnp.ndarray


def _normalize_log_weights(log_weights: jnp.ndarray) -> jnp.ndarray:
    return log_weights - jsp_special.logsumexp(log_weights)


def _ess_from_normalized_log_weights(log_weights: jnp.ndarray) -> jnp.ndarray:
    weights = jnp.exp(log_weights)
    return 1.0 / jnp.sum(weights * weights)


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _project_psd(matrix: jnp.ndarray) -> jnp.ndarray:
    matrix = _symmetrize(matrix)
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    eigvals = jnp.maximum(eigvals, 0.0)
    return _symmetrize((eigvecs * eigvals[None, :]) @ eigvecs.T)


def _sample_gaussian_particles(
    key: jnp.ndarray,
    mean: jnp.ndarray,
    covariance: jnp.ndarray,
    *,
    num_particles: int,
) -> jnp.ndarray:
    chol = jnp.linalg.cholesky(covariance)
    eps = random.normal(key, (num_particles, mean.shape[0]), dtype=mean.dtype)
    return mean[None, :] + eps @ chol.T


def _gaussian_log_prob_full(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    covariance: jnp.ndarray,
) -> jnp.ndarray:
    covariance = symmetrize_with_jitter(covariance, jitter=AUX_JITTER)
    chol = jnp.linalg.cholesky(covariance)
    diff = value - mean
    whitened = jla.solve_triangular(chol, diff, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    dim = diff.shape[-1]
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + whitened @ whitened)


def _gaussian_guide_log_prob(
    value: jnp.ndarray,
    precision: jnp.ndarray,
    information: jnp.ndarray,
    log_constant: jnp.ndarray,
) -> jnp.ndarray:
    return -0.5 * value @ precision @ value + information @ value + log_constant


def _observation_gaussian_guide(
    obs_increment_fn,
    context,
    reference_state: jnp.ndarray,
    time_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def _log_prob_at(state: jnp.ndarray) -> jnp.ndarray:
        return obs_increment_fn(context, state, time_idx)

    value, grad = jax.value_and_grad(_log_prob_at)(reference_state)
    neg_hess = _project_psd(-jax.hessian(_log_prob_at)(reference_state))
    information = grad + neg_hess @ reference_state
    log_constant = (
        value - grad @ reference_state - 0.5 * reference_state @ neg_hess @ reference_state
    )
    return neg_hess, information, log_constant


def _integrate_linear_gaussian_guide(
    transition_matrix: jnp.ndarray,
    transition_offset: jnp.ndarray,
    transition_covariance: jnp.ndarray,
    next_precision: jnp.ndarray,
    next_information: jnp.ndarray,
    next_log_constant: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    transition_offset = jnp.reshape(transition_offset, (transition_matrix.shape[0],))
    transition_covariance = symmetrize_with_jitter(
        transition_covariance,
        jitter=AUX_JITTER,
    )
    cov_chol = jnp.linalg.cholesky(transition_covariance)
    identity = jnp.eye(transition_covariance.shape[0], dtype=transition_covariance.dtype)
    cov_inv = jla.cho_solve((cov_chol, True), identity)
    guide_precision = cov_inv + next_precision
    guide_precision = symmetrize_with_jitter(guide_precision, jitter=AUX_JITTER)
    guide_chol = jnp.linalg.cholesky(guide_precision)
    guide_covariance = jla.cho_solve((guide_chol, True), identity)
    logdet_cov = 2.0 * jnp.sum(jnp.log(jnp.diag(cov_chol)))
    logdet_guide_cov = -2.0 * jnp.sum(jnp.log(jnp.diag(guide_chol)))

    projected_precision = cov_inv - cov_inv @ guide_covariance @ cov_inv
    projected_information = cov_inv @ guide_covariance @ next_information
    precision = transition_matrix.T @ projected_precision @ transition_matrix
    information = transition_matrix.T @ (
        projected_information - projected_precision @ transition_offset
    )
    log_constant = (
        next_log_constant
        + 0.5 * (logdet_guide_cov - logdet_cov)
        + 0.5 * next_information @ guide_covariance @ next_information
        - 0.5 * transition_offset @ projected_precision @ transition_offset
        + transition_offset @ projected_information
    )
    return _project_psd(precision), information, log_constant


def _build_bffg_guides(
    context,
    reference_trajectory: jnp.ndarray,
    obs_increment_fn,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_steps = int(reference_trajectory.shape[0])
    latent_dim = int(reference_trajectory.shape[-1])
    obs_precisions = []
    obs_informations = []
    obs_constants = []
    for time_idx in range(num_steps):
        precision_t, information_t, constant_t = _observation_gaussian_guide(
            obs_increment_fn,
            context,
            reference_trajectory[time_idx],
            jnp.asarray(time_idx, dtype=jnp.int32),
        )
        obs_precisions.append(precision_t)
        obs_informations.append(information_t)
        obs_constants.append(constant_t)

    precisions = [None] * num_steps
    informations = [None] * num_steps
    constants = [None] * num_steps
    future_precision = jnp.zeros((latent_dim, latent_dim), dtype=reference_trajectory.dtype)
    future_information = jnp.zeros((latent_dim,), dtype=reference_trajectory.dtype)
    future_constant = jnp.asarray(0.0, dtype=reference_trajectory.dtype)
    for time_idx in range(num_steps - 1, -1, -1):
        precision_t = obs_precisions[time_idx] + future_precision
        information_t = obs_informations[time_idx] + future_information
        constant_t = obs_constants[time_idx] + future_constant
        precisions[time_idx] = precision_t
        informations[time_idx] = information_t
        constants[time_idx] = constant_t
        if time_idx > 0:
            future_precision, future_information, future_constant = (
                _integrate_linear_gaussian_guide(
                    context.Ad[time_idx],
                    context.cd[time_idx],
                    context.Qd[time_idx],
                    precision_t,
                    information_t,
                    constant_t,
                )
            )

    return (
        jnp.stack(precisions),
        jnp.stack(informations),
        jnp.stack(constants),
    )


def _guided_gaussian_proposal(
    prior_mean: jnp.ndarray,
    prior_covariance: jnp.ndarray,
    guide_precision: jnp.ndarray,
    guide_information: jnp.ndarray,
    guide_log_constant: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    prior_covariance = symmetrize_with_jitter(prior_covariance, jitter=AUX_JITTER)
    prior_chol = jnp.linalg.cholesky(prior_covariance)
    identity = jnp.eye(prior_covariance.shape[0], dtype=prior_covariance.dtype)
    prior_precision = jla.cho_solve((prior_chol, True), identity)
    proposal_precision = prior_precision + guide_precision
    proposal_precision = symmetrize_with_jitter(proposal_precision, jitter=AUX_JITTER)
    proposal_chol = jnp.linalg.cholesky(proposal_precision)
    proposal_covariance = jla.cho_solve(
        (proposal_chol, True),
        identity,
    )
    proposal_information = prior_precision @ prior_mean + guide_information
    proposal_mean = proposal_covariance @ proposal_information
    prior_logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(prior_chol)))
    proposal_logdet = -2.0 * jnp.sum(jnp.log(jnp.diag(proposal_chol)))
    log_normalizer = (
        guide_log_constant
        + 0.5 * (proposal_logdet - prior_logdet)
        - 0.5 * prior_mean @ prior_precision @ prior_mean
        + 0.5 * proposal_information @ proposal_covariance @ proposal_information
    )
    return (
        proposal_mean,
        symmetrize_with_jitter(proposal_covariance, jitter=AUX_JITTER),
        log_normalizer,
    )


def initialize_particle_smoother_latents(
    bundle: dict[str, Any],
    init_positions: jnp.ndarray,
    *,
    seed: int,
    num_particles: int,
    guidance: str = "bffg",
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Initialize latent paths with a guided FFBSi particle smoother.

    ``guidance="bootstrap"`` uses the transition prior as the proposal.
    ``guidance="bffg"`` builds Gaussian backward information guides for
    ``p(y_{t:T} | x_t)`` and uses the resulting forward guided process with
    likelihood-ratio weights. The backward pass is standard FFBSi.
    """
    if num_particles < 2:
        raise ValueError("Particle-smoother latent init requires num_particles >= 2.")
    if guidance not in {"bootstrap", "bffg"}:
        raise ValueError(
            f"Unsupported particle-smoother guidance {guidance!r}. "
            "Supported: 'bootstrap' or 'bffg'."
        )

    init_positions = jnp.asarray(init_positions, dtype=bundle["flat_example"].dtype)
    if init_positions.ndim != 2 or init_positions.shape[1] != bundle["flat_example"].shape[0]:
        raise ValueError(
            "init_positions must have shape (num_chains, parameter_dim); got "
            f"{init_positions.shape}."
        )
    observations = bundle["observations"]
    times = bundle["times"]
    obs_increment_fn = bundle["observation_increment_log_prob_from_context_fn"]
    trajectory_log_prob_fn = bundle["trajectory_log_prob_from_context_fn"]
    prior_terms_fn = bundle["prior_terms_from_context_fn"]
    latent_context_runtime_fn = bundle["latent_context_runtime_fn"]
    initial_latent_from_context_fn = bundle["initial_latent_from_context_fn"]

    def _obs_log_prob_batch(context, particles: jnp.ndarray, time_idx: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda particle: obs_increment_fn(context, particle, time_idx))(particles)

    def _initialize_one(init_position: jnp.ndarray, key: jnp.ndarray) -> ParticleSmootherInitResult:
        context = latent_context_runtime_fn(init_position, times)
        num_steps = int(observations.shape[0])
        filter_key, backward_key = random.split(key)
        init_key, step_key = random.split(filter_key)
        init_mean, init_cov = _initial_latent_moments(context)
        if guidance == "bffg":
            reference_trajectory = initial_latent_from_context_fn(context)
            guide_precisions, guide_informations, guide_constants = _build_bffg_guides(
                context,
                reference_trajectory,
                obs_increment_fn,
            )
            proposal_mean0, proposal_cov0, proposal_log_norm0 = _guided_gaussian_proposal(
                init_mean,
                init_cov,
                guide_precisions[0],
                guide_informations[0],
                guide_constants[0],
            )
            particles0 = _sample_gaussian_particles(
                init_key,
                proposal_mean0,
                proposal_cov0,
                num_particles=num_particles,
            )
            guide_log0 = jax.vmap(
                lambda particle: _gaussian_guide_log_prob(
                    particle,
                    guide_precisions[0],
                    guide_informations[0],
                    guide_constants[0],
                )
            )(particles0)
            logw0_raw = (
                _obs_log_prob_batch(context, particles0, jnp.asarray(0, dtype=jnp.int32))
                - guide_log0
            )
            log_norm0 = proposal_log_norm0 + (
                jsp_special.logsumexp(logw0_raw)
                - jnp.log(jnp.asarray(num_particles, dtype=logw0_raw.dtype))
            )
        else:
            guide_precisions = jnp.zeros(
                (num_steps, init_mean.shape[0], init_mean.shape[0]),
                dtype=init_mean.dtype,
            )
            guide_informations = jnp.zeros((num_steps, init_mean.shape[0]), dtype=init_mean.dtype)
            guide_constants = jnp.zeros((num_steps,), dtype=init_mean.dtype)
            particles0 = _sample_gaussian_particles(
                init_key,
                init_mean,
                init_cov,
                num_particles=num_particles,
            )
            logw0_raw = _obs_log_prob_batch(context, particles0, jnp.asarray(0, dtype=jnp.int32))
            log_norm0 = jsp_special.logsumexp(logw0_raw) - jnp.log(
                jnp.asarray(num_particles, dtype=logw0_raw.dtype)
            )
        logw0 = _normalize_log_weights(logw0_raw)
        ess0 = _ess_from_normalized_log_weights(logw0)

        def _filter_step(carry, time_idx):
            prev_particles, prev_logw, log_norm_acc, min_ess = carry
            resample_key, particle_key = random.split(random.fold_in(step_key, time_idx))
            time_idx_i32 = time_idx.astype(jnp.int32)

            pred_means = prev_particles @ context.Ad[time_idx].T + context.cd[time_idx]
            if guidance == "bffg":
                proposal_means, proposal_covariances, guide_log = jax.vmap(
                    lambda pred_mean: _guided_gaussian_proposal(
                        pred_mean,
                        context.Qd[time_idx],
                        guide_precisions[time_idx],
                        guide_informations[time_idx],
                        guide_constants[time_idx],
                    )
                )(pred_means)
                ancestor_logits = prev_logw + guide_log
            else:
                proposal_means = pred_means
                proposal_covariances = jnp.broadcast_to(
                    symmetrize_with_jitter(context.Qd[time_idx], jitter=AUX_JITTER),
                    (num_particles, context.Qd.shape[-2], context.Qd.shape[-1]),
                )
                guide_log = jnp.zeros((num_particles,), dtype=prev_logw.dtype)
                ancestor_logits = prev_logw

            ancestors = random.categorical(
                resample_key,
                ancestor_logits,
                shape=(num_particles,),
            ).astype(jnp.int32)
            means = proposal_means[ancestors]
            covariances = proposal_covariances[ancestors]
            chols = jax.vmap(jnp.linalg.cholesky)(covariances)
            eps = random.normal(particle_key, means.shape, dtype=means.dtype)
            particles_t = means + jax.vmap(lambda chol, eps_t: chol @ eps_t)(chols, eps)
            obs_log = _obs_log_prob_batch(context, particles_t, time_idx_i32)
            if guidance == "bffg":
                guide_log_particles = jax.vmap(
                    lambda particle: _gaussian_guide_log_prob(
                        particle,
                        guide_precisions[time_idx],
                        guide_informations[time_idx],
                        guide_constants[time_idx],
                    )
                )(particles_t)
                raw_logw = obs_log - guide_log_particles
            else:
                raw_logw = obs_log - guide_log[ancestors]
            log_norm_t = jsp_special.logsumexp(ancestor_logits) + (
                jsp_special.logsumexp(raw_logw)
                - jnp.log(jnp.asarray(num_particles, dtype=raw_logw.dtype))
            )
            logw_t = _normalize_log_weights(raw_logw)
            ess_t = _ess_from_normalized_log_weights(logw_t)
            next_carry = (
                particles_t,
                logw_t,
                log_norm_acc + log_norm_t,
                jnp.minimum(min_ess, ess_t),
            )
            return next_carry, (particles_t, logw_t)

        if num_steps > 1:
            (_particles_last, logw_last, log_norm, min_ess), (particles_rest, logw_rest) = (
                jax.lax.scan(
                    _filter_step,
                    (particles0, logw0, log_norm0, ess0),
                    jnp.arange(1, num_steps, dtype=jnp.int32),
                )
            )
            particles_all = jnp.concatenate([particles0[None, :, :], particles_rest], axis=0)
            logw_all = jnp.concatenate([logw0[None, :], logw_rest], axis=0)
        else:
            logw_last = logw0
            log_norm = log_norm0
            min_ess = ess0
            particles_all = particles0[None, :, :]
            logw_all = logw0[None, :]

        backward_keys = random.split(backward_key, max(num_steps, 1))
        final_idx = random.categorical(backward_keys[-1], logw_last).astype(jnp.int32)
        final_state = particles_all[-1, final_idx]

        def _backward_step(carry, inputs):
            next_state = carry
            particles_t, logw_t, Ad_next, cd_next, Q_next, key_t = inputs
            transition_means = particles_t @ Ad_next.T + cd_next
            transition_log = jax.vmap(
                lambda mean_t: _gaussian_log_prob_full(next_state, mean_t, Q_next)
            )(transition_means)
            idx_t = random.categorical(key_t, logw_t + transition_log).astype(jnp.int32)
            state_t = particles_t[idx_t]
            return state_t, state_t

        if num_steps > 1:
            _state0, states_rev = jax.lax.scan(
                _backward_step,
                final_state,
                (
                    particles_all[:-1][::-1],
                    logw_all[:-1][::-1],
                    context.Ad[1:][::-1],
                    context.cd[1:][::-1],
                    context.Qd[1:][::-1],
                    backward_keys[:-1],
                ),
            )
            trajectory = jnp.concatenate([states_rev[::-1], final_state[None, :]], axis=0)
        else:
            trajectory = final_state[None, :]

        prior_terms = prior_terms_fn(context)
        trajectory_lp = trajectory_log_prob_fn(context, trajectory, prior_terms)
        return ParticleSmootherInitResult(
            trajectories=trajectory,
            log_normalizer=log_norm,
            min_ess=min_ess,
            trajectory_log_prob=trajectory_lp,
        )

    keys = random.split(random.PRNGKey(seed), init_positions.shape[0])
    result = jax.vmap(_initialize_one)(init_positions, keys)
    diagnostics = {
        "latent_init_method": "particle_smoother",
        "latent_init_algorithm": "ffbsi",
        "latent_init_guidance": guidance,
        "latent_init_num_particles": int(num_particles),
        "latent_init_log_normalizer": jax.device_get(result.log_normalizer).tolist(),
        "latent_init_min_ess": jax.device_get(result.min_ess).tolist(),
        "latent_init_trajectory_log_prob": jax.device_get(result.trajectory_log_prob).tolist(),
    }
    return result.trajectories, diagnostics


def initialize_ieks_latents(
    bundle: dict[str, Any],
    init_positions: jnp.ndarray,
    *,
    model: Any,
    seed: int,
    n_ieks_iters: int = 6,
    noise_scale: float = 0.0,
    reparam: Any = None,
    trace_key: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """One-shot IEKS-based latent init at each given parameter point.

    Runs the iterated extended Kalman smoother (Gauss-Newton MAP for the
    latent given theta) at each ``init_positions[c]``. Cheap and
    deterministic — produces the MAP latent trajectory given theta. Optional
    ``noise_scale`` adds Gaussian perturbation per coordinate for stochastic
    chain diversity.

    Compared with :func:`initialize_particle_smoother_latents`, this is
    ``O(n_ieks_iters * T)`` per chain rather than ``O(N * T)`` — roughly
    70x cheaper at typical (N=200, n_ieks_iters=6). The trade-off is that
    IEKS gives the MAP, not a posterior sample, so chains starting from the
    same theta land at the same x. Per-chain theta perturbation (e.g. via
    Pathfinder multistart) is the usual way to get cross-chain diversity.

    Interface matches :func:`initialize_particle_smoother_latents` apart from
    requiring ``model`` (needed to build the Laplace backend) and
    ``reparam``/``trace_key`` (must match Pathfinder's choices so the flat
    parameter ordering agrees).
    """
    from nof1_causal_lab.models.ssm.inference.methods.map import (
        _build_map_laplace_bundle,
    )

    observations = bundle["observations"]
    times = bundle["times"]
    trajectory_log_prob_fn = bundle["trajectory_log_prob_conditioned_from_context_fn"]
    prior_terms_fn = bundle["prior_terms_from_context_fn"]
    latent_context_runtime_fn = bundle["latent_context_runtime_fn"]
    laplace_mode_to_runtime_latent_trajectory_fn = bundle[
        "laplace_mode_to_runtime_latent_trajectory_fn"
    ]
    initial_observation_auxiliary_from_context_fn = bundle[
        "initial_observation_auxiliary_from_context_fn"
    ]

    if trace_key is None:
        trace_key = random.PRNGKey(seed)

    likelihood_backend = model.make_laplace_backend(n_ieks_iters)
    laplace_bundle = _build_map_laplace_bundle(
        model, observations, times, trace_key, likelihood_backend, reparam
    )
    neg_log_post_with_aux = laplace_bundle["neg_log_posterior_with_aux_fn"]

    init_positions = jnp.asarray(init_positions, dtype=bundle["flat_example"].dtype)
    num_chains = int(init_positions.shape[0])

    trajectories_list: list[jnp.ndarray] = []
    log_posteriors: list[float] = []
    for c in range(num_chains):
        _neg, aux = neg_log_post_with_aux(
            init_positions[c],
            observations,
            times,
            latent_mode_init=None,
        )
        if "latent_mode" not in aux:
            raise RuntimeError(
                "Laplace backend did not return 'latent_mode' in aux dict; "
                "IEKS init is unsupported for this likelihood. Use "
                "latent_init_method='particle_smoother' instead."
            )
        z_mode = laplace_mode_to_runtime_latent_trajectory_fn(
            jnp.asarray(aux["latent_mode"], dtype=bundle["flat_example"].dtype)
        )
        trajectories_list.append(z_mode)
        log_posteriors.append(float(jax.device_get(-_neg)))
    trajectories = jnp.stack(trajectories_list, axis=0)

    if noise_scale > 0.0:
        noise_key = random.PRNGKey(seed)
        chain_keys = random.split(noise_key, num_chains)
        noise = jax.vmap(lambda k, t: random.normal(k, t.shape, dtype=t.dtype))(
            chain_keys,
            trajectories,
        )
        trajectories = trajectories + noise_scale * noise

    traj_lps: list[float] = []
    for c in range(num_chains):
        context = latent_context_runtime_fn(init_positions[c], times)
        prior_terms = prior_terms_fn(context)
        observation_auxiliary = initial_observation_auxiliary_from_context_fn(
            context,
            trajectories[c],
        )
        traj_lps.append(
            float(
                jax.device_get(
                    trajectory_log_prob_fn(
                        context,
                        trajectories[c],
                        observation_auxiliary,
                        prior_terms,
                    )
                )
            )
        )

    diagnostics = {
        "latent_init_method": "ieks",
        "latent_init_algorithm": "iterated_extended_kalman_smoother",
        "latent_init_n_ieks_iters": int(n_ieks_iters),
        "latent_init_noise_scale": float(noise_scale),
        "latent_init_log_posterior_at_mode": log_posteriors,
        "latent_init_trajectory_log_prob": traj_lps,
    }
    return trajectories, diagnostics
