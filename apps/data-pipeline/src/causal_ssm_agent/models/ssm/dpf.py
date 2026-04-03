"""Differentiable Particle Filter with Learned Proposals (DPF).

Implements Method 3 from the algorithmic specification:

1. Training phase: Learn a neural proposal q_phi(z_t | z_{t-1}, y_t) via
   prior-predictive simulation. The proposal is trained to approximate the
   optimal proposal p(z_t | z_{t-1}, y_t) using the VSMC objective.

2. Inference phase: Run a particle filter with the learned proposal on the
   real data, producing an unbiased marginal likelihood estimate and weighted
   particle approximation to the state posterior.

The learned proposal replaces the bootstrap PF's prior proposal with a
data-informed proposal, dramatically reducing particle degeneracy for
non-Gaussian emissions.

Key design choices:
- Gaussian proposal network (MLP-parameterized mean and diagonal covariance)
  rather than normalizing flow, for computational tractability with JAX
- Soft resampling for differentiable training, standard systematic resampling
  at inference time
- Prior-predictive training for the N=1 setting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from numpyro.distributions import MultivariateNormal, Normal

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.likelihoods.emissions import (
    build_predictive_observation_sampler,
)
from causal_ssm_agent.models.likelihoods.kernels import (
    compile_measurement_semantics,
)
from causal_ssm_agent.models.likelihoods.particle import _systematic_resampling
from causal_ssm_agent.models.likelihoods.trajectory_observations import (
    advance_support_observation_state,
    project_response_trajectory,
    support_observation_log_prob,
)
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.tempered_core import run_tempered_smc

logger = get_prefect_logger(__name__)

if TYPE_CHECKING:
    from causal_ssm_agent.models.likelihoods.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )
    from causal_ssm_agent.models.ssm.inference import InferenceResult
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

# ---------------------------------------------------------------------------
# Proposal network: Equinox MLP-parameterized Gaussian
# ---------------------------------------------------------------------------


class ProposalNetwork(eqx.Module):
    """MLP-parameterized Gaussian proposal q_phi(z_t | z_{t-1}, y_t).

    Architecture: [z_{t-1}, y_t] -> 2-layer MLP -> (mu, log_sigma)
    """

    layers: list
    mu_head: eqx.nn.Linear
    log_sigma_head: eqx.nn.Linear

    def __init__(self, D_latent, D_obs, hidden_dim=64, *, key):
        k1, k2, k3, k4 = random.split(key, 4)
        input_dim = D_latent + D_obs
        self.layers = [
            eqx.nn.Linear(input_dim, hidden_dim, key=k1),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=k2),
        ]
        self.mu_head = eqx.nn.Linear(hidden_dim, D_latent, key=k3)
        # Initialize log_sigma bias to -1.0 for small initial variance
        log_sigma_linear = eqx.nn.Linear(hidden_dim, D_latent, key=k4)
        assert log_sigma_linear.bias is not None
        self.log_sigma_head = eqx.tree_at(
            lambda layer: layer.bias, log_sigma_linear, log_sigma_linear.bias - 1.0
        )

    def __call__(self, z_prev, y_t):
        """Forward pass returning (mu, log_sigma)."""
        x = jnp.concatenate([z_prev, y_t])
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        mu = self.mu_head(x)
        log_sigma = jnp.clip(self.log_sigma_head(x), -5.0, 2.0)
        return mu, log_sigma

    def sample(self, z_prev, y_t, rng_key):
        """Sample z_new and compute log q(z_new | z_prev, y_t)."""
        mu, log_sigma = self(z_prev, y_t)
        sigma = jnp.exp(log_sigma)
        eps = random.normal(rng_key, mu.shape)
        z_new = mu + sigma * eps  # reparameterization trick
        log_q = Normal(mu, sigma).log_prob(z_new).sum()
        return z_new, log_q

    def log_prob(self, z_prev, y_t, z_new):
        """Evaluate log q(z_new | z_prev, y_t) without sampling."""
        mu, log_sigma = self(z_prev, y_t)
        sigma = jnp.exp(log_sigma)
        return Normal(mu, sigma).log_prob(z_new).sum()


# ---------------------------------------------------------------------------
# Soft resampling (differentiable)
# ---------------------------------------------------------------------------


def _soft_resample(log_weights, particles, alpha=0.5):
    """Soft resampling: mixture of particle and weighted mean.

    z_tilde_n = alpha * z_n + (1-alpha) * sum_m W_m * z_m

    This is biased but differentiable. Used only during training.

    Args:
        log_weights: (N,) unnormalized log weights
        particles: (N, D) particle states
        alpha: mixture coefficient (1.0 = no resampling, 0.0 = full collapse)

    Returns:
        resampled: (N, D) soft-resampled particles
        new_log_weights: (N,) adjusted log weights
    """
    weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
    weighted_mean = jnp.sum(weights[:, None] * particles, axis=0)  # (D,)
    resampled = alpha * particles + (1 - alpha) * weighted_mean[None, :]
    # Adjust weights: new weight proportional to old weight / resampling weight
    # For soft resampling, the effective proposal is a mixture, so weights
    # are approximately uniform after soft resampling
    new_log_weights = jnp.zeros_like(log_weights)
    return resampled, new_log_weights


def _soft_resample_tree(log_weights, state_tree, alpha=0.5):
    """Soft-resample an arbitrary particle-state pytree."""
    weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))

    def _mix(arr):
        expand_shape = (arr.shape[0],) + (1,) * (arr.ndim - 1)
        weighted_mean = jnp.sum(weights.reshape(expand_shape) * arr, axis=0)
        return alpha * arr + (1.0 - alpha) * weighted_mean[None, ...]

    resampled = jax.tree.map(_mix, state_tree)
    new_log_weights = jnp.zeros_like(log_weights)
    return resampled, new_log_weights


class _SupportAwareDPFState(eqx.Module):
    latent: jnp.ndarray
    response: jnp.ndarray
    accum_sum: jnp.ndarray
    accum_sumsq: jnp.ndarray
    accum_weight: jnp.ndarray


# ---------------------------------------------------------------------------
# DPF: forward pass (differentiable particle filter)
# ---------------------------------------------------------------------------


def _dpf_forward(
    proposal_net,
    observations,
    obs_mask,
    Ad,
    Qd,
    cd,
    H,
    d_meas,
    R,
    init_mean,
    init_cov,
    emission_log_prob_fn,
    n_particles,
    rng_key,
    soft_resample_alpha=0.5,
    training=True,
):
    """Run differentiable particle filter with learned proposals.

    Args:
        proposal_net: ProposalNetwork module
        observations: (T, n_manifest) observations
        obs_mask: (T, n_manifest) mask
        Ad, Qd, cd: (T, D, D), (T, D, D), (T, D) discrete dynamics
        H, d_meas, R: measurement model parameters
        init_mean, init_cov: initial state distribution
        emission_log_prob_fn: callable for log p(y|z)
        n_particles: N
        rng_key: PRNG key
        soft_resample_alpha: alpha for soft resampling (training only)
        training: if True, use soft resampling; else standard systematic

    Returns:
        log_Z: scalar log normalizing constant estimate
    """
    T, _n_manifest = observations.shape
    D = init_mean.shape[0]
    jitter = jnp.eye(D) * 1e-6

    # Initialize particles from prior
    key_init, rng_key = random.split(rng_key)
    chol_P0 = jla.cholesky(init_cov + jitter, lower=True)
    init_keys = random.split(key_init, n_particles)
    particles = jax.vmap(lambda k: init_mean + chol_P0 @ random.normal(k, (D,)))(
        init_keys
    )  # (N, D)

    # Weight against first observation
    mask_float = obs_mask.astype(jnp.float32)
    log_weights = jax.vmap(
        lambda z: emission_log_prob_fn(observations[0], z, H, d_meas, R, mask_float[0])
    )(particles)  # (N,)

    log_Z = jax.nn.logsumexp(log_weights) - jnp.log(float(n_particles))

    def _pf_step(carry, inputs):
        particles_prev, log_weights_prev, log_Z_acc, rng_key = carry
        y_t, mask_t, Ad_t, Qd_t, cd_t = inputs

        # Resample
        if training:
            particles_resampled, _log_weights_resampled = _soft_resample(
                log_weights_prev, particles_prev, soft_resample_alpha
            )
        else:
            # Standard systematic resampling (non-differentiable, fine for inference)
            from causal_ssm_agent.models.likelihoods.particle import _systematic_resampling

            rng_key, resample_key = random.split(rng_key)
            idx = _systematic_resampling(resample_key, log_weights_prev, n_particles)
            particles_resampled = particles_prev[idx]

        # Propose: z_t ~ q_phi(z_t | z_{t-1}, y_t)
        rng_key, propose_key = random.split(rng_key)
        propose_keys = random.split(propose_key, n_particles)

        def _propose_and_weight(key, z_prev):
            z_new, log_q = proposal_net.sample(z_prev, y_t, key)

            # Transition log-prob: log p(z_new | z_prev)
            mean_trans = Ad_t @ z_prev + cd_t
            Qd_reg = Qd_t + jitter
            log_trans = MultivariateNormal(mean_trans, covariance_matrix=Qd_reg).log_prob(z_new)

            # Emission log-prob: log p(y_t | z_new)
            log_obs = emission_log_prob_fn(y_t, z_new, H, d_meas, R, mask_t)

            # Importance weight: log w = log p(y|z) + log p(z|z_prev) - log q(z|z_prev,y)
            log_w = log_obs + log_trans - log_q
            return z_new, log_w

        particles_new, log_w_new = jax.vmap(_propose_and_weight)(propose_keys, particles_resampled)

        # Accumulate log normalizing constant
        log_Z_step = jax.nn.logsumexp(log_w_new) - jnp.log(float(n_particles))
        log_Z_new = log_Z_acc + log_Z_step

        return (particles_new, log_w_new, log_Z_new, rng_key), None

    if T > 1:
        scan_inputs = (
            observations[1:],
            mask_float[1:],
            Ad[1:],
            Qd[1:],
            cd[1:],
        )

        (_particles_final, _log_weights_final, log_Z_total, _), _ = jax.lax.scan(
            _pf_step,
            (particles, log_weights, log_Z, rng_key),
            scan_inputs,
        )
    else:
        log_Z_total = log_Z

    return log_Z_total


def _dpf_forward_support_aware(
    proposal_net,
    observations,
    obs_mask,
    Ad,
    Qd,
    cd,
    H,
    d_meas,
    R,
    init_mean,
    init_cov,
    measurement_semantics,
    n_particles,
    rng_key,
    soft_resample_alpha=0.5,
    training=True,
):
    """Run DPF with interval-summary observation semantics."""
    T, _n_manifest = observations.shape
    D = init_mean.shape[0]
    jitter = jnp.eye(D) * 1e-6
    mask_float = obs_mask.astype(jnp.float32)
    obs_kernel = measurement_semantics.obs_kernel
    observation_operator = measurement_semantics.observation_operator
    mean_log_prob_fn = measurement_semantics.mean_log_prob_fn
    if mean_log_prob_fn is None or not observation_operator.requires_interval_summary_handling:
        raise ValueError(
            "_dpf_forward_support_aware requires compiled interval-summary measurement semantics"
        )
    assert observation_operator.prev_coeffs is not None
    assert observation_operator.curr_coeffs is not None
    assert observation_operator.interval_weights is not None
    assert observation_operator.emission_slots is not None
    prev_coeffs = jnp.asarray(observation_operator.prev_coeffs, dtype=observations.dtype)
    curr_coeffs = jnp.asarray(observation_operator.curr_coeffs, dtype=observations.dtype)
    interval_weights = jnp.asarray(observation_operator.interval_weights, dtype=observations.dtype)
    emission_slots = jnp.asarray(observation_operator.emission_slots, dtype=jnp.int32)

    key_init, rng_key = random.split(rng_key)
    chol_P0 = jla.cholesky(init_cov + jitter, lower=True)
    init_keys = random.split(key_init, n_particles)
    init_latents = jax.vmap(lambda k: init_mean + chol_P0 @ random.normal(k, (D,)))(init_keys)
    init_responses = jax.vmap(lambda z: obs_kernel.response_fn(H @ z + d_meas))(init_latents)
    zeros = observation_operator.empty_accumulators(observations.dtype, (n_particles,))
    states = _SupportAwareDPFState(
        latent=init_latents,
        response=init_responses,
        accum_sum=zeros,
        accum_sumsq=zeros,
        accum_weight=zeros,
    )

    log_weights = jax.vmap(
        lambda z: obs_kernel.emission_fn(
            observations[0],
            z,
            H,
            d_meas,
            R,
            mask_float[0] * observation_operator.point_like_mask(mask_float.dtype),
        )
    )(states.latent)
    log_Z = jax.nn.logsumexp(log_weights) - jnp.log(float(n_particles))

    def _pf_step(carry, inputs):
        state_prev, log_weights_prev, log_Z_acc, rng_key = carry
        y_t, mask_t, Ad_t, Qd_t, cd_t, prev_coeff_t, curr_coeff_t, weight_t, emission_slots_t = (
            inputs
        )

        if training:
            states_resampled, _ = _soft_resample_tree(
                log_weights_prev,
                state_prev,
                soft_resample_alpha,
            )
        else:
            rng_key, resample_key = random.split(rng_key)
            idx = _systematic_resampling(resample_key, log_weights_prev, n_particles)
            states_resampled = jax.tree.map(lambda arr: arr[idx], state_prev)

        rng_key, propose_key = random.split(rng_key)
        propose_keys = random.split(propose_key, n_particles)

        Qd_reg = Qd_t + jitter

        def _propose_and_weight(
            key, latent_prev, response_prev, accum_sum, accum_sumsq, accum_weight
        ):
            z_new, log_q = proposal_net.sample(latent_prev, y_t, key)
            mean_trans = Ad_t @ latent_prev + cd_t
            log_trans = MultivariateNormal(mean_trans, covariance_matrix=Qd_reg).log_prob(z_new)
            response_new = obs_kernel.response_fn(H @ z_new + d_meas)
            step_result = advance_support_observation_state(
                observation_operator,
                response_prev,
                accum_sum,
                accum_sumsq,
                accum_weight,
                response_new,
                mask_t,
                prev_coeff_t,
                curr_coeff_t,
                weight_t,
                emission_slots_t,
            )
            log_w = support_observation_log_prob(
                observation_operator,
                obs_kernel,
                mean_log_prob_fn,
                y_t,
                mask_t,
                z_new,
                H,
                d_meas,
                R,
                step_result.summary,
            )
            next_state = _SupportAwareDPFState(
                latent=z_new,
                response=response_new,
                accum_sum=step_result.next_accum_sum,
                accum_sumsq=step_result.next_accum_sumsq,
                accum_weight=step_result.next_accum_weight,
            )
            return next_state, log_w + log_trans - log_q

        states_new, log_w_new = jax.vmap(_propose_and_weight)(
            propose_keys,
            states_resampled.latent,
            states_resampled.response,
            states_resampled.accum_sum,
            states_resampled.accum_sumsq,
            states_resampled.accum_weight,
        )

        log_Z_step = jax.nn.logsumexp(log_w_new) - jnp.log(float(n_particles))
        log_Z_new = log_Z_acc + log_Z_step
        return (states_new, log_w_new, log_Z_new, rng_key), None

    if T > 1:
        scan_inputs = (
            observations[1:],
            mask_float[1:],
            Ad[1:],
            Qd[1:],
            cd[1:],
            prev_coeffs[1:],
            curr_coeffs[1:],
            interval_weights[1:],
            emission_slots[1:],
        )
        (_states_final, _log_weights_final, log_Z_total, _), _ = jax.lax.scan(
            _pf_step,
            (states, log_weights, log_Z, rng_key),
            scan_inputs,
        )
    else:
        log_Z_total = log_Z

    return log_Z_total


# ---------------------------------------------------------------------------
# Prior-predictive training
# ---------------------------------------------------------------------------


def _simulate_from_prior(
    rng_key,
    T,
    D,
    n_manifest,
    drift_prior_mu,
    drift_prior_sigma,
    diff_prior_sigma,
    H,
    d_meas,
    R,
    manifest_dist="gaussian",
    manifest_link="identity",
    extra_params=None,
    time_intervals=None,
    observation_support=None,
    obs_mask_template=None,
):
    """Simulate a single trajectory from the prior predictive distribution.

    Returns:
        observations: (T, n_manifest)
        Ad: (T, D, D)
        Qd: (T, D, D)
        cd: (T, D)
        init_mean: (D,)
        init_cov: (D, D)
    """
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

    k1, k2, k3, k4, k5 = random.split(rng_key, 5)

    # Sample drift matrix (diagonal negative, small off-diagonal)
    drift_diag = -jnp.abs(random.normal(k1, (D,)) * drift_prior_sigma + drift_prior_mu)
    drift = jnp.diag(drift_diag)

    # Sample diffusion
    diff_diag = jnp.abs(random.normal(k2, (D,)) * diff_prior_sigma + 0.3)
    diffusion_cov = jnp.diag(diff_diag**2)

    if time_intervals is None:
        time_intervals = jnp.ones((T,), dtype=jnp.float32)
    else:
        time_intervals = jnp.asarray(time_intervals, dtype=jnp.float32)

    Ad_all, Qd_all, cd_all = discretize_system_batched(drift, diffusion_cov, None, time_intervals)
    if cd_all is None:
        cd_all = jnp.zeros((T, D), dtype=jnp.float32)
    jitter = jnp.eye(D) * 1e-6
    chol_Qd = jax.vmap(lambda q_t: jla.cholesky(q_t + jitter, lower=True))(Qd_all)

    # Simulate latent trajectory
    init_mean = jnp.zeros(D)
    init_cov = jnp.eye(D)
    z0 = init_mean + jla.cholesky(init_cov + jitter, lower=True) @ random.normal(k3, (D,))

    def _sim_step(z_prev, inputs):
        Ad_t, chol_Qd_t, cd_t, key = inputs
        z_new = Ad_t @ z_prev + cd_t + chol_Qd_t @ random.normal(key, (D,))
        return z_new, z_new

    if T > 1:
        sim_keys = random.split(k4, T - 1)
        _, z_rest = jax.lax.scan(
            _sim_step,
            z0,
            (Ad_all[1:], chol_Qd[1:], cd_all[1:], sim_keys),
        )
        z_all = jnp.concatenate([z0[None], z_rest], axis=0)
    else:
        z_all = z0[None]

    measurement_semantics = compile_measurement_semantics(
        DistributionFamily(manifest_dist),
        manifest_cov=R,
        extra_params=extra_params,
        manifest_link=LinkFunction(manifest_link),
        observation_support=observation_support,
    )
    obs_kernel = measurement_semantics.obs_kernel
    responses = jax.vmap(lambda z_t: obs_kernel.response_fn(H @ z_t + d_meas))(z_all)
    try:
        predictive_sampler = build_predictive_observation_sampler(
            manifest_dist=manifest_dist,
            manifest_cov=R,
            manifest_link=manifest_link,
            extra_params=extra_params,
        )
    except ValueError as exc:
        raise NotImplementedError(
            f"Support-aware DPF proposal training does not support '{manifest_dist}'."
        ) from exc

    if obs_mask_template is None:
        obs_mask_template = jnp.ones((T, n_manifest), dtype=bool)

    expected_means, semantic_mask = project_response_trajectory(responses, observation_support)
    effective_mask = obs_mask_template & (semantic_mask > 0.5)
    observations = predictive_sampler.sample_mean_trajectory(k5, expected_means)
    observations = jnp.where(effective_mask, observations, jnp.zeros_like(observations))

    return observations, Ad_all, Qd_all, cd_all, init_mean, init_cov


def _train_proposal(
    D_latent,
    n_manifest,
    H,
    d_meas,
    R,
    manifest_dist="gaussian",
    manifest_link="identity",
    extra_params=None,
    observation_support=None,
    time_intervals=None,
    obs_mask_template=None,
    n_train_seqs=50,
    n_train_steps=200,
    n_particles_train=32,
    T_train=50,
    lr=1e-3,
    seed=0,
):
    """Train the proposal network on prior-predictive simulations.

    Args:
        D_latent: latent dimension
        n_manifest: observation dimension
        H, d_meas, R: measurement model
        manifest_dist: emission type (DistributionFamily enum or string)
        manifest_link: link function (LinkFunction enum or string)
        extra_params: emission hyperparameters
        n_train_seqs: number of training sequences
        n_train_steps: gradient steps
        n_particles_train: particles during training
        T_train: training sequence length
        lr: learning rate
        seed: random seed

    Returns:
        trained ProposalNetwork
    """
    import optax

    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

    rng_key = random.PRNGKey(seed)

    # Initialize proposal network
    rng_key, init_key = random.split(rng_key)
    proposal_net = ProposalNetwork(D_latent, n_manifest, key=init_key)

    # Pre-generate training sequences
    rng_key, data_key = random.split(rng_key)
    data_keys = random.split(data_key, n_train_seqs)

    measurement_semantics = compile_measurement_semantics(
        DistributionFamily(manifest_dist),
        manifest_cov=R,
        extra_params=extra_params,
        manifest_link=LinkFunction(manifest_link),
        observation_support=observation_support,
    )
    obs_kernel = measurement_semantics.obs_kernel
    init_mean = jnp.zeros(D_latent)
    init_cov = jnp.eye(D_latent)

    def _gen_one(key):
        obs, Ad, Qd, cd, _im, _ic = _simulate_from_prior(
            key,
            T_train,
            D_latent,
            n_manifest,
            drift_prior_mu=-0.5,
            drift_prior_sigma=0.3,
            diff_prior_sigma=0.3,
            H=H,
            d_meas=d_meas,
            R=R,
            manifest_dist=manifest_dist,
            manifest_link=manifest_link,
            extra_params=extra_params,
            time_intervals=time_intervals,
            observation_support=observation_support,
            obs_mask_template=obs_mask_template,
        )
        return obs, Ad, Qd, cd

    all_obs, all_Ad, all_Qd, all_cd = jax.vmap(_gen_one)(data_keys)

    if obs_mask_template is None:
        obs_mask_all = jnp.ones_like(all_obs, dtype=bool)
    else:
        obs_mask_all = jnp.broadcast_to(
            jnp.asarray(obs_mask_template, dtype=bool),
            all_obs.shape,
        )

    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(lr))
    opt_state = optimizer.init(eqx.filter(proposal_net, eqx.is_array))

    def _vsmc_loss(proposal_net, batch_idx, rng_key):
        """Negative VSMC objective for a batch of sequences."""
        obs = all_obs[batch_idx]
        Ad = all_Ad[batch_idx]
        Qd = all_Qd[batch_idx]
        cd = all_cd[batch_idx]
        mask = obs_mask_all[batch_idx]

        if (
            observation_support is not None
            and observation_support.requires_interval_summary_handling
        ):
            log_Z = _dpf_forward_support_aware(
                proposal_net,
                obs,
                mask,
                Ad,
                Qd,
                cd,
                H,
                d_meas,
                R,
                init_mean,
                init_cov,
                measurement_semantics,
                n_particles_train,
                rng_key,
                soft_resample_alpha=0.5,
                training=True,
            )
        else:
            log_Z = _dpf_forward(
                proposal_net,
                obs,
                mask,
                Ad,
                Qd,
                cd,
                H,
                d_meas,
                R,
                init_mean,
                init_cov,
                obs_kernel.emission_fn,
                n_particles_train,
                rng_key,
                soft_resample_alpha=0.5,
                training=True,
            )
        return -log_Z  # Minimize negative VSMC

    @eqx.filter_jit
    def _train_step(proposal_net, opt_state, batch_idx, rng_key):
        loss, grads = eqx.filter_value_and_grad(_vsmc_loss)(proposal_net, batch_idx, rng_key)
        updates, opt_state_new = optimizer.update(grads, opt_state)
        proposal_net_new = eqx.apply_updates(proposal_net, updates)
        return proposal_net_new, opt_state_new, loss

    logger.info(
        "  Training proposal: %s steps, %s sequences, %s particles...",
        n_train_steps,
        n_train_seqs,
        n_particles_train,
    )

    for step in range(n_train_steps):
        rng_key, step_key, batch_key = random.split(rng_key, 3)
        batch_idx = random.randint(batch_key, (), 0, n_train_seqs)
        proposal_net, opt_state, loss = _train_step(proposal_net, opt_state, batch_idx, step_key)
        if (step + 1) % 50 == 0:
            logger.info("    step %s/%s: VSMC loss = %.2f", step + 1, n_train_steps, float(loss))

    return proposal_net


# ---------------------------------------------------------------------------
# DPF Likelihood backend
# ---------------------------------------------------------------------------


class DPFLikelihood:
    """Differentiable Particle Filter likelihood backend with learned proposals.

    First trains a proposal network on prior-predictive data, then uses it
    for efficient importance sampling in the particle filter.
    """

    checkpoint_loglik = True

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        manifest_dist: DistributionFamily | str = "gaussian",
        manifest_link: LinkFunction | str = "identity",
        n_particles: int = 100,
        proposal_net: ProposalNetwork | None = None,
        rng_key: jax.Array | None = None,
        observation_support=None,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dist = manifest_dist
        self.manifest_link = manifest_link
        self.n_particles = n_particles
        self.proposal_net = proposal_net
        self.rng_key = rng_key if rng_key is not None else random.PRNGKey(0)
        self.observation_support = observation_support

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
    ) -> float:
        """Compute log-likelihood via DPF with learned proposals."""
        n = self.n_latent
        T = observations.shape[0]

        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)
        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift, ct_params.diffusion_cov, ct_params.cint, time_intervals
        )
        if cd is None:
            cd = jnp.zeros((T, n))

        measurement_semantics = compile_measurement_semantics(
            self.manifest_dist,
            manifest_cov=measurement_params.manifest_cov,
            extra_params=extra_params,
            manifest_link=self.manifest_link,
            observation_support=self.observation_support,
        )
        obs_kernel = measurement_semantics.obs_kernel

        if self.proposal_net is None:
            raise ValueError("Proposal not trained. Call _train_proposal first.")

        if measurement_semantics.requires_interval_summary_handling:
            return _dpf_forward_support_aware(
                self.proposal_net,
                clean_obs,
                obs_mask,
                Ad,
                Qd,
                cd,
                measurement_params.lambda_mat,
                measurement_params.manifest_means,
                measurement_params.manifest_cov,
                initial_state.mean,
                initial_state.cov,
                measurement_semantics,
                self.n_particles,
                self.rng_key,
                training=False,
            )

        log_Z = _dpf_forward(
            self.proposal_net,
            clean_obs,
            obs_mask,
            Ad,
            Qd,
            cd,
            measurement_params.lambda_mat,
            measurement_params.manifest_means,
            measurement_params.manifest_cov,
            initial_state.mean,
            initial_state.cov,
            obs_kernel.emission_fn,
            self.n_particles,
            self.rng_key,
            training=False,
        )
        return log_Z


# ---------------------------------------------------------------------------
# fit_dpf: full pipeline (train + inference)
# ---------------------------------------------------------------------------


def fit_dpf(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    n_outer: int = 100,
    n_csmc_particles: int = 20,
    n_mh_steps: int = 10,
    param_step_size: float = 0.1,
    n_warmup: int | None = None,
    target_accept: float | None = None,
    seed: int = 0,
    n_leapfrog: int = 5,
    adaptive_tempering: bool = True,
    target_ess_ratio: float = 0.5,
    waste_free: bool = False,
    # DPF-specific
    n_pf_particles: int = 100,
    n_train_seqs: int = 50,
    n_train_steps: int = 200,
    n_particles_train: int = 32,
    T_train: int | None = None,
    proposal_lr: float = 1e-3,
    reparam=None,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM via DPF with learned proposals + tempered SMC outer loop.

    Phase 1: Train a proposal network on prior-predictive simulations.
    Phase 2: Use the trained proposal in a particle filter for likelihood
             estimation, with tempered SMC for parameter inference.
    """
    T_data = observations.shape[0]
    observation_support = getattr(model, "observation_support", None)
    requires_interval_summary_handling = bool(
        observation_support is not None and observation_support.requires_interval_summary_handling
    )
    if requires_interval_summary_handling:
        if T_train is not None and T_train != T_data:
            raise ValueError(
                "Support-aware DPF proposal training requires T_train to match the "
                "prepared observation schedule."
            )
        T_train = T_data
    elif T_train is None:
        T_train = min(T_data, 50)

    rng_key = random.PRNGKey(seed)
    spec = model.spec

    # Phase 1: Train proposal network
    # Extract measurement model from spec
    with jax.ensure_compile_time_eval():
        if isinstance(spec.lambda_mat, jnp.ndarray):
            H_train = spec.lambda_mat
        else:
            H_train = jnp.eye(spec.n_manifest, spec.n_latent)
        if isinstance(spec.manifest_means, jnp.ndarray):
            d_train = spec.manifest_means
        elif spec.manifest_means is None:
            d_train = jnp.zeros(spec.n_manifest)
        else:
            d_train = jnp.zeros(spec.n_manifest)
        if isinstance(spec.manifest_var, jnp.ndarray):
            R_train = spec.manifest_var @ spec.manifest_var.T
        else:
            R_train = jnp.eye(spec.n_manifest) * 0.25

    logger.info("DPF: Phase 1 - Training proposal network...")
    rng_key, train_key = random.split(rng_key)
    train_time_intervals = None
    train_obs_mask = None
    if requires_interval_summary_handling:
        train_time_intervals = jnp.diff(times, prepend=times[0])
        first_dt = (
            jnp.maximum(times[1] - times[0], 1e-3)
            if times.shape[0] > 1
            else jnp.asarray(1.0, dtype=times.dtype)
        )
        train_time_intervals = train_time_intervals.at[0].set(first_dt)
        train_obs_mask = ~jnp.isnan(observations)
    proposal_net = _train_proposal(
        D_latent=spec.n_latent,
        n_manifest=spec.n_manifest,
        H=H_train,
        d_meas=d_train,
        R=R_train,
        manifest_dist=spec.manifest_dist,
        manifest_link=spec.manifest_link,
        observation_support=observation_support,
        time_intervals=train_time_intervals,
        obs_mask_template=train_obs_mask,
        n_train_seqs=n_train_seqs,
        n_train_steps=n_train_steps,
        n_particles_train=n_particles_train,
        T_train=T_train,
        lr=proposal_lr,
        seed=int(train_key[0]),
    )

    logger.info("DPF: Phase 2 - Parameter inference via tempered SMC...")

    # Phase 2: Parameter inference via shared tempered SMC loop
    rng_key, dpf_key = random.split(rng_key)
    if model.likelihood == "kalman":
        backend = model.make_likelihood_backend()
    else:
        backend = DPFLikelihood(
            n_latent=spec.n_latent,
            n_manifest=spec.n_manifest,
            manifest_dist=spec.manifest_dist,
            manifest_link=spec.manifest_link,
            n_particles=n_pf_particles,
            proposal_net=proposal_net,
            rng_key=dpf_key,
            observation_support=observation_support,
        )
    return run_tempered_smc(
        model,
        observations,
        times,
        n_outer=n_outer,
        n_csmc_particles=n_csmc_particles,
        n_mh_steps=n_mh_steps,
        param_step_size=param_step_size,
        n_warmup=n_warmup,
        target_accept=target_accept,
        seed=seed,
        adaptive_tempering=adaptive_tempering,
        target_ess_ratio=target_ess_ratio,
        waste_free=waste_free,
        n_leapfrog=n_leapfrog,
        method_name="dpf",
        likelihood_backend=backend,
        extra_diagnostics={
            "n_pf_particles": n_pf_particles,
            "n_train_seqs": n_train_seqs,
            "n_train_steps": n_train_steps,
            "proposal_net": proposal_net,
        },
        print_prefix="DPF",
        reparam=reparam,
    )
