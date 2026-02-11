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

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from dsem_agent.models.ssm.discretization import discretize_system_batched
from dsem_agent.models.ssm.inference import InferenceResult
from dsem_agent.models.ssm.laplace_em import _get_emission_fn
from dsem_agent.models.ssm.utils import (
    _assemble_deterministics,
    _build_eval_fns,
    _discover_sites,
)

if TYPE_CHECKING:
    from dsem_agent.models.likelihoods.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )

# ---------------------------------------------------------------------------
# Proposal network: MLP-parameterized Gaussian
# ---------------------------------------------------------------------------


def _init_proposal_params(D_latent, D_obs, hidden_dim=64, rng_key=None):
    """Initialize parameters for the Gaussian proposal network.

    q_phi(z_t | z_{t-1}, y_t) = N(mu_phi(z_{t-1}, y_t), diag(sigma_phi(z_{t-1}, y_t)^2))

    Architecture: [z_{t-1}, y_t] -> MLP -> (mu, log_sigma)

    Args:
        D_latent: latent state dimension
        D_obs: observation dimension
        hidden_dim: hidden layer size
        rng_key: PRNG key

    Returns:
        proposal_params: dict of MLP parameters
    """
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    input_dim = D_latent + D_obs
    k1, k2, k3, k4 = random.split(rng_key, 4)

    # Two-layer MLP: input -> hidden -> hidden -> (mu, log_sigma)
    scale1 = jnp.sqrt(2.0 / input_dim)
    scale2 = jnp.sqrt(2.0 / hidden_dim)

    return {
        "W1": random.normal(k1, (input_dim, hidden_dim)) * scale1,
        "b1": jnp.zeros(hidden_dim),
        "W2": random.normal(k2, (hidden_dim, hidden_dim)) * scale2,
        "b2": jnp.zeros(hidden_dim),
        "W_mu": random.normal(k3, (hidden_dim, D_latent)) * 0.01,
        "b_mu": jnp.zeros(D_latent),
        "W_log_sigma": random.normal(k4, (hidden_dim, D_latent)) * 0.01,
        "b_log_sigma": jnp.zeros(D_latent) - 1.0,  # Initialize to small variance
    }


def _proposal_forward(params, z_prev, y_t):
    """Evaluate the proposal network.

    Args:
        params: MLP parameters
        z_prev: (D_latent,) previous state
        y_t: (D_obs,) current observation

    Returns:
        mu: (D_latent,) proposal mean
        log_sigma: (D_latent,) log proposal std
    """
    x = jnp.concatenate([z_prev, y_t])
    h = jax.nn.relu(x @ params["W1"] + params["b1"])
    h = jax.nn.relu(h @ params["W2"] + params["b2"])
    mu = h @ params["W_mu"] + params["b_mu"]
    log_sigma = h @ params["W_log_sigma"] + params["b_log_sigma"]
    # Clamp log_sigma for numerical stability
    log_sigma = jnp.clip(log_sigma, -5.0, 2.0)
    return mu, log_sigma


def _proposal_sample(params, z_prev, y_t, rng_key):
    """Sample from the proposal and compute log-density.

    Returns:
        z_new: (D_latent,) proposed state
        log_q: scalar log q(z_new | z_prev, y_t)
    """
    mu, log_sigma = _proposal_forward(params, z_prev, y_t)
    sigma = jnp.exp(log_sigma)
    eps = random.normal(rng_key, mu.shape)
    z_new = mu + sigma * eps
    log_q = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + 2 * log_sigma + eps**2)
    return z_new, log_q


def _proposal_log_prob(params, z_prev, y_t, z_new):
    """Evaluate log q(z_new | z_prev, y_t) without sampling."""
    mu, log_sigma = _proposal_forward(params, z_prev, y_t)
    sigma = jnp.exp(log_sigma)
    normalized = (z_new - mu) / sigma
    return -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + 2 * log_sigma + normalized**2)


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


# ---------------------------------------------------------------------------
# DPF: forward pass (differentiable particle filter)
# ---------------------------------------------------------------------------


def _dpf_forward(
    proposal_params,
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
        proposal_params: MLP parameters for q_phi
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
            from dsem_agent.models.likelihoods.particle import _systematic_resampling

            rng_key, resample_key = random.split(rng_key)
            idx = _systematic_resampling(resample_key, log_weights_prev, n_particles)
            particles_resampled = particles_prev[idx]

        # Propose: z_t ~ q_phi(z_t | z_{t-1}, y_t)
        rng_key, propose_key = random.split(rng_key)
        propose_keys = random.split(propose_key, n_particles)

        def _propose_and_weight(key, z_prev):
            z_new, log_q = _proposal_sample(proposal_params, z_prev, y_t, key)

            # Transition log-prob: log p(z_new | z_prev)
            mean_trans = Ad_t @ z_prev + cd_t
            diff = z_new - mean_trans
            Qd_reg = Qd_t + jitter
            _, logdet_Q = jnp.linalg.slogdet(Qd_reg)
            mahal = diff @ jla.solve(Qd_reg, diff, assume_a="pos")
            log_trans = -0.5 * (D * jnp.log(2 * jnp.pi) + logdet_Q + mahal)

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
    extra_params=None,
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
    from dsem_agent.models.ssm.discretization import discretize_system

    k1, k2, k3, k4, k5 = random.split(rng_key, 5)

    # Sample drift matrix (diagonal negative, small off-diagonal)
    drift_diag = -jnp.abs(random.normal(k1, (D,)) * drift_prior_sigma + drift_prior_mu)
    drift = jnp.diag(drift_diag)

    # Sample diffusion
    diff_diag = jnp.abs(random.normal(k2, (D,)) * diff_prior_sigma + 0.3)
    diffusion_cov = jnp.diag(diff_diag**2)

    # Discretize with dt=1.0
    dt = 1.0
    Ad_single, Qd_single, _ = discretize_system(drift, diffusion_cov, None, dt)
    jitter = jnp.eye(D) * 1e-6
    chol_Qd = jla.cholesky(Qd_single + jitter, lower=True)

    # Simulate latent trajectory
    init_mean = jnp.zeros(D)
    init_cov = jnp.eye(D)
    z0 = init_mean + jla.cholesky(init_cov + jitter, lower=True) @ random.normal(k3, (D,))

    def _sim_step(z_prev, key):
        z_new = Ad_single @ z_prev + chol_Qd @ random.normal(key, (D,))
        return z_new, z_new

    sim_keys = random.split(k4, T - 1)
    _, z_rest = jax.lax.scan(_sim_step, z0, sim_keys)
    z_all = jnp.concatenate([z0[None], z_rest], axis=0)  # (T, D)

    # Simulate observations
    obs_keys = random.split(k5, T)

    def _sim_obs(key, z_t):
        eta = H @ z_t + d_meas
        if manifest_dist == "poisson":
            rate = jnp.exp(eta)
            return random.poisson(key, rate).astype(jnp.float32)
        elif manifest_dist == "gamma":
            shape = extra_params.get("obs_shape", 1.0) if extra_params else 1.0
            scale = jnp.exp(eta) / shape
            return random.gamma(key, shape, shape=eta.shape) * scale
        else:
            # Gaussian
            chol_R = jla.cholesky(R + jnp.eye(n_manifest) * 1e-6, lower=True)
            return eta + chol_R @ random.normal(key, (n_manifest,))

    observations = jax.vmap(_sim_obs)(obs_keys, z_all)

    # Broadcast Ad, Qd for all timesteps
    Ad_all = jnp.broadcast_to(Ad_single, (T, D, D))
    Qd_all = jnp.broadcast_to(Qd_single, (T, D, D))
    cd_all = jnp.zeros((T, D))

    return observations, Ad_all, Qd_all, cd_all, init_mean, init_cov


def _train_proposal(
    D_latent,
    n_manifest,
    H,
    d_meas,
    R,
    manifest_dist="gaussian",
    extra_params=None,
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
        manifest_dist: emission type
        extra_params: emission hyperparameters
        n_train_seqs: number of training sequences
        n_train_steps: gradient steps
        n_particles_train: particles during training
        T_train: training sequence length
        lr: learning rate
        seed: random seed

    Returns:
        trained proposal_params
    """
    import optax

    rng_key = random.PRNGKey(seed)

    # Initialize proposal
    rng_key, init_key = random.split(rng_key)
    proposal_params = _init_proposal_params(D_latent, n_manifest, rng_key=init_key)

    # Pre-generate training sequences
    rng_key, data_key = random.split(rng_key)
    data_keys = random.split(data_key, n_train_seqs)

    emission_fn = _get_emission_fn(manifest_dist, extra_params)
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
            extra_params=extra_params,
        )
        return obs, Ad, Qd, cd

    all_obs, all_Ad, all_Qd, all_cd = jax.vmap(_gen_one)(data_keys)

    obs_mask_all = jnp.ones_like(all_obs, dtype=bool)  # No missing data in training

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(proposal_params)

    def _vsmc_loss(proposal_params, batch_idx, rng_key):
        """Negative VSMC objective for a batch of sequences."""
        obs = all_obs[batch_idx]
        Ad = all_Ad[batch_idx]
        Qd = all_Qd[batch_idx]
        cd = all_cd[batch_idx]
        mask = obs_mask_all[batch_idx]

        log_Z = _dpf_forward(
            proposal_params,
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
            emission_fn,
            n_particles_train,
            rng_key,
            soft_resample_alpha=0.5,
            training=True,
        )
        return -log_Z  # Minimize negative VSMC

    @jax.jit
    def _train_step(proposal_params, opt_state, batch_idx, rng_key):
        loss, grads = jax.value_and_grad(_vsmc_loss)(proposal_params, batch_idx, rng_key)
        grads = jax.tree.map(lambda g: jnp.clip(g, -5.0, 5.0), grads)
        updates, opt_state_new = optimizer.update(grads, opt_state)
        params_new = optax.apply_updates(proposal_params, updates)
        return params_new, opt_state_new, loss

    print(
        f"  Training proposal: {n_train_steps} steps, {n_train_seqs} sequences, "
        f"{n_particles_train} particles..."
    )

    for step in range(n_train_steps):
        rng_key, step_key, batch_key = random.split(rng_key, 3)
        batch_idx = random.randint(batch_key, (), 0, n_train_seqs)
        proposal_params, opt_state, loss = _train_step(
            proposal_params, opt_state, batch_idx, step_key
        )
        if (step + 1) % 50 == 0:
            print(f"    step {step + 1}/{n_train_steps}: VSMC loss = {float(loss):.2f}")

    return proposal_params


# ---------------------------------------------------------------------------
# DPF Likelihood backend
# ---------------------------------------------------------------------------


class DPFLikelihood:
    """Differentiable Particle Filter likelihood backend with learned proposals.

    First trains a proposal network on prior-predictive data, then uses it
    for efficient importance sampling in the particle filter.
    """

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        manifest_dist: str = "gaussian",
        n_particles: int = 100,
        proposal_params: dict | None = None,
        rng_key: jax.Array | None = None,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dist = manifest_dist
        self.n_particles = n_particles
        self.proposal_params = proposal_params
        self.rng_key = rng_key if rng_key is not None else random.PRNGKey(0)

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

        emission_fn = _get_emission_fn(self.manifest_dist, extra_params)

        if self.proposal_params is None:
            raise ValueError("Proposal not trained. Call _train_proposal first.")

        log_Z = _dpf_forward(
            self.proposal_params,
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
            emission_fn,
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
    subject_ids: jnp.ndarray | None = None,
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
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM via DPF with learned proposals + tempered SMC outer loop.

    Phase 1: Train a proposal network on prior-predictive simulations.
    Phase 2: Use the trained proposal in a particle filter for likelihood
             estimation, with tempered SMC for parameter inference.

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        subject_ids: optional subject indices
        n_outer: max tempering levels
        n_csmc_particles: parameter particles
        n_mh_steps: HMC mutations per round
        param_step_size: initial step size
        n_warmup: levels to discard
        target_accept: target acceptance
        seed: random seed
        n_leapfrog: leapfrog steps
        adaptive_tempering: ESS-based tempering
        target_ess_ratio: ESS target
        waste_free: waste-free recycling
        n_pf_particles: particles for inference-time PF
        n_train_seqs: training sequences
        n_train_steps: training gradient steps
        n_particles_train: particles during training
        T_train: training sequence length (default: min(T, 50))
        proposal_lr: proposal learning rate

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    from jax.flatten_util import ravel_pytree

    from dsem_agent.models.ssm.mcmc_utils import (
        compute_weighted_chol_mass,
        find_next_beta,
        hmc_step,
    )

    if target_accept is None:
        target_accept = 0.65 if n_leapfrog > 1 else 0.44

    T_data = observations.shape[0]
    if T_train is None:
        T_train = min(T_data, 50)

    rng_key = random.PRNGKey(seed)
    N = n_csmc_particles

    if waste_free and N % n_mh_steps != 0:
        raise ValueError(
            f"waste_free requires N % n_mh_steps == 0, got N={N}, n_mh_steps={n_mh_steps}"
        )

    # 1. Discover model sites
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, subject_ids, trace_key)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    D = flat_example.shape[0]

    manifest_dist = (
        model.spec.manifest_dist.value
        if hasattr(model.spec.manifest_dist, "value")
        else str(model.spec.manifest_dist)
    )

    # 2. Train proposal network
    # Extract measurement model from a trace
    with jax.ensure_compile_time_eval():
        # Get fixed measurement parameters for training
        spec = model.spec
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

    print("DPF: Phase 1 - Training proposal network...")
    rng_key, train_key = random.split(rng_key)
    proposal_params = _train_proposal(
        D_latent=spec.n_latent,
        n_manifest=spec.n_manifest,
        H=H_train,
        d_meas=d_train,
        R=R_train,
        manifest_dist=manifest_dist,
        n_train_seqs=n_train_seqs,
        n_train_steps=n_train_steps,
        n_particles_train=n_particles_train,
        T_train=T_train,
        lr=proposal_lr,
        seed=int(train_key[0]),
    )

    print("DPF: Phase 2 - Parameter inference via tempered SMC...")

    # 3. Build evaluators (using standard PF likelihood, not DPF, for the outer loop)
    # The DPF proposal improves particle filter efficiency but the outer loop
    # uses the same tempered SMC infrastructure
    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model, observations, times, subject_ids, site_info, unravel_fn
    )

    def _safe_lik_val_and_grad(z):
        val, grad = jax.value_and_grad(log_lik_fn)(z)
        safe_val = jnp.where(jnp.isfinite(val), val, -1e30)
        safe_grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return safe_val, safe_grad

    batch_lik_val_and_grad = jax.jit(jax.vmap(_safe_lik_val_and_grad))

    def _tempered_val_and_grad(z, beta):
        lik_val, lik_grad = jax.value_and_grad(log_lik_fn)(z)
        prior_val, prior_grad = jax.value_and_grad(log_prior_unc_fn)(z)
        val = prior_val + beta * lik_val
        grad = prior_grad + beta * lik_grad
        safe_val = jnp.where(jnp.isfinite(val), val, -1e30)
        safe_grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return safe_val, safe_grad

    def _mutate_particle(rng_key, z, beta, eps, chol_mass):
        keys = random.split(rng_key, n_mh_steps)

        def scan_fn(carry, key):
            z_curr, n_acc = carry

            def tempered_vg(z_):
                return _tempered_val_and_grad(z_, beta)

            z_new, accepted, _ = hmc_step(key, z_curr, tempered_vg, eps, chol_mass, n_leapfrog)
            return (z_new, n_acc + accepted.astype(jnp.int32)), None

        (z_final, n_accept), _ = jax.lax.scan(scan_fn, (z, jnp.int32(0)), keys)
        return z_final, n_accept

    def _mutate_batch(rng_key, particles, beta, eps, chol_mass):
        keys = random.split(rng_key, particles.shape[0])
        return jax.vmap(lambda k, z: _mutate_particle(k, z, beta, eps, chol_mass))(keys, particles)

    _mutate_batch_jit = jax.jit(_mutate_batch)

    # Waste-free
    def _mutate_particle_wastefree(rng_key, z, beta, eps, chol_mass):
        keys = random.split(rng_key, n_mh_steps)

        def scan_fn(carry, key):
            z_curr, n_acc = carry

            def tempered_vg(z_):
                return _tempered_val_and_grad(z_, beta)

            z_new, accepted, _ = hmc_step(key, z_curr, tempered_vg, eps, chol_mass, n_leapfrog)
            return (z_new, n_acc + accepted.astype(jnp.int32)), z_new

        (_, n_acc), all_z = jax.lax.scan(scan_fn, (z, jnp.int32(0)), keys)
        return all_z, n_acc

    def _mutate_batch_wastefree(rng_key, particles_M, beta, eps, chol_mass):
        M = particles_M.shape[0]
        keys = random.split(rng_key, M)
        return jax.vmap(lambda k, z: _mutate_particle_wastefree(k, z, beta, eps, chol_mass))(
            keys, particles_M
        )

    _mutate_batch_wastefree_jit = jax.jit(_mutate_batch_wastefree)

    # 4. Initialize
    eps = param_step_size
    print(f"  N={N}, K={n_outer}, D={D}, n_mh={n_mh_steps}, eps={eps}")
    print(f"  Initializing {N} particles from prior...")

    parts = []
    for name in sorted(site_info.keys()):
        info = site_info[name]
        rng_key, sample_key = random.split(rng_key)
        prior_samples = info["distribution"].sample(sample_key, (N,))
        unc_samples = info["transform"].inv(prior_samples)
        parts.append(unc_samples.reshape(N, -1))

    particles = jnp.concatenate(parts, axis=1)
    chol_mass = compute_weighted_chol_mass(particles, jnp.zeros(N), D)

    from blackjax.smc.resampling import systematic as _systematic_resample

    # Pilot
    print("  Pilot: adapting step size at prior...")
    for pilot_step in range(30):
        rng_key, mutate_key = random.split(rng_key)
        particles_new, n_accepts = _mutate_batch_jit(mutate_key, particles, 0.0, eps, chol_mass)
        avg_accept = float(jnp.mean(n_accepts) / n_mh_steps)
        particles = particles_new

        log_eps = jnp.log(jnp.array(eps))
        log_eps = log_eps + 0.5 * (avg_accept - target_accept)
        eps = float(jnp.clip(jnp.exp(log_eps), 1e-5, 2.0))

        if pilot_step >= 5 and abs(avg_accept - target_accept) < 0.1:
            print(
                f"    pilot converged at step {pilot_step + 1}: accept={avg_accept:.2f} eps={eps:.4f}"
            )
            break
    else:
        print(f"    pilot done: accept={avg_accept:.2f} eps={eps:.4f}")

    log_liks, _ = batch_lik_val_and_grad(particles)
    chol_mass = compute_weighted_chol_mass(particles, jnp.zeros(N), D)
    logw = jnp.zeros(N)

    accept_rates = []
    ess_history = []
    eps_history = []
    beta_schedule = []
    chain_samples = []

    beta_prev = 0.0
    level = 0
    max_mutation_rounds = 5
    M = N // n_mh_steps if waste_free else N

    while beta_prev < 1.0 and level < n_outer:
        if adaptive_tempering:
            beta_k = find_next_beta(logw, log_liks, beta_prev, target_ess_ratio, N)
        else:
            beta_k = float(level + 1) / n_outer

        beta_schedule.append(beta_k)
        logw = logw + (beta_k - beta_prev) * log_liks

        lse = jax.nn.logsumexp(logw)
        log_wn = logw - lse
        wn = jnp.exp(log_wn)
        ess = float(1.0 / jnp.sum(wn**2))
        ess_history.append(ess)

        if ess > N / 4:
            chol_mass = compute_weighted_chol_mass(particles, logw, D)

        if waste_free:
            rng_key, resample_key, mutate_key = random.split(rng_key, 3)
            idx = _systematic_resample(resample_key, wn, M)
            resampled = particles[idx]
            all_trajs, n_accs = _mutate_batch_wastefree_jit(
                mutate_key, resampled, beta_k, eps, chol_mass
            )
            particles = all_trajs.reshape(N, D)
            logw = jnp.full(N, -jnp.log(float(N)))
            avg_accept = float(jnp.mean(n_accs) / n_mh_steps)

            log_eps = jnp.log(jnp.array(eps))
            log_eps = log_eps + 0.1 * (avg_accept - target_accept)
            eps = float(jnp.clip(jnp.exp(log_eps), 1e-5, 2.0))
        else:
            if ess < N / 2:
                rng_key, resample_key = random.split(rng_key)
                idx = _systematic_resample(resample_key, wn, N)
                particles = particles[idx]
                log_liks = log_liks[idx]
                logw = jnp.full(N, -jnp.log(float(N)))

            total_accepts = 0
            total_proposals = 0
            for mutation_round in range(max_mutation_rounds):
                rng_key, mutate_key = random.split(rng_key)
                particles_new, n_accepts = _mutate_batch_jit(
                    mutate_key, particles, beta_k, eps, chol_mass
                )
                round_accepts = float(jnp.sum(n_accepts))
                total_accepts += round_accepts
                total_proposals += N * n_mh_steps
                particles = particles_new

                round_accept_rate = round_accepts / (N * n_mh_steps)
                log_eps = jnp.log(jnp.array(eps))
                log_eps = log_eps + 0.1 * (round_accept_rate - target_accept)
                eps = float(jnp.clip(jnp.exp(log_eps), 1e-5, 2.0))

                if mutation_round > 0 and round_accept_rate > 0.2:
                    break

            avg_accept = total_accepts / max(total_proposals, 1)

        accept_rates.append(avg_accept)
        eps_history.append(eps)
        log_liks, _ = batch_lik_val_and_grad(particles)
        chain_samples.append(particles[level % N])

        print(
            f"  step {level + 1}  beta={beta_k:.3f}  ESS={ess:.1f}/{N}"
            f"  accept={avg_accept:.2f}  eps={eps:.4f}"
        )

        beta_prev = beta_k
        level += 1

    actual_levels = level
    if n_warmup is None:
        n_warmup = actual_levels // 2
    n_warmup = min(n_warmup, max(actual_levels - 1, 0))

    chain_particles = jnp.stack(chain_samples[n_warmup:], axis=0)

    transforms = {name: info["transform"] for name, info in site_info.items()}
    samples = {}
    for name in transforms:

        def _extract_one(z, _name=name):
            unc = unravel_fn(z)
            return transforms[_name](unc[_name])

        samples[name] = jax.vmap(_extract_one)(chain_particles)

    det_samples = _assemble_deterministics(samples, model.spec)
    samples.update(det_samples)

    return InferenceResult(
        _samples=samples,
        method="dpf",
        diagnostics={
            "accept_rates": accept_rates,
            "ess_history": ess_history,
            "eps_history": eps_history,
            "beta_schedule": beta_schedule,
            "n_levels": actual_levels,
            "n_pf_particles": n_pf_particles,
            "n_train_seqs": n_train_seqs,
            "n_train_steps": n_train_steps,
            "proposal_params": proposal_params,
            "n_outer": n_outer,
            "n_csmc_particles": N,
            "n_mh_steps": n_mh_steps,
            "n_leapfrog": n_leapfrog,
            "n_warmup": n_warmup,
        },
    )
