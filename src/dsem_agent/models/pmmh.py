"""Particle MCMC (PMMH) module for non-Gaussian/nonlinear state-space models.

Implements Particle Marginal Metropolis-Hastings (Andrieu et al., 2010)
for parameter inference when the particle filter is needed for likelihood
estimation. This is a completely separate inference path from NumPyro NUTS.

Architecture:
    SSMSpec → SSMProtocol adapter → bootstrap_filter (log p̂(y|θ))
    → PMMH kernel (MH with stochastic LL) → posterior samples

Components:
    - SSMProtocol: adapter mapping CT-SEM SSMSpec into PF-compatible functions
    - bootstrap_filter: reference pure-JAX bootstrap particle filter
    - pmmh_kernel: random-walk Metropolis-Hastings with particle filter LL
    - run_pmmh / run_pmmh_chains: samplers
    - tune_proposal: adaptive proposal covariance (Roberts & Rosenthal, 2001)
    - to_arviz: ArviZ InferenceData conversion

References:
    - Andrieu, Doucet & Holenstein (2010): Particle MCMC methods
    - Andrieu & Roberts (2009): Pseudo-marginal approach validity
    - Roberts & Rosenthal (2001): Optimal scaling for MH
"""

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from jax import lax

from dsem_agent.models.ssm.discretization import discretize_system

# =============================================================================
# SSM Protocol — adapter for CT-SEM
# =============================================================================


class SSMProtocol(Protocol):
    """Protocol for state-space models compatible with particle filtering."""

    def initial_sample(self, key: jax.Array, params: dict) -> jax.Array:
        """Sample from initial state distribution."""
        ...

    def transition_sample(
        self, key: jax.Array, x_prev: jax.Array, params: dict, dt: float
    ) -> jax.Array:
        """Sample x_t | x_{t-1} from transition distribution."""
        ...

    def observation_log_prob(
        self, y: jax.Array, x: jax.Array, params: dict, obs_mask: jax.Array
    ) -> float:
        """Compute log p(y_t | x_t) for observation model."""
        ...


class CTSEMAdapter:
    """Adapts CT-SEM parameters into SSMProtocol-compatible functions.

    Maps the CT-SEM structure (drift, diffusion, measurement) into
    initial_sample, transition_sample, and observation_log_prob.
    """

    def __init__(self, n_latent: int, n_manifest: int):
        self.n_latent = n_latent
        self.n_manifest = n_manifest

    def initial_sample(self, key: jax.Array, params: dict) -> jax.Array:
        """Sample η_0 ~ N(t0_mean, t0_cov)."""
        t0_mean = params["t0_mean"]
        t0_cov = params["t0_cov"]
        chol = jla.cholesky(t0_cov + jnp.eye(self.n_latent) * 1e-8, lower=True)
        return t0_mean + chol @ random.normal(key, (self.n_latent,))

    def transition_sample(
        self, key: jax.Array, x_prev: jax.Array, params: dict, dt: float
    ) -> jax.Array:
        """Sample η_t | η_{t-1} via CT→DT discretization.

        η_t ~ N(Ad * η_{t-1} + cd, Qd)
        """
        Ad, Qd, cd = discretize_system(
            params["drift"], params["diffusion_cov"], params.get("cint"), dt
        )
        mean = Ad @ x_prev
        if cd is not None:
            mean = mean + cd.flatten()
        chol = jla.cholesky(Qd + jnp.eye(self.n_latent) * 1e-8, lower=True)
        return mean + chol @ random.normal(key, (self.n_latent,))

    def observation_log_prob(
        self, y: jax.Array, x: jax.Array, params: dict, obs_mask: jax.Array
    ) -> float:
        """Compute log p(y | x) under measurement model.

        y = Λx + μ + ε, ε ~ N(0, R)
        Handles missing data via mask.
        """
        lambda_mat = params["lambda_mat"]
        manifest_means = params["manifest_means"]
        manifest_cov = params["manifest_cov"]

        y_pred = lambda_mat @ x + manifest_means
        innovation = (y - y_pred) * obs_mask.astype(jnp.float32)

        n_observed = jnp.sum(obs_mask.astype(jnp.float32))

        # Add large variance for missing dims
        large_var = 1e10
        mask_float = obs_mask.astype(jnp.float32)
        R_adj = manifest_cov + jnp.diag((1.0 - mask_float) * large_var)
        R_adj = 0.5 * (R_adj + R_adj.T) + jnp.eye(self.n_manifest) * 1e-8

        _, logdet = jnp.linalg.slogdet(R_adj)
        mahal = innovation @ jla.solve(R_adj, innovation, assume_a="pos")
        ll = -0.5 * (n_observed * jnp.log(2 * jnp.pi) + logdet + mahal)

        return jnp.where(n_observed > 0, ll, 0.0)


# =============================================================================
# Bootstrap Particle Filter
# =============================================================================


class PFResult(NamedTuple):
    """Result from particle filter."""

    log_likelihood: float
    final_particles: jnp.ndarray  # (n_particles, n_latent)
    final_log_weights: jnp.ndarray  # (n_particles,)


def bootstrap_filter(
    model: CTSEMAdapter,
    params: dict,
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    obs_mask: jnp.ndarray,
    n_particles: int,
    key: jax.Array,
    ess_threshold: float = 0.5,
) -> PFResult:
    """Bootstrap particle filter for log-likelihood estimation.

    Reference implementation in pure JAX. Returns unbiased log p̂(y|θ).

    Args:
        model: SSMProtocol-compatible model adapter
        params: dict of model parameters
        observations: (T, n_manifest) observed data
        time_intervals: (T,) time intervals (first is dummy ~1e-6)
        obs_mask: (T, n_manifest) boolean mask
        n_particles: number of particles
        key: JAX random key
        ess_threshold: resample when ESS/N < threshold

    Returns:
        PFResult with log-likelihood, final particles and weights
    """
    # Initialize particles from prior
    key, subkey = random.split(key)
    keys = random.split(subkey, n_particles)
    particles = jax.vmap(lambda k: model.initial_sample(k, params))(keys)
    log_weights = jnp.full(n_particles, -jnp.log(n_particles))

    def _systematic_resample(key, particles, log_weights):
        n = particles.shape[0]
        log_w_norm = log_weights - jax.scipy.special.logsumexp(log_weights)
        weights = jnp.exp(log_w_norm)
        cumsum = jnp.cumsum(weights)
        u = random.uniform(key) / n
        positions = u + jnp.arange(n) / n
        indices = jnp.searchsorted(cumsum, positions)
        indices = jnp.clip(indices, 0, n - 1)
        return particles[indices], jnp.full(n, -jnp.log(n))

    def _compute_ess(log_weights):
        n = log_weights.shape[0]
        log_w_norm = log_weights - jax.scipy.special.logsumexp(log_weights)
        return 1.0 / jnp.sum(jnp.exp(2 * log_w_norm)) / n

    def scan_fn(carry, inputs):
        particles, log_weights, total_ll, key = carry
        obs, dt, mask = inputs

        # Propagate particles through dynamics
        key, subkey = random.split(key)
        keys = random.split(subkey, n_particles)
        new_particles = jax.vmap(lambda k, x: model.transition_sample(k, x, params, dt))(
            keys, particles
        )

        # Weight by observation likelihood
        obs_clean = jnp.nan_to_num(obs, nan=0.0)
        particle_lls = jax.vmap(lambda x: model.observation_log_prob(obs_clean, x, params, mask))(
            new_particles
        )
        new_log_weights = log_weights + particle_lls

        # Log-likelihood increment
        ll_increment = jax.scipy.special.logsumexp(new_log_weights)
        total_ll = total_ll + ll_increment

        # Normalize
        new_log_weights = new_log_weights - jax.scipy.special.logsumexp(new_log_weights)

        # Adaptive resampling
        key, subkey = random.split(key)
        ess = _compute_ess(new_log_weights)

        resampled_p, resampled_w = _systematic_resample(subkey, new_particles, new_log_weights)
        final_particles = jnp.where(
            ess < ess_threshold,
            resampled_p,
            new_particles,
        )
        final_log_weights = jnp.where(
            ess < ess_threshold,
            resampled_w,
            new_log_weights,
        )

        return (final_particles, final_log_weights, total_ll, key), None

    init = (particles, log_weights, 0.0, key)
    (final_particles, final_log_weights, total_ll, _), _ = lax.scan(
        scan_fn, init, (observations, time_intervals, obs_mask)
    )

    return PFResult(
        log_likelihood=total_ll,
        final_particles=final_particles,
        final_log_weights=final_log_weights,
    )


# =============================================================================
# PMMH Kernel
# =============================================================================


class PMMHState(NamedTuple):
    """State of the PMMH sampler."""

    theta: jnp.ndarray  # flat parameter vector
    log_likelihood: float  # current log p̂(y|θ)
    log_prior: float  # current log p(θ)
    accepted: bool  # whether last proposal was accepted


class PMMHInfo(NamedTuple):
    """Diagnostics from PMMH sampling."""

    acceptance_rate: float
    log_likelihoods: jnp.ndarray  # (n_samples,)
    accepted: jnp.ndarray  # (n_samples,) boolean


def pmmh_kernel(
    model: CTSEMAdapter,
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    obs_mask: jnp.ndarray,
    log_prior_fn,
    unpack_fn,
    n_particles: int = 1000,
    proposal_cov: jnp.ndarray | None = None,
):
    """Create PMMH init and step functions.

    Args:
        model: SSMProtocol-compatible model adapter
        observations: (T, n_manifest) observed data
        time_intervals: (T,) time intervals
        obs_mask: (T, n_manifest) boolean mask
        log_prior_fn: θ -> log p(θ) (scalar)
        unpack_fn: flat θ vector -> params_dict
        n_particles: number of particles for PF
        proposal_cov: (d, d) random-walk proposal covariance

    Returns:
        (init_fn, step_fn) tuple
    """

    def _compute_log_likelihood(theta, key):
        params = unpack_fn(theta)
        result = bootstrap_filter(
            model,
            params,
            observations,
            time_intervals,
            obs_mask,
            n_particles,
            key,
        )
        return result.log_likelihood

    def init(theta_init, key):
        log_prior = log_prior_fn(theta_init)
        log_lik = _compute_log_likelihood(theta_init, key)
        return PMMHState(
            theta=theta_init,
            log_likelihood=log_lik,
            log_prior=log_prior,
            accepted=True,
        )

    def step(state, key):
        key_propose, key_pf, key_accept = random.split(key, 3)

        d = state.theta.shape[0]
        cov = proposal_cov if proposal_cov is not None else jnp.eye(d) * 0.01
        chol = jla.cholesky(cov, lower=True)

        # Propose
        theta_star = state.theta + chol @ random.normal(key_propose, (d,))

        # Compute log prior
        log_prior_star = log_prior_fn(theta_star)

        # Compute log-likelihood via particle filter
        log_lik_star = _compute_log_likelihood(theta_star, key_pf)

        # MH acceptance ratio
        log_alpha = (log_lik_star + log_prior_star) - (state.log_likelihood + state.log_prior)
        log_u = jnp.log(random.uniform(key_accept))
        accept = log_u < log_alpha

        new_state = PMMHState(
            theta=jnp.where(accept, theta_star, state.theta),
            log_likelihood=jnp.where(accept, log_lik_star, state.log_likelihood),
            log_prior=jnp.where(accept, log_prior_star, state.log_prior),
            accepted=accept,
        )
        return new_state

    return init, step


# =============================================================================
# Samplers
# =============================================================================


class PMMHResult(NamedTuple):
    """Full result from PMMH sampling."""

    samples: jnp.ndarray  # (n_samples, d)
    log_likelihoods: jnp.ndarray  # (n_samples,)
    accepted: jnp.ndarray  # (n_samples,) boolean
    acceptance_rate: float


def run_pmmh(
    model: CTSEMAdapter,
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    obs_mask: jnp.ndarray,
    log_prior_fn,
    unpack_fn,
    init_theta: jnp.ndarray,
    n_samples: int = 1000,
    n_warmup: int = 500,
    n_particles: int = 1000,
    proposal_cov: jnp.ndarray | None = None,
    seed: int = 0,
) -> PMMHResult:
    """Run PMMH sampler.

    Args:
        model: SSMProtocol-compatible model adapter
        observations: (T, n_manifest) observed data
        time_intervals: (T,) time intervals
        obs_mask: (T, n_manifest) boolean mask
        log_prior_fn: θ -> log p(θ)
        unpack_fn: flat θ vector -> params_dict
        init_theta: initial parameter vector
        n_samples: number of posterior samples (after warmup)
        n_warmup: number of warmup samples (discarded)
        n_particles: particles for PF
        proposal_cov: (d, d) proposal covariance
        seed: random seed

    Returns:
        PMMHResult with samples and diagnostics
    """
    init_fn, step_fn = pmmh_kernel(
        model,
        observations,
        time_intervals,
        obs_mask,
        log_prior_fn,
        unpack_fn,
        n_particles=n_particles,
        proposal_cov=proposal_cov,
    )

    key = random.PRNGKey(seed)
    key, init_key = random.split(key)

    # Initialize
    state = init_fn(init_theta, init_key)

    # Warmup (not recorded)
    def warmup_step(state, key):
        new_state = step_fn(state, key)
        return new_state, new_state.accepted

    key, warmup_key = random.split(key)
    warmup_keys = random.split(warmup_key, n_warmup)
    state, _warmup_accepted = lax.scan(warmup_step, state, warmup_keys)

    # Sampling
    def sample_step(state, key):
        new_state = step_fn(state, key)
        return new_state, (new_state.theta, new_state.log_likelihood, new_state.accepted)

    sample_keys = random.split(key, n_samples)
    _, (samples, log_liks, accepted) = lax.scan(sample_step, state, sample_keys)

    return PMMHResult(
        samples=samples,
        log_likelihoods=log_liks,
        accepted=accepted,
        acceptance_rate=jnp.mean(accepted.astype(float)),
    )


def tune_proposal(
    samples: jnp.ndarray,
    target_acceptance: float = 0.234,
    current_acceptance: float | None = None,
    scale_factor: float = 2.38,
) -> jnp.ndarray:
    """Tune proposal covariance from samples.

    Uses the empirical covariance scaled by 2.38²/d (Roberts & Rosenthal, 2001).

    Args:
        samples: (n_samples, d) parameter samples
        target_acceptance: target acceptance rate (0.234 for RWM)
        current_acceptance: current acceptance rate (for additional scaling)
        scale_factor: base scaling factor (2.38 for optimal RWM)

    Returns:
        (d, d) tuned proposal covariance
    """
    d = samples.shape[1]
    emp_cov = jnp.cov(samples.T)

    # Optimal scaling: (2.38)^2 / d
    scale = scale_factor**2 / d

    # Additional adaptive scaling if acceptance rate is known
    if current_acceptance is not None:
        # Scale up if accepting too much, down if too little
        ratio = current_acceptance / target_acceptance
        scale = scale * jnp.clip(ratio, 0.1, 10.0)

    return emp_cov * scale


# =============================================================================
# ArviZ Integration
# =============================================================================


def to_arviz(result: PMMHResult, var_names: list[str]):
    """Convert PMMH result to ArviZ InferenceData.

    Args:
        result: PMMHResult from run_pmmh
        var_names: names for each dimension of θ

    Returns:
        arviz.InferenceData
    """
    import arviz as az

    samples = result.samples  # (n_samples, d)

    # Build posterior dict
    posterior = {}
    for i, name in enumerate(var_names):
        # ArviZ expects (chain, draw) shape
        posterior[name] = samples[:, i][None, :]  # (1, n_samples)

    # Sample stats
    sample_stats = {
        "lp": result.log_likelihoods[None, :],  # (1, n_samples)
        "accepted": result.accepted.astype(float)[None, :],
    }

    return az.from_dict(posterior=posterior, sample_stats=sample_stats)
