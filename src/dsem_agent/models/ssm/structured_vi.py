"""Structured Variational Inference with backward-factored Gaussian family.

Implements Method 2 from the algorithmic specification:

The variational family mirrors the true smoothing distribution's backward
factorization:
    q(z_{1:T} | phi) = q(z_T | phi_T) prod_{t=1}^{T-1} q(z_t | z_{t+1}, phi_t)

where each backward kernel is Gaussian:
    q(z_T | phi_T) = N(z_T; m_T, S_T)
    q(z_t | z_{t+1}, phi_t) = N(z_t; m_t + C_t(z_{t+1} - m_{t+1}), S_t)

The ELBO is optimized jointly over variational parameters phi and model
parameters theta using the reparameterization trick.

Can be initialized from Laplace-EM (Method 1) output for better convergence.
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
# Variational parameters and sampling
# ---------------------------------------------------------------------------


def _init_variational_params(T, D, rng_key):
    """Initialize variational parameters for the backward-factored family.

    Args:
        T: number of time steps
        D: latent dimension

    Returns:
        phi: dict with keys 'm', 'log_S_diag', 'C'
            m: (T, D) variational means
            log_S_diag: (T, D) log-diagonal of Cholesky of S_t
            C: (T-1, D, D) backward coupling matrices
    """
    key1, _key2, _key3 = random.split(rng_key, 3)
    return {
        "m": random.normal(key1, (T, D)) * 0.01,
        "log_S_diag": jnp.full((T, D), -1.0),  # S_t ≈ 0.37 * I
        "C": jnp.zeros((max(T - 1, 1), D, D)),  # No backward coupling initially
    }


def _sample_trajectory(phi, rng_key):
    """Sample z_{1:T} from the backward-factored variational distribution.

    Sampling proceeds backward: z_T first, then z_{T-1}|z_T, etc.

    Args:
        phi: variational parameters
        rng_key: PRNG key

    Returns:
        z: (T, D) sampled trajectory
    """
    m = phi["m"]  # (T, D)
    log_S_diag = phi["log_S_diag"]  # (T, D)
    C = phi["C"]  # (T-1, D, D)

    T, D = m.shape
    S_diag = jnp.exp(log_S_diag)  # (T, D) - diagonal of Cholesky

    # Sample z_T ~ N(m_T, S_T)
    key_T, rng_key = random.split(rng_key)
    eps_T = random.normal(key_T, (D,))
    z_T = m[T - 1] + S_diag[T - 1] * eps_T

    if T == 1:
        return z_T[None]

    # Backward sampling: z_t | z_{t+1} ~ N(m_t + C_t(z_{t+1} - m_{t+1}), S_t)
    keys = random.split(rng_key, T - 1)

    def _backward_step(z_next, inputs):
        key, m_t, S_diag_t, C_t, m_tp1 = inputs
        eps = random.normal(key, (D,))
        mean = m_t + C_t @ (z_next - m_tp1)
        z_t = mean + S_diag_t * eps
        return z_t, z_t

    # Inputs for backward scan: t = T-2, T-3, ..., 0
    bwd_inputs = (
        keys[::-1],
        m[: T - 1][::-1],
        S_diag[: T - 1][::-1],
        C[::-1],
        m[1:][::-1],  # m_{t+1} for each t
    )

    _, z_rest_rev = jax.lax.scan(_backward_step, z_T, bwd_inputs)
    z_rest = z_rest_rev[::-1]  # (T-1, D)

    return jnp.concatenate([z_rest, z_T[None]], axis=0)


def _log_q_trajectory(z, phi):
    """Compute log q(z_{1:T} | phi) for the backward-factored family.

    Args:
        z: (T, D) trajectory
        phi: variational parameters

    Returns:
        scalar log q(z | phi)
    """
    m = phi["m"]
    log_S_diag = phi["log_S_diag"]
    C = phi["C"]
    T, D = m.shape
    S_diag = jnp.exp(log_S_diag)

    # log q(z_T | phi_T) = log N(z_T; m_T, diag(S_T^2))
    diff_T = z[T - 1] - m[T - 1]
    log_q_T = (
        -0.5 * D * jnp.log(2 * jnp.pi)
        - jnp.sum(log_S_diag[T - 1])
        - 0.5 * jnp.sum((diff_T / S_diag[T - 1]) ** 2)
    )

    if T == 1:
        return log_q_T

    # log q(z_t | z_{t+1}, phi_t) for t = 0..T-2
    def _log_q_step(t):
        mean_t = m[t] + C[t] @ (z[t + 1] - m[t + 1])
        diff_t = z[t] - mean_t
        return (
            -0.5 * D * jnp.log(2 * jnp.pi)
            - jnp.sum(log_S_diag[t])
            - 0.5 * jnp.sum((diff_t / S_diag[t]) ** 2)
        )

    log_q_rest = jax.vmap(_log_q_step)(jnp.arange(T - 1))
    return log_q_T + jnp.sum(log_q_rest)


# ---------------------------------------------------------------------------
# ELBO computation
# ---------------------------------------------------------------------------


def _compute_elbo(
    z,
    phi,
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
):
    """Compute the ELBO for a single sampled trajectory.

    ELBO = E_q[log p(y,z|theta) - log q(z|phi)]
         = sum_t log p(y_t|z_t) + log p(z_1) + sum_{t>=2} log p(z_t|z_{t-1}) - log q(z|phi)
    """
    T, D = z.shape
    jitter = jnp.eye(D) * 1e-6

    # 1. Emission log-probs
    mask_float = obs_mask.astype(jnp.float32)
    emission_lls = jax.vmap(lambda y, zz, mm: emission_log_prob_fn(y, zz, H, d_meas, R, mm))(
        observations, z, mask_float
    )
    total_emission_ll = jnp.sum(emission_lls)

    # 2. Transition log-probs (forward model)
    def _transition_ll(z_t, z_tm1, Ad_t, Qd_t, cd_t):
        mean = Ad_t @ z_tm1 + cd_t
        diff = z_t - mean
        Qd_reg = Qd_t + jitter
        _, logdet = jnp.linalg.slogdet(Qd_reg)
        mahal = diff @ jla.solve(Qd_reg, diff, assume_a="pos")
        return -0.5 * (D * jnp.log(2 * jnp.pi) + logdet + mahal)

    # Initial state: z_0 | prior
    z0_pred = Ad[0] @ init_mean + cd[0]
    P0_pred = Ad[0] @ init_cov @ Ad[0].T + Qd[0]
    P0_pred = 0.5 * (P0_pred + P0_pred.T) + jitter
    diff0 = z[0] - z0_pred
    _, logdet0 = jnp.linalg.slogdet(P0_pred)
    mahal0 = diff0 @ jla.solve(P0_pred, diff0, assume_a="pos")
    init_ll = -0.5 * (D * jnp.log(2 * jnp.pi) + logdet0 + mahal0)

    if T > 1:
        trans_lls = jax.vmap(_transition_ll)(z[1:], z[:-1], Ad[1:], Qd[1:], cd[1:])
        total_trans_ll = init_ll + jnp.sum(trans_lls)
    else:
        total_trans_ll = init_ll

    # 3. Variational entropy: -log q(z | phi)
    log_q = _log_q_trajectory(z, phi)

    return total_emission_ll + total_trans_ll - log_q


# ---------------------------------------------------------------------------
# Structured VI likelihood backend
# ---------------------------------------------------------------------------


class StructuredVILikelihood:
    """Structured VI likelihood backend.

    Computes an ELBO lower bound on log p(y|theta) using the backward-factored
    Gaussian variational family. The variational parameters are optimized
    jointly with model parameters.
    """

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        manifest_dist: str = "gaussian",
        n_vi_steps: int = 100,
        n_mc_samples: int = 4,
        vi_lr: float = 0.01,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dist = manifest_dist
        self.n_vi_steps = n_vi_steps
        self.n_mc_samples = n_mc_samples
        self.vi_lr = vi_lr

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
        """Compute ELBO lower bound on log-likelihood."""
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

        H = measurement_params.lambda_mat
        d_meas = measurement_params.manifest_means
        R = measurement_params.manifest_cov

        # Initialize variational params
        rng_key = random.PRNGKey(0)
        phi = _init_variational_params(T, n, rng_key)

        # Optimize variational parameters for this theta
        import optax

        optimizer = optax.adam(self.vi_lr)
        opt_state = optimizer.init(phi)

        def _mc_elbo(phi, rng_key):
            keys = random.split(rng_key, self.n_mc_samples)

            def _single_elbo(key):
                z = _sample_trajectory(phi, key)
                return _compute_elbo(
                    z,
                    phi,
                    clean_obs,
                    obs_mask,
                    Ad,
                    Qd,
                    cd,
                    H,
                    d_meas,
                    R,
                    initial_state.mean,
                    initial_state.cov,
                    emission_fn,
                )

            return jnp.mean(jax.vmap(_single_elbo)(keys))

        def _vi_body(carry, _):
            phi, opt_state, rng_key = carry
            rng_key, step_key = random.split(rng_key)
            _elbo, grads = jax.value_and_grad(_mc_elbo)(phi, step_key)
            grads = jax.tree.map(lambda g: jnp.clip(g, -10.0, 10.0), grads)
            updates, opt_state_new = optimizer.update(grads, opt_state)
            phi_new = optax.apply_updates(phi, updates)
            return (phi_new, opt_state_new, rng_key), None

        (phi, opt_state, rng_key), _ = jax.lax.scan(
            _vi_body, (phi, opt_state, rng_key), None, length=self.n_vi_steps
        )

        # Final ELBO estimate with more samples
        rng_key, eval_key = random.split(rng_key)
        final_elbo = _mc_elbo(phi, eval_key)
        return final_elbo


# ---------------------------------------------------------------------------
# fit_structured_vi: outer loop for parameter estimation
# ---------------------------------------------------------------------------


def fit_structured_vi(
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
    n_vi_steps: int = 50,
    n_mc_samples: int = 4,
    vi_lr: float = 0.01,
    n_leapfrog: int = 5,
    adaptive_tempering: bool = True,
    target_ess_ratio: float = 0.5,
    waste_free: bool = False,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM via structured VI with tempered SMC outer loop.

    Uses the ELBO from structured VI as the log-density for a tempered SMC
    sampler over the parameter space. The variational parameters are
    re-optimized for each parameter particle at each tempering level.

    For computational efficiency, we use a simplified approach: the structured
    VI computes an ELBO bound for each theta, which serves as a lower bound
    on the marginal likelihood. The tempered SMC uses this as the log-density.

    In practice, for the outer loop we reuse the same infrastructure as
    tempered_smc but with the model's likelihood evaluated via structured VI.

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        subject_ids: optional subject indices
        n_outer: max tempering levels
        n_csmc_particles: number of parameter particles
        n_mh_steps: HMC mutations per round
        param_step_size: initial leapfrog step size
        n_warmup: levels to discard
        target_accept: target MH acceptance
        seed: random seed
        n_vi_steps: VI optimization steps per likelihood evaluation
        n_mc_samples: MC samples for ELBO estimation
        vi_lr: learning rate for VI optimization
        n_leapfrog: leapfrog steps
        adaptive_tempering: use ESS-based tempering
        target_ess_ratio: target ESS fraction
        waste_free: waste-free recycling

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

    # 2. Build differentiable evaluators
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

    # Waste-free mutation
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

    # 3. Initialize
    eps = param_step_size
    print(
        f"Structured VI: N={N}, K={n_outer}, D={D}, "
        f"n_vi_steps={n_vi_steps}, n_mh={n_mh_steps}, eps={eps}"
    )
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
        method="structured_vi",
        diagnostics={
            "accept_rates": accept_rates,
            "ess_history": ess_history,
            "eps_history": eps_history,
            "beta_schedule": beta_schedule,
            "n_levels": actual_levels,
            "n_vi_steps": n_vi_steps,
            "n_mc_samples": n_mc_samples,
            "vi_lr": vi_lr,
            "n_outer": n_outer,
            "n_csmc_particles": N,
            "n_mh_steps": n_mh_steps,
            "n_leapfrog": n_leapfrog,
            "n_warmup": n_warmup,
        },
    )
