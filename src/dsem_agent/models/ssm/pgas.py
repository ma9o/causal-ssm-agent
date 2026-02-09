"""PGAS: Particle Gibbs with Ancestor Sampling + Gradient-Informed Proposals.

Combines three elements:
1. PGAS outer loop: Gibbs-alternate between latent trajectories and parameters
2. Gradient-informed CSMC: Langevin proposals inside conditional SMC for states
3. MALA parameter updates: Gradient-informed MH for the parameter conditional

The CSMC sweep uses the PGAS kernel (Lindsten, Jordan & Schoen, 2014) with
gradient-informed state proposals that shift the bootstrap transition toward
high-likelihood regions via nabla_x log p(y_t | x_t). The ancestor sampling step
uses the model transition density (unaffected by the proposal choice).

The parameter update uses MALA (Metropolis-Adjusted Langevin Algorithm)
targeting p(theta | x_{1:T}, y_{1:T}), which is cheaply evaluable given fixed
trajectories (no particle filter needed).

Novel combination: nobody has published PGAS + gradient CSMC proposals +
Hess-MC2-style parameter updates together. Septier & Peters did gradient
proposals in regular SMC; Lindsten et al. did PGAS with bootstrap proposals;
Murphy et al. did Hess-MC2 for parameter-only SMC. This module unifies all three.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from blackjax.smc.resampling import systematic as _systematic_resample
from jax.flatten_util import ravel_pytree

from dsem_agent.models.likelihoods.particle import SSMAdapter
from dsem_agent.models.ssm.discretization import discretize_system_batched
from dsem_agent.models.ssm.hessmc2 import _assemble_deterministics, _discover_sites
from dsem_agent.models.ssm.inference import InferenceResult

# ---------------------------------------------------------------------------
# Transition log-prob
# ---------------------------------------------------------------------------


def _transition_log_prob(x_curr, x_prev, Ad_t, chol_t, cd_t):
    """Log N(x_curr; Ad_t @ x_prev + cd_t, Qd_t) where Qd_t = chol_t @ chol_t.T."""
    mean = Ad_t @ x_prev + cd_t
    diff = x_curr - mean
    Linv_d = jla.solve_triangular(chol_t, diff, lower=True)
    logdet = jnp.sum(jnp.log(jnp.diag(chol_t)))
    n = x_curr.shape[0]
    return -0.5 * jnp.dot(Linv_d, Linv_d) - logdet - 0.5 * n * jnp.log(2 * jnp.pi)


# ---------------------------------------------------------------------------
# Helper: extract all SSM matrices from det dict + spec fallbacks
# ---------------------------------------------------------------------------


def _extract_matrices(det, con, spec):
    """Extract SSM matrices from _assemble_deterministics output + spec fallbacks."""
    n_l, n_m = spec.n_latent, spec.n_manifest

    drift = det["drift"][0]
    diff_chol = det["diffusion"][0]
    diff_cov = diff_chol @ diff_chol.T

    lambda_mat = (
        det["lambda"][0]
        if "lambda" in det
        else spec.lambda_mat
        if isinstance(spec.lambda_mat, jnp.ndarray)
        else jnp.eye(n_m, n_l)
    )
    manifest_cov = (
        det["manifest_cov"][0]
        if "manifest_cov" in det
        else spec.manifest_var @ spec.manifest_var.T
        if isinstance(spec.manifest_var, jnp.ndarray)
        else jnp.eye(n_m)
    )
    manifest_means = (
        con["manifest_means"]
        if "manifest_means" in con
        else spec.manifest_means
        if isinstance(spec.manifest_means, jnp.ndarray)
        else jnp.zeros(n_m)
    )
    t0_mean = (
        det["t0_means"][0]
        if "t0_means" in det
        else spec.t0_means
        if isinstance(spec.t0_means, jnp.ndarray)
        else jnp.zeros(n_l)
    )
    t0_cov = (
        det["t0_cov"][0]
        if "t0_cov" in det
        else spec.t0_var @ spec.t0_var.T
        if isinstance(spec.t0_var, jnp.ndarray)
        else jnp.eye(n_l)
    )
    cint = (
        det["cint"][0]
        if "cint" in det
        else spec.cint
        if isinstance(spec.cint, jnp.ndarray)
        else None
    )

    return drift, diff_cov, cint, lambda_mat, manifest_means, manifest_cov, t0_mean, t0_cov


def _params_to_matrices(z_unc, unravel_fn, transforms, spec):
    """Convert unconstrained flat vector to SSM matrices."""
    unc = unravel_fn(z_unc)
    con = {name: transforms[name](unc[name]) for name in unc}
    samples_1 = {name: con[name][None] for name in con}
    det = _assemble_deterministics(samples_1, spec)
    return _extract_matrices(det, con, spec)


# ---------------------------------------------------------------------------
# Simulate initial trajectory from prior
# ---------------------------------------------------------------------------


def _simulate_trajectory(key, drift, diff_cov, cint, t0_mean, t0_cov, dt_array, n_latent):
    """Forward-simulate a trajectory from the model (prior, no conditioning on obs)."""
    T = dt_array.shape[0]
    Ad, Qd, cd = discretize_system_batched(drift, diff_cov, cint, dt_array)
    if cd is None:
        cd = jnp.zeros((T, n_latent))
    jitter = jnp.eye(n_latent) * 1e-6
    chol_Qd = jax.vmap(lambda Q: jla.cholesky(Q + jitter, lower=True))(Qd)
    chol_t0 = jla.cholesky(t0_cov + jitter, lower=True)

    key, init_key = random.split(key)
    x0 = t0_mean + chol_t0 @ random.normal(init_key, (n_latent,))

    def step(x_prev, inputs):
        k, Ad_t, chol_t, cd_t = inputs
        x_t = Ad_t @ x_prev + cd_t + chol_t @ random.normal(k, (n_latent,))
        return x_t, x_t

    keys = random.split(key, T - 1)
    _, traj_rest = jax.lax.scan(step, x0, (keys, Ad[1:], chol_Qd[1:], cd[1:]))

    return jnp.concatenate([x0[None], traj_rest], axis=0)


# ---------------------------------------------------------------------------
# CSMC sweep with gradient-informed proposals and ancestor sampling
# ---------------------------------------------------------------------------


def _csmc_sweep(
    key,
    ref_traj,
    Ad,
    Qd,
    chol_Qd,
    cd,
    t0_mean,
    t0_cov,
    observations,
    obs_mask_float,
    n_particles,
    langevin_step_size,
    obs_lp_fn,
    grad_obs_fn,
):
    """One CSMC sweep (Algorithm 2 from PGAS paper) with gradient proposals.

    Runs a conditional SMC sampler where particle N-1 is pinned to ref_traj.
    Free particles (0..N-2) use Langevin-shifted proposals that incorporate
    the observation gradient. Ancestor sampling connects the reference
    trajectory to the particle histories for path diversity.

    Uses systematic resampling (blackjax) for free particles (lower variance
    than multinomial), and categorical sampling for the single-draw ancestor
    sampling step.
    """
    _T, n_l = ref_traj.shape
    N = n_particles
    jitter = jnp.eye(n_l) * 1e-6

    # --- Initialize at t=0 ---
    chol_t0 = jla.cholesky(t0_cov + jitter, lower=True)
    key, init_key = random.split(key)
    init_keys = random.split(init_key, N - 1)

    init_free = t0_mean + jax.vmap(lambda k: chol_t0 @ random.normal(k, (n_l,)))(init_keys)
    particles_0 = jnp.concatenate([init_free, ref_traj[0:1]], axis=0)

    log_w_0 = jax.vmap(lambda x: obs_lp_fn(x, observations[0], obs_mask_float[0]))(particles_0)

    # --- Scan over t = 1, ..., T-1 ---
    def scan_step(carry, inputs):
        particles_prev, log_w_prev, key = carry
        Ad_t, Qd_t, chol_t, cd_t, y_t, mask_t, ref_x_t = inputs

        # ---- Systematic resampling for free particles (lower variance) ----
        key, rkey = random.split(key)
        wn = jnp.exp(log_w_prev - jax.nn.logsumexp(log_w_prev))
        free_ancestors = _systematic_resample(rkey, wn, N - 1)
        parent_states = particles_prev[free_ancestors]

        # ---- Propagate free particles with gradient-informed proposals ----
        prior_means = jax.vmap(lambda x: Ad_t @ x)(parent_states) + cd_t

        def compute_shift(m):
            g = grad_obs_fn(m, y_t, mask_t)
            g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            raw_shift = langevin_step_size * Qd_t @ g
            # Clip shift to at most 1 std of process noise to prevent
            # positive feedback when obs noise is underestimated.
            scaled = jla.solve_triangular(chol_t, raw_shift, lower=True)
            norm = jnp.sqrt(jnp.dot(scaled, scaled) + 1e-10)
            clip = jnp.minimum(1.0, 1.0 / norm)
            return raw_shift * clip

        shifts = jax.vmap(compute_shift)(prior_means)
        proposal_means = prior_means + shifts

        key, nkey = random.split(key)
        z = random.normal(nkey, (N - 1, n_l))
        new_x_free = proposal_means + jax.vmap(lambda zi: chol_t @ zi)(z)

        new_particles = jnp.concatenate([new_x_free, ref_x_t[None]], axis=0)

        # ---- Weights for free particles: g(y|x) * f/q ----
        obs_ll_free = jax.vmap(lambda x: obs_lp_fn(x, y_t, mask_t))(new_x_free)

        diff_f = new_x_free - prior_means
        diff_q = new_x_free - proposal_means

        def log_ratio(df, dq):
            Linv_df = jla.solve_triangular(chol_t, df, lower=True)
            Linv_dq = jla.solve_triangular(chol_t, dq, lower=True)
            return -0.5 * (jnp.dot(Linv_df, Linv_df) - jnp.dot(Linv_dq, Linv_dq))

        log_fq = jax.vmap(log_ratio)(diff_f, diff_q)
        log_w_free = obs_ll_free + log_fq

        # Reference particle weight: observation likelihood only (no proposal ratio)
        ref_obs_ll = obs_lp_fn(ref_x_t, y_t, mask_t)
        new_log_w = jnp.concatenate([log_w_free, jnp.array([ref_obs_ll])])

        # ---- Ancestor sampling for reference (Eq 33: w * f(ref|x_prev)) ----
        def log_trans_to_ref(x_prev):
            mean = Ad_t @ x_prev + cd_t
            diff = ref_x_t - mean
            Linv_d = jla.solve_triangular(chol_t, diff, lower=True)
            return -0.5 * jnp.dot(Linv_d, Linv_d)

        anc_log_w = log_w_prev + jax.vmap(log_trans_to_ref)(particles_prev)
        key, anc_key = random.split(key)
        ref_ancestor = random.categorical(anc_key, anc_log_w)

        full_ancestors = jnp.concatenate(
            [free_ancestors, ref_ancestor[None].astype(free_ancestors.dtype)]
        )

        return (new_particles, new_log_w, key), (new_particles, full_ancestors)

    scan_inputs = (
        Ad[1:],
        Qd[1:],
        chol_Qd[1:],
        cd[1:],
        observations[1:],
        obs_mask_float[1:],
        ref_traj[1:],
    )

    init_carry = (particles_0, log_w_0, key)
    final_carry, (all_particles, all_ancestors) = jax.lax.scan(scan_step, init_carry, scan_inputs)

    all_particles_full = jnp.concatenate([particles_0[None], all_particles], axis=0)

    # ---- Select trajectory from final weights ----
    _, final_log_w, final_key = final_carry
    sel_key, out_key = random.split(final_key)
    selected = random.categorical(sel_key, final_log_w)

    # ---- Trace back through ancestor indices ----
    def traceback_step(k, inputs):
        ancestors_t, particles_t = inputs
        x = particles_t[k]
        parent = ancestors_t[k]
        return parent, x

    reversed_ancestors = all_ancestors[::-1]
    reversed_particles = all_particles_full[1:][::-1]
    final_k, traj_rev = jax.lax.scan(
        traceback_step, selected, (reversed_ancestors, reversed_particles)
    )

    x_0 = all_particles_full[0, final_k]
    new_traj = jnp.concatenate([x_0[None], traj_rev[::-1]], axis=0)

    return new_traj, out_key


# ---------------------------------------------------------------------------
# Trajectory-conditioned log-posterior for parameter updates
# ---------------------------------------------------------------------------


def _traj_log_post(
    z,
    trajectory,
    dt_array,
    observations,
    obs_mask_float,
    distributions,
    unravel_fn,
    transforms,
    spec,
    adapter,
):
    """Log p(theta | x_{1:T}, y_{1:T}) given fixed trajectory.

    Cheaply evaluable (no PF needed):
    log p(theta | x, y) = log prior(theta) + log p(x_1 | theta)
                        + sum_t log f(x_t | x_{t-1}, theta)
                        + sum_t log g(y_t | x_t, theta)
    """
    T = observations.shape[0]
    n_l = spec.n_latent
    jitter_l = jnp.eye(n_l) * 1e-6

    unc = unravel_fn(z)
    con = {name: transforms[name](unc[name]) for name in unc}

    # 1. Log-prior + Jacobian
    lp_prior = sum(jnp.sum(distributions[name].log_prob(con[name])) for name in unc)
    lp_jac = sum(
        jnp.sum(transforms[name].log_abs_det_jacobian(unc[name], con[name])) for name in unc
    )

    # 2. Assemble matrices
    samples_1 = {name: con[name][None] for name in con}
    det = _assemble_deterministics(samples_1, spec)
    drift, diff_cov, cint, lambda_mat, manifest_means, manifest_cov, t0_mean, t0_cov = (
        _extract_matrices(det, con, spec)
    )

    # 3. Log p(x_1 | theta)
    chol_t0 = jla.cholesky(t0_cov + jitter_l, lower=True)
    dx0 = trajectory[0] - t0_mean
    Linv_dx0 = jla.solve_triangular(chol_t0, dx0, lower=True)
    lp_init = -0.5 * jnp.dot(Linv_dx0, Linv_dx0) - jnp.sum(jnp.log(jnp.diag(chol_t0)))

    # 4. Log transition densities
    Ad, Qd, cd_all = discretize_system_batched(drift, diff_cov, cint, dt_array)
    if cd_all is None:
        cd_all = jnp.zeros((T, n_l))
    chol_Qd = jax.vmap(lambda Q: jla.cholesky(Q + jitter_l, lower=True))(Qd)

    def scan_trans(lp, inputs):
        x_prev, x_curr, Ad_t, chol_t, cd_t = inputs
        lp_t = _transition_log_prob(x_curr, x_prev, Ad_t, chol_t, cd_t)
        return lp + lp_t, None

    lp_trans, _ = jax.lax.scan(
        scan_trans,
        0.0,
        (trajectory[:-1], trajectory[1:], Ad[1:], chol_Qd[1:], cd_all[1:]),
    )

    # 5. Log observation densities via SSMAdapter (supports all noise families)
    params = {
        "lambda_mat": lambda_mat,
        "manifest_means": manifest_means,
        "manifest_cov": manifest_cov,
    }

    def obs_lp_single(x, y, mask):
        return adapter.observation_log_prob(y, x, params, mask)

    lp_obs = jnp.sum(jax.vmap(obs_lp_single)(trajectory, observations, obs_mask_float))

    total = lp_prior + lp_jac + lp_init + lp_trans + lp_obs
    return jnp.where(jnp.isfinite(total), total, -1e30)


# ---------------------------------------------------------------------------
# MALA MH step for parameter updates
# ---------------------------------------------------------------------------


def _mala_step(key, theta, val, grad, log_post_fn, step_size):
    """One MALA Metropolis-Hastings step."""
    key, noise_key, accept_key = random.split(key, 3)
    z = random.normal(noise_key, theta.shape)

    eps = step_size
    theta_star = theta + 0.5 * eps**2 * grad + eps * z

    val_star, grad_star = jax.value_and_grad(log_post_fn)(theta_star)
    grad_star = jnp.nan_to_num(grad_star, nan=0.0, posinf=0.0, neginf=0.0)

    diff_fwd = theta_star - theta - 0.5 * eps**2 * grad
    log_q_fwd = -0.5 * jnp.sum(diff_fwd**2) / eps**2

    diff_rev = theta - theta_star - 0.5 * eps**2 * grad_star
    log_q_rev = -0.5 * jnp.sum(diff_rev**2) / eps**2

    log_alpha = val_star - val + log_q_rev - log_q_fwd
    u = random.uniform(accept_key)
    accept = (jnp.log(u) < log_alpha) & jnp.isfinite(val_star)

    new_theta = jnp.where(accept, theta_star, theta)
    new_val = jnp.where(accept, val_star, val)
    new_grad = jnp.where(accept, grad_star, grad)

    return new_theta, new_val, new_grad, accept, key


# ---------------------------------------------------------------------------
# Main PGAS sampler
# ---------------------------------------------------------------------------


def fit_pgas(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    n_outer: int = 50,
    n_csmc_particles: int = 20,
    n_mh_steps: int = 5,
    langevin_step_size: float = 0.0,
    param_step_size: float = 0.1,
    n_warmup: int | None = None,
    seed: int = 0,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM via PGAS with gradient-informed CSMC and MALA parameter updates.

    Gibbs loop:
      1. Sample x_{1:T} | theta, y via CSMC with gradient proposals + ancestor sampling
      2. Update theta | x_{1:T}, y via MALA MH steps (no PF needed)

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        subject_ids: optional subject indices
        n_outer: number of Gibbs iterations
        n_csmc_particles: N for CSMC (including reference particle)
        n_mh_steps: MALA steps per parameter update
        langevin_step_size: step size for gradient shift in CSMC proposals
        param_step_size: MALA step size for parameter updates
        n_warmup: warmup iterations to discard (default: n_outer // 2)
        seed: random seed

    Returns:
        InferenceResult with posterior samples
    """
    rng_key = random.PRNGKey(seed)
    N_csmc = n_csmc_particles
    T = observations.shape[0]
    n_l = model.spec.n_latent
    n_m = model.spec.n_manifest

    if n_warmup is None:
        n_warmup = n_outer // 2

    # Observation mask
    obs_mask = ~jnp.isnan(observations)
    obs_mask_float = obs_mask.astype(jnp.float32)
    clean_obs = jnp.nan_to_num(observations, nan=0.0)

    # Time intervals
    dt_array = jnp.diff(times, prepend=times[0])
    dt_array = dt_array.at[0].set(1e-6)

    print(f"PGAS: n_outer={n_outer}, N_csmc={N_csmc}, n_mh={n_mh_steps}, n_l={n_l}")

    # 1. Discover model sites
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, clean_obs, times, subject_ids, trace_key)
    transforms = {name: info["transform"] for name, info in site_info.items()}
    distributions = {name: info["distribution"] for name, info in site_info.items()}

    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    D = flat_example.shape[0]
    print(f"  D={D} parameters, T={T} time steps")

    # 2. SSMAdapter for observation model (supports Gaussian, Poisson, Student-t, Gamma)
    adapter = SSMAdapter(
        n_l,
        n_m,
        manifest_dist=model.spec.manifest_dist.value,
        diffusion_dist=model.spec.diffusion_dist.value,
    )

    # 3. Build JIT-compiled CSMC sweep
    # obs_lp_fn and grad_obs_fn close over the adapter (Python object, traced at compile time).
    # The measurement matrices are passed as explicit arguments to _csmc_sweep via the wrapper.
    def _do_csmc(
        key, ref_traj, Ad, Qd, chol_Qd, cd, lam, means, cov, t0_mean, t0_cov, obs, mask, step_size
    ):
        params = {"lambda_mat": lam, "manifest_means": means, "manifest_cov": cov}

        def obs_lp(x, y, m):
            return adapter.observation_log_prob(y, x, params, m)

        grad_obs = jax.grad(obs_lp, argnums=0)

        return _csmc_sweep(
            key,
            ref_traj,
            Ad,
            Qd,
            chol_Qd,
            cd,
            t0_mean,
            t0_cov,
            obs,
            mask,
            N_csmc,
            step_size,
            obs_lp,
            grad_obs,
        )

    jit_csmc = jax.jit(_do_csmc)

    # 4. Build checkpointed trajectory log-posterior + JIT value_and_grad
    @jax.checkpoint
    def _log_post(z, trajectory):
        return _traj_log_post(
            z,
            trajectory,
            dt_array,
            clean_obs,
            obs_mask_float,
            distributions,
            unravel_fn,
            transforms,
            model.spec,
            adapter,
        )

    @jax.jit
    def _val_grad(z, trajectory):
        val, grad = jax.value_and_grad(_log_post)(z, trajectory)
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return val, grad

    # JIT the MALA step (log_post_fn captured via closure over _log_post)
    @jax.jit
    def _jit_mala_step(key, theta, val, grad, trajectory, step_size):
        def lp(z):
            return _log_post(z, trajectory)

        return _mala_step(key, theta, val, grad, lp, step_size)

    # 5. Initialize parameters at prior mean (more stable than random sample)
    parts = []
    for name in sorted(site_info.keys()):
        info = site_info[name]
        prior_mean = info["distribution"].mean
        unc_mean = info["transform"].inv(prior_mean)
        parts.append(unc_mean.reshape(-1))
    theta_unc = jnp.concatenate(parts)

    # 6. Initialize trajectory from prior simulation
    drift, diff_cov, cint, lambda_mat, manifest_means, manifest_cov, t0_mean, t0_cov = (
        _params_to_matrices(theta_unc, unravel_fn, transforms, model.spec)
    )

    rng_key, sim_key = random.split(rng_key)
    trajectory = _simulate_trajectory(
        sim_key, drift, diff_cov, cint, t0_mean, t0_cov, dt_array, n_l
    )

    # Storage
    theta_chain = []
    accept_rates = []
    current_step_size = param_step_size

    print("  Starting PGAS loop...")

    # 7. PGAS Gibbs loop
    for n in range(n_outer):
        # --- Step A: CSMC sweep (sample trajectory given theta) ---
        drift, diff_cov, cint, lambda_mat, manifest_means, manifest_cov, t0_mean, t0_cov = (
            _params_to_matrices(theta_unc, unravel_fn, transforms, model.spec)
        )

        Ad, Qd, cd = discretize_system_batched(drift, diff_cov, cint, dt_array)
        if cd is None:
            cd = jnp.zeros((T, n_l))
        _jitter = jnp.eye(n_l) * 1e-6
        chol_Qd = jax.vmap(lambda Q, j=_jitter: jla.cholesky(Q + j, lower=True))(Qd)

        rng_key, csmc_key = random.split(rng_key)
        trajectory, _ = jit_csmc(
            csmc_key,
            trajectory,
            Ad,
            Qd,
            chol_Qd,
            cd,
            lambda_mat,
            manifest_means,
            manifest_cov,
            t0_mean,
            t0_cov,
            clean_obs,
            obs_mask_float,
            langevin_step_size,
        )

        # --- Step B: Parameter update (MALA steps given trajectory) ---
        val, grad = _val_grad(theta_unc, trajectory)

        n_accepted = 0
        for _ in range(n_mh_steps):
            rng_key, mh_key = random.split(rng_key)
            theta_unc, val, grad, accepted, _ = _jit_mala_step(
                mh_key, theta_unc, val, grad, trajectory, current_step_size
            )
            n_accepted += int(accepted)

        accept_rate = n_accepted / n_mh_steps
        accept_rates.append(accept_rate)

        # Adapt step size (target 25-50% acceptance)
        # During warmup: aggressive adaptation. Post-warmup: gentler to preserve ergodicity.
        if n > 0 and n % 5 == 0:
            recent_rates = accept_rates[-5:]
            avg_rate = sum(recent_rates) / len(recent_rates)
            if n < n_warmup:
                if avg_rate < 0.15:
                    current_step_size *= 0.5
                elif avg_rate > 0.6:
                    current_step_size *= 2.0
                elif avg_rate < 0.25:
                    current_step_size *= 0.8
                elif avg_rate > 0.5:
                    current_step_size *= 1.3
            else:
                # Gentler post-warmup adaptation to avoid frozen chains
                if avg_rate < 0.05:
                    current_step_size *= 0.5
                elif avg_rate < 0.15:
                    current_step_size *= 0.8
                elif avg_rate > 0.6:
                    current_step_size *= 1.2

        theta_chain.append(theta_unc.copy())

        if (n + 1) % max(1, n_outer // 5) == 0:
            print(
                f"  iter {n + 1}/{n_outer}  log_post={float(val):.1f}"
                f"  accept={accept_rate:.2f}  step={current_step_size:.4f}"
            )

    # 8. Extract posterior samples (discard warmup)
    theta_samples_unc = jnp.stack(theta_chain[n_warmup:])

    samples = {}
    for name in transforms:

        def _extract_one(z, _name=name):
            unc = unravel_fn(z)
            return transforms[_name](unc[_name])

        samples[name] = jax.vmap(_extract_one)(theta_samples_unc)

    det_samples = _assemble_deterministics(samples, model.spec)
    samples.update(det_samples)

    return InferenceResult(
        _samples=samples,
        method="pgas",
        diagnostics={
            "accept_rates": accept_rates,
            "n_outer": n_outer,
            "n_warmup": n_warmup,
            "n_csmc_particles": N_csmc,
            "n_mh_steps": n_mh_steps,
            "langevin_step_size": langevin_step_size,
            "param_step_size": current_step_size,
        },
    )
