"""PGAS: Particle Gibbs with Ancestor Sampling + Gradient-Informed Proposals.

Combines three elements:
1. PGAS outer loop: Gibbs-alternate between latent trajectories and parameters
2. Gradient-informed CSMC: Langevin proposals inside conditional SMC for states
3. HMC/MALA parameter updates: Gradient-informed MH for the parameter conditional

The CSMC sweep uses the PGAS kernel (Lindsten, Jordan & Schoen, 2014) with
gradient-informed state proposals that shift the bootstrap transition toward
high-likelihood regions via nabla_x log p(y_t | x_t). The ancestor sampling step
uses the model transition density (unaffected by the proposal choice).

Upgrades:
  - **Preconditioned MALA/HMC**: Uses running covariance mass matrix from theta
    chain, replacing identity-mass MALA. Shared hmc_step from mcmc_utils.
  - **Locally optimal proposal**: For Gaussian observations, analytically computes
    the optimal CSMC proposal p(x_t | x_{t-1}, y_t) which incorporates observation
    information directly into the proposal.
  - **Block parameter sampling**: Splits parameters into blocks by site name,
    updates each block independently with separate step sizes and mass matrices.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from blackjax.smc.resampling import systematic as _systematic_resample
from jax.flatten_util import ravel_pytree
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.optim import ClippedAdam

from causal_ssm_agent.models.likelihoods.particle import SSMAdapter
from causal_ssm_agent.models.ssm.constants import MIN_DT
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference import InferenceResult
from causal_ssm_agent.models.ssm.mcmc_utils import (
    compute_weighted_chol_mass,
    dual_averaging_init,
    dual_averaging_update,
    hmc_step,
)
from causal_ssm_agent.models.ssm.utils import _assemble_deterministics, _discover_sites

# ---------------------------------------------------------------------------
# SVI warmstart for mass matrix initialization
# ---------------------------------------------------------------------------


def _svi_warmstart(
    model_fn,
    observations,
    times,
    D,
    blocks=None,
    num_steps=500,
    learning_rate=0.01,
    seed=42,
):
    """Run quick SVI to estimate posterior covariance for mass matrix init.

    Uses AutoMultivariateNormal guide, which parameterizes the posterior in the
    same unconstrained space as PGAS (both use ravel_pytree on sorted dict keys).

    Args:
        model_fn: NumPyro model function (with likelihood_backend already bound)
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        D: total number of unconstrained parameters
        blocks: optional list of block dicts (for per-block mass initialization)
        num_steps: SVI optimization steps
        learning_rate: Adam learning rate
        seed: random seed

    Returns:
        init_theta: (D,) SVI posterior mean in unconstrained space
        chol_mass_full: (D, D) Cholesky of precision matrix
        block_chol_masses: dict of block_name -> (block_size, block_size) Cholesky,
            or None if blocks is None
    """
    guide = AutoMultivariateNormal(model_fn)
    optimizer = ClippedAdam(step_size=learning_rate)
    svi = SVI(model_fn, guide, optimizer, Trace_ELBO())

    rng_key = random.PRNGKey(seed)
    svi_result = svi.run(rng_key, num_steps, observations, times, progress_bar=False)

    # Extract guide parameters
    init_theta = svi_result.params["auto_loc"]  # (D,) posterior mean
    scale_tril = svi_result.params["auto_scale_tril"]  # (D, D) lower-tri Cholesky of cov

    # Check for NaN/Inf
    if not (jnp.all(jnp.isfinite(init_theta)) and jnp.all(jnp.isfinite(scale_tril))):
        print("  SVI warmstart: NaN detected, falling back to identity mass matrix")
        return None, None, None

    # Full covariance and precision
    reg = 1e-3
    full_cov = scale_tril @ scale_tril.T

    # Full mass matrix: precision = inv(cov + reg*I)
    cov_reg = full_cov + reg * jnp.eye(D)
    L_cov = jla.cholesky(cov_reg, lower=True)
    prec = jla.cho_solve((L_cov, True), jnp.eye(D))
    chol_mass_full = jla.cholesky(prec, lower=True)

    # Per-block mass matrices from marginal covariance sub-blocks
    block_chol_masses = None
    if blocks is not None:
        block_chol_masses = {}
        for block in blocks:
            s = block["slice"]
            bsize = block["size"]
            block_cov = full_cov[s, s] + reg * jnp.eye(bsize)
            L_b = jla.cholesky(block_cov, lower=True)
            block_prec = jla.cho_solve((L_b, True), jnp.eye(bsize))
            block_chol_masses[block["name"]] = jla.cholesky(block_prec, lower=True)

    final_loss = float(svi_result.losses[-1])
    print(f"  SVI warmstart: {num_steps} steps, final ELBO={final_loss:.0f}")

    return init_theta, chol_mass_full, block_chol_masses


# ---------------------------------------------------------------------------
# Transition log-prob
# ---------------------------------------------------------------------------


def _transition_log_prob(x_curr, x_prev, Ad_t, chol_t, cd_t):
    """Log N(x_curr; Ad_t @ x_prev + cd_t, Qd_t) where Qd_t = chol_t @ chol_t.T."""
    from numpyro.distributions import MultivariateNormal

    mean = Ad_t @ x_prev + cd_t
    return MultivariateNormal(mean, scale_tril=chol_t).log_prob(x_curr)


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
    gaussian_obs=False,
    lambda_mat=None,
    manifest_means=None,
    R_adj=None,
):
    """One CSMC sweep (Algorithm 2 from PGAS paper) with gradient proposals.

    Runs a conditional SMC sampler where particle N-1 is pinned to ref_traj.
    Free particles (0..N-2) use Langevin-shifted proposals that incorporate
    the observation gradient. Ancestor sampling connects the reference
    trajectory to the particle histories for path diversity.

    When gaussian_obs=True and R_adj is provided, uses the locally optimal
    proposal that analytically incorporates observation information.
    """
    _T, n_l = ref_traj.shape
    N = n_particles
    n_m = observations.shape[1]
    jitter = jnp.eye(n_l) * 1e-6
    jitter_obs = jnp.eye(n_m) * 1e-6

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

        # ---- Propagate free particles ----
        prior_means = jax.vmap(lambda x: Ad_t @ x)(parent_states) + cd_t

        if gaussian_obs:
            # Locally optimal proposal for Gaussian observations
            # R_adj varies per timestep for missing data
            R_adj_t = R_adj[0]  # base R_adj (missing data handled via mask inflation)
            # Adjust R_inv for this timestep's mask
            mask_inf = (1.0 - mask_t) * 1e10
            R_adj_t_inflated = R_adj_t + jnp.diag(mask_inf)
            R_inv_t = jnp.linalg.inv(R_adj_t_inflated + jitter_obs)
            LtRinvL_t = lambda_mat.T @ R_inv_t @ lambda_mat
            LtRinv_t = lambda_mat.T @ R_inv_t

            Qd_inv_t = jnp.linalg.inv(Qd_t + jitter)
            S_inv_t = Qd_inv_t + LtRinvL_t
            S_t = jnp.linalg.inv(S_inv_t)
            chol_S_t = jla.cholesky(S_t + jitter, lower=True)

            residual = y_t - manifest_means
            info_obs = LtRinv_t @ residual

            def propose_optimal(prior_mean, k):
                m = S_t @ (Qd_inv_t @ prior_mean + info_obs)
                return m + chol_S_t @ random.normal(k, (n_l,))

            key, nkey = random.split(key)
            prop_keys = random.split(nkey, N - 1)
            new_x_free = jax.vmap(propose_optimal)(prior_means, prop_keys)

            new_particles = jnp.concatenate([new_x_free, ref_x_t[None]], axis=0)

            # Weights for optimal proposal: p(y|x_prev) = N(y; Lambda@prior_mean+mu, Lambda@Qd@Lambda^T+R)
            pred_cov = lambda_mat @ Qd_t @ lambda_mat.T + R_adj_t_inflated
            chol_pred = jla.cholesky(pred_cov + jitter_obs, lower=True)

            def log_pred_lik(prior_mean):
                pred_mean = lambda_mat @ prior_mean + manifest_means
                diff = y_t - pred_mean
                Linv_diff = jla.solve_triangular(chol_pred, diff, lower=True)
                logdet = jnp.sum(jnp.log(jnp.diag(chol_pred)))
                return (
                    -0.5 * jnp.dot(Linv_diff, Linv_diff) - logdet - 0.5 * n_m * jnp.log(2 * jnp.pi)
                )

            log_w_free = jax.vmap(log_pred_lik)(prior_means)

        else:
            # Standard gradient-informed proposal (bootstrap + Langevin shift)
            def compute_shift(m):
                g = grad_obs_fn(m, y_t, mask_t)
                g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                raw_shift = langevin_step_size * Qd_t @ g
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

            # Weights for free particles: g(y|x) * f/q
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
# Main PGAS sampler
# ---------------------------------------------------------------------------


def fit_pgas(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    n_outer: int = 50,
    n_csmc_particles: int = 30,
    n_mh_steps: int = 5,
    langevin_step_size: float = 0.0,
    param_step_size: float = 0.1,
    n_warmup: int | None = None,
    seed: int = 0,
    n_leapfrog: int = 5,
    block_sampling: bool = False,
    svi_warmstart: bool = True,
    svi_num_steps: int = 500,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM via PGAS with gradient-informed CSMC and HMC parameter updates.

    Gibbs loop:
      1. Sample x_{1:T} | theta, y via CSMC with gradient proposals + ancestor sampling
      2. Update theta | x_{1:T}, y via block HMC/MALA steps (no PF needed)

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        n_outer: number of Gibbs iterations
        n_csmc_particles: N for CSMC (including reference particle).
            For good mixing, use N >= T/2 (Lindsten et al. 2014).
        n_mh_steps: HMC/MALA steps per parameter update
        langevin_step_size: step size for gradient shift in CSMC proposals
        param_step_size: HMC/MALA step size for parameter updates
        n_warmup: warmup iterations to discard (default: n_outer // 2)
        seed: random seed
        n_leapfrog: number of leapfrog steps (1 = MALA, >1 = HMC)
        block_sampling: update parameter blocks independently
        svi_warmstart: run SVI to initialize mass matrix and parameters
        svi_num_steps: number of SVI steps for warmstart

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
    dt_array = jnp.maximum(dt_array, MIN_DT)

    # Detect Gaussian observations for optimal proposal
    gaussian_obs = model.spec.manifest_dist == "gaussian"

    block_tag = "+block" if block_sampling else ""
    hmc_tag = f"+HMC(L={n_leapfrog})" if n_leapfrog > 1 else ""
    opt_tag = "+optimal" if gaussian_obs else ""
    svi_tag = "+svi_init" if svi_warmstart else ""
    print(
        f"PGAS [precond{block_tag}{hmc_tag}{opt_tag}{svi_tag}]: "
        f"n_outer={n_outer}, N_csmc={N_csmc}, n_mh={n_mh_steps}, n_l={n_l}"
    )

    # 1. Discover model sites
    backend = model.make_likelihood_backend()
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, clean_obs, times, trace_key, backend)
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

    # 3. Build block structure for parameter sampling
    blocks = []
    offset = 0
    for name in sorted(site_info.keys()):
        size = site_info[name]["value"].reshape(-1).shape[0]
        blocks.append(
            {
                "name": name,
                "slice": slice(offset, offset + size),
                "size": size,
                "step_size": param_step_size,
                "chol_mass": jnp.eye(size),
            }
        )
        offset += size

    # 4. Build JIT-compiled CSMC sweep
    def _do_csmc(
        key, ref_traj, Ad, Qd, chol_Qd, cd, lam, means, cov, t0_mean, t0_cov, obs, mask, step_size
    ):
        params = {"lambda_mat": lam, "manifest_means": means, "manifest_cov": cov}

        def obs_lp(x, y, m):
            return adapter.observation_log_prob(y, x, params, m)

        grad_obs = jax.grad(obs_lp, argnums=0)

        # Precompute R_adj for optimal proposal
        R_adj = None
        if gaussian_obs:
            R_adj = cov[None]  # (1, n_m, n_m) — base manifest_cov

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
            gaussian_obs=gaussian_obs,
            lambda_mat=lam if gaussian_obs else None,
            manifest_means=means if gaussian_obs else None,
            R_adj=R_adj,
        )

    jit_csmc = jax.jit(_do_csmc)

    # 5. Build checkpointed trajectory log-posterior + JIT value_and_grad
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

    # JIT'd HMC step for full-vector updates (non-block mode)
    @jax.jit
    def _jit_hmc_step(key, theta, trajectory, step_size, chol_mass):
        def lp_vg(z):
            v, g = jax.value_and_grad(_log_post)(z, trajectory)
            return jnp.where(jnp.isfinite(v), v, -1e30), jnp.nan_to_num(g)

        return hmc_step(key, theta, lp_vg, step_size, chol_mass, n_leapfrog)

    # Per-block JIT'd HMC steps (block_start and block_size must be concrete)
    def _make_block_hmc_fn(block_start, block_size):
        @jax.jit
        def _block_hmc(key, theta_full, trajectory, step_size, chol_mass_block):
            z_block = theta_full[block_start : block_start + block_size]

            def block_lp_vg(z_b):
                theta_new = theta_full.at[block_start : block_start + block_size].set(z_b)
                v, g_full = jax.value_and_grad(_log_post)(theta_new, trajectory)
                g_block = g_full[block_start : block_start + block_size]
                return jnp.where(jnp.isfinite(v), v, -1e30), jnp.nan_to_num(g_block)

            z_new, accepted, log_target_new = hmc_step(
                key, z_block, block_lp_vg, step_size, chol_mass_block, n_leapfrog
            )
            theta_updated = theta_full.at[block_start : block_start + block_size].set(z_new)
            theta_result = jnp.where(accepted, theta_updated, theta_full)
            return theta_result, accepted, log_target_new

        return _block_hmc

    # Create per-block JIT functions with concrete slice indices
    block_hmc_fns = []
    for block in blocks:
        s = block["slice"]
        block_hmc_fns.append(_make_block_hmc_fn(s.start, block["size"]))

    # 6. Initialize parameters and mass matrix
    chol_mass_full = jnp.eye(D)

    model_fn = functools.partial(model.model, likelihood_backend=backend)

    if svi_warmstart:
        svi_theta, svi_chol_mass, svi_block_chols = _svi_warmstart(
            model_fn,
            observations,
            times,
            D,
            blocks=blocks if block_sampling else None,
            num_steps=svi_num_steps,
            seed=seed + 1000,
        )
        if svi_theta is not None:
            theta_unc = svi_theta
            chol_mass_full = svi_chol_mass
            if svi_block_chols is not None:
                for block in blocks:
                    block["chol_mass"] = svi_block_chols[block["name"]]
        else:
            # SVI failed, fall back to prior mean + identity mass
            svi_warmstart = False

    if not svi_warmstart:
        # Initialize at prior mean (original behavior)
        parts = []
        for name in sorted(site_info.keys()):
            info = site_info[name]
            prior_mean = info["distribution"].mean
            unc_mean = info["transform"].inv(prior_mean)
            parts.append(unc_mean.reshape(-1))
        theta_unc = jnp.concatenate(parts)

    # 7. Initialize trajectory from current parameters
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
    block_accept_rates = {b["name"]: [] for b in blocks} if block_sampling else {}

    # Dual averaging for step size adaptation
    target_accept = 0.65 if n_leapfrog > 1 else 0.574
    da_state = dual_averaging_init(param_step_size)
    current_step_size = param_step_size
    # Per-block dual averaging states
    if block_sampling:
        for block in blocks:
            block["da_state"] = dual_averaging_init(block["step_size"])

    # Mass matrix update parameters
    mass_update_interval = 10
    min_mass_samples = max(2 * D, 20)

    print("  Starting PGAS loop...")

    # 8. PGAS Gibbs loop
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

        # --- Step B: Parameter update ---
        if block_sampling and len(blocks) > 1:
            # Block parameter updates
            n_accepted_total = 0
            for bi, block in enumerate(blocks):
                block_accepted = 0
                for _ in range(n_mh_steps):
                    rng_key, mh_key = random.split(rng_key)
                    theta_unc, accepted, _ = block_hmc_fns[bi](
                        mh_key,
                        theta_unc,
                        trajectory,
                        block["step_size"],
                        block["chol_mass"],
                    )
                    block_accepted += int(accepted)
                block_rate = block_accepted / n_mh_steps
                block_accept_rates[block["name"]].append(block_rate)
                n_accepted_total += block_accepted

                # Per-block dual averaging step size adaptation during warmup
                if n < n_warmup:
                    block["da_state"] = dual_averaging_update(
                        block["da_state"], block_rate, target_accept
                    )
                    block["step_size"] = block["da_state"].eps
                elif n == n_warmup:
                    # Fix step size at averaged value after warmup
                    block["step_size"] = block["da_state"].eps_bar

            accept_rate = n_accepted_total / (n_mh_steps * len(blocks))
        else:
            # Joint parameter update (original behavior)
            n_accepted = 0
            for _ in range(n_mh_steps):
                rng_key, mh_key = random.split(rng_key)
                theta_new, accepted, _ = _jit_hmc_step(
                    mh_key, theta_unc, trajectory, current_step_size, chol_mass_full
                )
                theta_unc = jnp.where(accepted, theta_new, theta_unc)
                n_accepted += int(accepted)

            accept_rate = n_accepted / n_mh_steps

            # Joint dual averaging step size adaptation during warmup
            if n < n_warmup:
                da_state = dual_averaging_update(da_state, accept_rate, target_accept)
                current_step_size = da_state.eps
            elif n == n_warmup:
                # Fix step size at averaged value after warmup
                current_step_size = da_state.eps_bar

        accept_rates.append(accept_rate)

        # Mass matrix update during warmup
        theta_chain.append(theta_unc.copy())
        if n % mass_update_interval == 0 and n < n_warmup and len(theta_chain) >= min_mass_samples:
            recent = jnp.stack(theta_chain[-min_mass_samples:])
            if block_sampling and len(blocks) > 1:
                # Per-block mass matrix
                for block in blocks:
                    s = block["slice"]
                    block_samples = recent[:, s]
                    block["chol_mass"] = compute_weighted_chol_mass(
                        block_samples, jnp.zeros(len(block_samples)), block["size"]
                    )
            else:
                chol_mass_full = compute_weighted_chol_mass(recent, jnp.zeros(len(recent)), D)

        if (n + 1) % max(1, n_outer // 5) == 0:
            print(
                f"  iter {n + 1}/{n_outer}  accept={accept_rate:.2f}  step={current_step_size:.4f}"
            )

    # 9. Extract posterior samples (discard warmup)
    theta_samples_unc = jnp.stack(theta_chain[n_warmup:])

    samples = {}
    for name in transforms:

        def _extract_one(z, _name=name):
            unc = unravel_fn(z)
            return transforms[_name](unc[_name])

        samples[name] = jax.vmap(_extract_one)(theta_samples_unc)

    det_samples = _assemble_deterministics(samples, model.spec)
    samples.update(det_samples)

    diagnostics = {
        "accept_rates": accept_rates,
        "n_outer": n_outer,
        "n_warmup": n_warmup,
        "n_csmc_particles": N_csmc,
        "n_mh_steps": n_mh_steps,
        "n_leapfrog": n_leapfrog,
        "langevin_step_size": langevin_step_size,
        "param_step_size": current_step_size,
        "block_sampling": block_sampling,
        "gaussian_obs": gaussian_obs,
        "svi_warmstart": svi_warmstart,
    }
    if block_sampling and len(blocks) > 1:
        diagnostics["block_accept_rates"] = block_accept_rates
        diagnostics["block_step_sizes"] = {b["name"]: b["step_size"] for b in blocks}

    return InferenceResult(
        _samples=samples,
        method="pgas",
        diagnostics=diagnostics,
    )
