"""Block Rao-Blackwell particle filter for mixed Gaussian/non-Gaussian dynamics.

BIRCH-style graph decomposition for SSMs: partition latent variables into a
Gaussian block (analytically marginalized via Kalman filter) and a sampled block
(propagated via bootstrap PF). This gives variance reduction over full bootstrap
PF whenever a subset of latents has Gaussian dynamics.

Architecture:
- Each particle carries a BlockRBState: Kalman sufficient statistics for the
  Gaussian block + a point sample for the sampled block.
- init_sample: RBState for G-block from prior, sample S-block from prior.
- propagate_sample: Kalman predict on G-block (with S-block as known input),
  sample S-block (with G-block posterior mean as point estimate).
- log_potential: integrate p(y | x_g, x_s) over x_g via quadrature, fix x_s.

Cross-coupling:
- S -> G: sampled value x_s enters Kalman predict as known input (exact).
- G -> S: Gaussian posterior mean m_g used as point estimate (approximation).
- Shared observations: quadrature marginalizes over G-block only, fixing x_s.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from causal_ssm_agent.models.likelihoods.emissions import get_emission_fn
from causal_ssm_agent.models.likelihoods.rao_blackwell import (
    _kalman_predict,
    _linearized_update,
    _unscented_sigma_points,
)

# =============================================================================
# BlockRBState — hybrid state carried per particle
# =============================================================================


class BlockRBState(NamedTuple):
    """Per-particle hybrid state for block Rao-Blackwell PF.

    Gaussian block: Kalman sufficient statistics (analytically marginalized).
    Sampled block: point samples (propagated via bootstrap PF).
    """

    # Gaussian block — Kalman sufficient statistics
    g_mean: jnp.ndarray  # (n_g,) — updated (posterior) mean
    g_cov: jnp.ndarray  # (n_g, n_g) — updated covariance
    g_pred_mean: jnp.ndarray  # (n_g,) — predicted mean (for weighting)
    g_pred_cov: jnp.ndarray  # (n_g, n_g) — predicted covariance

    # Sampled block — point samples
    s_sample: jnp.ndarray  # (n_s,) — current sampled state


# =============================================================================
# Partitioning utilities
# =============================================================================


def partition_indices(
    diffusion_dists: list[str],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Partition latent variable indices into Gaussian and sampled blocks.

    Args:
        diffusion_dists: Per-variable diffusion distribution, e.g.
            ["gaussian", "student_t", "gaussian"].

    Returns:
        g_idx: Integer array of Gaussian variable indices.
        s_idx: Integer array of sampled (non-Gaussian) variable indices.
    """
    import numpy as np

    g_idx = []
    s_idx = []
    for i, d in enumerate(diffusion_dists):
        if d == "gaussian":
            g_idx.append(i)
        else:
            s_idx.append(i)
    # Return as numpy arrays to avoid JIT tracing issues when called at
    # construction time inside a traced function.
    return np.array(g_idx, dtype=np.int32), np.array(s_idx, dtype=np.int32)


def extract_subblocks(
    A: jnp.ndarray,
    Q: jnp.ndarray,
    c: jnp.ndarray,
    g_idx: jnp.ndarray,
    s_idx: jnp.ndarray,
) -> dict:
    """Extract sub-blocks of dynamics matrices for Gaussian/sampled partition.

    Given full matrices A (drift), Q (diffusion cov), c (intercept), partition
    them into 2x2 block structure indexed by g (Gaussian) and s (sampled).

    Args:
        A: (n, n) discrete drift matrix.
        Q: (n, n) discrete diffusion covariance.
        c: (n,) discrete intercept.
        g_idx: Gaussian variable indices.
        s_idx: Sampled variable indices.

    Returns:
        Dict with keys: A_gg, A_gs, A_sg, A_ss, Q_gg, Q_ss, c_g, c_s.
    """
    return {
        "A_gg": A[jnp.ix_(g_idx, g_idx)],
        "A_gs": A[jnp.ix_(g_idx, s_idx)],
        "A_sg": A[jnp.ix_(s_idx, g_idx)],
        "A_ss": A[jnp.ix_(s_idx, s_idx)],
        "Q_gg": Q[jnp.ix_(g_idx, g_idx)],
        "Q_ss": Q[jnp.ix_(s_idx, s_idx)],
        "c_g": c[g_idx],
        "c_s": c[s_idx],
    }


def extract_obs_subblocks(
    H: jnp.ndarray,
    g_idx: jnp.ndarray,
    s_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract observation matrix columns for each block.

    Args:
        H: (n_manifest, n_latent) observation matrix.
        g_idx: Gaussian variable indices.
        s_idx: Sampled variable indices.

    Returns:
        H_g: (n_manifest, n_g) columns for Gaussian block.
        H_s: (n_manifest, n_s) columns for sampled block.
    """
    return H[:, g_idx], H[:, s_idx]


# =============================================================================
# Callback factory — public API
# =============================================================================


def make_block_rb_callbacks(
    n_latent: int,
    n_manifest: int,  # noqa: ARG001
    manifest_dist: str,
    params: dict,
    extra_params: dict,
    m0: jnp.ndarray,
    P0: jnp.ndarray,
    g_idx: jnp.ndarray,
    s_idx: jnp.ndarray,
    quadrature: str = "unscented",
    n_quadrature: int = 5,
):
    """Build Feynman-Kac callbacks for block Rao-Blackwell particle filter.

    Partitions the state space into Gaussian (Kalman-marginalized) and sampled
    (bootstrap PF) blocks. Returns callbacks compatible with cuthbert's
    build_filter.

    Args:
        n_latent: Total latent state dimension.
        n_manifest: Observation dimension.
        manifest_dist: Observation distribution.
        params: Dict with lambda_mat, manifest_means, manifest_cov, etc.
        extra_params: Dict with obs_df, obs_shape, proc_df, etc.
        m0: Initial state mean (n_latent,).
        P0: Initial state covariance (n_latent, n_latent).
        g_idx: Indices of Gaussian (Kalman-marginalized) variables.
        s_idx: Indices of sampled (bootstrap PF) variables.
        quadrature: "unscented" or "gauss_hermite".
        n_quadrature: Number of quadrature points per dimension.

    Returns:
        (init_sample, propagate_sample, log_potential) — callback triple.
    """
    n_g = g_idx.shape[0]
    n_s = s_idx.shape[0]

    H = params["lambda_mat"]  # (n_manifest, n_latent)
    d = params["manifest_means"]  # (n_manifest,)
    R = params["manifest_cov"]  # (n_manifest, n_manifest)

    # Observation matrix sub-blocks
    H_g, H_s = extract_obs_subblocks(H, g_idx, s_idx)

    # Initial state sub-blocks
    m0_g = m0[g_idx]
    m0_s = m0[s_idx]
    P0_gg = P0[jnp.ix_(g_idx, g_idx)]
    P0_ss = P0[jnp.ix_(s_idx, s_idx)]

    # Merge extra_params into params for observation model
    obs_params = {**params, **extra_params}

    # Process noise df for sampled block
    proc_df = extra_params.get("proc_df", 5.0)

    # Diffusion distribution for sampled block
    diffusion_dist = extra_params.get("diffusion_dist_s", "student_t")

    def init_sample(key, _model_inputs):
        """Initialize: Kalman stats for G-block, sample for S-block."""
        chol_P0_ss = jla.cholesky(P0_ss + jnp.eye(n_s) * 1e-6, lower=True)
        s_sample = m0_s + chol_P0_ss @ random.normal(key, (n_s,))
        return BlockRBState(
            g_mean=m0_g,
            g_cov=P0_gg,
            g_pred_mean=m0_g,
            g_pred_cov=P0_gg,
            s_sample=s_sample,
        )

    def propagate_sample(key, state, model_inputs):
        """Propagate hybrid state: Kalman predict on G, sample S.

        1. Extract sub-blocks from pre-discretized dynamics.
        2. Kalman predict on G-block with S-block as known input.
        3. Sample S-block with G-block posterior mean as point estimate.
        4. Linearized update on G-block (EKF-style, for covariance bounding).
        """
        Ad_t = model_inputs["Ad"]  # (n, n) full discrete drift
        Qd_t = model_inputs["Qd"]  # (n, n) full discrete diffusion cov
        cd_t = model_inputs["cd"]  # (n,) full discrete intercept
        y_t = model_inputs["observation"]  # (n_manifest,)
        mask_t = model_inputs["obs_mask"]  # (n_manifest,)

        # Extract sub-blocks
        A_gg = Ad_t[jnp.ix_(g_idx, g_idx)]
        A_gs = Ad_t[jnp.ix_(g_idx, s_idx)]
        A_sg = Ad_t[jnp.ix_(s_idx, g_idx)]
        A_ss = Ad_t[jnp.ix_(s_idx, s_idx)]
        Q_gg = Qd_t[jnp.ix_(g_idx, g_idx)]
        Q_ss = Qd_t[jnp.ix_(s_idx, s_idx)]
        c_g = cd_t[g_idx]
        c_s = cd_t[s_idx]

        x_s = state.s_sample
        m_g = state.g_mean

        # --- Step 1: Kalman predict on G-block ---
        # m_pred_g = A_gg @ m_g + A_gs @ x_s + c_g
        # The S->G coupling enters as a known input.
        c_g_eff = c_g + A_gs @ x_s  # effective intercept includes S-block input
        m_pred_g, P_pred_g = _kalman_predict(m_g, state.g_cov, A_gg, Q_gg, c_g_eff)

        # --- Step 2: Sample S-block ---
        # x_s_new = A_sg @ m_g + A_ss @ x_s + c_s + noise
        # G->S coupling uses posterior mean as point estimate.
        mean_s = A_sg @ m_g + A_ss @ x_s + c_s
        chol_Q_ss = jla.cholesky(Q_ss + jnp.eye(n_s) * 1e-6, lower=True)

        if diffusion_dist == "student_t":
            df = jnp.maximum(proc_df, 2.1)
            key_z, key_chi2 = random.split(key)
            z = random.normal(key_z, (n_s,))
            chi2_sample = random.gamma(key_chi2, df / 2.0) * 2.0
            scale = jnp.sqrt((df - 2.0) / chi2_sample)
            x_s_new = mean_s + chol_Q_ss @ (z * scale)
        else:
            x_s_new = mean_s + chol_Q_ss @ random.normal(key, (n_s,))

        # --- Step 3: Linearized update on G-block ---
        # Condition on observation to keep covariances bounded.
        # Absorb the S-block contribution into the intercept so the
        # effective observation model for x_g is:
        #   y = H_g @ x_g + d_eff + noise,  d_eff = d + H_s @ x_s_new
        # For Gaussian obs this is equivalent to subtracting H_s @ x_s from y.
        # For non-Gaussian obs (Poisson, Gamma) it is essential: the link
        # function is nonlinear, so we cannot subtract H_s @ x_s from y —
        # we must include it in the linear predictor that enters exp().
        d_eff = d + H_s @ x_s_new

        if manifest_dist == "gaussian":
            from causal_ssm_agent.models.likelihoods.rao_blackwell import (
                _kalman_update_gaussian,
            )

            m_upd_g, P_upd_g, _ = _kalman_update_gaussian(
                m_pred_g, P_pred_g, H_g, R, d_eff, y_t, mask_t
            )
        else:
            m_upd_g, P_upd_g = _linearized_update(
                m_pred_g, P_pred_g, y_t, H_g, d_eff, mask_t, manifest_dist, obs_params
            )

        return BlockRBState(
            g_mean=m_upd_g,
            g_cov=P_upd_g,
            g_pred_mean=m_pred_g,
            g_pred_cov=P_pred_g,
            s_sample=x_s_new,
        )

    def log_potential(_state_prev, state, model_inputs):
        """Compute log observation weight.

        Integrates p(y | x_g, x_s) over x_g via quadrature, fixing x_s at
        the particle's sampled value. This is the marginal likelihood
        contribution from this timestep.
        """
        y_t = model_inputs["observation"]
        mask_t = model_inputs["obs_mask"]
        x_s = state.s_sample
        mask_float = mask_t.astype(jnp.float32)
        n_observed = jnp.sum(mask_float)

        if manifest_dist == "gaussian":
            # Exact: y ~ N(H_g @ x_g + H_s @ x_s + d, R)
            # Marginalize x_g ~ N(m_pred_g, P_pred_g):
            # y | x_s ~ N(H_g @ m_pred_g + H_s @ x_s + d, H_g @ P_pred_g @ H_g' + R)
            y_pred = H_g @ state.g_pred_mean + H_s @ x_s + d
            S = H_g @ state.g_pred_cov @ H_g.T + R

            n_m = H.shape[0]
            large_var = 1e10
            S = 0.5 * (S + S.T) + jnp.eye(n_m) * 1e-8
            S = S + jnp.diag((1.0 - mask_float) * large_var)

            v = (y_t - y_pred) * mask_float
            _, logdet = jnp.linalg.slogdet(S)
            n_missing = n_m - n_observed
            logdet = logdet - n_missing * jnp.log(large_var)
            mahal = v @ jnp.linalg.solve(S, v)
            log_w = -0.5 * (n_observed * jnp.log(2 * jnp.pi) + logdet + mahal)
            return jnp.where(n_observed > 0, log_w, 0.0)
        else:
            # Non-Gaussian obs: quadrature over G-block, fixing S-block.
            # Generate sigma points for G-block.
            if quadrature == "unscented":
                sigma_pts_g, sigma_wts = _unscented_sigma_points(
                    state.g_pred_mean, state.g_pred_cov
                )
            else:
                from causal_ssm_agent.models.likelihoods.rao_blackwell import (
                    _multivariate_gauss_hermite,
                )

                gh_nodes, gh_wts = _multivariate_gauss_hermite(n_quadrature, n_g)
                L = jla.cholesky(state.g_pred_cov + jnp.eye(n_g) * 1e-8, lower=True)
                sigma_pts_g = state.g_pred_mean[None, :] + gh_nodes @ L.T
                sigma_wts = gh_wts

            # For each sigma point, reconstruct full latent state and evaluate obs LL.
            emission_fn = get_emission_fn(manifest_dist, obs_params)

            def eval_one(x_g):
                # Reconstruct full state: place G and S at their indices.
                full_x = jnp.zeros(n_latent)
                full_x = full_x.at[g_idx].set(x_g)
                full_x = full_x.at[s_idx].set(x_s)
                return emission_fn(y_t, full_x, H, d, R, mask_float)

            log_obs_vals = jax.vmap(eval_one)(sigma_pts_g)
            log_weights = jnp.log(jnp.maximum(sigma_wts, 1e-30))
            log_integral = jax.nn.logsumexp(log_obs_vals + log_weights)

            return jnp.where(n_observed > 0, log_integral, 0.0)

    return init_sample, propagate_sample, log_potential
