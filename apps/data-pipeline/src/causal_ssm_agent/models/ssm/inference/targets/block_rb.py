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

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np

from causal_ssm_agent.models.ssm.covariance_utils import (
    inflate_missing_variance,
    symmetrize_with_jitter,
)
from causal_ssm_agent.models.ssm.inference.targets.base import CHOL_JITTER, MISSING_DATA_LARGE_VAR
from causal_ssm_agent.models.ssm.inference.targets.rao_blackwell import (
    _kalman_predict,
    _kalman_update_gaussian,
    _linearized_update,
    _unscented_sigma_points,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.inference.targets.kernels import (
        ObservationKernel,
        TransitionKernel,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Partition latent variable indices into Gaussian and sampled blocks.

    Args:
        diffusion_dists: Per-variable diffusion distribution, e.g.
            ["gaussian", "student_t", "gaussian"].

    Returns:
        g_idx: Integer array of Gaussian variable indices.
        s_idx: Integer array of sampled (non-Gaussian) variable indices.
    """
    g_list: list[int] = []
    s_list: list[int] = []
    for i, d in enumerate(diffusion_dists):
        if d == "gaussian":
            g_list.append(i)
        else:
            s_list.append(i)
    # Return as numpy arrays to avoid JIT tracing issues when called at
    # construction time inside a traced function.
    return np.array(g_list, dtype=np.int32), np.array(s_list, dtype=np.int32)


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
    g_idx: jnp.ndarray | np.ndarray,
    s_idx: jnp.ndarray | np.ndarray,
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
    params: dict,
    m0: jnp.ndarray,
    P0: jnp.ndarray,
    g_idx: jnp.ndarray | np.ndarray,
    s_idx: jnp.ndarray | np.ndarray,
    obs_kernel: ObservationKernel,
    trans_kernel: TransitionKernel,
    quadrature: str = "unscented",
    n_quadrature: int = 5,
):
    """Build Feynman-Kac callbacks for block Rao-Blackwell particle filter.

    Partitions the state space into Gaussian (Kalman-marginalized) and sampled
    (bootstrap PF) blocks. Returns callbacks compatible with cuthbert's
    build_filter.

    Args:
        n_latent: Total latent state dimension.
        params: Dict with lambda_mat, manifest_means, manifest_cov, etc.
        m0: Initial state mean (n_latent,).
        P0: Initial state covariance (n_latent, n_latent).
        g_idx: Indices of Gaussian (Kalman-marginalized) variables.
        s_idx: Indices of sampled (bootstrap PF) variables.
        obs_kernel: Pre-resolved observation model.
        trans_kernel: Pre-resolved transition noise model (for sampled block).
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

    def init_sample(key, model_inputs):  # noqa: ARG001
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
        Ad_t = model_inputs["Ad"]
        Qd_t = model_inputs["Qd"]
        cd_t = model_inputs["cd"]
        y_t = model_inputs["observation"]
        mask_t = model_inputs["obs_mask"]

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
        c_g_eff = c_g + A_gs @ x_s
        m_pred_g, P_pred_g = _kalman_predict(m_g, state.g_cov, A_gg, Q_gg, c_g_eff)

        # --- Step 2: Sample S-block using transition kernel ---
        mean_s = A_sg @ m_g + A_ss @ x_s + c_s
        chol_Q_ss = jla.cholesky(Q_ss + jnp.eye(n_s) * 1e-6, lower=True)
        x_s_new = mean_s + trans_kernel.sample_noise_fn(key, chol_Q_ss)

        # --- Step 3: Linearized update on G-block ---
        # Absorb S-block contribution into effective intercept.
        d_eff = d + H_s @ x_s_new

        if obs_kernel.is_gaussian:
            m_upd_g, P_upd_g, _ = _kalman_update_gaussian(
                m_pred_g, P_pred_g, H_g, R, d_eff, y_t, mask_t
            )
        else:
            m_upd_g, P_upd_g = _linearized_update(
                m_pred_g, P_pred_g, y_t, H_g, d_eff, mask_t, obs_kernel
            )

        return BlockRBState(
            g_mean=m_upd_g,
            g_cov=P_upd_g,
            g_pred_mean=m_pred_g,
            g_pred_cov=P_pred_g,
            s_sample=x_s_new,
        )

    def log_potential(state_prev, state, model_inputs):  # noqa: ARG001
        """Compute log observation weight.

        Integrates p(y | x_g, x_s) over x_g via quadrature, fixing x_s at
        the particle's sampled value.
        """
        y_t = model_inputs["observation"]
        mask_t = model_inputs["obs_mask"]
        x_s = state.s_sample
        mask_float = mask_t.astype(jnp.float64)
        n_observed = jnp.sum(mask_float)

        if obs_kernel.is_gaussian:
            y_pred = H_g @ state.g_pred_mean + H_s @ x_s + d
            S = H_g @ state.g_pred_cov @ H_g.T + R

            S = symmetrize_with_jitter(S)
            S = inflate_missing_variance(S, mask_float)

            v = (y_t - y_pred) * mask_float
            _, logdet = jnp.linalg.slogdet(S)
            n_missing = H.shape[0] - n_observed
            logdet = logdet - n_missing * jnp.log(MISSING_DATA_LARGE_VAR)
            mahal = v @ jnp.linalg.solve(S, v)
            log_w = -0.5 * (n_observed * jnp.log(2 * jnp.pi) + logdet + mahal)
            return jnp.where(n_observed > 0, log_w, 0.0)
        if quadrature == "unscented":
            sigma_pts_g, sigma_wts = _unscented_sigma_points(state.g_pred_mean, state.g_pred_cov)
        else:
            from causal_ssm_agent.models.ssm.inference.targets.rao_blackwell import (
                _multivariate_gauss_hermite,
            )

            gh_nodes, gh_wts = _multivariate_gauss_hermite(n_quadrature, n_g)
            L = jla.cholesky(state.g_pred_cov + jnp.eye(n_g) * CHOL_JITTER, lower=True)
            sigma_pts_g = state.g_pred_mean[None, :] + gh_nodes @ L.T
            sigma_wts = gh_wts

        def eval_one(x_g):
            full_x = jnp.zeros(n_latent)
            full_x = full_x.at[g_idx].set(x_g)
            full_x = full_x.at[s_idx].set(x_s)
            return obs_kernel.emission_fn(y_t, full_x, H, d, R, mask_float)

        log_obs_vals = jax.vmap(eval_one)(sigma_pts_g)
        log_weights = jnp.log(jnp.maximum(sigma_wts, 1e-30))
        log_integral = jax.nn.logsumexp(log_obs_vals + log_weights)

        return jnp.where(n_observed > 0, log_integral, 0.0)

    return init_sample, propagate_sample, log_potential
