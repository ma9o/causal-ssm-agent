"""Rao-Blackwell particle filter callbacks for Gaussian dynamics.

When the latent dynamics are linear-Gaussian (diffusion_dist == "gaussian"),
the Kalman filter can analytically marginalize the latent state inside each
particle. Particles are only needed for the non-Gaussian observation model.
This gives strictly lower variance than the bootstrap PF at no extra cost.

Architecture:
- Each particle carries an RBState (Kalman sufficient statistics) instead of
  a point sample.
- init_sample returns RBState(m0, P0, m0, P0) — no sampling needed.
- propagate_sample runs a Kalman predict step (deterministic, no noise sampling).
- log_potential evaluates log int p(y|x) N(x|m_pred, P_pred) dx via quadrature.

The linearized update (EKF-style) conditions the Kalman state on the
observation to keep covariances bounded across time.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

# =============================================================================
# RBState — Kalman sufficient statistics carried per particle
# =============================================================================


class RBState(NamedTuple):
    """Per-particle Kalman state for Rao-Blackwell PF.

    mean/cov: updated (posterior) moments after conditioning on y_{t-1}.
    pred_mean/pred_cov: predicted (prior) moments before observing y_t.
    """

    mean: jnp.ndarray  # (n_latent,) — updated mean
    cov: jnp.ndarray  # (n_latent, n_latent) — updated covariance
    pred_mean: jnp.ndarray  # (n_latent,) — predicted mean (for weighting)
    pred_cov: jnp.ndarray  # (n_latent, n_latent) — predicted covariance


# =============================================================================
# Quadrature utilities
# =============================================================================


def _gauss_hermite_1d(n_points: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Hermite nodes and weights for 1D integration against N(0,1).

    Returns nodes x_i and weights w_i such that:
        sum_i w_i * f(x_i) ≈ int f(x) N(x|0,1) dx

    Uses numpy for the static computation, then wraps as JAX arrays.
    """
    import numpy as np

    # numpy's hermgauss uses the physicist convention: weight = exp(-x^2)
    # We need the probabilist convention: weight = exp(-x^2/2) / sqrt(2*pi)
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_points)
    # Convert: x_phys → x_prob = x_phys * sqrt(2)
    nodes_np = nodes_np * np.sqrt(2.0)
    # Adjust weights: divide by sqrt(pi) to account for the change of variable
    # int f(x) N(x|0,1) dx = (1/sqrt(pi)) * int f(t*sqrt(2)) exp(-t^2) dt
    weights_np = weights_np / np.sqrt(np.pi)
    return jnp.array(nodes_np), jnp.array(weights_np)


def _multivariate_gauss_hermite(n_points: int, dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Tensor-product Gauss-Hermite quadrature for dim-dimensional N(0,I).

    Returns:
        nodes: (n_points^dim, dim) — quadrature nodes
        weights: (n_points^dim,) — quadrature weights
    """
    nodes_1d, weights_1d = _gauss_hermite_1d(n_points)

    # Build tensor product via meshgrid
    grids = jnp.meshgrid(*([nodes_1d] * dim), indexing="ij")
    nodes = jnp.stack([g.ravel() for g in grids], axis=-1)  # (n^d, d)

    weight_grids = jnp.meshgrid(*([weights_1d] * dim), indexing="ij")
    weights = weight_grids[0].ravel()
    for i in range(1, dim):
        weights = weights * weight_grids[i].ravel()

    return nodes, weights


def _unscented_sigma_points(mean: jnp.ndarray, cov: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unscented transform sigma points and weights.

    Generates 2n+1 sigma points for an n-dimensional Gaussian N(mean, cov).

    Uses standard parameters: alpha=1, beta=0, kappa=0 (equal weight scheme).

    Returns:
        points: (2n+1, n) — sigma points
        weights: (2n+1,) — associated weights (sum to 1)
    """
    n = mean.shape[0]
    lam = n  # lambda = alpha^2 * (n + kappa) - n, with alpha=1, kappa=n

    # Cholesky of scaled covariance
    scaled_cov = (n + lam) * cov
    L = jla.cholesky(scaled_cov + jnp.eye(n) * 1e-8, lower=True)  # (n, n)

    # Sigma points: mean, mean + L_i, mean - L_i
    points = jnp.zeros((2 * n + 1, n))
    points = points.at[0].set(mean)
    for i in range(n):
        points = points.at[1 + i].set(mean + L[:, i])
        points = points.at[1 + n + i].set(mean - L[:, i])

    # Weights
    w0 = lam / (n + lam)
    wi = 1.0 / (2.0 * (n + lam))
    weights = jnp.full(2 * n + 1, wi)
    weights = weights.at[0].set(w0)

    return points, weights


# =============================================================================
# Kalman sub-operations (pure JAX)
# =============================================================================


def _kalman_predict(
    m: jnp.ndarray,
    P: jnp.ndarray,
    F: jnp.ndarray,
    Q: jnp.ndarray,
    c: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Kalman predict step.

    m_pred = F @ m + c
    P_pred = F @ P @ F.T + Q
    """
    m_pred = F @ m + c
    P_pred = F @ P @ F.T + Q
    # Symmetrize for numerical stability
    P_pred = 0.5 * (P_pred + P_pred.T)
    return m_pred, P_pred


def _kalman_update_gaussian(
    m: jnp.ndarray,
    P: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    d: jnp.ndarray,
    y: jnp.ndarray,
    obs_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Kalman update with Gaussian observations. Returns log marginal likelihood.

    y ~ N(H @ x + d, R)

    Handles missing data via variance inflation on unobserved channels.
    Returns updated (m, P) and the log marginal likelihood of y.
    """
    n_manifest = H.shape[0]
    large_var = 1e10
    mask_float = obs_mask.astype(jnp.float32)

    # Inflate R for missing observations
    R_adj = R + jnp.diag((1.0 - mask_float) * large_var)
    R_adj = 0.5 * (R_adj + R_adj.T) + jnp.eye(n_manifest) * 1e-8

    # Innovation
    y_pred = H @ m + d
    v = (y - y_pred) * mask_float  # zero out missing

    # Innovation covariance S = H P H' + R_adj
    S = H @ P @ H.T + R_adj
    S = 0.5 * (S + S.T) + jnp.eye(n_manifest) * 1e-8

    # Kalman gain K = P H' S^{-1}
    K = P @ H.T @ jnp.linalg.inv(S)

    # Update
    m_upd = m + K @ v
    P_upd = P - K @ S @ K.T
    P_upd = 0.5 * (P_upd + P_upd.T)

    # Log marginal likelihood: log N(v | 0, S)
    n_observed = jnp.sum(mask_float)
    _sign, logdet = jnp.linalg.slogdet(S)
    # Subtract out the contribution of inflated missing dimensions
    n_missing = n_manifest - n_observed
    logdet = logdet - n_missing * jnp.log(large_var)

    mahal = v @ jnp.linalg.solve(S, v)
    log_marg = -0.5 * (n_observed * jnp.log(2 * jnp.pi) + logdet + mahal)
    log_marg = jnp.where(n_observed > 0, log_marg, 0.0)

    return m_upd, P_upd, log_marg


def _linearized_update(
    m: jnp.ndarray,
    P: jnp.ndarray,
    y: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    obs_mask: jnp.ndarray,
    manifest_dist: str,
    params: dict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """EKF-style linearized update for non-Gaussian observations.

    Linearizes the observation model around the predicted mean to compute
    approximate posterior moments. This keeps covariances bounded without
    affecting the particle weights (weights use quadrature, not this update).
    """
    n = m.shape[0]
    n_manifest = H.shape[0]
    mask_float = obs_mask.astype(jnp.float32)

    # Compute the predicted observation and pseudo-Jacobian
    eta_pred = H @ m + d  # predicted linear predictor

    if manifest_dist == "poisson":
        # g(eta) = exp(eta), so g'(eta) = exp(eta)
        rate = jnp.exp(eta_pred)
        # Pseudo-observation noise: Var[y|eta] = rate (Poisson variance)
        R_pseudo = jnp.diag(rate + 1e-8)
        # Pseudo-innovation: y - E[y|eta] = y - rate
        v = (y - rate) * mask_float
    elif manifest_dist == "student_t":
        # Location-scale: y ~ t(df, eta, scale)
        # Linearization: same as Gaussian but with heavier tails
        manifest_cov = params.get("manifest_cov", jnp.eye(n_manifest) * 0.1)
        R_pseudo = manifest_cov + jnp.eye(n_manifest) * 1e-8
        v = (y - eta_pred) * mask_float
    elif manifest_dist == "gamma":
        # g(eta) = exp(eta), mean = exp(eta)
        mean_pred = jnp.exp(eta_pred)
        shape = params.get("obs_shape", 1.0)
        # Var[y|eta] = mean^2 / shape (Gamma variance)
        R_pseudo = jnp.diag(mean_pred**2 / (shape + 1e-8) + 1e-8)
        v = (y - mean_pred) * mask_float
    else:
        # Fallback: Gaussian
        manifest_cov = params.get("manifest_cov", jnp.eye(n_manifest) * 0.1)
        R_pseudo = manifest_cov + jnp.eye(n_manifest) * 1e-8
        v = (y - eta_pred) * mask_float

    # Inflate for missing
    large_var = 1e10
    R_pseudo = R_pseudo + jnp.diag((1.0 - mask_float) * large_var)

    # Standard Kalman update with pseudo-observation model
    S = H @ P @ H.T + R_pseudo
    S = 0.5 * (S + S.T) + jnp.eye(n_manifest) * 1e-8
    K = P @ H.T @ jnp.linalg.inv(S)

    m_upd = m + K @ v
    P_upd = P - K @ S @ K.T
    P_upd = 0.5 * (P_upd + P_upd.T)

    # Ensure P stays positive definite
    P_upd = P_upd + jnp.eye(n) * 1e-8

    return m_upd, P_upd


# =============================================================================
# Observation weight functions
# =============================================================================


def _obs_weight_gaussian(
    y: jnp.ndarray,
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    d: jnp.ndarray,
    obs_mask: jnp.ndarray,
) -> float:
    """Exact log marginal for Gaussian observations: log N(y | H m + d, H P H' + R)."""
    n_manifest = H.shape[0]
    mask_float = obs_mask.astype(jnp.float32)

    y_pred = H @ pred_mean + d
    v = (y - y_pred) * mask_float

    S = H @ pred_cov @ H.T + R
    S = 0.5 * (S + S.T) + jnp.eye(n_manifest) * 1e-8

    # Inflate for missing
    large_var = 1e10
    S = S + jnp.diag((1.0 - mask_float) * large_var)

    n_observed = jnp.sum(mask_float)
    _, logdet = jnp.linalg.slogdet(S)
    n_missing = n_manifest - n_observed
    logdet = logdet - n_missing * jnp.log(large_var)

    mahal = v @ jnp.linalg.solve(S, v)
    log_w = -0.5 * (n_observed * jnp.log(2 * jnp.pi) + logdet + mahal)
    return jnp.where(n_observed > 0, log_w, 0.0)


def _obs_weight_quadrature(
    y: jnp.ndarray,
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    obs_mask: jnp.ndarray,
    manifest_dist: str,
    params: dict,
    quadrature: str = "unscented",
    n_quadrature: int = 5,
) -> float:
    """Compute log int p(y|x) N(x|pred_mean, pred_cov) dx via sigma-point quadrature.

    Transforms sigma points from latent space to observation space, evaluates
    the observation log-likelihood at each point, and averages.
    """
    n_latent = pred_mean.shape[0]
    mask_float = obs_mask.astype(jnp.float32)
    n_observed = jnp.sum(mask_float)

    if quadrature == "unscented":
        # Sigma points in latent space
        sigma_pts, sigma_wts = _unscented_sigma_points(pred_mean, pred_cov)
    else:
        # Gauss-Hermite in latent space
        gh_nodes, gh_wts = _multivariate_gauss_hermite(n_quadrature, n_latent)
        # Transform from N(0,I) to N(pred_mean, pred_cov)
        L = jla.cholesky(pred_cov + jnp.eye(n_latent) * 1e-8, lower=True)
        sigma_pts = pred_mean[None, :] + gh_nodes @ L.T  # (K, n_latent)
        sigma_wts = gh_wts

    # Evaluate observation log-likelihood at each sigma point
    def _log_obs_single(x):
        """Log p(y|x) for a single latent state x."""
        eta = H @ x + d  # linear predictor

        if manifest_dist == "poisson":
            rate = jnp.exp(eta)
            log_probs = jax.scipy.stats.poisson.logpmf(y, rate)
        elif manifest_dist == "student_t":
            manifest_cov = params.get("manifest_cov", jnp.eye(y.shape[0]) * 0.1)
            scale = jnp.sqrt(jnp.diag(manifest_cov))
            df = params.get("obs_df", 5.0)
            log_probs = jax.scipy.stats.t.logpdf(y, df, loc=eta, scale=scale)
        elif manifest_dist == "gamma":
            mean = jnp.exp(eta)
            shape = params.get("obs_shape", 1.0)
            scale = mean / shape
            log_probs = jax.scipy.stats.gamma.logpdf(y, shape, scale=scale)
        else:
            # Gaussian fallback
            manifest_cov = params.get("manifest_cov", jnp.eye(y.shape[0]) * 0.1)
            log_probs = jax.scipy.stats.norm.logpdf(
                y, loc=eta, scale=jnp.sqrt(jnp.diag(manifest_cov))
            )

        # Mask missing observations
        return jnp.sum(jnp.where(mask_float > 0.5, log_probs, 0.0))

    # Vectorize over sigma points
    log_obs_vals = jax.vmap(_log_obs_single)(sigma_pts)  # (K,)

    # Log-sum-exp with weights: log(sum_i w_i * exp(log_obs_i))
    # = logsumexp(log_obs_i + log(w_i))
    log_weights = jnp.log(jnp.maximum(sigma_wts, 1e-30))
    log_integral = jax.nn.logsumexp(log_obs_vals + log_weights)

    return jnp.where(n_observed > 0, log_integral, 0.0)


# =============================================================================
# Callback factory — public API
# =============================================================================


def make_rb_callbacks(
    n_latent: int,  # noqa: ARG001
    n_manifest: int,  # noqa: ARG001
    manifest_dist: str,
    params: dict,
    extra_params: dict,
    m0: jnp.ndarray,
    P0: jnp.ndarray,
    quadrature: str = "unscented",
    n_quadrature: int = 5,
):
    """Build Feynman-Kac callbacks for Rao-Blackwell particle filter.

    Returns (init_sample, propagate_sample, log_potential) closures compatible
    with cuthbert's build_filter. The particle state is an RBState pytree
    (cuthbert supports arbitrary pytree states).

    Args:
        n_latent: Latent state dimension.
        n_manifest: Observation dimension.
        manifest_dist: Observation distribution ("gaussian", "poisson", "student_t", "gamma").
        params: Dict with lambda_mat, manifest_means, manifest_cov, etc.
        extra_params: Dict with obs_df, obs_shape, etc.
        m0: Initial state mean (n_latent,).
        P0: Initial state covariance (n_latent, n_latent).
        quadrature: "unscented" or "gauss_hermite".
        n_quadrature: Number of quadrature points per dimension (for GH).

    Returns:
        (init_sample, propagate_sample, log_potential) — callback triple.
    """
    H = params["lambda_mat"]  # (n_manifest, n_latent)
    d = params["manifest_means"]  # (n_manifest,)
    R = params["manifest_cov"]  # (n_manifest, n_manifest)

    # Merge extra_params into params for observation model
    obs_params = {**params, **extra_params}

    def init_sample(_key, _model_inputs):
        """Initialize particle as Kalman sufficient statistics (no sampling)."""
        return RBState(
            mean=m0,
            cov=P0,
            pred_mean=m0,
            pred_cov=P0,
        )

    def propagate_sample(_key, state, model_inputs):
        """Kalman predict + linearized update (deterministic, no noise sampling).

        1. Predict: propagate Kalman state through linear dynamics.
        2. Linearized update: condition on current observation (EKF-style)
           to keep covariances bounded. This does NOT affect weights.
        """
        F_t = model_inputs["Ad"]  # (n, n)
        Q_t = model_inputs["Qd"]  # (n, n)
        c_t = model_inputs["cd"]  # (n,)
        y_t = model_inputs["observation"]  # (n_manifest,)
        mask_t = model_inputs["obs_mask"]  # (n_manifest,)

        # Kalman predict
        m_pred, P_pred = _kalman_predict(state.mean, state.cov, F_t, Q_t, c_t)

        # Linearized update to condition on observation
        if manifest_dist == "gaussian":
            m_upd, P_upd, _ = _kalman_update_gaussian(m_pred, P_pred, H, R, d, y_t, mask_t)
        else:
            m_upd, P_upd = _linearized_update(
                m_pred, P_pred, y_t, H, d, mask_t, manifest_dist, obs_params
            )

        return RBState(
            mean=m_upd,
            cov=P_upd,
            pred_mean=m_pred,
            pred_cov=P_pred,
        )

    def log_potential(_state_prev, state, model_inputs):
        """Compute log observation weight via analytical/quadrature integration.

        Uses the PREDICTED (prior) moments to compute:
            log int p(y|x) N(x | m_pred, P_pred) dx
        """
        y_t = model_inputs["observation"]
        mask_t = model_inputs["obs_mask"]

        if manifest_dist == "gaussian":
            return _obs_weight_gaussian(y_t, state.pred_mean, state.pred_cov, H, R, d, mask_t)
        else:
            return _obs_weight_quadrature(
                y_t,
                state.pred_mean,
                state.pred_cov,
                H,
                d,
                mask_t,
                manifest_dist,
                obs_params,
                quadrature=quadrature,
                n_quadrature=n_quadrature,
            )

    return init_sample, propagate_sample, log_potential
