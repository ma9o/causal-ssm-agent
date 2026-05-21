"""Rung-3 abduction: recover the latent state at the evidence boundary.

The smoother runs on posterior-mean discretised parameters; the recovered
state at ``evidence_end_idx`` becomes the initial condition for the
counterfactual forward simulation. This is conceptually separate from
the forward simulator (Diffrax) — it consumes observations rather than
producing trajectories — so it lives next to the simulation modules but
does not share their code path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jax import vmap

from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.discretization import (
    discretize_system_with_inputs_batched,
)

if TYPE_CHECKING:
    from cuthbert.gaussian.types import LinearizedKalmanFilterState
    from cuthbertlib.linearize.moments import MeanAndCholCovFunc
    from cuthbertlib.types import ArrayTreeLike
    from jax import Array
    from jax.typing import ArrayLike

logger = get_prefect_logger(__name__)


def _kalman_filter_smoother_per_step_linearised(
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    observations: jnp.ndarray,
    H: jnp.ndarray,
    d_meas: jnp.ndarray,
    R: jnp.ndarray,
    *,
    jitter: float = 1e-6,
    obs_kernel: Any = None,
) -> jnp.ndarray:
    """Forward Kalman filter + RTS smoother on a per-step linearised LGSSM.

    Returns smoothed state means of shape ``(T, n_latent)``. Used by the
    composite EKS / IEKS abduction below.

    Two paths:

    - **Gaussian observation** (``obs_kernel is None`` or
      ``obs_kernel.is_gaussian``): delegates to :func:`_kalman_smooth_states`
      which runs cuthbert's square-root filter + smoother. Standard
      linear Kalman recursion.
    - **EKF observation** (non-Gaussian ``obs_kernel``): per-step
      linearisation of the response function — ``H_eff = ∂response/∂η · H``
      and ``R_eff = variance_fn(μ_pred)``. This branch stays hand-rolled
      because the EKF observation update depends on the *predicted mean*
      at each step, which is the algorithm's intrinsic shape; cuthbert's
      filter expects a fixed observation operator.
    """
    use_ekf = obs_kernel is not None and not obs_kernel.is_gaussian
    if not use_ekf:
        return _kalman_smooth_states(
            observations=observations,
            Ad=Ad,
            Qd=Qd,
            cd=cd,
            H=H,
            d=d_meas,
            R=R,
            init_mean=init_mean,
            init_cov=init_cov,
        )

    import jax

    n_latent = Ad.shape[1]
    eye = jnp.eye(n_latent)

    def _forward_step(carry, inputs):
        m, P = carry
        Ad_t, Qd_t, cd_t, y_t = inputs
        m_pred = Ad_t @ m + cd_t
        P_pred = Ad_t @ P @ Ad_t.T + Qd_t
        P_pred = 0.5 * (P_pred + P_pred.T) + jitter * eye
        eta_pred = H @ m_pred + d_meas
        response_fn = obs_kernel.response_fn
        mu_pred = response_fn(eta_pred)
        J_resp = jax.jacfwd(response_fn)(eta_pred)
        H_eff = J_resp @ H
        R_eff = jnp.asarray(obs_kernel.variance_fn(mu_pred))
        if R_eff.ndim == 1:
            R_eff = jnp.diag(R_eff)
        y_resid = y_t - mu_pred
        S = H_eff @ P_pred @ H_eff.T + R_eff
        S = 0.5 * (S + S.T) + jitter * jnp.eye(S.shape[0])
        chol_S = jnp.linalg.cholesky(S)
        K = P_pred @ H_eff.T @ jax.scipy.linalg.cho_solve(
            (chol_S, True), jnp.eye(H_eff.shape[0])
        )
        m_new = m_pred + K @ y_resid
        P_new = P_pred - K @ H_eff @ P_pred
        P_new = 0.5 * (P_new + P_new.T)
        return (m_new, P_new), (m_new, P_new, m_pred, P_pred)

    init_carry = (init_mean, init_cov)
    _, (filt_means, filt_covs, pred_means, pred_covs) = jax.lax.scan(
        _forward_step, init_carry, (Ad, Qd, cd, observations)
    )

    def _backward_step(carry, inputs):
        sm_mean, sm_cov = carry
        filt_m, filt_P, pred_m, pred_P, Ad_next = inputs
        chol_pred = jnp.linalg.cholesky(pred_P + jitter * eye)
        G_T = jax.scipy.linalg.cho_solve((chol_pred, True), Ad_next @ filt_P)
        G = G_T.T
        sm_mean_new = filt_m + G @ (sm_mean - pred_m)
        sm_cov_new = filt_P + G @ (sm_cov - pred_P) @ G.T
        sm_cov_new = 0.5 * (sm_cov_new + sm_cov_new.T)
        return (sm_mean_new, sm_cov_new), (sm_mean_new, sm_cov_new)

    init_sm = (filt_means[-1], filt_covs[-1])
    backward_inputs = (
        filt_means[:-1],
        filt_covs[:-1],
        pred_means[1:],
        pred_covs[1:],
        Ad[1:],
    )
    _, (sm_means_rev, _sm_covs_rev) = jax.lax.scan(
        _backward_step, init_sm, backward_inputs, reverse=True
    )
    return jnp.concatenate(
        [sm_means_rev, filt_means[-1:].reshape(1, -1)], axis=0
    )


def approximate_abducted_state_composite_eks(
    canonical: Any,
    param_samples: list,
    runtime_times: jnp.ndarray,
    observations: jnp.ndarray,
    evidence_end_idx: int,
    *,
    x_lin: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """EKS-based rung-3 abduction for composite specs.

    For each posterior parameter draw:

    1. Linearise the composite vector field at ``x_lin`` (default:
       ``canonical.init_mean`` broadcast over time).
    2. Run a forward Kalman filter on the linearised LGSSM with the
       actual observations.
    3. Run an RTS smoother backward.
    4. Take the smoothed mean at ``evidence_end_idx``.

    Return the average across parameter draws — gives a *deterministic
    conditional mean* per parameter draw, in contrast to
    :func:`approximate_abducted_state_composite` which averages
    composite MCMC trajectory samples (one stochastic draw per draw).
    Statistically cleaner when the MCMC has not fully converged.

    Args:
        canonical: ``RuntimeSSM`` carrying observation operator
            (``H``, ``d_meas``, ``R``), init distribution, diffusion,
            and the composite vector field.
        param_samples: Per-iteration param tuples (typically
            ``fit_result.diagnostics["param_samples"]``).
        runtime_times: ``(T,)`` observation times — needed for
            per-step linearisation.
        observations: ``(T, n_m)`` actual observations.
        evidence_end_idx: Index along ``T`` to read the abducted state from.
        x_lin: Optional ``(T, n_latent)`` linearisation trajectory.
            Defaults to ``canonical.init_mean`` broadcast.

    Returns:
        Same dict shape as :func:`approximate_abducted_state`:
        ``{"state": Array, "method": "composite_eks", "warning": None}``.
    """
    from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.composite_kalman import (
        composite_latent_context_at_trajectory,
    )

    if not param_samples:
        raise ValueError("approximate_abducted_state_composite_eks needs non-empty param_samples.")

    n_latent = canonical.init_mean.shape[0]
    T = observations.shape[0]
    if x_lin is None:
        x_lin = jnp.broadcast_to(canonical.init_mean, (T, n_latent))

    per_draw_smoothed: list[jnp.ndarray] = []
    for params in param_samples:
        ctx = composite_latent_context_at_trajectory(
            vector_field=canonical.vector_field,
            vf_params=params,
            x_traj=x_lin,
            init_mean=canonical.init_mean,
            init_cov=canonical.init_cov,
            diffusion_cov=canonical.diffusion_cov,
            runtime_times=runtime_times,
            H=canonical.H,
            d_meas=canonical.d_meas,
            R=canonical.R,
        )
        sm_means = _kalman_filter_smoother_per_step_linearised(
            ctx.Ad, ctx.Qd, ctx.cd, ctx.init_mean, ctx.init_cov,
            observations, ctx.H, ctx.d_meas, ctx.R,
            obs_kernel=canonical.obs_kernel,
        )
        per_draw_smoothed.append(sm_means[evidence_end_idx])

    stacked = jnp.stack(per_draw_smoothed, axis=0)
    return {
        "state": jnp.mean(stacked, axis=0),
        "method": "composite_eks",
        "warning": None,
    }


def approximate_abducted_state_composite_ieks(
    canonical: Any,
    param_samples: list,
    runtime_times: jnp.ndarray,
    observations: jnp.ndarray,
    evidence_end_idx: int,
    *,
    n_iters: int = 3,
    tol: float = 1e-3,
    x_lin: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """IEKS (Iterated Extended Kalman Smoother) rung-3 abduction.

    Per parameter draw:

    1. Initialise ``x_lin`` (default: ``canonical.init_mean`` broadcast).
    2. Linearise composite VF at ``x_lin`` → per-step ``(Ad, Qd, cd)``.
    3. Run forward Kalman filter + RTS smoother on linearised LGSSM.
    4. Compute new ``x_lin = smoothed_means``.
    5. Stop if ``||x_lin_new − x_lin|| < tol`` (per draw), else go to 2.
    6. Use the converged smoothed mean at ``evidence_end_idx``.

    Average across parameter draws.

    Quality upgrade over the single-pass EKS — the iterative re-
    linearisation finds the true fixed point of the smoothed-trajectory
    estimator. For highly non-linear systems (Hill saturation, etc.)
    the single-pass EKS at a fixed ``x_lin = init_mean`` can be quite
    biased; IEKS finds the consistent smoothed trajectory.

    Args:
        canonical: ``RuntimeSSM`` carrying observation operator,
            init distribution, diffusion, and composite vector field.
        param_samples: Per-iteration param tuples.
        runtime_times: ``(T,)`` observation times.
        observations: ``(T, n_m)`` actual observations.
        evidence_end_idx: Index along ``T`` to read the abducted state from.
        n_iters: Max IEKS iterations per draw (3 typically converges).
        tol: L-infinity convergence threshold on ``||x_lin_new − x_lin||``.
        x_lin: Optional initial linearisation trajectory.

    Returns the same dict shape as the other abduction estimators
    (``state``, ``method="composite_ieks"``, ``warning``).
    """
    from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.composite_kalman import (
        composite_latent_context_at_trajectory,
    )

    if not param_samples:
        raise ValueError("approximate_abducted_state_composite_ieks needs non-empty param_samples.")

    n_latent = canonical.init_mean.shape[0]
    T = observations.shape[0]
    if x_lin is None:
        x_lin_init = jnp.broadcast_to(canonical.init_mean, (T, n_latent))
    else:
        x_lin_init = x_lin

    per_draw_smoothed: list[jnp.ndarray] = []
    per_draw_iters: list[int] = []
    for params in param_samples:
        x_lin_iter = x_lin_init
        n_iter_used = n_iters
        sm_means_final = x_lin_iter
        for it in range(n_iters):
            ctx = composite_latent_context_at_trajectory(
                vector_field=canonical.vector_field,
                vf_params=params,
                x_traj=x_lin_iter,
                init_mean=canonical.init_mean,
                init_cov=canonical.init_cov,
                diffusion_cov=canonical.diffusion_cov,
                runtime_times=runtime_times,
                H=canonical.H,
                d_meas=canonical.d_meas,
                R=canonical.R,
            )
            sm_means = _kalman_filter_smoother_per_step_linearised(
                ctx.Ad, ctx.Qd, ctx.cd, ctx.init_mean, ctx.init_cov,
                observations, ctx.H, ctx.d_meas, ctx.R,
                obs_kernel=canonical.obs_kernel,
            )
            delta = float(jnp.max(jnp.abs(sm_means - x_lin_iter)))
            x_lin_iter = sm_means
            sm_means_final = sm_means
            if delta < tol:
                n_iter_used = it + 1
                break
        per_draw_smoothed.append(sm_means_final[evidence_end_idx])
        per_draw_iters.append(n_iter_used)

    stacked = jnp.stack(per_draw_smoothed, axis=0)
    return {
        "state": jnp.mean(stacked, axis=0),
        "method": "composite_ieks",
        "warning": None,
        "n_iters_per_draw": per_draw_iters,
    }


def approximate_abducted_state_composite(
    composite_result: Any,
    evidence_end_idx: int,
) -> dict[str, Any]:
    """Rung-3 abduction for the composite (non-linear) path.

    The composite MCMC's trajectory samples already represent draws from
    the smoothing posterior ``p(x_t | y_{1:T}, θ)`` — that's the
    auxiliary-Kalman MH's whole job. So the abducted state at the
    evidence boundary is simply the posterior-mean trajectory at
    ``evidence_end_idx``, averaged across MCMC iterations.

    Mirrors the return shape of :func:`approximate_abducted_state`:
    a dict with ``state`` (1-D vector), ``method`` (string),
    ``warning`` (None or human-readable note).

    This unblocks the Stage 6 ``simulate_counterfactual`` endpoint for
    composite-fitted artifacts — previously it returned a structured
    "not yet implemented" error.

    Args:
        composite_result: ``InferenceResult`` from ``fit_composite_aux_kalman``.
            Must have ``diagnostics["trajectory_samples"]`` of shape
            ``(n_draws, T, n_latent)``.
        evidence_end_idx: Index along ``T`` to take the abducted state from.
    """
    trajectory_samples = composite_result.diagnostics.get("trajectory_samples")
    if trajectory_samples is None:
        raise ValueError(
            "approximate_abducted_state_composite expects a composite "
            "InferenceResult with diagnostics['trajectory_samples'] populated."
        )
    T = trajectory_samples.shape[1]
    if evidence_end_idx < 0 or evidence_end_idx >= T:
        raise ValueError(
            f"evidence_end_idx={evidence_end_idx} out of bounds for "
            f"trajectory of length {T}."
        )
    abducted_mean = jnp.mean(trajectory_samples[:, evidence_end_idx, :], axis=0)
    return {
        "state": abducted_mean,
        "method": "composite_trajectory_marginal",
        "warning": None,
    }


def approximate_abducted_state(
    samples: dict[str, jnp.ndarray],
    ssm_model: Any,
    spec: Any,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    evidence_start_idx: int,
    evidence_end_idx: int,
) -> dict[str, Any]:
    """Approximate rung-3 abduction from observed history.

    Uses a Kalman smoother on posterior-mean parameters when available.
    Falls back to a least-squares inversion of the contemporaneous
    observation model at the evidence boundary.
    """
    from nof1_causal_lab.models.ssm.inference.utils import _assemble_single_deterministics

    posterior_means = {name: jnp.mean(value, axis=0) for name, value in samples.items()}
    det_values = _assemble_single_deterministics(posterior_means, spec)

    det_values["manifest_means"] = posterior_means.get(
        "manifest_means", spec.manifest_means_block.template
    )

    evidence_obs = observations[evidence_start_idx : evidence_end_idx + 1]
    evidence_times = times[evidence_start_idx : evidence_end_idx + 1]
    smoothed = _try_smoother(
        ssm_model,
        evidence_obs,
        evidence_times,
        posterior_means,
        det_values,
    )
    if smoothed is not None:
        return {
            "state": smoothed[-1],
            "method": "kalman_smoother",
            "warning": None,
        }

    lambda_mat = det_values.get("lambda")
    if lambda_mat is None:
        lambda_template = spec.lambda_block.template
        lambda_mat = lambda_template if isinstance(lambda_template, jnp.ndarray) else None
    if lambda_mat is None:
        return {
            "state": jnp.zeros(spec.n_latent),
            "method": "zero_state",
            "warning": "Could not reconstruct observation operator; using zero latent state.",
        }

    obs_t = observations[evidence_end_idx]
    obs_mask = ~jnp.isnan(obs_t)
    if not bool(jnp.any(obs_mask)):
        return {
            "state": jnp.zeros(spec.n_latent),
            "method": "zero_state",
            "warning": "Evidence boundary has no observed values; using zero latent state.",
        }

    manifest_means = det_values["manifest_means"]
    H_obs = lambda_mat[obs_mask]
    y_obs = obs_t[obs_mask] - manifest_means[obs_mask]
    state = jnp.linalg.pinv(H_obs) @ y_obs
    return {
        "state": state,
        "method": "observation_pseudoinverse",
        "warning": (
            "Kalman smoother unavailable; counterfactual state estimated from the final "
            "observed measurement slice."
        ),
    }


def _kalman_smooth_states(
    observations: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
) -> jnp.ndarray:
    """Kalman filter + RTS smoother for linear Gaussian SSM via cuthbert.

    Returns smoothed state means ``(T, D)``. Handles missing data (NaN) via
    variance inflation.
    """
    from cuthbert.filtering import filter as cuthbert_filter
    from cuthbert.gaussian.moments import build_filter, build_smoother
    from cuthbert.smoothing import smoother as cuthbert_smoother

    from nof1_causal_lab.models.ssm.inference.targets.base import preprocess_missing_data

    T, n_m = observations.shape
    n = Ad.shape[1]
    dtype = jnp.asarray(observations).dtype
    jitter_n = 1e-6 * jnp.eye(n, dtype=dtype)
    jitter_m = 1e-6 * jnp.eye(n_m, dtype=dtype)

    clean_obs, R_adjusted, _obs_mask = preprocess_missing_data(observations, R, None)

    chol_Qd = vmap(lambda Q: jnp.linalg.cholesky(Q.astype(dtype) + jitter_n))(Qd)
    chol_R = jnp.linalg.cholesky(R_adjusted.astype(dtype) + jitter_m)
    chol_P0 = jnp.linalg.cholesky(init_cov.astype(dtype) + jitter_n)

    H_arr = H.astype(dtype)
    d_arr = d.astype(dtype)

    def _prepend_init(steps: jnp.ndarray) -> jnp.ndarray:
        head = jnp.zeros((1, *steps.shape[1:]), dtype=dtype)
        return jnp.concatenate([head, steps], axis=0)

    model_inputs = {
        "m0": jnp.broadcast_to(init_mean.astype(dtype), (T + 1, n)),
        "chol_P0": jnp.broadcast_to(chol_P0, (T + 1, n, n)),
        "F": _prepend_init(Ad.astype(dtype)),
        "c": _prepend_init(cd.astype(dtype)),
        "chol_Q": _prepend_init(chol_Qd),
        "H": _prepend_init(jnp.broadcast_to(H_arr, (T, n_m, n))),
        "d": _prepend_init(jnp.broadcast_to(d_arr, (T, n_m))),
        "chol_R": _prepend_init(chol_R),
        "y": _prepend_init(clean_obs.astype(dtype)),
    }

    def get_init_params(model_inputs: ArrayTreeLike) -> tuple[Array, Array]:
        return model_inputs["m0"], model_inputs["chol_P0"]

    def get_dynamics_params(
        state: LinearizedKalmanFilterState, model_inputs: ArrayTreeLike
    ) -> tuple[MeanAndCholCovFunc, Array]:
        F_t, c_t, chol_Q_t = model_inputs["F"], model_inputs["c"], model_inputs["chol_Q"]

        def dynamics_fn(x: ArrayLike) -> tuple[Array, Array]:
            return F_t @ x + c_t, chol_Q_t

        return dynamics_fn, state.mean

    def get_observation_params(
        state: LinearizedKalmanFilterState, model_inputs: ArrayTreeLike
    ) -> tuple[MeanAndCholCovFunc, Array, Array]:
        H_t, d_t, chol_R_t, y_t = (
            model_inputs["H"],
            model_inputs["d"],
            model_inputs["chol_R"],
            model_inputs["y"],
        )

        def obs_fn(x: ArrayLike) -> tuple[Array, Array]:
            return H_t @ x + d_t, chol_R_t

        return obs_fn, state.mean, y_t

    filter_obj = build_filter(
        get_init_params=get_init_params,
        get_dynamics_params=get_dynamics_params,
        get_observation_params=get_observation_params,
        associative=False,
    )
    filter_states = cuthbert_filter(filter_obj, model_inputs)

    smoother_obj = build_smoother(
        get_dynamics_params=get_dynamics_params,
    )
    smoothed_states = cuthbert_smoother(smoother_obj, filter_states)

    return smoothed_states.mean[1:]


def _try_smoother(
    ssm_model: Any,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    site_values: dict,
    det_values: dict,
) -> jnp.ndarray | None:
    """Try running Kalman smoother with estimated parameters."""
    spec = ssm_model.spec
    n_l = spec.n_latent

    try:
        from nof1_causal_lab.models.ssm.dynamics.composite import (
            compile_composite,
            pack_component_params_from_samples,
        )
        from nof1_causal_lab.models.ssm.inference.targets.affine import (
            derive_affine_dynamics,
        )
        from nof1_causal_lab.models.ssm.inference.targets.base import RuntimeDynamics

        diffusion_chol = det_values["diffusion"]
        diffusion_cov = diffusion_chol @ diffusion_chol.T
        compiled = compile_composite(spec.drift_spec)
        vf_params = pack_component_params_from_samples(
            spec.drift_spec,
            site_values,
            det_values,
        )
        affine = derive_affine_dynamics(
            RuntimeDynamics(
                vector_field=compiled.vector_field,
                vf_params=vf_params,
                diffusion_cov=diffusion_cov,
                input_effect=det_values.get("input_effect"),
            )
        )
        drift = affine.drift
        lambda_mat = det_values["lambda"]
        manifest_cov = det_values["manifest_cov"]
        t0_mean = det_values["t0_means"]
        t0_cov = det_values["t0_cov"]
        cint = affine.cint

        manifest_means_val = det_values.get(
            "manifest_means",
            spec.manifest_means_block.assemble(),
        )

        time_intervals = jnp.diff(times, prepend=times[0])
        time_intervals = jnp.maximum(time_intervals, MIN_DT)

        transition_inputs = getattr(ssm_model, "transition_inputs", None)
        if transition_inputs is not None:
            transition_inputs = transition_inputs[: times.shape[0]]

        Ad_all, Qd_all, cd_all = discretize_system_with_inputs_batched(
            drift,
            diffusion_cov,
            cint,
            affine.input_effect,
            transition_inputs,
            time_intervals,
        )
        cd_for_smoother = cd_all if cd_all is not None else jnp.zeros((len(time_intervals), n_l))

        smoothed = _kalman_smooth_states(
            observations,
            Ad_all,
            Qd_all,
            cd_for_smoother,
            lambda_mat,
            manifest_means_val,
            manifest_cov,
            t0_mean,
            t0_cov,
        )

        if not jnp.all(jnp.isfinite(smoothed)):
            logger.warning("Kalman smoother produced NaN/Inf states")
            return None

        logger.info(
            "Kalman smoother: states shape=%s, range=[%.3f, %.3f]",
            smoothed.shape,
            float(smoothed.min()),
            float(smoothed.max()),
        )
        return smoothed

    except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as e:
        logger.warning("Kalman smoother failed: %s", e)
        return None
