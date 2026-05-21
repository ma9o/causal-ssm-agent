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
        compiled = compile_composite(spec.dynamics_spec)
        vf_params = pack_component_params_from_samples(
            spec.dynamics_spec,
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
