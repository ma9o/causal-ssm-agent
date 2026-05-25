"""Blocked MCMC driver: latent trajectory updates + parameter kernels."""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)

from nof1_causal_lab.models.ssm.inference.latent_trace import (
    LATENT_MOVE_FIELD_NAMES,
    MH_LOG_ALPHA_FIELD_NAMES,
    PIT_PARTICLE_TRACE_FIELD_NAMES,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.auxiliary_kalman import (
    _latent_mh_step_eq8_runtime,
    _latent_mh_step_eq10_11_runtime,
)

_DEFAULT_MIN_SCALE = 1e-6
_DEFAULT_MAX_SCALE = 1e3
_MAX_INITIAL_PARAM_STEP_SIZE_ITERS = 20
_PARTICLE_LATENT_ADAPTATION_WINDOW = 100
_PARTICLE_LATENT_ADAPTATION_TOLERANCE = 0.05
_PARTICLE_LATENT_ADAPTATION_RHO = 0.5
_PARTICLE_LATENT_ADAPTATION_GAMMA = -0.5
_PARTICLE_LATENT_ADAPTATION_MIN_RATE = 1e-3


class AuxKalmanMCMCState(NamedTuple):
    position: jnp.ndarray
    latent_context: Any
    latent_trajectory: jnp.ndarray
    observation_auxiliary: Any
    trajectory_log_prob: jnp.ndarray
    complete_log_posterior: jnp.ndarray
    latent_delta: jnp.ndarray
    param_step_size: jnp.ndarray
    # BlackJAX dual-averaging state. Carried but not updated when
    # adaptation_scheme == "simple"; evolves only during warmup otherwise.
    latent_da: DualAveragingAdaptationState
    param_da: DualAveragingAdaptationState


@dataclass(frozen=True)
class AuxKalmanMCMCResult:
    """Minimal MCMC-compatible wrapper for auxiliary Kalman MCMC outputs."""

    chain_samples: dict[str, jnp.ndarray]
    chain_extra_fields: dict[str, jnp.ndarray]
    num_chains: int
    num_samples: int
    backend: str = "aux_kalman_mcmc"

    def get_samples(self, group_by_chain: bool = False) -> dict[str, jnp.ndarray]:
        if group_by_chain:
            return self.chain_samples
        return {
            name: values.reshape((self.num_chains * self.num_samples, *values.shape[2:]))
            for name, values in self.chain_samples.items()
        }

    def get_extra_fields(self, group_by_chain: bool = False) -> dict[str, jnp.ndarray]:
        if group_by_chain:
            return self.chain_extra_fields
        return {
            name: values.reshape((self.num_chains * self.num_samples, *values.shape[2:]))
            for name, values in self.chain_extra_fields.items()
        }


def _adapt_scale(
    scale: jnp.ndarray,
    *,
    accepted: jnp.ndarray,
    target_accept: float,
    adaptation_rate: float,
    min_scale: float = 1e-6,
    max_scale: float = 1e3,
) -> jnp.ndarray:
    """Simple exponential step-size adaptation on per-step binary accept."""
    dtype = scale.dtype
    accepted = jnp.asarray(accepted, dtype=dtype)
    target_accept = jnp.asarray(target_accept, dtype=dtype)
    adaptation_rate = jnp.asarray(adaptation_rate, dtype=dtype)
    min_scale = jnp.asarray(min_scale, dtype=dtype)
    max_scale = jnp.asarray(max_scale, dtype=dtype)
    factor = jnp.exp(adaptation_rate * (accepted - target_accept))
    return jnp.clip(scale * factor, min_scale, max_scale)


def _latent_summary_from_chain_moments(
    chain_means: jnp.ndarray,
    chain_stds: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    pooled_mean = jnp.mean(chain_means, axis=0)
    pooled_second_moment = jnp.mean(chain_stds * chain_stds + chain_means * chain_means, axis=0)
    pooled_var = jnp.maximum(pooled_second_moment - pooled_mean * pooled_mean, 0.0)
    return {
        "chain_mean": chain_means,
        "chain_std": chain_stds,
        "mean": pooled_mean,
        "std": jnp.sqrt(pooled_var),
    }


def _hostify_chain_array(values: jnp.ndarray) -> list[float] | list[list[float]]:
    host = jax.device_get(values)
    if host.ndim == 1:
        return [float(value) for value in host]
    return [[float(item) for item in row] for row in host]


def _clip_scale(
    scale: jnp.ndarray,
    *,
    min_scale: float | None,
    max_scale: float | None,
) -> jnp.ndarray:
    clipped = scale
    if min_scale is not None:
        min_value = jnp.nextafter(
            jnp.asarray(min_scale, dtype=clipped.dtype),
            jnp.asarray(jnp.inf, dtype=clipped.dtype),
        )
        clipped = jnp.maximum(clipped, min_value)
    if max_scale is not None:
        clipped = jnp.minimum(clipped, jnp.asarray(max_scale, dtype=clipped.dtype))
    return clipped


def _normalize_dual_averaging_state_shape(
    da_state: DualAveragingAdaptationState,
    reference_scale: jnp.ndarray,
) -> DualAveragingAdaptationState:
    target_shape = reference_scale.shape
    target_dtype = reference_scale.dtype
    return DualAveragingAdaptationState(
        log_step_size=jnp.broadcast_to(
            jnp.asarray(da_state.log_step_size, dtype=target_dtype),
            target_shape,
        ),
        log_step_size_avg=jnp.broadcast_to(
            jnp.asarray(da_state.log_step_size_avg, dtype=target_dtype),
            target_shape,
        ),
        step=da_state.step,
        avg_error=jnp.broadcast_to(
            jnp.asarray(da_state.avg_error, dtype=target_dtype),
            target_shape,
        ),
        mu=jnp.broadcast_to(
            jnp.asarray(da_state.mu, dtype=target_dtype),
            target_shape,
        ),
    )


def _clip_dual_averaging_state(
    da_state: DualAveragingAdaptationState,
    *,
    min_scale: float | None,
    max_scale: float | None,
) -> DualAveragingAdaptationState:
    if min_scale is None and max_scale is None:
        return da_state

    log_min = (
        None if min_scale is None else jnp.log(jnp.asarray(min_scale, dtype=da_state.mu.dtype))
    )
    log_max = (
        None if max_scale is None else jnp.log(jnp.asarray(max_scale, dtype=da_state.mu.dtype))
    )

    def _clip_log_value(value: jnp.ndarray) -> jnp.ndarray:
        clipped = value
        if log_min is not None:
            clipped = jnp.maximum(clipped, log_min)
        if log_max is not None:
            clipped = jnp.minimum(clipped, log_max)
        return clipped

    return DualAveragingAdaptationState(
        log_step_size=_clip_log_value(da_state.log_step_size),
        log_step_size_avg=_clip_log_value(da_state.log_step_size_avg),
        step=da_state.step,
        avg_error=da_state.avg_error,
        mu=_clip_log_value(da_state.mu),
    )


def _identity_project_latent_trajectory(latent_trajectory: jnp.ndarray) -> jnp.ndarray:
    return latent_trajectory


_PER_T_FIELD_NAMES = MH_LOG_ALPHA_FIELD_NAMES

_LATENT_FIELD_NAMES = LATENT_MOVE_FIELD_NAMES + PIT_PARTICLE_TRACE_FIELD_NAMES

_PARAMETER_FIELD_NAMES = (
    "accept_prob",
    "diverging",
    "num_steps",
    "energy",
    "gibbs_block_count",
    "residual_dim",
)


@dataclass(frozen=True)
class _AuxKalmanMCMCRunnerStatic:
    public_latent_trajectory_runtime_fn: Any
    latent_context_runtime_fn: Any
    initial_latent_from_context_fn: Any
    initial_observation_auxiliary_from_context_runtime_fn: Any
    refresh_observation_auxiliary_from_context_runtime_fn: Any
    complete_log_posterior_from_context_runtime_fn: Any
    log_prior_unc_fn: Any
    prior_terms_from_context_fn: Any
    observation_grad_from_context_runtime_fn: Any
    observation_log_prob_and_grad_from_context_runtime_fn: Any
    observation_log_prob_per_t_from_context_runtime_fn: Any
    trajectory_log_prob_from_context_runtime_fn: Any
    complete_log_posterior_runtime_fn: Any
    dim: int
    latent_kernel_name: str
    latent_step_fn: Any
    latent_proposal_family: str
    latent_parallel_filter: bool
    parameter_kernel_name: str
    parameter_step_fn: Any
    parameter_preconditioned: bool
    latent_target_accept: float
    param_target_accept: float
    adaptation_rate: float
    initial_param_scale: float
    initial_latent_scale_mode: str
    latent_min_scale: float | None
    latent_max_scale: float | None
    param_min_scale: float
    param_max_scale: float
    parameter_residual_dim: int
    use_dual_averaging: bool
    use_windowed_latent_adaptation: bool
    retain_latent_paths: bool
    emit_per_t_log_alpha: bool
    compute_latent_posterior_summary: bool


def _initialize_latent_delta(
    initial_latent_scale_value,
    init_latent: jnp.ndarray,
    *,
    initial_latent_scale_mode: str,
    scale_dtype,
    latent_min_scale: float | None,
    latent_max_scale: float | None,
) -> jnp.ndarray:
    if initial_latent_scale_mode == "direct":
        init_latent_delta = jnp.asarray(initial_latent_scale_value, dtype=scale_dtype)
    elif initial_latent_scale_mode == "per_time_constant":
        init_latent_delta = jnp.full(
            (init_latent.shape[0],),
            jnp.asarray(initial_latent_scale_value, dtype=scale_dtype),
            dtype=scale_dtype,
        )
    else:
        raise ValueError(
            "Unknown latent initial scale mode "
            f"{initial_latent_scale_mode!r}; expected 'direct' or 'per_time_constant'."
        )
    return _clip_scale(
        init_latent_delta,
        min_scale=latent_min_scale,
        max_scale=latent_max_scale,
    )


def _expand_chain_statistic(values: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    if reference.ndim <= values.ndim:
        return values
    return jnp.reshape(values, values.shape + (1,) * (reference.ndim - values.ndim))


def _initialize_aux_kalman_mcmc_chain_state(
    init_position: jnp.ndarray,
    initial_latent_scale_value,
    *,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    static: _AuxKalmanMCMCRunnerStatic,
    initial_latent_trajectory: jnp.ndarray | None = None,
) -> AuxKalmanMCMCState:
    init_context = static.latent_context_runtime_fn(init_position, times)
    predictive_latent = static.initial_latent_from_context_fn(init_context)
    if initial_latent_trajectory is None:
        init_latent = predictive_latent
    else:
        init_latent = jnp.asarray(initial_latent_trajectory, dtype=predictive_latent.dtype)
        if init_latent.shape != predictive_latent.shape:
            raise ValueError(
                "initial_latent_trajectory shape must match the model latent trajectory; "
                f"got {init_latent.shape}, expected {predictive_latent.shape}."
            )
    init_observation_auxiliary = static.initial_observation_auxiliary_from_context_runtime_fn(
        init_context,
        init_latent,
        observations,
    )
    init_complete, init_traj = static.complete_log_posterior_from_context_runtime_fn(
        init_position,
        init_context,
        init_latent,
        init_observation_auxiliary,
        observations,
    )
    scale_dtype = init_latent.dtype
    init_latent_delta = _initialize_latent_delta(
        initial_latent_scale_value,
        init_latent,
        initial_latent_scale_mode=static.initial_latent_scale_mode,
        scale_dtype=scale_dtype,
        latent_min_scale=static.latent_min_scale,
        latent_max_scale=static.latent_max_scale,
    )
    init_param_step_size = _clip_scale(
        jnp.asarray(static.initial_param_scale, dtype=scale_dtype),
        min_scale=static.param_min_scale,
        max_scale=static.param_max_scale,
    )

    da_latent_init, _da_latent_update, _ = dual_averaging_adaptation(
        target=float(static.latent_target_accept)
    )
    da_param_init, _da_param_update, _ = dual_averaging_adaptation(
        target=float(static.param_target_accept)
    )
    return AuxKalmanMCMCState(
        position=init_position,
        latent_context=init_context,
        latent_trajectory=init_latent,
        observation_auxiliary=init_observation_auxiliary,
        trajectory_log_prob=init_traj,
        complete_log_posterior=init_complete,
        latent_delta=init_latent_delta,
        param_step_size=init_param_step_size,
        latent_da=_normalize_dual_averaging_state_shape(
            da_latent_init(init_latent_delta),
            init_latent_delta,
        ),
        param_da=da_param_init(init_param_step_size),
    )


def _stack_chain_states(states: list[AuxKalmanMCMCState]) -> AuxKalmanMCMCState:
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values, axis=0), *states)


@functools.partial(jax.jit, static_argnames=("static",))
def _sample_public_latent_batch(
    states: AuxKalmanMCMCState,
    keys: jnp.ndarray,
    observations: jnp.ndarray,
    *,
    static: _AuxKalmanMCMCRunnerStatic,
) -> jnp.ndarray:
    return jax.vmap(
        lambda state, key: static.public_latent_trajectory_runtime_fn(
            state.latent_context,
            state.latent_trajectory,
            state.observation_auxiliary,
            observations,
            key,
        )
    )(states, keys)


@functools.partial(jax.jit, static_argnames=("static",))
def _run_batched_observation_auxiliary_step(
    states: AuxKalmanMCMCState,
    step_keys: jnp.ndarray,
    observations: jnp.ndarray,
    *,
    static: _AuxKalmanMCMCRunnerStatic,
) -> tuple[AuxKalmanMCMCState, dict[str, jnp.ndarray]]:
    def step_fn(state, key):
        observation_auxiliary = static.refresh_observation_auxiliary_from_context_runtime_fn(
            state.latent_context,
            state.latent_trajectory,
            state.observation_auxiliary,
            observations,
            key,
        )
        trajectory_lp = static.trajectory_log_prob_from_context_runtime_fn(
            state.latent_context,
            state.latent_trajectory,
            observation_auxiliary,
            observations,
        )
        complete_lp = static.log_prior_unc_fn(state.position) + trajectory_lp
        next_state = state._replace(
            observation_auxiliary=observation_auxiliary,
            trajectory_log_prob=trajectory_lp,
            complete_log_posterior=complete_lp,
        )
        return next_state, {"refreshed": jnp.asarray(1.0, dtype=state.position.dtype)}

    return jax.vmap(step_fn)(states, step_keys)


@functools.partial(jax.jit, static_argnames=("static",))
def _run_batched_aux_kalman_mcmc_latent_step(
    states: AuxKalmanMCMCState,
    step_keys: jnp.ndarray,
    observations: jnp.ndarray,
    *,
    static: _AuxKalmanMCMCRunnerStatic,
) -> tuple[AuxKalmanMCMCState, dict[str, jnp.ndarray]]:
    if static.latent_kernel_name == "pit_particle_mgrad":
        return jax.vmap(static.latent_step_fn)(states, step_keys)

    if static.latent_proposal_family == "eq8":

        def step_fn(state, key):
            return _latent_mh_step_eq8_runtime(
                state,
                key,
                observations,
                prior_terms_from_context_fn=static.prior_terms_from_context_fn,
                log_prior_unc_fn=static.log_prior_unc_fn,
                observation_grad_from_context_runtime_fn=(
                    static.observation_grad_from_context_runtime_fn
                ),
                observation_log_prob_and_grad_from_context_runtime_fn=(
                    static.observation_log_prob_and_grad_from_context_runtime_fn
                ),
                observation_log_prob_per_t_from_context_runtime_fn=(
                    static.observation_log_prob_per_t_from_context_runtime_fn
                ),
                parallel=static.latent_parallel_filter,
                emit_per_t_log_alpha=static.emit_per_t_log_alpha,
            )

    else:

        def step_fn(state, key):
            return _latent_mh_step_eq10_11_runtime(
                state,
                key,
                observations,
                prior_terms_from_context_fn=static.prior_terms_from_context_fn,
                log_prior_unc_fn=static.log_prior_unc_fn,
                observation_grad_from_context_runtime_fn=(
                    static.observation_grad_from_context_runtime_fn
                ),
                observation_log_prob_and_grad_from_context_runtime_fn=(
                    static.observation_log_prob_and_grad_from_context_runtime_fn
                ),
                observation_log_prob_per_t_from_context_runtime_fn=(
                    static.observation_log_prob_per_t_from_context_runtime_fn
                ),
                parallel=static.latent_parallel_filter,
                emit_per_t_log_alpha=static.emit_per_t_log_alpha,
            )

    return jax.vmap(step_fn)(states, step_keys)


@functools.partial(jax.jit, static_argnames=("static",))
def _run_batched_aux_kalman_mcmc_parameter_step(
    states: AuxKalmanMCMCState,
    step_keys: jnp.ndarray,
    *,
    static: _AuxKalmanMCMCRunnerStatic,
) -> tuple[AuxKalmanMCMCState, dict[str, jnp.ndarray]]:
    return jax.vmap(static.parameter_step_fn)(states, step_keys)


@functools.partial(jax.jit, static_argnames=("static",))
def _auto_tune_initial_param_step_size_batched(
    states: AuxKalmanMCMCState,
    step_keys: jnp.ndarray,
    *,
    static: _AuxKalmanMCMCRunnerStatic,
) -> tuple[AuxKalmanMCMCState, dict[str, jnp.ndarray]]:
    """Find a reasonable initial residual-NUTS step size for each chain."""
    if static.parameter_residual_dim == 0:
        return states, {
            "accept_prob": jnp.ones_like(states.param_step_size),
            "num_search_steps": jnp.zeros_like(states.param_step_size),
        }

    target_accept = jnp.asarray(static.param_target_accept, dtype=states.param_step_size.dtype)

    def _one_chain(state, key):
        initial_scale = _clip_scale(
            state.param_step_size,
            min_scale=static.param_min_scale,
            max_scale=static.param_max_scale,
        )

        def _accept_prob(search_idx: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
            trial_state = state._replace(param_step_size=scale)
            _next_state, info = static.parameter_step_fn(
                trial_state,
                random.fold_in(key, search_idx),
            )
            accept = jnp.asarray(info["accept_prob"], dtype=scale.dtype)
            accept = jnp.where(jnp.isfinite(accept), accept, jnp.asarray(0.0, dtype=scale.dtype))
            return jnp.clip(accept, 0.0, 1.0)

        initial_accept = _accept_prob(jnp.asarray(0, dtype=jnp.int32), initial_scale)
        initial_direction = jnp.where(
            target_accept < initial_accept,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
        )

        def _continue(carry) -> jnp.ndarray:
            (
                search_idx,
                direction,
                previous_direction,
                _previous_scale,
                _previous_accept,
                scale,
                _accept,
            ) = carry
            not_too_large = (scale < static.param_max_scale) | (direction <= 0)
            not_too_small = (scale > static.param_min_scale) | (direction >= 0)
            not_crossed = (previous_direction == 0) | (direction == previous_direction)
            return (
                (search_idx < _MAX_INITIAL_PARAM_STEP_SIZE_ITERS)
                & not_too_large
                & not_too_small
                & not_crossed
            )

        def _update(carry):
            (
                search_idx,
                direction,
                _previous_direction,
                _previous_scale,
                _previous_accept,
                scale,
                accept_current,
            ) = carry
            proposed_scale = jnp.where(direction > 0, 2.0 * scale, 0.5 * scale)
            proposed_scale = _clip_scale(
                proposed_scale,
                min_scale=static.param_min_scale,
                max_scale=static.param_max_scale,
            )
            accept = _accept_prob(search_idx, proposed_scale)
            next_direction = jnp.where(
                target_accept < accept,
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(-1, dtype=jnp.int32),
            )
            return (
                search_idx + 1,
                next_direction,
                direction,
                scale,
                accept_current,
                proposed_scale,
                accept,
            )

        (
            final_search_idx,
            final_direction,
            previous_direction,
            previous_scale,
            previous_accept,
            candidate_scale,
            candidate_accept,
        ) = (
            jax.lax.while_loop(
                _continue,
                _update,
                (
                    jnp.asarray(1, dtype=jnp.int32),
                    initial_direction,
                    jnp.asarray(0, dtype=jnp.int32),
                    initial_scale,
                    initial_accept,
                    initial_scale,
                    initial_accept,
                ),
            )
        )
        has_previous = previous_direction != 0
        previous_error = jnp.abs(previous_accept - target_accept)
        candidate_error = jnp.abs(candidate_accept - target_accept)
        selected_previous = has_previous & (previous_error <= candidate_error)
        tuned_scale = jnp.where(selected_previous, previous_scale, candidate_scale)
        tuned_accept = jnp.where(selected_previous, previous_accept, candidate_accept)
        crossed = has_previous & (final_direction != previous_direction)
        return (
            tuned_scale,
            tuned_accept,
            final_search_idx,
            candidate_accept,
            previous_accept,
            selected_previous.astype(tuned_scale.dtype),
            crossed.astype(tuned_scale.dtype),
        )

    (
        tuned_scale,
        accept_prob,
        num_search_steps,
        candidate_accept_prob,
        previous_accept_prob,
        selected_previous,
        crossed,
    ) = jax.vmap(_one_chain)(states, step_keys)
    return states._replace(param_step_size=tuned_scale), {
        "accept_prob": accept_prob,
        "num_search_steps": num_search_steps.astype(states.param_step_size.dtype),
        "candidate_accept_prob": candidate_accept_prob,
        "previous_accept_prob": previous_accept_prob,
        "selected_previous": selected_previous,
        "crossed": crossed,
    }


def _reset_param_dual_averaging_state(
    states: AuxKalmanMCMCState,
    *,
    target_accept: float,
) -> AuxKalmanMCMCState:
    da_param_init, _da_param_update, _ = dual_averaging_adaptation(target=float(target_accept))
    return states._replace(param_da=jax.vmap(da_param_init)(states.param_step_size))


@functools.partial(jax.jit, static_argnames=("static",))
def _apply_windowed_latent_update_batched(
    states: AuxKalmanMCMCState,
    latent_accept: jnp.ndarray,
    latent_accept_window: jnp.ndarray,
    warmup_step: jnp.ndarray,
    *,
    static: _AuxKalmanMCMCRunnerStatic,
) -> tuple[AuxKalmanMCMCState, jnp.ndarray, jnp.ndarray]:
    """mGRAD released-code per-time multiplicative latent delta adaptation."""
    accept = jnp.asarray(latent_accept, dtype=states.latent_delta.dtype)
    if accept.shape != states.latent_delta.shape:
        accept = _expand_chain_statistic(accept, states.latent_delta)

    window_idx = jnp.mod(warmup_step, latent_accept_window.shape[1])
    latent_accept_window = latent_accept_window.at[:, window_idx, ...].set(accept)
    denominator = jnp.minimum(
        warmup_step + jnp.asarray(1, dtype=warmup_step.dtype),
        jnp.asarray(latent_accept_window.shape[1], dtype=warmup_step.dtype),
    )
    window_accept_rate = jnp.sum(latent_accept_window, axis=1) / jnp.asarray(
        denominator,
        dtype=states.latent_delta.dtype,
    )
    target_accept = jnp.asarray(static.latent_target_accept, dtype=states.latent_delta.dtype)
    error = window_accept_rate - target_accept
    normalized_error = error / jnp.maximum(target_accept, jnp.asarray(1e-6, dtype=target_accept.dtype))
    learning_rate = jnp.maximum(
        jnp.asarray(_PARTICLE_LATENT_ADAPTATION_RHO, dtype=states.latent_delta.dtype)
        * jnp.power(
            jnp.asarray(warmup_step + 1, dtype=states.latent_delta.dtype),
            jnp.asarray(_PARTICLE_LATENT_ADAPTATION_GAMMA, dtype=states.latent_delta.dtype),
        ),
        jnp.asarray(_PARTICLE_LATENT_ADAPTATION_MIN_RATE, dtype=states.latent_delta.dtype),
    )
    current_delta = _clip_scale(
        states.latent_delta,
        min_scale=static.latent_min_scale,
        max_scale=static.latent_max_scale,
    )
    next_delta = jnp.where(
        jnp.abs(error) < jnp.asarray(
            _PARTICLE_LATENT_ADAPTATION_TOLERANCE,
            dtype=states.latent_delta.dtype,
        ),
        current_delta,
        current_delta + learning_rate * current_delta * normalized_error,
    )
    next_latent_delta = _clip_scale(
        next_delta,
        min_scale=static.latent_min_scale,
        max_scale=static.latent_max_scale,
    )
    return states._replace(latent_delta=next_latent_delta), latent_accept_window, window_accept_rate


def _apply_param_dual_averaging_update_batched(
    states: AuxKalmanMCMCState,
    param_accept: jnp.ndarray,
    *,
    static: _AuxKalmanMCMCRunnerStatic,
    da_param_update,
    is_final_warmup: bool,
) -> AuxKalmanMCMCState:
    updated_param_da = jax.vmap(
        lambda da_state, accepted: _clip_dual_averaging_state(
            da_param_update(da_state, accepted),
            min_scale=static.param_min_scale,
            max_scale=static.param_max_scale,
        )
    )(states.param_da, param_accept)
    scale_dtype = states.latent_delta.dtype
    param_live = jnp.exp(updated_param_da.log_step_size).astype(scale_dtype)
    param_frozen = jnp.exp(updated_param_da.log_step_size_avg).astype(scale_dtype)
    return states._replace(
        param_step_size=_clip_scale(
            jnp.where(is_final_warmup, param_frozen, param_live),
            min_scale=static.param_min_scale,
            max_scale=static.param_max_scale,
        ),
        param_da=updated_param_da,
    )


def _apply_dual_averaging_update_batched(
    states: AuxKalmanMCMCState,
    latent_accept: jnp.ndarray,
    param_accept: jnp.ndarray,
    *,
    static: _AuxKalmanMCMCRunnerStatic,
    da_latent_update,
    da_param_update,
    is_final_warmup: bool,
) -> AuxKalmanMCMCState:
    updated_latent_da = jax.vmap(
        lambda da_state, accepted, reference_scale: _clip_dual_averaging_state(
            _normalize_dual_averaging_state_shape(
                da_latent_update(da_state, accepted),
                reference_scale,
            ),
            min_scale=static.latent_min_scale,
            max_scale=static.latent_max_scale,
        )
    )(states.latent_da, latent_accept, states.latent_delta)
    states = _apply_param_dual_averaging_update_batched(
        states,
        param_accept,
        static=static,
        da_param_update=da_param_update,
        is_final_warmup=is_final_warmup,
    )
    scale_dtype = states.latent_delta.dtype
    latent_live = jnp.exp(updated_latent_da.log_step_size).astype(scale_dtype)
    latent_frozen = jnp.exp(updated_latent_da.log_step_size_avg).astype(scale_dtype)
    next_latent_delta = _clip_scale(
        jnp.where(is_final_warmup, latent_frozen, latent_live),
        min_scale=static.latent_min_scale,
        max_scale=static.latent_max_scale,
    )
    return states._replace(
        latent_delta=next_latent_delta,
        latent_da=updated_latent_da,
    )


def _apply_simple_adaptation_update_batched(
    states: AuxKalmanMCMCState,
    latent_accept: jnp.ndarray,
    param_accept: jnp.ndarray,
    *,
    static: _AuxKalmanMCMCRunnerStatic,
) -> AuxKalmanMCMCState:
    latent_accept_for_scale = _expand_chain_statistic(latent_accept, states.latent_delta)
    return states._replace(
        latent_delta=_adapt_scale(
            states.latent_delta,
            accepted=latent_accept_for_scale,
            target_accept=static.latent_target_accept,
            adaptation_rate=static.adaptation_rate,
            min_scale=1e-6 if static.latent_min_scale is None else float(static.latent_min_scale),
            max_scale=1e3 if static.latent_max_scale is None else float(static.latent_max_scale),
        ),
        param_step_size=_adapt_scale(
            states.param_step_size,
            accepted=param_accept,
            target_accept=static.param_target_accept,
            adaptation_rate=static.adaptation_rate,
        ),
    )


def _stack_sample_history(
    history: list[jnp.ndarray],
    *,
    num_chains: int,
    trailing_shape: tuple[int, ...],
    dtype,
) -> jnp.ndarray:
    if not history:
        return jnp.zeros((num_chains, 0, *trailing_shape), dtype=dtype)
    stacked = jnp.stack(history, axis=0)
    return jnp.swapaxes(stacked, 0, 1)


def run_aux_kalman_mcmc(
    bundle: dict[str, Any],
    *,
    latent_kernel: dict[str, Any],
    parameter_kernel: dict[str, Any],
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    adaptation_rate: float,
    init_scale: float,
    retain_latent_paths: bool,
    adaptation_scheme: str = "dual_averaging",
    init_positions: jnp.ndarray | None = None,
    initial_latent_trajectories: jnp.ndarray | None = None,
    emit_per_t_log_alpha: bool = False,
    compute_latent_posterior_summary: bool = True,
) -> dict[str, Any]:
    """Blocked MCMC: latent trajectory step + hybrid Gibbs/NUTS parameter step.

    Each block has its own accept signal so ``latent_delta`` and
    ``param_step_size`` adapt against their own target acceptances, avoiding
    the degenerate "both scales race to zero" corner that a joint-MH shared
    adaptation falls into.

    Parameters
    ----------
    adaptation_scheme:
        ``"dual_averaging"`` (default) — Hoffman & Gelman (2014) dual
        averaging on log step size, with √t damping and a decaying running-mean
        weight. ``"simple"`` keeps the original exponential update on the raw
        per-step binary accept for ablations and reproduction runs.
    init_positions:
        Optional ``(num_chains, dim)`` array of unconstrained parameter
        positions. When ``None`` the sampler perturbs ``bundle["flat_example"]``
        with ``init_scale·randn``; when provided (e.g. from Pathfinder) the
        chains start exactly from those positions.
    initial_latent_trajectories:
        Optional ``(num_chains, T, latent_dim)`` array of complete latent paths
        used to seed the latent block. When ``None`` each chain uses the
        bundle's predictive mean rollout.
    """
    total_steps = num_warmup + num_samples
    if total_steps <= 0:
        raise ValueError("aux_kalman_mcmc requires at least one warmup or posterior draw step.")
    if adaptation_scheme not in {"simple", "dual_averaging"}:
        raise ValueError(
            f"Unknown adaptation_scheme {adaptation_scheme!r}; expected 'simple' or 'dual_averaging'."
        )
    project_latent_trajectory = bundle.get(
        "project_latent_trajectory_fn",
        _identity_project_latent_trajectory,
    )
    latent_kernel_name = latent_kernel.get("name", "kalman")
    if latent_kernel_name not in {"kalman", "pit_particle_mgrad"}:
        raise ValueError(
            f"Unsupported latent kernel name {latent_kernel_name!r}; "
            "expected 'kalman' or 'pit_particle_mgrad'."
        )
    if latent_kernel_name == "pit_particle_mgrad" and latent_kernel.get("step_fn") is None:
        raise ValueError("pit_particle_mgrad latent kernel requires a 'step_fn'.")
    latent_target_accept = latent_kernel["target_accept"]
    parameter_kernel_name = parameter_kernel.get("name", "hybrid_gibbs_nuts")
    if parameter_kernel_name != "hybrid_gibbs_nuts":
        raise ValueError(
            f"Unsupported parameter kernel name {parameter_kernel_name!r}; "
            "expected 'hybrid_gibbs_nuts'."
        )
    if parameter_kernel.get("step_fn") is None:
        raise ValueError(f"{parameter_kernel_name} parameter kernel requires a 'step_fn'.")
    param_target_accept = parameter_kernel["target_accept"]
    initial_latent_scale_value = latent_kernel.get(
        "initial_scale_value",
        latent_kernel["initial_scale"],
    )
    initial_latent_scale_mode = latent_kernel.get("initial_scale_mode", "direct")
    initial_param_scale = float(parameter_kernel["initial_scale"])
    use_dual_averaging = adaptation_scheme == "dual_averaging"
    use_windowed_latent_adaptation = use_dual_averaging and latent_kernel_name == "pit_particle_mgrad"
    parameter_residual_dim = int(parameter_kernel.get("residual_dim", 0))
    latent_min_scale = latent_kernel.get("min_scale")
    latent_max_scale = latent_kernel.get("max_scale")
    if latent_min_scale is None:
        latent_min_scale = _DEFAULT_MIN_SCALE
    if latent_max_scale is None:
        latent_max_scale = _DEFAULT_MAX_SCALE
    param_min_scale = float(parameter_kernel.get("min_scale", _DEFAULT_MIN_SCALE))
    param_max_scale = float(parameter_kernel.get("max_scale", _DEFAULT_MAX_SCALE))
    if initial_latent_scale_mode not in {"direct", "per_time_constant"}:
        raise ValueError(
            "Unknown latent initial scale mode "
            f"{initial_latent_scale_mode!r}; expected 'direct' or 'per_time_constant'."
        )
    if (
        latent_min_scale is not None
        and latent_max_scale is not None
        and float(latent_min_scale) > float(latent_max_scale)
    ):
        raise ValueError(
            "latent_kernel min_scale must be <= max_scale; got "
            f"{latent_min_scale} > {latent_max_scale}."
        )
    if param_min_scale > param_max_scale:
        raise ValueError(
            "parameter_kernel min_scale must be <= max_scale; got "
            f"{param_min_scale} > {param_max_scale}."
        )

    observations = bundle["observations"]
    times = bundle["times"]
    static = _AuxKalmanMCMCRunnerStatic(
        public_latent_trajectory_runtime_fn=bundle.get(
            "public_latent_trajectory_runtime_fn",
            lambda _context, latent_trajectory, _observation_auxiliary, _observations, _key: (
                project_latent_trajectory(latent_trajectory)
            ),
        ),
        latent_context_runtime_fn=bundle.get(
            "latent_context_runtime_fn",
            lambda z, _runtime_times: bundle["latent_context_fn"](z),
        ),
        initial_latent_from_context_fn=bundle["initial_latent_from_context_fn"],
        initial_observation_auxiliary_from_context_runtime_fn=bundle[
            "initial_observation_auxiliary_from_context_runtime_fn"
        ],
        refresh_observation_auxiliary_from_context_runtime_fn=bundle[
            "refresh_observation_auxiliary_from_context_runtime_fn"
        ],
        complete_log_posterior_from_context_runtime_fn=bundle[
            "complete_log_posterior_conditioned_from_context_runtime_fn"
        ],
        log_prior_unc_fn=bundle["log_prior_unc_fn"],
        prior_terms_from_context_fn=bundle["prior_terms_from_context_fn"],
        observation_grad_from_context_runtime_fn=bundle[
            "observation_grad_conditioned_from_context_runtime_fn"
        ],
        observation_log_prob_and_grad_from_context_runtime_fn=bundle[
            "observation_log_prob_and_grad_conditioned_from_context_runtime_fn"
        ],
        observation_log_prob_per_t_from_context_runtime_fn=bundle[
            "observation_log_prob_per_t_conditioned_from_context_runtime_fn"
        ],
        trajectory_log_prob_from_context_runtime_fn=bundle[
            "trajectory_log_prob_conditioned_from_context_runtime_fn"
        ],
        complete_log_posterior_runtime_fn=bundle["complete_log_posterior_conditioned_runtime_fn"],
        dim=int(bundle["flat_example"].shape[0]),
        latent_kernel_name=latent_kernel_name,
        latent_step_fn=latent_kernel.get("step_fn"),
        latent_proposal_family=latent_kernel.get("proposal_family", "eq8"),
        latent_parallel_filter=bool(latent_kernel.get("parallel", True)),
        parameter_kernel_name=parameter_kernel_name,
        parameter_step_fn=parameter_kernel["step_fn"],
        parameter_preconditioned=bool(parameter_kernel.get("preconditioned", False)),
        latent_target_accept=latent_target_accept,
        param_target_accept=param_target_accept,
        adaptation_rate=adaptation_rate,
        initial_param_scale=initial_param_scale,
        initial_latent_scale_mode=initial_latent_scale_mode,
        latent_min_scale=latent_min_scale,
        latent_max_scale=latent_max_scale,
        param_min_scale=param_min_scale,
        param_max_scale=param_max_scale,
        parameter_residual_dim=parameter_residual_dim,
        use_dual_averaging=use_dual_averaging,
        use_windowed_latent_adaptation=use_windowed_latent_adaptation,
        retain_latent_paths=retain_latent_paths,
        emit_per_t_log_alpha=emit_per_t_log_alpha,
        compute_latent_posterior_summary=compute_latent_posterior_summary,
    )

    base_key = random.PRNGKey(seed)
    init_key, tune_key, chain_key = random.split(base_key, 3)
    if init_positions is None:
        init_keys = random.split(init_key, num_chains)
        init_noise = jax.vmap(
            lambda key: random.normal(
                key,
                bundle["flat_example"].shape,
                dtype=bundle["flat_example"].dtype,
            )
        )(init_keys)
        chain_init_positions = bundle["flat_example"][None, ...] + init_scale * init_noise
    else:
        chain_init_positions = jnp.asarray(init_positions, dtype=bundle["flat_example"].dtype)
        if chain_init_positions.shape != (num_chains, int(bundle["flat_example"].shape[0])):
            raise ValueError(
                "init_positions must have shape (num_chains, dim); got "
                f"{chain_init_positions.shape} with num_chains={num_chains} and "
                f"dim={int(bundle['flat_example'].shape[0])}."
            )
    if initial_latent_trajectories is not None:
        chain_initial_latents = jnp.asarray(
            initial_latent_trajectories,
            dtype=observations.dtype,
        )
        if chain_initial_latents.shape[0] != num_chains:
            raise ValueError(
                "initial_latent_trajectories must have leading dimension num_chains; got "
                f"{chain_initial_latents.shape[0]} with num_chains={num_chains}."
            )
    else:
        chain_initial_latents = None

    states = _stack_chain_states(
        [
            _initialize_aux_kalman_mcmc_chain_state(
                chain_init_positions[chain_idx],
                initial_latent_scale_value,
                observations=observations,
                times=times,
                static=static,
                initial_latent_trajectory=(
                    None if chain_initial_latents is None else chain_initial_latents[chain_idx]
                ),
            )
            for chain_idx in range(num_chains)
        ]
    )
    param_step_size_initial_guess = states.param_step_size
    param_step_size_auto_tuned = bool(use_dual_averaging and parameter_residual_dim > 0)
    if param_step_size_auto_tuned:
        states, param_step_size_tuning_info = _auto_tune_initial_param_step_size_batched(
            states,
            random.split(tune_key, num_chains),
            static=static,
        )
        states = _reset_param_dual_averaging_state(
            states,
            target_accept=param_target_accept,
        )
        print(
            "aux_kalman_mcmc progress: "
            "parameter initial step-size auto-tuned "
            f"guess_range=[{float(jnp.min(param_step_size_initial_guess)):.3g},"
            f"{float(jnp.max(param_step_size_initial_guess)):.3g}] "
            f"tuned_range=[{float(jnp.min(states.param_step_size)):.3g},"
            f"{float(jnp.max(states.param_step_size)):.3g}] "
            "accept_range="
            f"[{float(jnp.min(param_step_size_tuning_info['accept_prob'])):.3g},"
            f"{float(jnp.max(param_step_size_tuning_info['accept_prob'])):.3g}] "
            "search_steps_range="
            f"[{float(jnp.min(param_step_size_tuning_info['num_search_steps'])):.0f},"
            f"{float(jnp.max(param_step_size_tuning_info['num_search_steps'])):.0f}] "
            "selected_previous="
            f"{int(jnp.sum(param_step_size_tuning_info['selected_previous']))}/"
            f"{num_chains}",
            flush=True,
        )
    else:
        param_step_size_tuning_info = {
            "accept_prob": jnp.zeros_like(states.param_step_size),
            "num_search_steps": jnp.zeros_like(states.param_step_size),
            "candidate_accept_prob": jnp.zeros_like(states.param_step_size),
            "previous_accept_prob": jnp.zeros_like(states.param_step_size),
            "selected_previous": jnp.zeros_like(states.param_step_size),
            "crossed": jnp.zeros_like(states.param_step_size),
        }
    initial_param_step_size = states.param_step_size
    latent_accept_window = jnp.zeros(
        (
            num_chains,
            _PARTICLE_LATENT_ADAPTATION_WINDOW,
            *states.latent_delta.shape[1:],
        ),
        dtype=states.latent_delta.dtype,
    )
    latent_window_accept_rate = jnp.zeros_like(states.latent_delta)
    step_keys = random.split(chain_key, total_steps * num_chains).reshape(
        total_steps, num_chains, 2
    )
    need_public_latent = compute_latent_posterior_summary or retain_latent_paths
    if use_dual_averaging:
        _, da_latent_update, _ = dual_averaging_adaptation(target=float(latent_target_accept))
        _, da_param_update, _ = dual_averaging_adaptation(target=float(param_target_accept))

    if compute_latent_posterior_summary:
        public_example = _sample_public_latent_batch(
            states,
            random.split(init_key, num_chains),
            observations,
            static=static,
        )
        latent_sum = jnp.zeros_like(public_example)
        latent_sumsq = jnp.zeros_like(public_example)
        sample_count = jnp.asarray(0, dtype=jnp.int32)

    position_history: list[jnp.ndarray] = []
    latent_accept_history: list[jnp.ndarray] = []
    latent_extra_history: dict[str, list[jnp.ndarray]] = {name: [] for name in _LATENT_FIELD_NAMES}
    param_accept_history: list[jnp.ndarray] = []
    complete_lp_history: list[jnp.ndarray] = []
    latent_paths_history: list[jnp.ndarray] = []
    per_t_history: dict[str, list[jnp.ndarray]] = {name: [] for name in _PER_T_FIELD_NAMES}
    parameter_extra_history: dict[str, list[jnp.ndarray]] = {
        name: [] for name in _PARAMETER_FIELD_NAMES
    }

    progress_started = time.monotonic()
    progress_every = max(1, min(250, total_steps // 20))
    print(
        "aux_kalman_mcmc progress: "
        f"chains={num_chains} warmup={num_warmup} samples={num_samples} "
        f"total_steps={total_steps} adaptation={adaptation_scheme} "
        f"latent_adaptation={'windowed' if use_windowed_latent_adaptation else adaptation_scheme} "
        f"latent_kernel={latent_kernel_name} parameter_kernel={parameter_kernel_name} "
        f"progress_every={progress_every}",
        flush=True,
    )

    for step_idx in range(total_steps):
        aux_latent_param_keys = jax.vmap(lambda key: random.split(key, 4))(step_keys[step_idx])
        obs_aux_keys = aux_latent_param_keys[:, 0, :]
        latent_keys = aux_latent_param_keys[:, 1, :]
        param_keys = aux_latent_param_keys[:, 2, :]
        public_latent_keys = aux_latent_param_keys[:, 3, :]
        states, _obs_aux_info = _run_batched_observation_auxiliary_step(
            states,
            obs_aux_keys,
            observations,
            static=static,
        )
        states, latent_info = _run_batched_aux_kalman_mcmc_latent_step(
            states,
            latent_keys,
            observations,
            static=static,
        )
        states, param_info = _run_batched_aux_kalman_mcmc_parameter_step(
            states,
            param_keys,
            static=static,
        )
        if (
            step_idx == 0
            or (step_idx + 1) % progress_every == 0
            or step_idx + 1 == num_warmup
            or step_idx + 1 == total_steps
        ):
            latent_accept_now = jax.device_get(jnp.mean(latent_info["accepted"]))
            param_accept_now = jax.device_get(jnp.mean(param_info["accepted"]))
            latent_delta_now = jax.device_get(states.latent_delta)
            param_step_now = jax.device_get(states.param_step_size)
            complete_lp_now = jax.device_get(states.complete_log_posterior)
            phase = "warmup" if step_idx < num_warmup else "sample"
            elapsed = time.monotonic() - progress_started
            print(
                "aux_kalman_mcmc progress: "
                f"step={step_idx + 1}/{total_steps} phase={phase} elapsed={elapsed:.1f}s "
                f"latent_accept_now={float(latent_accept_now):.3f} "
                f"param_accept_now={float(param_accept_now):.3f} "
                f"latent_delta_range=[{float(jnp.min(latent_delta_now)):.3g},"
                f"{float(jnp.max(latent_delta_now)):.3g}] "
                f"param_step_range=[{float(jnp.min(param_step_now)):.3g},"
                f"{float(jnp.max(param_step_now)):.3g}] "
                f"complete_lp_range=[{float(jnp.min(complete_lp_now)):.3g},"
                f"{float(jnp.max(complete_lp_now)):.3g}]",
                flush=True,
            )

        if step_idx < num_warmup:
            if use_dual_averaging:
                if use_windowed_latent_adaptation:
                    states, latent_accept_window, latent_window_accept_rate = (
                        _apply_windowed_latent_update_batched(
                            states,
                            latent_info["accepted"],
                            latent_accept_window,
                            jnp.asarray(step_idx, dtype=jnp.int32),
                            static=static,
                        )
                    )
                    states = _apply_param_dual_averaging_update_batched(
                        states,
                        param_info["accepted"],
                        static=static,
                        da_param_update=da_param_update,
                        is_final_warmup=step_idx == (num_warmup - 1),
                    )
                else:
                    states = _apply_dual_averaging_update_batched(
                        states,
                        latent_info["accepted"],
                        param_info["accepted"],
                        static=static,
                        da_latent_update=da_latent_update,
                        da_param_update=da_param_update,
                        is_final_warmup=step_idx == (num_warmup - 1),
                    )
            else:
                states = _apply_simple_adaptation_update_batched(
                    states,
                    latent_info["accepted"],
                    param_info["accepted"],
                    static=static,
                )
            continue

        position_history.append(states.position)
        latent_accept_history.append(latent_info["accepted"])
        for latent_field_name in _LATENT_FIELD_NAMES:
            if latent_field_name in latent_info:
                latent_extra_history[latent_field_name].append(latent_info[latent_field_name])
        param_accept_history.append(param_info["accepted"])
        complete_lp_history.append(states.complete_log_posterior)
        for param_field_name in _PARAMETER_FIELD_NAMES:
            if param_field_name in param_info:
                parameter_extra_history[param_field_name].append(param_info[param_field_name])

        if need_public_latent:
            public_latent = _sample_public_latent_batch(
                states,
                public_latent_keys,
                observations,
                static=static,
            )
            if compute_latent_posterior_summary:
                latent_sum = latent_sum + public_latent
                latent_sumsq = latent_sumsq + public_latent * public_latent
                sample_count = sample_count + 1
            if retain_latent_paths:
                latent_paths_history.append(public_latent)

        if emit_per_t_log_alpha:
            for per_t_name in _PER_T_FIELD_NAMES:
                if per_t_name in latent_info:
                    per_t_history[per_t_name].append(latent_info[per_t_name])

    position_shape = tuple(chain_init_positions.shape[1:])
    grouped_positions = _stack_sample_history(
        position_history,
        num_chains=num_chains,
        trailing_shape=position_shape,
        dtype=chain_init_positions.dtype,
    )
    chain_extra_fields = {
        "latent_accept_prob": _stack_sample_history(
            latent_accept_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "parameter_accept_prob": _stack_sample_history(
            param_accept_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
    }
    for latent_field_name in _LATENT_FIELD_NAMES:
        if latent_extra_history[latent_field_name]:
            chain_extra_fields[latent_field_name] = _stack_sample_history(
                latent_extra_history[latent_field_name],
                num_chains=num_chains,
                trailing_shape=tuple(latent_extra_history[latent_field_name][0].shape[1:]),
                dtype=latent_extra_history[latent_field_name][0].dtype,
            )
    for per_t_name in _PER_T_FIELD_NAMES:
        if per_t_history[per_t_name]:
            chain_extra_fields[per_t_name] = _stack_sample_history(
                per_t_history[per_t_name],
                num_chains=num_chains,
                trailing_shape=tuple(per_t_history[per_t_name][0].shape[1:]),
                dtype=per_t_history[per_t_name][0].dtype,
            )
    for param_field_name in _PARAMETER_FIELD_NAMES:
        if parameter_extra_history[param_field_name]:
            chain_extra_fields[param_field_name] = _stack_sample_history(
                parameter_extra_history[param_field_name],
                num_chains=num_chains,
                trailing_shape=(),
                dtype=parameter_extra_history[param_field_name][0].dtype,
            )

    complete_log_posterior_history = _stack_sample_history(
        complete_lp_history,
        num_chains=num_chains,
        trailing_shape=(),
        dtype=states.complete_log_posterior.dtype,
    )

    latent_summary = None
    if compute_latent_posterior_summary:
        denom = jnp.maximum(sample_count, 1).astype(latent_sum.dtype)
        chain_mean = latent_sum / denom
        chain_var = jnp.maximum(latent_sumsq / denom - chain_mean * chain_mean, 0.0)
        latent_summary = _latent_summary_from_chain_moments(chain_mean, jnp.sqrt(chain_var))

    latent_paths = None
    if retain_latent_paths:
        latent_trailing_shape = (
            tuple(latent_paths_history[0].shape[1:])
            if latent_paths_history
            else tuple(
                _sample_public_latent_batch(
                    states,
                    random.split(init_key, num_chains),
                    observations,
                    static=static,
                ).shape[1:]
            )
        )
        latent_dtype = (
            latent_paths_history[0].dtype
            if latent_paths_history
            else _sample_public_latent_batch(
                states,
                random.split(init_key, num_chains),
                observations,
                static=static,
            ).dtype
        )
        latent_paths = _stack_sample_history(
            latent_paths_history,
            num_chains=num_chains,
            trailing_shape=latent_trailing_shape,
            dtype=latent_dtype,
        )

    return {
        "grouped_positions": grouped_positions,
        "chain_extra_fields": chain_extra_fields,
        "initial_param_step_size": _hostify_chain_array(initial_param_step_size),
        "param_step_size_initial_guess": _hostify_chain_array(param_step_size_initial_guess),
        "param_step_size_auto_tuned": param_step_size_auto_tuned,
        "param_step_size_tuning_accept_prob": _hostify_chain_array(
            param_step_size_tuning_info["accept_prob"]
        ),
        "param_step_size_tuning_steps": _hostify_chain_array(
            param_step_size_tuning_info["num_search_steps"]
        ),
        "param_step_size_tuning_candidate_accept_prob": _hostify_chain_array(
            param_step_size_tuning_info["candidate_accept_prob"]
        ),
        "param_step_size_tuning_previous_accept_prob": _hostify_chain_array(
            param_step_size_tuning_info["previous_accept_prob"]
        ),
        "param_step_size_tuning_selected_previous": _hostify_chain_array(
            param_step_size_tuning_info["selected_previous"]
        ),
        "param_step_size_tuning_crossed": _hostify_chain_array(
            param_step_size_tuning_info["crossed"]
        ),
        "latent_adaptation_method": (
            "windowed_multiplicative_per_time"
            if use_windowed_latent_adaptation
            else adaptation_scheme
        ),
        "latent_window_adaptation_window_size": (
            _PARTICLE_LATENT_ADAPTATION_WINDOW if use_windowed_latent_adaptation else 0
        ),
        "latent_window_acceptance_mean": float(jnp.mean(latent_window_accept_rate)),
        "latent_window_acceptance_min": float(jnp.min(latent_window_accept_rate)),
        "latent_window_acceptance_max": float(jnp.max(latent_window_accept_rate)),
        "final_latent_delta": _hostify_chain_array(states.latent_delta),
        "final_param_step_size": _hostify_chain_array(states.param_step_size),
        "latent_posterior_summary": latent_summary,
        "latent_paths": latent_paths,
        "complete_log_posterior_history": complete_log_posterior_history,
        "final_complete_log_posterior": states.complete_log_posterior,
        "post_warmup_complete_log_posterior_mean": jnp.mean(
            complete_log_posterior_history,
            axis=1,
        )
        if num_samples > 0
        else jnp.zeros((num_chains,), dtype=states.complete_log_posterior.dtype),
    }
