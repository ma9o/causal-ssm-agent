"""Blocked MCMC driver: latent trajectory updates + parameter kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
)

_DEFAULT_MIN_SCALE = 1e-6
_DEFAULT_MAX_SCALE = 1e3
_MAX_INITIAL_PARAM_STEP_SIZE_ITERS = 20
_PARTICLE_LATENT_ADAPTATION_WINDOW = 100
_PARTICLE_LATENT_ADAPTATION_TOLERANCE = 0.05
_PARTICLE_LATENT_ADAPTATION_RHO = 0.5
_PARTICLE_LATENT_ADAPTATION_GAMMA = -0.5
_PARTICLE_LATENT_ADAPTATION_MIN_RATE = 1e-3


class TrajectoryMCMCState(NamedTuple):
    position: jnp.ndarray
    latent_context: Any
    latent_trajectory: jnp.ndarray
    trajectory_log_prob: jnp.ndarray
    complete_log_posterior: jnp.ndarray
    latent_delta: jnp.ndarray
    param_step_size: jnp.ndarray
    # BlackJAX dual-averaging state. Carried but not updated when
    # adaptation_scheme == "simple"; evolves only during warmup otherwise.
    latent_da: DualAveragingAdaptationState
    param_da: DualAveragingAdaptationState


@dataclass(frozen=True)
class TrajectoryMCMCResult:
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


def _stack_chain_states(states: list[TrajectoryMCMCState]) -> TrajectoryMCMCState:
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values, axis=0), *states)


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
