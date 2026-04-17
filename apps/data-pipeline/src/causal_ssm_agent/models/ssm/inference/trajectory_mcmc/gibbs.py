"""Generic blocked trajectory-MCMC runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random


class AuxGibbsState(NamedTuple):
    position: jnp.ndarray
    latent_trajectory: jnp.ndarray
    trajectory_log_prob: jnp.ndarray
    complete_log_posterior: jnp.ndarray
    latent_delta: jnp.ndarray
    param_step_size: jnp.ndarray


@dataclass(frozen=True)
class AuxGibbsMCMCResult:
    """Minimal MCMC-compatible wrapper for auxiliary Gibbs outputs."""

    chain_samples: dict[str, jnp.ndarray]
    chain_extra_fields: dict[str, jnp.ndarray]
    num_chains: int
    num_samples: int
    backend: str = "aux_gibbs"

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


def run_aux_gibbs(
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
) -> dict[str, Any]:
    """Run blocked auxiliary Gibbs updates for parameters and trajectories."""
    total_steps = num_warmup + num_samples
    if total_steps <= 0:
        raise ValueError("aux_gibbs requires at least one warmup or posterior draw step.")

    def _run_chain(
        init_position: jnp.ndarray,
        chain_key: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        init_latent = bundle["initial_latent_fn"](init_position)
        init_complete, init_traj = bundle["complete_log_posterior_with_aux_fn"](
            init_position,
            init_latent,
        )
        init_state = AuxGibbsState(
            position=init_position,
            latent_trajectory=init_latent,
            trajectory_log_prob=init_traj,
            complete_log_posterior=init_complete,
            latent_delta=jnp.asarray(latent_kernel["initial_scale"], dtype=init_latent.dtype),
            param_step_size=jnp.asarray(parameter_kernel["initial_scale"], dtype=init_latent.dtype),
        )
        latent_sum0 = jnp.zeros_like(init_latent)
        latent_sumsq0 = jnp.zeros_like(init_latent)
        post_count0 = jnp.asarray(0, dtype=jnp.int32)
        step_keys = random.split(chain_key, total_steps)
        step_idx = jnp.arange(total_steps, dtype=jnp.int32)

        if retain_latent_paths:

            def _scan_step(carry, inputs):
                state, latent_sum, latent_sumsq, post_count = carry
                iter_idx, step_key = inputs
                latent_key, param_key = random.split(step_key)
                state_after_latent, latent_info = latent_kernel["step_fn"](state, latent_key)
                state_after_param, param_info = parameter_kernel["step_fn"](
                    state_after_latent,
                    param_key,
                )
                warmup = iter_idx < num_warmup
                next_latent_delta = jax.lax.cond(
                    warmup,
                    lambda _: _adapt_scale(
                        state_after_param.latent_delta,
                        accepted=latent_info["accepted"],
                        target_accept=latent_kernel["target_accept"],
                        adaptation_rate=adaptation_rate,
                    ),
                    lambda _: state_after_param.latent_delta,
                    operand=None,
                )
                next_param_step = jax.lax.cond(
                    warmup,
                    lambda _: _adapt_scale(
                        state_after_param.param_step_size,
                        accepted=param_info["accepted"],
                        target_accept=parameter_kernel["target_accept"],
                        adaptation_rate=adaptation_rate,
                    ),
                    lambda _: state_after_param.param_step_size,
                    operand=None,
                )
                next_state = state_after_param._replace(
                    latent_delta=next_latent_delta,
                    param_step_size=next_param_step,
                )
                keep = (iter_idx >= num_warmup).astype(next_state.latent_trajectory.dtype)
                next_latent_sum = latent_sum + keep * next_state.latent_trajectory
                next_latent_sumsq = latent_sumsq + keep * (
                    next_state.latent_trajectory * next_state.latent_trajectory
                )
                next_post_count = post_count + (iter_idx >= num_warmup).astype(jnp.int32)
                return (
                    next_state,
                    next_latent_sum,
                    next_latent_sumsq,
                    next_post_count,
                ), (
                    next_state.position,
                    latent_info["accepted"],
                    param_info["accepted"],
                    next_state.latent_delta,
                    next_state.param_step_size,
                    next_state.latent_trajectory,
                )

            final_carry, history = jax.lax.scan(
                _scan_step,
                (init_state, latent_sum0, latent_sumsq0, post_count0),
                (step_idx, step_keys),
            )
            final_state, latent_sum, latent_sumsq, post_count = final_carry
            (
                positions,
                latent_accept,
                param_accept,
                latent_delta_hist,
                param_step_hist,
                latent_hist,
            ) = history
        else:

            def _scan_step(carry, inputs):
                state, latent_sum, latent_sumsq, post_count = carry
                iter_idx, step_key = inputs
                latent_key, param_key = random.split(step_key)
                state_after_latent, latent_info = latent_kernel["step_fn"](state, latent_key)
                state_after_param, param_info = parameter_kernel["step_fn"](
                    state_after_latent,
                    param_key,
                )
                warmup = iter_idx < num_warmup
                next_latent_delta = jax.lax.cond(
                    warmup,
                    lambda _: _adapt_scale(
                        state_after_param.latent_delta,
                        accepted=latent_info["accepted"],
                        target_accept=latent_kernel["target_accept"],
                        adaptation_rate=adaptation_rate,
                    ),
                    lambda _: state_after_param.latent_delta,
                    operand=None,
                )
                next_param_step = jax.lax.cond(
                    warmup,
                    lambda _: _adapt_scale(
                        state_after_param.param_step_size,
                        accepted=param_info["accepted"],
                        target_accept=parameter_kernel["target_accept"],
                        adaptation_rate=adaptation_rate,
                    ),
                    lambda _: state_after_param.param_step_size,
                    operand=None,
                )
                next_state = state_after_param._replace(
                    latent_delta=next_latent_delta,
                    param_step_size=next_param_step,
                )
                keep = (iter_idx >= num_warmup).astype(next_state.latent_trajectory.dtype)
                next_latent_sum = latent_sum + keep * next_state.latent_trajectory
                next_latent_sumsq = latent_sumsq + keep * (
                    next_state.latent_trajectory * next_state.latent_trajectory
                )
                next_post_count = post_count + (iter_idx >= num_warmup).astype(jnp.int32)
                return (
                    next_state,
                    next_latent_sum,
                    next_latent_sumsq,
                    next_post_count,
                ), (
                    next_state.position,
                    latent_info["accepted"],
                    param_info["accepted"],
                    next_state.latent_delta,
                    next_state.param_step_size,
                )

            final_carry, history = jax.lax.scan(
                _scan_step,
                (init_state, latent_sum0, latent_sumsq0, post_count0),
                (step_idx, step_keys),
            )
            final_state, latent_sum, latent_sumsq, post_count = final_carry
            positions, latent_accept, param_accept, latent_delta_hist, param_step_hist = history
            latent_hist = None

        denom = jnp.maximum(post_count, 1).astype(init_latent.dtype)
        chain_mean = latent_sum / denom
        chain_var = jnp.maximum(latent_sumsq / denom - chain_mean * chain_mean, 0.0)
        return {
            "positions": positions[num_warmup:],
            "latent_accept": latent_accept[num_warmup:],
            "param_accept": param_accept[num_warmup:],
            "latent_delta_history": latent_delta_hist,
            "param_step_size_history": param_step_hist,
            "final_latent_delta": final_state.latent_delta,
            "final_param_step_size": final_state.param_step_size,
            "latent_mean": chain_mean,
            "latent_std": jnp.sqrt(chain_var),
            "latent_paths": None if latent_hist is None else latent_hist[num_warmup:],
        }

    run_chain_jit = jax.jit(_run_chain)
    base_key = random.PRNGKey(seed)
    init_key, chain_key = random.split(base_key)
    init_keys = random.split(init_key, num_chains)
    chain_keys = random.split(chain_key, num_chains)

    chain_results: list[dict[str, jnp.ndarray]] = []
    for chain_idx in range(num_chains):
        if bundle["dim"] == 0:
            init_position = bundle["flat_example"]
        else:
            init_position = bundle["flat_example"] + init_scale * random.normal(
                init_keys[chain_idx],
                bundle["flat_example"].shape,
                dtype=bundle["flat_example"].dtype,
            )
        chain_results.append(run_chain_jit(init_position, chain_keys[chain_idx]))

    grouped_positions = jnp.stack([result["positions"] for result in chain_results], axis=0)
    chain_extra_fields = {
        "latent_accept_prob": jnp.stack(
            [result["latent_accept"] for result in chain_results], axis=0
        ),
        "parameter_accept_prob": jnp.stack(
            [result["param_accept"] for result in chain_results],
            axis=0,
        ),
    }
    latent_summary = _latent_summary_from_chain_moments(
        jnp.stack([result["latent_mean"] for result in chain_results], axis=0),
        jnp.stack([result["latent_std"] for result in chain_results], axis=0),
    )
    latent_paths = None
    if retain_latent_paths:
        latent_paths = jnp.stack([result["latent_paths"] for result in chain_results], axis=0)

    return {
        "grouped_positions": grouped_positions,
        "chain_extra_fields": chain_extra_fields,
        "final_latent_delta": [float(result["final_latent_delta"]) for result in chain_results],
        "final_param_step_size": [
            float(result["final_param_step_size"]) for result in chain_results
        ],
        "latent_posterior_summary": latent_summary,
        "latent_paths": latent_paths,
    }
