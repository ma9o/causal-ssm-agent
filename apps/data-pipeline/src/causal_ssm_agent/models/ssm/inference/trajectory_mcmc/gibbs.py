"""Blocked Gibbs driver: eq-8 auxiliary-Kalman latent + MALA parameter updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random


class AuxGibbsState(NamedTuple):
    position: jnp.ndarray
    latent_context: Any
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
    """Blocked Gibbs: eq-8 aux-Kalman latent step + MALA parameter step.

    Each block has its own accept signal so ``latent_delta`` and
    ``param_step_size`` adapt against their own target acceptances, avoiding
    the degenerate "both scales race to zero" corner that a joint-MH shared
    adaptation falls into.
    """
    total_steps = num_warmup + num_samples
    if total_steps <= 0:
        raise ValueError("aux_gibbs requires at least one warmup or posterior draw step.")
    project_latent_trajectory = bundle.get("project_latent_trajectory_fn", lambda x: x)
    latent_step_fn = latent_kernel["step_fn"]
    parameter_step_fn = parameter_kernel["step_fn"]

    def _run_chain(
        init_position: jnp.ndarray,
        chain_key: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        init_context = bundle["latent_context_fn"](init_position)
        init_latent = bundle["initial_latent_from_context_fn"](init_context)
        init_latent_public = project_latent_trajectory(init_latent)
        init_complete, init_traj = bundle["complete_log_posterior_from_context_fn"](
            init_position,
            init_context,
            init_latent,
        )
        init_state = AuxGibbsState(
            position=init_position,
            latent_context=init_context,
            latent_trajectory=init_latent,
            trajectory_log_prob=init_traj,
            complete_log_posterior=init_complete,
            latent_delta=jnp.asarray(
                latent_kernel["initial_scale"], dtype=init_latent.dtype
            ),
            param_step_size=jnp.asarray(
                parameter_kernel["initial_scale"], dtype=init_latent.dtype
            ),
        )
        latent_sum0 = jnp.zeros_like(init_latent_public)
        latent_sumsq0 = jnp.zeros_like(init_latent_public)
        post_count0 = jnp.asarray(0, dtype=jnp.int32)
        step_keys = random.split(chain_key, total_steps)
        step_idx = jnp.arange(total_steps, dtype=jnp.int32)

        def _scan_step(carry, inputs):
            state, latent_sum, latent_sumsq, post_count = carry
            iter_idx, step_key = inputs
            latent_key, param_key = random.split(step_key)
            state_after_latent, latent_info = latent_step_fn(state, latent_key)
            state_after_param, param_info = parameter_step_fn(
                state_after_latent, param_key
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
            next_latent_public = project_latent_trajectory(next_state.latent_trajectory)
            keep = (iter_idx >= num_warmup).astype(next_state.latent_trajectory.dtype)
            next_latent_sum = latent_sum + keep * next_latent_public
            next_latent_sumsq = latent_sumsq + keep * (next_latent_public * next_latent_public)
            next_post_count = post_count + (iter_idx >= num_warmup).astype(jnp.int32)
            history_entry = (
                next_state.position,
                latent_info["accepted"],
                param_info["accepted"],
                next_state.latent_delta,
                next_state.param_step_size,
            )
            if retain_latent_paths:
                history_entry = (*history_entry, next_latent_public)
            return (
                next_state,
                next_latent_sum,
                next_latent_sumsq,
                next_post_count,
            ), history_entry

        final_carry, history = jax.lax.scan(
            _scan_step,
            (init_state, latent_sum0, latent_sumsq0, post_count0),
            (step_idx, step_keys),
        )
        final_state, latent_sum, latent_sumsq, post_count = final_carry
        if retain_latent_paths:
            (
                positions,
                latent_accept,
                param_accept,
                latent_delta_hist,
                param_step_hist,
                latent_hist,
            ) = history
        else:
            (
                positions,
                latent_accept,
                param_accept,
                latent_delta_hist,
                param_step_hist,
            ) = history
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

    base_key = random.PRNGKey(seed)
    init_key, chain_key = random.split(base_key)
    init_keys = random.split(init_key, num_chains)
    chain_keys = random.split(chain_key, num_chains)
    init_noise = jax.vmap(
        lambda key: random.normal(
            key,
            bundle["flat_example"].shape,
            dtype=bundle["flat_example"].dtype,
        )
    )(init_keys)
    init_positions = bundle["flat_example"][None, ...] + init_scale * init_noise

    run_chains_jit = jax.jit(jax.vmap(_run_chain, in_axes=(0, 0)))
    chain_results = run_chains_jit(init_positions, chain_keys)

    grouped_positions = chain_results["positions"]
    chain_extra_fields = {
        "latent_accept_prob": chain_results["latent_accept"],
        "parameter_accept_prob": chain_results["param_accept"],
    }
    latent_summary = _latent_summary_from_chain_moments(
        chain_results["latent_mean"],
        chain_results["latent_std"],
    )
    latent_paths = None
    if retain_latent_paths:
        latent_paths = chain_results["latent_paths"]

    return {
        "grouped_positions": grouped_positions,
        "chain_extra_fields": chain_extra_fields,
        "final_latent_delta": [float(value) for value in chain_results["final_latent_delta"]],
        "final_param_step_size": [float(value) for value in chain_results["final_param_step_size"]],
        "latent_posterior_summary": latent_summary,
        "latent_paths": latent_paths,
    }
