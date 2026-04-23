"""Blocked Gibbs driver: eq-8 auxiliary-Kalman latent + MALA parameter updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)


class AuxGibbsState(NamedTuple):
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
        clipped = jnp.maximum(clipped, jnp.asarray(min_scale, dtype=clipped.dtype))
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
    adaptation_scheme: str = "dual_averaging",
    init_positions: jnp.ndarray | None = None,
    emit_per_t_log_alpha: bool = False,
) -> dict[str, Any]:
    """Blocked Gibbs: eq-8 aux-Kalman latent step + MALA parameter step.

    Each block has its own accept signal so ``latent_delta`` and
    ``param_step_size`` adapt against their own target acceptances, avoiding
    the degenerate "both scales race to zero" corner that a joint-MH shared
    adaptation falls into.

    Parameters
    ----------
    adaptation_scheme:
        ``"simple"`` (default) — the original exponential update on the raw
        per-step binary accept. ``"dual_averaging"`` — Hoffman & Gelman (2014)
        dual averaging on log step size, with √t damping and a decaying
        running-mean weight. Dual averaging converges in ~100 steps where the
        simple scheme takes ~2000.
    init_positions:
        Optional ``(num_chains, dim)`` array of unconstrained parameter
        positions. When ``None`` the sampler perturbs ``bundle["flat_example"]``
        with ``init_scale·randn``; when provided (e.g. from Pathfinder) the
        chains start exactly from those positions.
    """
    total_steps = num_warmup + num_samples
    if total_steps <= 0:
        raise ValueError("aux_gibbs requires at least one warmup or posterior draw step.")
    if adaptation_scheme not in {"simple", "dual_averaging"}:
        raise ValueError(
            f"Unknown adaptation_scheme {adaptation_scheme!r}; expected 'simple' or 'dual_averaging'."
        )
    project_latent_trajectory = bundle.get("project_latent_trajectory_fn", lambda x: x)
    latent_step_fn = latent_kernel["step_fn"]
    parameter_step_fn = parameter_kernel["step_fn"]
    latent_target_accept = latent_kernel["target_accept"]
    param_target_accept = parameter_kernel["target_accept"]
    initial_latent_scale = latent_kernel["initial_scale"]
    initial_param_scale = float(parameter_kernel["initial_scale"])
    use_dual_averaging = adaptation_scheme == "dual_averaging"
    latent_scale_init_from_latent = latent_kernel.get("initial_scale_from_latent_fn")
    latent_min_scale = latent_kernel.get("min_scale")
    latent_max_scale = latent_kernel.get("max_scale")
    if (
        latent_min_scale is not None
        and latent_max_scale is not None
        and float(latent_min_scale) > float(latent_max_scale)
    ):
        raise ValueError(
            "latent_kernel min_scale must be <= max_scale; got "
            f"{latent_min_scale} > {latent_max_scale}."
        )

    # BlackJAX dual-averaging primitives (Hoffman & Gelman 2014).
    da_latent_init, da_latent_update, _ = dual_averaging_adaptation(
        target=float(latent_target_accept)
    )
    da_param_init, da_param_update, _ = dual_averaging_adaptation(target=float(param_target_accept))

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
        scale_dtype = init_latent.dtype
        if latent_scale_init_from_latent is None:
            init_latent_delta = jnp.asarray(initial_latent_scale, dtype=scale_dtype)
        else:
            init_latent_delta = jnp.asarray(
                latent_scale_init_from_latent(init_latent, scale_dtype),
                dtype=scale_dtype,
            )
        init_latent_delta = _clip_scale(
            init_latent_delta,
            min_scale=latent_min_scale,
            max_scale=latent_max_scale,
        )
        init_state = AuxGibbsState(
            position=init_position,
            latent_context=init_context,
            latent_trajectory=init_latent,
            trajectory_log_prob=init_traj,
            complete_log_posterior=init_complete,
            latent_delta=init_latent_delta,
            param_step_size=jnp.asarray(initial_param_scale, dtype=scale_dtype),
            # BlackJAX's dual averaging computes log/exp which promote to
            # float64; the normalization target must therefore be float64 so
            # the scan carry dtype matches the update output.
            latent_da=_normalize_dual_averaging_state_shape(
                da_latent_init(init_latent_delta),
                jnp.zeros(init_latent_delta.shape, dtype=jnp.float64),
            ),
            param_da=da_param_init(initial_param_scale),
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
            state_after_param, param_info = parameter_step_fn(state_after_latent, param_key)
            warmup = iter_idx < num_warmup
            is_final_warmup = iter_idx == (num_warmup - 1)

            if use_dual_averaging:
                updated_latent_da = _clip_dual_averaging_state(
                    _normalize_dual_averaging_state_shape(
                        da_latent_update(
                            state_after_param.latent_da,
                            latent_info["accepted"],
                        ),
                        state_after_param.latent_delta,
                    ),
                    min_scale=latent_min_scale,
                    max_scale=latent_max_scale,
                )
                updated_param_da = da_param_update(
                    state_after_param.param_da, param_info["accepted"]
                )
                # During warmup use the current (non-averaged) log-δ for
                # exploration; at the final warmup step, freeze the scale at
                # the running mean log-δ̄. Post-warmup: scale stays frozen.
                # BlackJAX's dual averaging state is float64; cast scales
                # back to the chain's scale dtype so the scan carry is stable.
                latent_live = jnp.exp(updated_latent_da.log_step_size).astype(scale_dtype)
                latent_frozen = jnp.exp(updated_latent_da.log_step_size_avg).astype(scale_dtype)
                param_live = jnp.exp(updated_param_da.log_step_size).astype(scale_dtype)
                param_frozen = jnp.exp(updated_param_da.log_step_size_avg).astype(scale_dtype)
                latent_live = _clip_scale(
                    latent_live,
                    min_scale=latent_min_scale,
                    max_scale=latent_max_scale,
                )
                latent_frozen = _clip_scale(
                    latent_frozen,
                    min_scale=latent_min_scale,
                    max_scale=latent_max_scale,
                )
                next_latent_delta = jnp.where(
                    warmup,
                    jnp.where(is_final_warmup, latent_frozen, latent_live),
                    state_after_param.latent_delta,
                )
                next_param_step = jnp.where(
                    warmup,
                    jnp.where(is_final_warmup, param_frozen, param_live),
                    state_after_param.param_step_size,
                )
                next_latent_da = jax.tree_util.tree_map(
                    lambda new_leaf, old_leaf: jnp.where(warmup, new_leaf, old_leaf),
                    updated_latent_da,
                    state_after_param.latent_da,
                )
                next_param_da = jax.tree_util.tree_map(
                    lambda new_leaf, old_leaf: jnp.where(warmup, new_leaf, old_leaf),
                    updated_param_da,
                    state_after_param.param_da,
                )
            else:
                next_latent_delta = jax.lax.cond(
                    warmup,
                    lambda _: _adapt_scale(
                        state_after_param.latent_delta,
                        accepted=latent_info["accepted"],
                        target_accept=latent_target_accept,
                        adaptation_rate=adaptation_rate,
                        min_scale=1e-6 if latent_min_scale is None else float(latent_min_scale),
                        max_scale=1e3 if latent_max_scale is None else float(latent_max_scale),
                    ),
                    lambda _: state_after_param.latent_delta,
                    operand=None,
                )
                next_param_step = jax.lax.cond(
                    warmup,
                    lambda _: _adapt_scale(
                        state_after_param.param_step_size,
                        accepted=param_info["accepted"],
                        target_accept=param_target_accept,
                        adaptation_rate=adaptation_rate,
                    ),
                    lambda _: state_after_param.param_step_size,
                    operand=None,
                )
                next_latent_da = state_after_param.latent_da
                next_param_da = state_after_param.param_da

            next_state = state_after_param._replace(
                latent_delta=next_latent_delta,
                param_step_size=next_param_step,
                latent_da=next_latent_da,
                param_da=next_param_da,
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
                next_state.complete_log_posterior,
            )
            if retain_latent_paths:
                history_entry = (*history_entry, next_latent_public)
            if emit_per_t_log_alpha:
                history_entry = (
                    *history_entry,
                    latent_info["log_alpha_per_t"],
                    latent_info["log_alpha_obs_per_t"],
                    latent_info["log_alpha_fwd_minus_rev_per_t"],
                    latent_info["log_alpha_q_per_t"],
                    latent_info["log_alpha"],
                )
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
        per_t_names = (
            "log_alpha_per_t",
            "log_alpha_obs_per_t",
            "log_alpha_fwd_minus_rev_per_t",
            "log_alpha_q_per_t",
            "log_alpha",
        )
        if retain_latent_paths and emit_per_t_log_alpha:
            (
                positions,
                latent_accept,
                param_accept,
                latent_delta_hist,
                param_step_hist,
                complete_lp_hist,
                latent_hist,
                *per_t_fields,
            ) = history
        elif retain_latent_paths:
            (
                positions,
                latent_accept,
                param_accept,
                latent_delta_hist,
                param_step_hist,
                complete_lp_hist,
                latent_hist,
            ) = history
            per_t_fields = []
        elif emit_per_t_log_alpha:
            (
                positions,
                latent_accept,
                param_accept,
                latent_delta_hist,
                param_step_hist,
                complete_lp_hist,
                *per_t_fields,
            ) = history
            latent_hist = None
        else:
            (
                positions,
                latent_accept,
                param_accept,
                latent_delta_hist,
                param_step_hist,
                complete_lp_hist,
            ) = history
            latent_hist = None
            per_t_fields = []
        per_t_history = dict(zip(per_t_names, per_t_fields, strict=True)) if per_t_fields else {}

        denom = jnp.maximum(post_count, 1).astype(init_latent.dtype)
        chain_mean = latent_sum / denom
        chain_var = jnp.maximum(latent_sumsq / denom - chain_mean * chain_mean, 0.0)
        chain_result = {
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
            "complete_log_posterior_history": complete_lp_hist,
            "final_complete_log_posterior": final_state.complete_log_posterior,
            "post_warmup_complete_log_posterior_mean": jnp.mean(complete_lp_hist[num_warmup:]),
        }
        if per_t_history:
            for name, arr in per_t_history.items():
                chain_result[name] = arr[num_warmup:]
        return chain_result

    base_key = random.PRNGKey(seed)
    init_key, chain_key = random.split(base_key)
    chain_keys = random.split(chain_key, num_chains)
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

    run_chains_jit = jax.jit(jax.vmap(_run_chain, in_axes=(0, 0)))
    chain_results = run_chains_jit(chain_init_positions, chain_keys)

    grouped_positions = chain_results["positions"]
    chain_extra_fields = {
        "latent_accept_prob": chain_results["latent_accept"],
        "parameter_accept_prob": chain_results["param_accept"],
    }
    for per_t_name in (
        "log_alpha_per_t",
        "log_alpha_obs_per_t",
        "log_alpha_fwd_minus_rev_per_t",
        "log_alpha_q_per_t",
        "log_alpha",
    ):
        if per_t_name in chain_results:
            chain_extra_fields[per_t_name] = chain_results[per_t_name]
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
        "final_latent_delta": _hostify_chain_array(chain_results["final_latent_delta"]),
        "final_param_step_size": _hostify_chain_array(chain_results["final_param_step_size"]),
        "latent_posterior_summary": latent_summary,
        "latent_paths": latent_paths,
        "complete_log_posterior_history": chain_results["complete_log_posterior_history"],
        "final_complete_log_posterior": chain_results["final_complete_log_posterior"],
        "post_warmup_complete_log_posterior_mean": chain_results[
            "post_warmup_complete_log_posterior_mean"
        ],
    }
