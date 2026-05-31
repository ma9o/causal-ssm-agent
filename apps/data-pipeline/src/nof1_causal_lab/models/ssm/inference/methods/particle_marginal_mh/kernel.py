"""Particle marginal Metropolis-Hastings kernel."""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
from blackjax.adaptation.step_size import dual_averaging_adaptation

from nof1_causal_lab.models.ssm.inference.mcmc_state import (
    _adapt_scale,
    _clip_dual_averaging_state,
    _clip_scale,
    _stack_sample_history,
)
from nof1_causal_lab.models.ssm.inference.methods._pmcmc_shared import (
    parameter_jump_rms,
    preconditioned_random_walk_proposal,
)
from nof1_causal_lab.models.ssm.inference.methods.particle_marginal_mh.particle_filter import (
    estimate_bootstrap_log_likelihood,
)

_DEFAULT_MIN_SCALE = 1e-6
_DEFAULT_MAX_SCALE = 1e3
_DEFAULT_TARGET_ACCEPT = 0.234


class PMMHState(NamedTuple):
    position: jnp.ndarray
    log_prior: jnp.ndarray
    estimated_log_likelihood: jnp.ndarray
    estimated_log_posterior: jnp.ndarray
    param_step_size: jnp.ndarray
    param_da: Any


@dataclass(frozen=True)
class PMMHKernel:
    step_fn: Any
    num_particles: int
    initial_param_step_size: float
    target_accept: float
    min_scale: float
    max_scale: float
    preconditioned: bool
    diagnostic_metrics: frozenset[str]


def _has_metric(metrics: frozenset[str], name: str) -> bool:
    return name in metrics


def build_particle_marginal_mh_kernel(
    bundle: dict[str, Any],
    *,
    num_particles: int,
    param_step_size: float,
    target_accept: float | None = None,
    min_scale: float = _DEFAULT_MIN_SCALE,
    max_scale: float = _DEFAULT_MAX_SCALE,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    diagnostic_metrics: tuple[str, ...] | list[str] | None = None,
    diagnostic_metrics_all: bool = False,
) -> PMMHKernel:
    if num_particles < 2:
        raise ValueError("particle_marginal_mh requires num_particles >= 2.")
    if min_scale > max_scale:
        raise ValueError(
            f"particle_marginal_mh min_scale must be <= max_scale; got {min_scale} > {max_scale}."
        )
    if param_step_size <= 0.0:
        raise ValueError(
            f"particle_marginal_mh param_step_size must be positive; got {param_step_size}."
        )
    if target_accept is None:
        target_accept = _DEFAULT_TARGET_ACCEPT
    if not (0.0 < target_accept < 1.0):
        raise ValueError(
            f"particle_marginal_mh target_accept must be in (0, 1); got {target_accept}."
        )
    metrics = frozenset(diagnostic_metrics or ())
    if diagnostic_metrics_all:
        metrics = frozenset({"parameter_movement", "particle_filter", "likelihood_noise"})
    preconditioner = (
        None
        if parameter_preconditioner_chol is None
        else jnp.asarray(parameter_preconditioner_chol, dtype=bundle["flat_example"].dtype)
    )
    log_prior_unc_fn = bundle["log_prior_unc_fn"]
    return_particle_diagnostics = _has_metric(metrics, "particle_filter")

    def _estimate(key: jnp.ndarray, position: jnp.ndarray):
        return estimate_bootstrap_log_likelihood(
            key,
            position,
            bundle=bundle,
            num_particles=num_particles,
            return_particle_diagnostics=return_particle_diagnostics,
        )

    def _step_fn(state: PMMHState, key: jnp.ndarray):
        proposal_key, pf_key, accept_key = random.split(key, 3)
        step_size = _clip_scale(
            state.param_step_size,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        proposed_position = preconditioned_random_walk_proposal(
            proposal_key,
            state.position,
            step_size,
            parameter_preconditioner_chol=preconditioner,
            variance_factor=4.0,
        )
        proposed_log_prior = log_prior_unc_fn(proposed_position)
        proposed_log_likelihood, pf_diagnostics = _estimate(pf_key, proposed_position)
        proposed_log_posterior = proposed_log_prior + proposed_log_likelihood
        log_alpha = proposed_log_posterior - state.estimated_log_posterior
        log_uniform = jnp.log(random.uniform(accept_key, dtype=state.position.dtype))
        accepted = (log_uniform < log_alpha).astype(state.position.dtype)

        def _select(proposed, current):
            return jnp.where(accepted.astype(bool), proposed, current)

        next_position = _select(proposed_position, state.position)
        next_log_prior = _select(proposed_log_prior, state.log_prior)
        next_log_likelihood = _select(proposed_log_likelihood, state.estimated_log_likelihood)
        next_log_posterior = _select(proposed_log_posterior, state.estimated_log_posterior)
        next_state = state._replace(
            position=next_position,
            log_prior=next_log_prior,
            estimated_log_likelihood=next_log_likelihood,
            estimated_log_posterior=next_log_posterior,
        )
        step_info = {
            "parameter_accepted": accepted,
            "estimated_log_likelihood": next_log_likelihood,
            "log_prior": next_log_prior,
            "estimated_log_posterior": next_log_posterior,
            "log_alpha": log_alpha,
            "proposed_estimated_log_likelihood": proposed_log_likelihood,
            "pf_ess_min": pf_diagnostics["pf_ess_min"].astype(state.position.dtype),
            "pf_ess_mean": pf_diagnostics["pf_ess_mean"].astype(state.position.dtype),
            "pf_log_weight_range_max": pf_diagnostics["pf_log_weight_range_max"].astype(
                state.position.dtype
            ),
            "pf_log_weight_variance_mean": pf_diagnostics["pf_log_weight_variance_mean"].astype(
                state.position.dtype
            ),
            "pf_log_likelihood_increment_variance": pf_diagnostics[
                "pf_log_likelihood_increment_variance"
            ].astype(state.position.dtype),
        }
        if _has_metric(metrics, "parameter_movement"):
            step_info["parameter_jump_rms"] = parameter_jump_rms(next_position, state.position)
        if return_particle_diagnostics:
            step_info.update(
                {
                    "pf_ess_by_t": pf_diagnostics["pf_ess_by_t"].astype(state.position.dtype),
                    "pf_log_weight_range_by_t": pf_diagnostics["pf_log_weight_range_by_t"].astype(
                        state.position.dtype
                    ),
                    "pf_log_weight_variance_by_t": pf_diagnostics[
                        "pf_log_weight_variance_by_t"
                    ].astype(state.position.dtype),
                    "pf_log_likelihood_increment_by_t": pf_diagnostics[
                        "pf_log_likelihood_increment_by_t"
                    ].astype(state.position.dtype),
                }
            )
        return next_state, step_info

    return PMMHKernel(
        step_fn=_step_fn,
        num_particles=num_particles,
        initial_param_step_size=float(param_step_size),
        target_accept=float(target_accept),
        min_scale=float(min_scale),
        max_scale=float(max_scale),
        preconditioned=parameter_preconditioner_chol is not None,
        diagnostic_metrics=metrics,
    )


def _stack_chain_states(states: list[PMMHState]) -> PMMHState:
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values, axis=0), *states)


@functools.partial(jax.jit, static_argnames=("step_fn",))
def _run_batched_step(
    states: PMMHState,
    step_keys: jnp.ndarray,
    *,
    step_fn,
) -> tuple[PMMHState, dict[str, jnp.ndarray]]:
    return jax.vmap(lambda state, key: step_fn(state, key))(states, step_keys)


def _initialize_chain_state(
    init_position: jnp.ndarray,
    key: jnp.ndarray,
    *,
    bundle: dict[str, Any],
    num_particles: int,
    param_step_size: float,
    param_min_scale: float,
    param_max_scale: float,
    param_target_accept: float,
) -> PMMHState:
    log_prior = bundle["log_prior_unc_fn"](init_position)
    estimated_log_likelihood, _pf_diagnostics = estimate_bootstrap_log_likelihood(
        key,
        init_position,
        bundle=bundle,
        num_particles=num_particles,
        return_particle_diagnostics=False,
    )
    estimated_log_posterior = log_prior + estimated_log_likelihood
    param_step_value = _clip_scale(
        jnp.asarray(param_step_size, dtype=init_position.dtype),
        min_scale=param_min_scale,
        max_scale=param_max_scale,
    )
    da_init, _da_update, _ = dual_averaging_adaptation(target=float(param_target_accept))
    return PMMHState(
        position=init_position,
        log_prior=log_prior,
        estimated_log_likelihood=estimated_log_likelihood,
        estimated_log_posterior=estimated_log_posterior,
        param_step_size=param_step_value,
        param_da=da_init(param_step_value),
    )


def run_particle_marginal_mh(
    bundle: dict[str, Any],
    *,
    kernel: PMMHKernel,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    adaptation_rate: float,
    init_scale: float,
    init_positions: jnp.ndarray | None = None,
    adaptation_scheme: str = "simple",
) -> dict[str, Any]:
    if adaptation_scheme not in {"simple", "dual_averaging"}:
        raise ValueError(
            f"Unknown adaptation_scheme {adaptation_scheme!r}; expected 'simple' or 'dual_averaging'."
        )
    total_steps = int(num_warmup + num_samples)
    if total_steps <= 0:
        raise ValueError("particle_marginal_mh requires at least one MCMC step.")
    dim = int(bundle["flat_example"].shape[0])
    base_key = random.PRNGKey(seed)
    init_key, chain_key = random.split(base_key)
    if init_positions is None:
        init_keys = random.split(init_key, num_chains)
        init_noise = jax.vmap(
            lambda key: random.normal(
                key,
                bundle["flat_example"].shape,
                dtype=bundle["flat_example"].dtype,
            )
        )(init_keys)
        chain_init_positions = bundle["flat_example"][None, :] + init_scale * init_noise
    else:
        chain_init_positions = jnp.asarray(init_positions, dtype=bundle["flat_example"].dtype)
        if chain_init_positions.shape != (num_chains, dim):
            raise ValueError(
                "init_positions must have shape (num_chains, dim); got "
                f"{chain_init_positions.shape} with num_chains={num_chains} and dim={dim}."
            )

    init_pf_keys = random.split(init_key, num_chains)
    states = _stack_chain_states(
        [
            _initialize_chain_state(
                chain_init_positions[chain_idx],
                init_pf_keys[chain_idx],
                bundle=bundle,
                num_particles=kernel.num_particles,
                param_step_size=kernel.initial_param_step_size,
                param_min_scale=kernel.min_scale,
                param_max_scale=kernel.max_scale,
                param_target_accept=kernel.target_accept,
            )
            for chain_idx in range(num_chains)
        ]
    )
    initial_param_step_size = states.param_step_size
    use_dual_averaging = adaptation_scheme == "dual_averaging"
    da_param_update = (
        dual_averaging_adaptation(target=float(kernel.target_accept))[1]
        if use_dual_averaging
        else None
    )
    step_keys = random.split(chain_key, total_steps * num_chains).reshape(
        total_steps, num_chains, 2
    )

    position_history: list[jnp.ndarray] = []
    parameter_accept_history: list[jnp.ndarray] = []
    estimated_log_likelihood_history: list[jnp.ndarray] = []
    log_prior_history: list[jnp.ndarray] = []
    estimated_log_posterior_history: list[jnp.ndarray] = []
    log_alpha_history: list[jnp.ndarray] = []
    proposed_log_likelihood_history: list[jnp.ndarray] = []
    parameter_jump_rms_history: list[jnp.ndarray] = []
    pf_ess_min_history: list[jnp.ndarray] = []
    pf_ess_mean_history: list[jnp.ndarray] = []
    pf_log_weight_range_max_history: list[jnp.ndarray] = []
    pf_log_weight_variance_mean_history: list[jnp.ndarray] = []
    pf_log_likelihood_increment_variance_history: list[jnp.ndarray] = []
    pf_ess_by_t_history: list[jnp.ndarray] = []
    pf_log_weight_range_by_t_history: list[jnp.ndarray] = []
    pf_log_weight_variance_by_t_history: list[jnp.ndarray] = []
    pf_log_likelihood_increment_by_t_history: list[jnp.ndarray] = []

    progress_started = time.monotonic()
    progress_every = max(1, min(250, total_steps // 20))
    print(
        "particle_marginal_mh progress: "
        f"chains={num_chains} warmup={num_warmup} samples={num_samples} "
        f"total_steps={total_steps} n_particles={kernel.num_particles} "
        f"progress_every={progress_every}",
        flush=True,
    )
    first_step_seconds: float | None = None
    sampling_loop_started = time.monotonic()
    for step_idx in range(total_steps):
        step_started = time.monotonic()
        if step_idx == 0:
            print("particle_marginal_mh progress: first step compile/run start", flush=True)
        states, step_info = _run_batched_step(
            states,
            step_keys[step_idx],
            step_fn=kernel.step_fn,
        )
        if (
            step_idx == 0
            or (step_idx + 1) % progress_every == 0
            or step_idx + 1 == num_warmup
            or step_idx + 1 == total_steps
        ):
            phase = "warmup" if step_idx < num_warmup else "sample"
            elapsed = time.monotonic() - progress_started
            accept_now = jax.device_get(jnp.mean(step_info["parameter_accepted"]))
            step_now = jax.device_get(states.param_step_size)
            lp_now = jax.device_get(states.estimated_log_posterior)
            ll_now = jax.device_get(states.estimated_log_likelihood)
            print(
                "particle_marginal_mh progress: "
                f"step={step_idx + 1}/{total_steps} phase={phase} elapsed={elapsed:.1f}s "
                f"parameter_accept_now={float(accept_now):.3f} "
                f"param_step_range=[{float(jnp.min(step_now)):.3g},"
                f"{float(jnp.max(step_now)):.3g}] "
                f"estimated_lp_range=[{float(jnp.min(lp_now)):.3g},"
                f"{float(jnp.max(lp_now)):.3g}] "
                f"estimated_ll_range=[{float(jnp.min(ll_now)):.3g},"
                f"{float(jnp.max(ll_now)):.3g}]",
                flush=True,
            )
        if step_idx == 0:
            states.estimated_log_posterior.block_until_ready()
            first_step_seconds = time.monotonic() - step_started
            print(
                "particle_marginal_mh progress: "
                f"first step compile/run complete elapsed={first_step_seconds:.1f}s",
                flush=True,
            )

        position_history.append(states.position)
        parameter_accept_history.append(step_info["parameter_accepted"])
        estimated_log_likelihood_history.append(step_info["estimated_log_likelihood"])
        log_prior_history.append(step_info["log_prior"])
        estimated_log_posterior_history.append(step_info["estimated_log_posterior"])
        log_alpha_history.append(step_info["log_alpha"])
        proposed_log_likelihood_history.append(step_info["proposed_estimated_log_likelihood"])
        pf_ess_min_history.append(step_info["pf_ess_min"])
        pf_ess_mean_history.append(step_info["pf_ess_mean"])
        pf_log_weight_range_max_history.append(step_info["pf_log_weight_range_max"])
        pf_log_weight_variance_mean_history.append(step_info["pf_log_weight_variance_mean"])
        pf_log_likelihood_increment_variance_history.append(
            step_info["pf_log_likelihood_increment_variance"]
        )
        if _has_metric(kernel.diagnostic_metrics, "parameter_movement"):
            parameter_jump_rms_history.append(step_info["parameter_jump_rms"])
        if _has_metric(kernel.diagnostic_metrics, "particle_filter"):
            pf_ess_by_t_history.append(step_info["pf_ess_by_t"])
            pf_log_weight_range_by_t_history.append(step_info["pf_log_weight_range_by_t"])
            pf_log_weight_variance_by_t_history.append(step_info["pf_log_weight_variance_by_t"])
            pf_log_likelihood_increment_by_t_history.append(
                step_info["pf_log_likelihood_increment_by_t"]
            )

        if step_idx < num_warmup:
            if use_dual_averaging:
                assert da_param_update is not None
                updated_param_da = jax.vmap(
                    lambda da_state, accepted: _clip_dual_averaging_state(
                        da_param_update(da_state, accepted),
                        min_scale=kernel.min_scale,
                        max_scale=kernel.max_scale,
                    )
                )(states.param_da, step_info["parameter_accepted"])
                if step_idx == num_warmup - 1:
                    next_param_step = jnp.exp(updated_param_da.log_step_size_avg)
                else:
                    next_param_step = jnp.exp(updated_param_da.log_step_size)
                states = states._replace(
                    param_step_size=_clip_scale(
                        next_param_step.astype(states.param_step_size.dtype),
                        min_scale=kernel.min_scale,
                        max_scale=kernel.max_scale,
                    ),
                    param_da=updated_param_da,
                )
            else:
                states = states._replace(
                    param_step_size=_adapt_scale(
                        states.param_step_size,
                        accepted=step_info["parameter_accepted"],
                        target_accept=kernel.target_accept,
                        adaptation_rate=adaptation_rate,
                        min_scale=kernel.min_scale,
                        max_scale=kernel.max_scale,
                    )
                )

    states.estimated_log_posterior.block_until_ready()
    sampling_loop_seconds = time.monotonic() - sampling_loop_started
    all_grouped_positions = _stack_sample_history(
        position_history,
        num_chains=num_chains,
        trailing_shape=(dim,),
        dtype=chain_init_positions.dtype,
    )
    grouped_positions = all_grouped_positions[:, num_warmup:]
    all_chain_extra_fields = {
        "parameter_accept_prob": _stack_sample_history(
            parameter_accept_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "estimated_log_likelihood": _stack_sample_history(
            estimated_log_likelihood_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "log_prior": _stack_sample_history(
            log_prior_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "estimated_log_posterior": _stack_sample_history(
            estimated_log_posterior_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "log_alpha": _stack_sample_history(
            log_alpha_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "proposed_estimated_log_likelihood": _stack_sample_history(
            proposed_log_likelihood_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "pf_ess_min": _stack_sample_history(
            pf_ess_min_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "pf_ess_mean": _stack_sample_history(
            pf_ess_mean_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "pf_log_weight_range_max": _stack_sample_history(
            pf_log_weight_range_max_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "pf_log_weight_variance_mean": _stack_sample_history(
            pf_log_weight_variance_mean_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
        "pf_log_likelihood_increment_variance": _stack_sample_history(
            pf_log_likelihood_increment_variance_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        ),
    }
    if _has_metric(kernel.diagnostic_metrics, "parameter_movement"):
        all_chain_extra_fields["parameter_jump_rms"] = _stack_sample_history(
            parameter_jump_rms_history,
            num_chains=num_chains,
            trailing_shape=(),
            dtype=chain_init_positions.dtype,
        )
    if _has_metric(kernel.diagnostic_metrics, "particle_filter"):
        time_count = int(bundle["observations"].shape[0])
        all_chain_extra_fields.update(
            {
                "pf_ess_by_t": _stack_sample_history(
                    pf_ess_by_t_history,
                    num_chains=num_chains,
                    trailing_shape=(time_count,),
                    dtype=chain_init_positions.dtype,
                ),
                "pf_log_weight_range_by_t": _stack_sample_history(
                    pf_log_weight_range_by_t_history,
                    num_chains=num_chains,
                    trailing_shape=(time_count,),
                    dtype=chain_init_positions.dtype,
                ),
                "pf_log_weight_variance_by_t": _stack_sample_history(
                    pf_log_weight_variance_by_t_history,
                    num_chains=num_chains,
                    trailing_shape=(time_count,),
                    dtype=chain_init_positions.dtype,
                ),
                "pf_log_likelihood_increment_by_t": _stack_sample_history(
                    pf_log_likelihood_increment_by_t_history,
                    num_chains=num_chains,
                    trailing_shape=(time_count,),
                    dtype=chain_init_positions.dtype,
                ),
            }
        )
    chain_extra_fields = {
        name: values[:, num_warmup:] for name, values in all_chain_extra_fields.items()
    }
    warmup_chain_extra_fields = {
        name: values[:, :num_warmup] for name, values in all_chain_extra_fields.items()
    }
    estimated_log_posterior_history = all_chain_extra_fields["estimated_log_posterior"]
    post_warmup_estimated_log_posterior_mean = (
        jnp.mean(estimated_log_posterior_history[:, num_warmup:], axis=1)
        if num_samples > 0
        else jnp.full((num_chains,), jnp.nan, dtype=chain_init_positions.dtype)
    )
    return {
        "grouped_positions": grouped_positions,
        "chain_extra_fields": chain_extra_fields,
        "warmup_chain_extra_fields": warmup_chain_extra_fields,
        "all_chain_extra_fields": all_chain_extra_fields,
        "estimated_log_posterior_history": estimated_log_posterior_history[:, num_warmup:],
        "warmup_estimated_log_posterior_history": estimated_log_posterior_history[:, :num_warmup],
        "all_estimated_log_posterior_history": estimated_log_posterior_history,
        "initial_param_step_size": initial_param_step_size,
        "final_param_step_size": states.param_step_size,
        "first_step_seconds": 0.0 if first_step_seconds is None else first_step_seconds,
        "sampling_loop_seconds": sampling_loop_seconds,
        "post_warmup_estimated_log_posterior_mean": post_warmup_estimated_log_posterior_mean,
    }
