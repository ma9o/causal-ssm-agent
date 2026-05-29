"""Latent-smoother contract types and per-step context for MPGibbs."""

# References: see kernel.py (parameter proposal) and the smoothers/ package (latent
# backends) for the papers in docs/papers/ behind each component.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp

_LATENT_SMOOTHER_PLAIN = "plain"
_LATENT_SMOOTHER_AMALA = "amala"
_LATENT_SMOOTHER_MGRAD = "mgrad"
_LATENT_SMOOTHERS = (_LATENT_SMOOTHER_PLAIN, _LATENT_SMOOTHER_AMALA, _LATENT_SMOOTHER_MGRAD)


@dataclass(frozen=True)
class MPGibbsLatentSmoother:
    """Static metadata for an MPGibbs latent smoother implementation."""

    name: str
    algorithm: str
    family: str
    selection: str
    parallel: bool


class MPGibbsLatentSmootherResult(NamedTuple):
    """Production result contract for MPGibbs latent smoothers."""

    latent_path: jnp.ndarray
    final_label_log_probs: jnp.ndarray
    origin_path: jnp.ndarray
    diagnostics: dict[str, jnp.ndarray]


def _resolve_latent_smoother(name: str) -> MPGibbsLatentSmoother:
    if name == _LATENT_SMOOTHER_PLAIN:
        return MPGibbsLatentSmoother(
            name=name,
            algorithm="blocked_posterior_mixture_backward_csmc",
            family="posterior_mixture_csmc",
            selection="blocked_backward_sampling",
            parallel=False,
        )
    if name == _LATENT_SMOOTHER_AMALA:
        return MPGibbsLatentSmoother(
            name=name,
            algorithm="sequential_particle_amala_csmc",
            family="posterior_mixture_particle_amala",
            selection="augmented_target_backward_sampling",
            parallel=False,
        )
    if name == _LATENT_SMOOTHER_MGRAD:
        return MPGibbsLatentSmoother(
            name=name,
            algorithm="sequential_particle_mgrad",
            family="particle_mgrad",
            selection="backward_sampling_forced_move",
            parallel=False,
        )
    allowed = ", ".join(repr(candidate) for candidate in _LATENT_SMOOTHERS)
    raise ValueError(
        f"marginal_particle_gibbs latent_smoother must be one of {allowed}; got {name!r}."
    )


@dataclass(frozen=True)
class MPGibbsStatic:
    """Build-time configuration and bundle callables for the smoother context."""

    latent_context_runtime_fn: Any
    log_prior_unc_fn: Any
    initial_latent_moments_fn: Any
    obs_increment_fn: Any
    trajectory_log_prob_fn: Any
    prior_terms_from_context_fn: Any
    initial_observation_auxiliary_fn: Any
    runtime_observations: Any
    runtime_times: Any
    num_particles: int
    num_parameter_particles: int
    latent_block_size: int
    latent_delta: float
    amala_q_scale: float
    amala_kappa: float
    amala_grad_clip: float
    mgrad_latent_kernel: Any
    diagnostic_metrics: frozenset[str]


@dataclass(frozen=True)
class SmootherContext:
    """Per-step latent-smoother inputs.

    Explicit interface replacing the implicit closure capture the smoothers
    previously relied on. Built once per joint step by
    :func:`build_smoother_context`; consumed by the smoother modules.
    """

    contexts: Any
    parameter_particles: jnp.ndarray
    parameter_log_probs: jnp.ndarray
    initial_label_log_probs: jnp.ndarray
    init_means: jnp.ndarray
    init_chols: jnp.ndarray
    init_logdets: jnp.ndarray
    transition_chols: jnp.ndarray
    transition_logdets: jnp.ndarray
    num_steps: int
    num_free_particles: int
    num_parameter_particles: int
    block_size: int
    num_blocks: int
    latent_dtype: Any
    traj_dtype: Any
    complete_dtype: Any
    state: Any
    obs_increment_fn: Any
    runtime_observations: Any
    initial_observation_auxiliary_fn: Any
    trajectory_log_prob_fn: Any
    prior_terms_from_context_fn: Any
    log_prior_unc_fn: Any
    mgrad_latent_kernel: Any
    amala_q_scale: float
    amala_kappa: float
    amala_grad_clip: float
    diagnostic_metrics: frozenset[str]
    transition_log_probs_from_fixed_prev: Callable
    segment_terminal_label_log_probs: Callable
    path_future_tail_log_probs: Callable
    trajectory_label_log_probs: Callable
