"""Latent-smoother contract types and per-step context for MPGibbs."""

# References: see kernel.py (parameter proposal) and the smoothers/ package (latent
# backends) for the papers in docs/papers/ behind each component.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp

    from nof1_causal_lab.models.ssm.shapes import Array, Float

_LATENT_SMOOTHER_PLAIN = "plain"
_LATENT_SMOOTHER_DSMC = "dsmc"
_DSMC_LEAF_PROPOSAL_AMALA = "amala"
_DSMC_LEAF_PROPOSAL_AMALA_PLUS = "amala_plus"
_DSMC_LEAF_PROPOSAL_AMALA_EXACT = "amala_exact"
_DSMC_LEAF_PROPOSALS = (
    _DSMC_LEAF_PROPOSAL_AMALA,
    _DSMC_LEAF_PROPOSAL_AMALA_PLUS,
    _DSMC_LEAF_PROPOSAL_AMALA_EXACT,
)
_LATENT_SMOOTHERS = (
    _LATENT_SMOOTHER_PLAIN,
    _LATENT_SMOOTHER_DSMC,
)


@dataclass(frozen=True)
class MPGibbsLatentSmoother:
    """Static metadata for an MPGibbs latent smoother implementation."""

    name: str
    algorithm: str
    family: str
    selection: str
    parallel: bool
    backward_sampling: bool


class MPGibbsLatentSmootherResult(NamedTuple):
    """Production result contract for MPGibbs latent smoothers."""

    latent_path: Float[Array, "T D"]
    final_label_log_probs: Float[Array, " K"]
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
            backward_sampling=True,
        )
    if name == _LATENT_SMOOTHER_DSMC:
        return MPGibbsLatentSmoother(
            name=name,
            algorithm="conditional_desequentialized_smc",
            family="posterior_mixture_dsmc",
            selection="tree_stitch_combination",
            parallel=True,
            backward_sampling=False,
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
    runtime_observations: Any
    runtime_times: Any
    num_particles: int
    num_parameter_particles: int
    latent_block_size: int
    latent_delta: float
    amala_kappa: float
    amala_grad_clip: float
    dsmc_leaf_proposal: str
    transition_initial_log_prob_fn: Any
    transition_log_prob_fn: Any
    transition_log_probs_for_pairs_fn: Any
    transition_pairwise_log_probs_fn: Any
    transition_sample_fn: Any
    diagnostic_metrics: frozenset[str]


@dataclass(frozen=True)
class SmootherContext:
    """Per-step latent-smoother inputs.

    Explicit interface replacing the implicit closure capture the smoothers
    previously relied on. Built once per joint step by
    :func:`build_smoother_context`; consumed by the smoother modules.
    """

    contexts: Any
    parameter_particles: Float[Array, "K U"]
    parameter_log_probs: Float[Array, " K"]
    initial_label_log_probs: Float[Array, " K"]
    init_means: Float[Array, "K D"]
    init_chols: Float[Array, "K D D"]
    init_logdets: Float[Array, " K"]
    num_steps: int
    num_free_particles: int
    num_parameter_particles: int
    block_size: int
    num_blocks: int
    latent_dtype: Any
    traj_dtype: Any
    complete_dtype: Any
    obs_increment_fn: Any
    runtime_observations: Any
    trajectory_log_prob_fn: Any
    prior_terms_from_context_fn: Any
    log_prior_unc_fn: Any
    amala_delta: Float[Array, " D"]
    amala_kappa: float
    amala_grad_clip: float
    dsmc_leaf_proposal: str
    diagnostic_metrics: frozenset[str]
    initial_value_grad_by_param: Callable
    transition_current_value_grad_by_param: Callable
    transition_next_value_grad_by_param: Callable
    selected_transition_log_probs: Callable
    pairwise_transition_log_probs: Callable
    transition_log_probs_from_fixed_prev: Callable
    transition_log_probs_by_param: Callable
    transition_log_probs_to_next_by_param: Callable
    sample_transition_by_label: Callable
    segment_terminal_label_log_probs: Callable
    path_future_tail_log_probs: Callable
    trajectory_label_log_probs: Callable
