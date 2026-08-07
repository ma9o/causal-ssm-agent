"""Typed configuration contracts for state-space model inference."""

from __future__ import annotations

from typing import Literal, TypedDict

import jax  # noqa: TC002 - Pydantic resolves these array annotations at runtime.
from pydantic import ConfigDict, TypeAdapter, with_config


@with_config(ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class _MarginalParticleGibbsMethodOptions(TypedDict, total=False):
    """Method-specific options shared by partial and resolved configs."""

    n_particles: int
    n_parameter_particles: int
    latent_smoother: Literal["dsmc"]
    latent_delta: float
    parameter_proposal: Literal["random_walk", "pseudo_langevin"]
    amala_delta_init: float
    amala_delta_min: float
    amala_delta_max: float
    amala_target_accept: float
    amala_adaptation_window: int
    amala_adaptation_tolerance: float
    amala_adaptation_rho: float
    amala_adaptation_rho_min: float
    amala_adaptation_gamma: float
    amala_kappa: float
    amala_grad_clip: float
    dsmc_leaf_proposal: Literal["amala_exact", "paid_mix"]
    latent_block_coords: int | None
    paid_mix_z_weight: float
    paid_mix_pilot_weight: float
    paid_mix_pilot_var_scale: float
    paid_mix_wide_mult: float
    latent_sign_flip_moves: bool
    diagnostic_metrics_all: bool
    diagnostic_metrics: tuple[str, ...] | list[str]
    param_step_size: float
    param_step_size_min: float
    param_step_size_max: float
    param_target_accept: float | None
    adaptation_rate: float
    adaptation_scheme: Literal["simple", "dual_averaging"]
    init_method: Literal["random", "pathfinder"]
    latent_init_method: Literal["predictive"]
    pathfinder_num_elbo_samples: int
    pathfinder_maxiter: int
    n_pathfinder_starts: int
    pathfinder_parallel_workers: int | None
    pathfinder_init_scale: float | None
    auto_preconditioner_method: Literal["map", "none", "pathfinder"]
    auto_preconditioner_maxiter: int
    parameter_preconditioner_chol: jax.Array | None
    initial_positions_override: jax.Array | None
    initial_latent_trajectories: jax.Array | None
    init_scale: float
    retain_latent_paths: bool
    compute_latent_posterior_summary: bool
    n_ieks_iters: int
    profile_dir: str | None
    profile_compile_analysis: bool
    profile_runtime_trace: bool
    profile_trace_start_step: int
    profile_trace_steps: int


class MarginalParticleGibbsOptions(_MarginalParticleGibbsMethodOptions, total=False):
    """Optional keyword surface accepted by marginalized Particle Gibbs."""

    num_warmup: int
    num_samples: int
    num_chains: int
    seed: int


class SamplerConfigOverride(MarginalParticleGibbsOptions, total=False):
    """Partial sampler configuration accepted at runtime boundaries."""

    method: Literal["marginal_particle_gibbs"]


class SamplerConfig(_MarginalParticleGibbsMethodOptions):
    """Resolved flat inference config with required common controls."""

    method: Literal["marginal_particle_gibbs"]
    num_warmup: int
    num_samples: int
    num_chains: int
    seed: int


type SamplerConfigInput = SamplerConfig | SamplerConfigOverride


_SAMPLER_CONFIG_ADAPTER = TypeAdapter(SamplerConfig)


def validate_sampler_config(config: object) -> SamplerConfig:
    """Validate a resolved sampler config at its untyped construction boundary."""
    return _SAMPLER_CONFIG_ADAPTER.validate_python(config)
