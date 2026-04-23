"""Samplers for pedagogical visualisation on ``synthetic_posteriors`` targets.

Each sampler exposes a ``run_*(target, config) -> SamplerTrace`` entry point that
returns a uniform record the notebook can plot without knowing what sampler
produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blackjax
import blackjax.smc.resampling as resampling
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from synthetic_posteriors.targets import Target

Array = jax.Array


@dataclass(frozen=True)
class SamplerTrace:
    """Uniform plotting record.

    - ``positions`` (N, D): flat array of all points to plot.
    - ``stage`` (N,): integer stage index (iteration for MCMC, tempering step for SMC).
    - ``killed`` (N,) bool: ``True`` if this point was aborted (divergent / out-of-support).
    - ``connect``: draw a polyline between consecutive points in ``positions``.
    - ``stage_label``: axis label for the colour gradient.
    - ``summary``: one-line string for a footer annotation.
    """

    positions: Array
    stage: Array
    killed: Array
    connect: bool
    stage_label: str
    summary: str


# ─── NUTS ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NUTSConfig:
    warmup: int = 400
    num_samples: int = 500
    target_accept: float = 0.8
    initial_position: tuple[float, ...] = (0.0, 0.0)
    seed: int = 0
    max_tree_depth: int = 10


def run_nuts(target: Target, config: NUTSConfig | None = None) -> SamplerTrace:
    cfg = config or NUTSConfig()

    def log_prob(x: Array) -> Array:
        return target.log_prob(x)

    key = jax.random.PRNGKey(cfg.seed)
    warm_key, sample_key = jax.random.split(key)
    init = jnp.asarray(cfg.initial_position, dtype=jnp.float64)

    adapt = blackjax.window_adaptation(
        blackjax.nuts,
        log_prob,
        target_acceptance_rate=cfg.target_accept,
    )
    (state, warmup_params), _ = adapt.run(warm_key, init, num_steps=cfg.warmup)
    kernel = blackjax.nuts(log_prob, max_num_doublings=cfg.max_tree_depth, **warmup_params)

    def step(carry_state, step_key):
        new_state, info = kernel.step(step_key, carry_state)
        return new_state, (new_state.position, info.acceptance_rate, info.is_divergent)

    keys = jax.random.split(sample_key, cfg.num_samples)
    _, (positions, accepts, divergences) = jax.lax.scan(step, state, keys)

    accept_mean = float(jnp.mean(accepts))
    n_div = int(jnp.sum(divergences))
    return SamplerTrace(
        positions=positions,
        stage=jnp.arange(cfg.num_samples),
        killed=divergences,
        connect=True,
        stage_label="iteration",
        summary=f"NUTS · accept̄={accept_mean:.2f} · divergences={n_div}/{cfg.num_samples}",
    )


# ─── Adaptive-tempered SMC (fixed linear schedule) ───────────────────────────


@dataclass(frozen=True)
class SMCConfig:
    num_particles: int = 180
    num_stages: int = 12
    num_mcmc_steps: int = 6
    hmc_step_size: float = 0.25
    hmc_integration_steps: int = 8
    prior_scale: float = 3.5
    seed: int = 0


def run_smc(target: Target, config: SMCConfig | None = None) -> SamplerTrace:
    cfg = config or SMCConfig()
    dim = 2
    prior_scale = cfg.prior_scale

    def logprior(x: Array) -> Array:
        return jax.scipy.stats.norm.logpdf(x, 0.0, prior_scale).sum()

    def loglikelihood(x: Array) -> Array:
        return target.log_prob(x) - logprior(x)

    hmc_params = {
        "step_size": jnp.full((cfg.num_particles,), cfg.hmc_step_size),
        "inverse_mass_matrix": jnp.tile(jnp.ones(dim), (cfg.num_particles, 1)),
        "num_integration_steps": jnp.full((cfg.num_particles,), cfg.hmc_integration_steps),
    }
    smc = blackjax.tempered_smc(
        logprior_fn=logprior,
        loglikelihood_fn=loglikelihood,
        mcmc_step_fn=blackjax.hmc.build_kernel(),
        mcmc_init_fn=blackjax.hmc.init,
        mcmc_parameters=hmc_params,
        resampling_fn=resampling.systematic,
        num_mcmc_steps=cfg.num_mcmc_steps,
    )

    key = jax.random.PRNGKey(cfg.seed)
    init_key, scan_key = jax.random.split(key)
    initial_particles = jax.random.normal(init_key, (cfg.num_particles, dim)) * prior_scale
    state = smc.init(initial_particles)

    lambdas = jnp.linspace(0.0, 1.0, cfg.num_stages + 1)[1:]

    def step(carry, lmbda):
        st, k = carry
        k, sub = jax.random.split(k)
        new_st, _ = smc.step(sub, st, lmbda)
        return (new_st, k), new_st.particles

    _, particles_history = jax.lax.scan(step, (state, scan_key), lambdas)
    # particles_history: (num_stages, num_particles, D)
    all_stages = jnp.concatenate([initial_particles[None], particles_history], axis=0)
    num_total = all_stages.shape[0]
    positions = all_stages.reshape(-1, dim)
    stage_arr = jnp.repeat(jnp.arange(num_total), cfg.num_particles)
    # mark particles that landed outside target support (log_prob -inf or NaN)
    lp_final = jax.vmap(target.log_prob)(positions)
    killed = ~jnp.isfinite(lp_final)

    return SamplerTrace(
        positions=positions,
        stage=stage_arr,
        killed=killed,
        connect=False,
        stage_label="tempering step",
        summary=(
            f"SMC · {cfg.num_particles} particles × {num_total} stages "
            f"(prior σ={prior_scale}) · dead={int(killed.sum())}"
        ),
    )
