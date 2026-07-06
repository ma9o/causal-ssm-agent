"""Compile-stable prior predictive runtime.

Builds prior predictive samples directly from compiled prior semantics or
``PriorRegistry`` without tracing back through ``SSMModel.model()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.artifacts.model_spec import DistributionFamily
from nof1_causal_lab.models.predictive_simulation import (
    sample_predictive_observations_from_linear_predictors,
)
from nof1_causal_lab.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
    stable_cholesky,
)
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.runtime import pack_vector_field_params_from_samples
from nof1_causal_lab.models.ssm.dynamics.simulator import SimulationConfig, simulate
from nof1_causal_lab.models.ssm.dynamics.spec import (
    compile_dynamics,
)
from nof1_causal_lab.models.ssm.inference.targets.observation_families import (
    any_family_needs_level_metadata,
)
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    assemble_deterministics_from_registry,
    assemble_extra_params_from_registry,
    load_prior_runtime_bundle,
    sample_prior_unconstrained,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec


class _InputDrivenVectorField(eqx.Module):
    """Vector-field wrapper adding piecewise-constant known-input forcing."""

    base: Any
    input_effect: jnp.ndarray
    times: jnp.ndarray
    transition_inputs: jnp.ndarray
    n_latent: int = eqx.field(static=True)

    def __call__(self, t: jnp.ndarray, eta: jnp.ndarray, args):
        drift = self.base(t, eta, args)
        idx = jnp.clip(
            jnp.searchsorted(self.times, t, side="right"),
            1,
            self.transition_inputs.shape[0] - 1,
        )
        return drift + self.input_effect @ self.transition_inputs[idx]

    def initial_condition(self, eta0: jnp.ndarray, args, t0: jnp.ndarray | float = 0.0):
        return self.base.initial_condition(eta0, args, t0)

    def steady_state_residual(self, eta: jnp.ndarray, args):
        return self.base.steady_state_residual(eta, args)

    def linearize(
        self,
        x_lin: jnp.ndarray,
        args,
        t: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if t is None:
            t = jnp.asarray(0.0)
        f_at_x = self(t, x_lin, args)
        jacobian = jax.jacfwd(lambda x: self(t, x, args))(x_lin)
        intercept = f_at_x - jacobian @ x_lin
        return jacobian, intercept


def _ensure_discrete_metadata(spec: SSMSpec) -> None:
    """Require hydrated level counts before sampling discrete emissions."""
    needs_levels = any_family_needs_level_metadata(spec.manifest_dists)
    if needs_levels and spec.manifest_level_counts is None:
        raise ValueError(
            "Prior predictive for ordered/categorical emissions requires hydrated "
            "manifest_level_counts."
        )


def _assemble_extra_params_batched(
    spec: SSMSpec,
    constrained_samples: dict[str, jnp.ndarray],
    runtime: PriorRuntimeBundle,
    *,
    n_draws: int,
) -> dict[str, jnp.ndarray]:
    """Assemble per-draw observation/process hyperparameters."""
    registry = runtime.site_runtime.registry
    if not any(site.assembly_group == "likelihood" for site in registry):
        return {}

    def _assemble_one(draw_idx):
        sampled_values = {
            site_name: values[draw_idx] for site_name, values in constrained_samples.items()
        }
        return assemble_extra_params_from_registry(spec, sampled_values, registry)

    return jax.vmap(_assemble_one)(jnp.arange(n_draws, dtype=jnp.int32))


def _ensure_gaussian_process_diffusion(spec: SSMSpec) -> None:
    non_gaussian = [
        str(dist.value if isinstance(dist, DistributionFamily) else dist)
        for dist in spec.diffusion_dists
        if DistributionFamily(dist) != DistributionFamily.GAUSSIAN
    ]
    if non_gaussian:
        raise ValueError(
            "Vector-field prior predictive simulation currently requires Gaussian process "
            f"diffusion; got {non_gaussian}."
        )


# One compiled program shared by every prior draw: array leaves (params,
# state, key, per-draw CFL step size inside SimulationConfig) are traced,
# structure is static. The bare exact-engine ``simulate`` builds fresh
# closures per call, which — combined with per-draw host-float step sizes —
# used to retrace AND recompile XLA once per draw (~0.8s × n_draws per
# submit_construct).
_simulate_jit = eqx.filter_jit(simulate)


def _prepare_vector_field_for_draw(
    base_vector_field,
    *,
    input_effect: jnp.ndarray,
    times: jnp.ndarray,
    transition_inputs: jnp.ndarray | None,
):
    if input_effect.shape[1] == 0:
        return base_vector_field
    if transition_inputs is None:
        raise ValueError("SSM has known input effects but transition_inputs was not provided.")
    transition_inputs = jnp.asarray(transition_inputs, dtype=input_effect.dtype)
    if transition_inputs.shape != (times.shape[0], input_effect.shape[1]):
        raise ValueError(
            "transition_inputs must have shape "
            f"({times.shape[0]}, {input_effect.shape[1]}), got {transition_inputs.shape}"
        )
    return _InputDrivenVectorField(
        base=base_vector_field,
        input_effect=input_effect,
        times=times,
        transition_inputs=transition_inputs,
        n_latent=base_vector_field.n_latent,
    )


def _linear_predictors_from_latents(
    latent_trajectory: jnp.ndarray,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(lambda eta_t: lambda_mat @ eta_t + manifest_means)(latent_trajectory)


# Explicit (Heun) SDE integration of a linear relaxation ``dy = -a·y dt + …`` is
# stable only for ``a·Δt ≲ 2``; past that the step map amplifies and the path
# diverges. The default step ``span/200`` is far too coarse for a fast construct
# over a long record (e.g. a τ ≈ 0.15 d node across 60 d ⇒ ``a·Δt ≈ 2``), and a
# tail draw with even faster decay blows the trajectory up to ``inf`` — which then
# trips the log-link overflow guard and aborts the whole prior predictive. The DAG
# drift is triangular, so its Jacobian's spectral radius is bounded by the largest
# diagonal relaxation rate (the sampled ``*_decay`` sites; NodePotential reuses the
# decay site for its stiffness). Cap the SDE step at a CFL-safe fraction of the
# fastest relaxation time per draw. This only ever *refines* the step (finer ⇒
# smaller discretization error, the exact-engine's one controllable error term),
# never coarsens it.
_SDE_CFL_SAFETY = 0.25
_SDE_MAX_STEPS = 16384


def _predictive_sde_config(draw: dict[str, jnp.ndarray], span: float) -> SimulationConfig:
    base_dt = span / 200.0 if span > 0.0 else None
    if base_dt is None:
        return SimulationConfig()
    # Traced (not host) arithmetic: the per-draw step size stays a jnp scalar
    # so every draw reuses ONE compiled program — a host float here bakes into
    # the XLA graph as a constant and forces a retrace + recompile per draw.
    decay_maxes = [
        jnp.max(jnp.abs(value)) for name, value in draw.items() if name.endswith("_decay")
    ]
    if decay_maxes:
        max_rate = jnp.stack(decay_maxes).max()
        capped = jnp.minimum(base_dt, _SDE_CFL_SAFETY / jnp.maximum(max_rate, 1e-30))
        sde_dt = jnp.where(max_rate > 0.0, capped, base_dt)
    else:
        sde_dt = jnp.asarray(base_dt)
    # Keep the step count within the solver budget for pathologically fast draws.
    sde_dt = jnp.maximum(sde_dt, span / _SDE_MAX_STEPS)
    return SimulationConfig(sde_dt=sde_dt, max_steps=_SDE_MAX_STEPS + 16)


def _simulate_vector_field_predictive_latents(
    spec: SSMSpec,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    *,
    transition_inputs: jnp.ndarray | None,
    seed: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _ensure_gaussian_process_diffusion(spec)
    compiled = compile_dynamics(spec.dynamics_spec)
    n_draws = int(next(iter(samples.values())).shape[0])
    draw_keys = random.split(random.PRNGKey(seed), n_draws)
    latents = []
    linear_predictors = []

    span = float(times[-1] - times[0]) if int(times.shape[0]) > 1 else 0.0
    intervention = Intervention.none()

    for draw_idx in range(n_draws):
        key_init, key_latent = random.split(draw_keys[draw_idx])
        draw = {name: values[draw_idx] for name, values in samples.items()}
        vf_params = pack_vector_field_params_from_samples(spec, draw, draw)
        t0_chol = stable_cholesky(
            draw["t0_cov"],
            min_eigenvalue=INITIAL_STATE_COV_MIN_EIGENVALUE,
        )
        eta0 = draw["t0_means"] + t0_chol @ random.normal(key_init, (spec.n_latent,))
        diffusion_chol = draw["diffusion"]
        # ``input_effect`` is a deterministic recorded only when the model has
        # exogenous inputs; posterior samples for input-free models omit it. Fall
        # back to a zero-width effect (no input forcing). Prior-predictive samples
        # always assemble it, so this leaves that path unchanged.
        input_effect = draw.get("input_effect", jnp.zeros((spec.n_latent, 0), dtype=eta0.dtype))
        if int(times.shape[0]) == 1:
            latent_trajectory = eta0[None, :]
        else:
            vector_field = _prepare_vector_field_for_draw(
                compiled.vector_field,
                input_effect=input_effect,
                times=times,
                transition_inputs=transition_inputs,
            )
            latent_trajectory = _simulate_jit(
                vector_field,
                vf_params,
                intervention,
                eta0,
                times,
                config=_predictive_sde_config(draw, span),
                key=key_latent,
                diffusion_cov=diffusion_chol @ diffusion_chol.T,
            )
        latents.append(latent_trajectory)
        linear_predictors.append(
            _linear_predictors_from_latents(
                latent_trajectory,
                draw["lambda"],
                draw["manifest_means"],
            )
        )

    return jnp.stack(latents), jnp.stack(linear_predictors)


def sample_prior_predictive_from_runtime(
    spec: SSMSpec,
    runtime: PriorRuntimeBundle,
    times: jnp.ndarray,
    *,
    observation_support=None,
    observation_mask: jnp.ndarray | None = None,
    transition_inputs: jnp.ndarray | None = None,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from a prepared runtime bundle."""
    _ensure_discrete_metadata(spec)

    z_samples, _rng_key = sample_prior_unconstrained(
        random.PRNGKey(seed),
        runtime.site_runtime.registry,
        runtime.prior_state,
        n_samples=num_samples,
    )
    constrained_samples = runtime.site_runtime.constrain_batched(z_samples)
    deterministic_samples = assemble_deterministics_from_registry(
        constrained_samples,
        spec,
        runtime.site_runtime.registry,
        n_draws=num_samples,
    )
    extra_params = _assemble_extra_params_batched(
        spec,
        constrained_samples,
        runtime,
        n_draws=num_samples,
    )

    samples: dict[str, jnp.ndarray] = {}
    samples.update(constrained_samples)
    samples.update(deterministic_samples)
    samples.update(extra_params)
    latents, linear_predictors = _simulate_vector_field_predictive_latents(
        spec,
        samples,
        times,
        transition_inputs=transition_inputs,
        seed=seed,
    )
    observations, observations_mask = sample_predictive_observations_from_linear_predictors(
        linear_predictors,
        samples,
        times,
        manifest_dists=spec.manifest_dists,
        manifest_links=spec.manifest_links,
        manifest_level_counts=spec.manifest_level_counts,
        observation_support=observation_support,
        observation_mask=observation_mask,
        n_subsample=num_samples,
        rng_seed=seed,
        manifest_names=list(spec.manifest_names) if spec.manifest_names is not None else None,
    )
    samples["latents"] = latents
    samples["linear_predictors"] = linear_predictors
    samples["observations"] = observations
    samples["observations_mask"] = observations_mask
    return samples


def simulate_posterior_predictive_observations(
    spec: SSMSpec,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    *,
    observation_support=None,
    observation_mask: jnp.ndarray | None = None,
    transition_inputs: jnp.ndarray | None = None,
    n_subsample: int = 50,
    rng_seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-simulate posterior-predictive observations through the exact field.

    Subsamples ``n_subsample`` posterior draws and integrates each through the
    *true* (Diffrax) vector field — the same nonlinearity-preserving simulator the
    prior-predictive path uses, never a linearised drift matrix — then samples
    observations from the emission families. Returns ``(observations,
    effective_mask)`` with a leading subsample axis.
    """
    n_draws = int(next(iter(samples.values())).shape[0]) if samples else 0
    n_use = min(n_subsample, n_draws)
    indices = jnp.linspace(0, n_draws - 1, n_use).astype(int)
    sub = {name: jnp.asarray(value)[indices] for name, value in samples.items()}
    _latents, linear_predictors = _simulate_vector_field_predictive_latents(
        spec,
        sub,
        times,
        transition_inputs=transition_inputs,
        seed=rng_seed,
    )
    return sample_predictive_observations_from_linear_predictors(
        linear_predictors,
        sub,
        times,
        manifest_dists=spec.manifest_dists,
        manifest_links=spec.manifest_links,
        manifest_level_counts=spec.manifest_level_counts,
        observation_support=observation_support,
        observation_mask=observation_mask,
        n_subsample=n_use,
        rng_seed=rng_seed,
        manifest_names=list(spec.manifest_names) if spec.manifest_names is not None else None,
    )


def sample_prior_predictive_from_compiled_semantics(
    spec: SSMSpec,
    compiled_prior_semantics: dict,
    times: jnp.ndarray,
    *,
    observation_support=None,
    observation_mask: jnp.ndarray | None = None,
    transition_inputs: jnp.ndarray | None = None,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from serialized compiled semantics."""
    runtime = load_prior_runtime_bundle(compiled_prior_semantics)
    return sample_prior_predictive_from_runtime(
        spec,
        runtime,
        times,
        observation_support=observation_support,
        observation_mask=observation_mask,
        transition_inputs=transition_inputs,
        num_samples=num_samples,
        seed=seed,
    )
