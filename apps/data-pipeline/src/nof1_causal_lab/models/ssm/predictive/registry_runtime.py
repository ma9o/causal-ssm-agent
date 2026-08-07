"""Compile-stable prior predictive runtime.

Builds prior predictive samples directly from compiled prior semantics or
``PriorRegistry`` without tracing back through ``SSMModel.model()``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily
from nof1_causal_lab.models.predictive_simulation import (
    sample_predictive_observations_from_linear_predictors,
)
from nof1_causal_lab.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
    stable_cholesky,
)
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.runtime import pack_vector_field_params_from_samples
from nof1_causal_lab.models.ssm.dynamics.serialization import dynamics_spec_to_dict
from nof1_causal_lab.models.ssm.dynamics.simulator import SimulationConfig, simulate
from nof1_causal_lab.models.ssm.dynamics.spec import (
    compile_dynamics,
)
from nof1_causal_lab.models.ssm.execution.observation_families import (
    any_family_needs_level_metadata,
)
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    assemble_deterministics_from_registry,
    assemble_extra_params_from_registry,
    build_site_registry,
    sample_prior_unconstrained,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec

logger = logging.getLogger(__name__)


class PredictiveKeys(NamedTuple):
    """Independent random streams for one predictive simulation."""

    parameters: jax.Array
    latents: jax.Array
    observations: jax.Array


def predictive_keys(seed: int) -> PredictiveKeys:
    """Derive independent parameter, latent, and observation streams."""
    parameter_key, latent_key, observation_key = random.split(random.PRNGKey(seed), 3)
    return PredictiveKeys(parameter_key, latent_key, observation_key)


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
    registry,
    *,
    n_draws: int,
) -> dict[str, jnp.ndarray]:
    """Assemble per-draw observation/process hyperparameters."""
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
_PREDICTIVE_MICROBATCH_SIZE = 32
_LATENT_CACHE_MAX_ENTRIES = 2
_LATENT_CACHE_ENGINE_VERSION = 1
_latent_cache: OrderedDict[str, jax.Array] = OrderedDict()
_latent_cache_lock = threading.Lock()


def _update_array_digest(digest: Any, label: str, value: Any) -> None:
    array = np.asarray(jax.device_get(value))
    digest.update(label.encode())
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))


def _prior_predictive_latent_cache_key(
    spec: SSMSpec,
    vf_params: Any,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    transition_inputs: jnp.ndarray | None,
    rng_key: jax.Array,
) -> str:
    """Fingerprint only inputs that can change the latent trajectories."""
    digest = hashlib.sha256()
    digest.update(f"latent-cache-v{_LATENT_CACHE_ENGINE_VERSION}".encode())
    digest.update(
        json.dumps(
            dynamics_spec_to_dict(spec.dynamics_spec),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(jax.default_backend().encode())
    for leaf_index, leaf in enumerate(jax.tree.leaves(vf_params)):
        _update_array_digest(digest, f"vf:{leaf_index}", leaf)
    for name in ("t0_cov", "t0_means", "diffusion", "input_effect"):
        _update_array_digest(digest, name, samples[name])
    for name in sorted(name for name in samples if name.endswith("_decay")):
        _update_array_digest(digest, name, samples[name])
    _update_array_digest(digest, "times", times)
    if transition_inputs is None:
        digest.update(b"transition-inputs:none")
    else:
        _update_array_digest(digest, "transition-inputs", transition_inputs)
    _update_array_digest(digest, "rng-key", rng_key)
    return digest.hexdigest()


def _cached_latents(key: str) -> jax.Array | None:
    with _latent_cache_lock:
        cached = _latent_cache.pop(key, None)
        if cached is not None:
            _latent_cache[key] = cached
        return cached


def _cache_latents(key: str, latents: jax.Array) -> None:
    with _latent_cache_lock:
        _latent_cache[key] = latents
        _latent_cache.move_to_end(key)
        while len(_latent_cache) > _LATENT_CACHE_MAX_ENTRIES:
            _latent_cache.popitem(last=False)


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
    return SimulationConfig(
        sde_dt=sde_dt,
        max_steps=_SDE_MAX_STEPS + 16,
        use_indexed_brownian_path=True,
    )


def _predictive_draw_order(
    samples: dict[str, jnp.ndarray], span: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Order draws by expected solver work and return the inverse permutation."""
    n_draws = int(next(iter(samples.values())).shape[0])
    if span <= 0.0:
        identity = jnp.arange(n_draws)
        return identity, identity

    decay_maxes = []
    for name, values in samples.items():
        if not name.endswith("_decay"):
            continue
        draw_axes = tuple(range(1, values.ndim))
        decay_maxes.append(
            jnp.max(jnp.abs(values), axis=draw_axes) if draw_axes else jnp.abs(values)
        )

    if decay_maxes:
        max_rate = jnp.stack(decay_maxes).max(axis=0)
        base_dt = span / 200.0
        sde_dt = jnp.minimum(base_dt, _SDE_CFL_SAFETY / jnp.maximum(max_rate, 1e-30))
        sde_dt = jnp.maximum(sde_dt, span / _SDE_MAX_STEPS)
        step_counts = jnp.ceil(span / sde_dt)
    else:
        step_counts = jnp.full(n_draws, 200.0)

    order = jnp.argsort(step_counts, stable=True)
    return order, jnp.argsort(order)


def _simulate_vector_field_predictive_latent_draw(
    n_latent: int,
    base_vector_field,
    vf_params,
    draw: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    transition_inputs: jnp.ndarray | None,
    key: jnp.ndarray,
    span: float,
) -> jnp.ndarray:
    """Simulate one exact latent path without binding it to an emission model."""
    key_init, key_latent = random.split(key)
    t0_chol = stable_cholesky(
        draw["t0_cov"],
        min_eigenvalue=INITIAL_STATE_COV_MIN_EIGENVALUE,
    )
    eta0 = draw["t0_means"] + t0_chol @ random.normal(key_init, (n_latent,))
    diffusion_chol = draw["diffusion"]
    input_effect = draw.get("input_effect", jnp.zeros((n_latent, 0), dtype=eta0.dtype))
    if int(times.shape[0]) == 1:
        latent_trajectory = eta0[None, :]
    else:
        vector_field = _prepare_vector_field_for_draw(
            base_vector_field,
            input_effect=input_effect,
            times=times,
            transition_inputs=transition_inputs,
        )
        latent_trajectory = simulate(
            vector_field,
            vf_params,
            Intervention.none(),
            eta0,
            times,
            config=_predictive_sde_config(draw, span),
            key=key_latent,
            diffusion_cov=diffusion_chol @ diffusion_chol.T,
        )
    return latent_trajectory


# Compile the draw loop once, executing similarly expensive draws in bounded
# vmapped chunks. A full vmap makes every draw pay for the slowest draw's Diffrax
# loop; CFL sorting plus microbatching retains vectorized execution without
# coupling all 200 draws to one pathological tail timestep.
def _simulate_vector_field_predictive_draws_microbatched(
    n_latent: int,
    base_vector_field,
    vf_params,
    draws: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    transition_inputs: jnp.ndarray | None,
    keys: jnp.ndarray,
    span: float,
) -> jnp.ndarray:
    def _simulate_one(args):
        draw_params, draw, key = args
        return _simulate_vector_field_predictive_latent_draw(
            n_latent,
            base_vector_field,
            draw_params,
            draw,
            times,
            transition_inputs,
            key,
            span,
        )

    return jax.lax.map(
        _simulate_one,
        (vf_params, draws, keys),
        batch_size=_PREDICTIVE_MICROBATCH_SIZE,
    )


_simulate_vector_field_predictive_draws = eqx.filter_jit(
    _simulate_vector_field_predictive_draws_microbatched
)


def _simulate_vector_field_predictive_latents(
    spec: SSMSpec,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    *,
    transition_inputs: jnp.ndarray | None,
    rng_key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _ensure_gaussian_process_diffusion(spec)
    compiled = compile_dynamics(spec.dynamics_spec)
    n_draws = int(next(iter(samples.values())).shape[0])
    draw_keys = random.split(rng_key, n_draws)
    # Pack component parameters before entering the compiled simulation. Closing
    # over ``spec`` keeps its structural dataclasses out of JIT's static-argument
    # cache, while vmap broadcasts any component-fixed scalar across draws.
    vf_params = jax.vmap(lambda draw: pack_vector_field_params_from_samples(spec, draw))(samples)
    span = float(times[-1] - times[0]) if int(times.shape[0]) > 1 else 0.0
    cache_key = _prior_predictive_latent_cache_key(
        spec,
        vf_params,
        samples,
        times,
        transition_inputs,
        rng_key,
    )
    latents = _cached_latents(cache_key)
    if latents is None:
        order, inverse_order = _predictive_draw_order(samples, span)
        sorted_latents = _simulate_vector_field_predictive_draws(
            spec.n_latent,
            compiled.vector_field,
            jax.tree.map(lambda value: value[order], vf_params),
            jax.tree.map(lambda value: value[order], samples),
            times,
            transition_inputs,
            draw_keys[order],
            span,
        )
        latents = sorted_latents[inverse_order]
        _cache_latents(cache_key, latents)
        logger.info("Prior-predictive latent cache miss %s", cache_key[:12])
    else:
        logger.info("Prior-predictive latent cache hit %s", cache_key[:12])
    linear_predictors = jax.vmap(_linear_predictors_from_latents)(
        latents,
        samples["lambda"],
        samples["manifest_means"],
    )
    return latents, linear_predictors


def sample_prior_parameters_from_runtime(
    spec: SSMSpec,
    runtime: PriorRuntimeBundle,
    *,
    num_samples: int,
    rng_key: jax.Array,
) -> dict[str, jnp.ndarray]:
    """Sample and assemble the parameter layer of a prior predictive run."""
    z_samples, _rng_key = sample_prior_unconstrained(
        rng_key,
        runtime.site_runtime.registry,
        runtime.prior_state,
        n_samples=num_samples,
    )
    constrained_samples = runtime.site_runtime.constrain_batched(z_samples)
    deterministic_samples = assemble_deterministics_from_registry(
        constrained_samples,
        spec,
        n_draws=num_samples,
    )
    extra_params = _assemble_extra_params_batched(
        spec,
        constrained_samples,
        runtime.site_runtime.registry,
        n_draws=num_samples,
    )
    return {
        **constrained_samples,
        **deterministic_samples,
        **extra_params,
    }


def simulate_prior_predictive_latents(
    spec: SSMSpec,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    *,
    transition_inputs: jnp.ndarray | None,
    rng_key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate exact nonlinear latent paths and their linear predictors."""
    return _simulate_vector_field_predictive_latents(
        spec,
        samples,
        times,
        transition_inputs=transition_inputs,
        rng_key=rng_key,
    )


def sample_prior_predictive_emissions(
    spec: SSMSpec,
    samples: dict[str, jnp.ndarray],
    linear_predictors: jnp.ndarray,
    times: jnp.ndarray,
    *,
    observation_support,
    observation_mask: jnp.ndarray | None,
    num_samples: int,
    rng_key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample the observation layer conditional on cached latent predictors."""
    return sample_predictive_observations_from_linear_predictors(
        linear_predictors,
        samples,
        times,
        manifest_dists=spec.manifest_dists,
        manifest_links=spec.manifest_links,
        manifest_level_counts=spec.manifest_level_counts,
        observation_support=observation_support,
        observation_mask=observation_mask,
        n_subsample=num_samples,
        rng_key=rng_key,
        manifest_names=list(spec.manifest_names) if spec.manifest_names is not None else None,
    )


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

    keys = predictive_keys(seed)

    samples = sample_prior_parameters_from_runtime(
        spec,
        runtime,
        num_samples=num_samples,
        rng_key=keys.parameters,
    )
    latents, linear_predictors = simulate_prior_predictive_latents(
        spec,
        samples,
        times,
        transition_inputs=transition_inputs,
        rng_key=keys.latents,
    )
    observations, observations_mask, expected_observations = sample_prior_predictive_emissions(
        spec,
        samples,
        linear_predictors,
        times,
        observation_support=observation_support,
        observation_mask=observation_mask,
        num_samples=num_samples,
        rng_key=keys.observations,
    )
    samples["latents"] = latents
    samples["linear_predictors"] = linear_predictors
    samples["observations"] = observations
    samples["observations_mask"] = observations_mask
    samples["expected_observations"] = expected_observations
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
    seed: int = 42,
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
    sub.update(
        _assemble_extra_params_batched(
            spec,
            sub,
            build_site_registry(spec),
            n_draws=n_use,
        )
    )
    keys = predictive_keys(seed)
    _latents, linear_predictors = _simulate_vector_field_predictive_latents(
        spec,
        sub,
        times,
        transition_inputs=transition_inputs,
        rng_key=keys.latents,
    )
    observations, effective_mask, _expected = sample_predictive_observations_from_linear_predictors(
        linear_predictors,
        sub,
        times,
        manifest_dists=spec.manifest_dists,
        manifest_links=spec.manifest_links,
        manifest_level_counts=spec.manifest_level_counts,
        observation_support=observation_support,
        observation_mask=observation_mask,
        n_subsample=n_use,
        rng_key=keys.observations,
        manifest_names=list(spec.manifest_names) if spec.manifest_names is not None else None,
    )
    return observations, effective_mask
