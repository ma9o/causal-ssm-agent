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
from nof1_causal_lab.models.ssm.dynamics.composite import (
    compile_composite,
    pack_component_params_from_samples,
)
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.simulator import simulate
from nof1_causal_lab.models.ssm.inference.targets.observation_families import (
    any_family_needs_level_metadata,
)
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    assemble_deterministics_from_registry,
    assemble_extra_params_from_registry,
    build_prior_runtime_bundle,
    load_prior_runtime_bundle,
    sample_prior_unconstrained,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.priors import PriorRegistry


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

    def initial_condition(self, eta0: jnp.ndarray, args):
        return self.base.initial_condition(eta0, args)

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

    return jax.vmap(_assemble_one)(jnp.arange(n_draws, dtype=jnp.int64))


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


def _simulate_vector_field_predictive_latents(
    spec: SSMSpec,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    *,
    transition_inputs: jnp.ndarray | None,
    seed: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _ensure_gaussian_process_diffusion(spec)
    compiled = compile_composite(spec.drift_spec)
    n_draws = int(next(iter(samples.values())).shape[0])
    draw_keys = random.split(random.PRNGKey(seed), n_draws)
    latents = []
    linear_predictors = []

    for draw_idx in range(n_draws):
        key_init, key_latent = random.split(draw_keys[draw_idx])
        draw = {name: values[draw_idx] for name, values in samples.items()}
        vf_params = pack_component_params_from_samples(spec.drift_spec, draw, draw)
        t0_chol = stable_cholesky(
            draw["t0_cov"],
            min_eigenvalue=INITIAL_STATE_COV_MIN_EIGENVALUE,
        )
        eta0 = draw["t0_means"] + t0_chol @ random.normal(key_init, (spec.n_latent,))
        diffusion_chol = draw["diffusion"]
        if int(times.shape[0]) == 1:
            latent_trajectory = eta0[None, :]
        else:
            vector_field = _prepare_vector_field_for_draw(
                compiled.vector_field,
                input_effect=draw["input_effect"],
                times=times,
                transition_inputs=transition_inputs,
            )
            latent_trajectory = simulate(
                vector_field,
                vf_params,
                Intervention.none(),
                eta0,
                times,
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


def sample_prior_predictive_from_priors(
    spec: SSMSpec,
    priors: PriorRegistry | None,
    times: jnp.ndarray,
    *,
    observation_support=None,
    observation_mask: jnp.ndarray | None = None,
    transition_inputs: jnp.ndarray | None = None,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from a ``PriorRegistry`` directly."""
    runtime = build_prior_runtime_bundle(spec, priors)
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
