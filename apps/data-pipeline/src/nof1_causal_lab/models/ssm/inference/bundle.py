"""Shared runtime bundle for the particle inference methods.

:func:`build_particle_runtime_bundle` discretizes the continuous-time SDE at the
observation times and assembles the prior, observation, and trajectory log-prob
closures (and their gradients) that ``marginal_particle_gibbs`` and
``particle_marginal_mh`` consume. Only point measurements are supported; models
with interval-summary observations are rejected here and handled by the Laplace
backend instead.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.dynamics.linearisation import infer_linearisation
from nof1_causal_lab.models.ssm.inference.shared import _trace_public_sites
from nof1_causal_lab.models.ssm.inference.targets.kernels import compile_measurement_semantics
from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
    GaussianTrajectoryPriorTerms,
    _predictive_latent_init,
    build_gaussian_trajectory_prior_terms,
)
from nof1_causal_lab.models.ssm.inference.targets.spec_metadata import has_student_t_diffusion
from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
    trajectory_observation_log_prob,
    trajectory_observation_log_probs,
)
from nof1_causal_lab.models.ssm.inference.targets.transitions import build_discrete_transitions
from nof1_causal_lab.models.ssm.inference.utils import (
    _assemble_likelihood_inputs,
    _build_original_sample_resolver,
    _discover_sites,
    _DummyLikelihoodBackend,
    build_unconstrained_site_transform,
)
from nof1_causal_lab.models.ssm.parameterization import build_site_registry

# Match laplace/shared.py's default jitter so the runtime bundle and the target
# trajectory log-prob agree on the covariance being evaluated.
AUX_JITTER = 1e-6


class LatentContext(NamedTuple):
    Ad: jnp.ndarray | None
    Qd: jnp.ndarray | None
    cd: jnp.ndarray | None
    vf_params: Any
    diffusion_cov: jnp.ndarray
    input_effect: jnp.ndarray | None
    time_intervals: jnp.ndarray
    runtime_times: jnp.ndarray
    transition_inputs: jnp.ndarray | None
    init_mean: jnp.ndarray
    init_cov: jnp.ndarray
    H: jnp.ndarray
    d_meas: jnp.ndarray
    R: jnp.ndarray
    extra_params: dict[str, jnp.ndarray] | None


def _transition_start_linearization_states(
    latent_trajectory: jnp.ndarray,
    init_mean: jnp.ndarray,
) -> jnp.ndarray:
    """Return per-transition start states for local dynamics linearization."""
    return jnp.concatenate((init_mean[None, :], latent_trajectory[:-1]), axis=0)


def _build_context_discrete_transitions(
    dynamics,
    time_intervals: jnp.ndarray,
    *,
    init_mean: jnp.ndarray,
    transition_inputs: jnp.ndarray | None,
):
    """Build transitions for a latent context, including trajectory-dependent drift."""
    if infer_linearisation(dynamics.vector_field) == "constant":
        return build_discrete_transitions(
            dynamics,
            time_intervals,
            transition_inputs=transition_inputs,
        )

    init_ref = jnp.broadcast_to(
        init_mean[None, :],
        (time_intervals.shape[0], dynamics.vector_field.n_latent),
    )
    initial_transitions = build_discrete_transitions(
        dynamics,
        time_intervals,
        linearization_states=init_ref,
        transition_inputs=transition_inputs,
    )
    initial_cd = (
        jnp.zeros(
            (initial_transitions.Ad.shape[0], initial_transitions.Ad.shape[1]),
            dtype=initial_transitions.Ad.dtype,
        )
        if initial_transitions.cd is None
        else jnp.asarray(initial_transitions.cd)
    )
    predictive_path = _predictive_latent_init(
        initial_transitions.Ad,
        initial_cd,
        init_mean,
    )
    return build_discrete_transitions(
        dynamics,
        time_intervals,
        linearization_states=_transition_start_linearization_states(predictive_path, init_mean),
        transition_inputs=transition_inputs,
    )


def _shape_dtype_signature(array: jnp.ndarray) -> tuple[tuple[int, ...], str]:
    return tuple(array.shape), str(jnp.dtype(array.dtype))


def build_particle_runtime_bundle(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    scheme: str,
    trace_key: jnp.ndarray,
    reparam,
) -> dict[str, Any]:
    """Assemble all static helpers needed by the particle inference methods.

    ``scheme`` is the latent discretization the calling inference method requests
    (CT→DT); it is not a property of the model.
    """
    ssm = model
    declared_target = ssm.trajectory_target(scheme)
    if has_student_t_diffusion(ssm.spec):
        raise ValueError(
            "Particle inference currently requires Gaussian latent diffusion for every state."
        )
    observation_support = getattr(ssm, "observation_support", None)
    manifest_links = ssm.spec.manifest_links or [
        LinkFunction.IDENTITY for _ in range(ssm.spec.n_manifest)
    ]
    if observation_support is not None and observation_support.requires_interval_summary_handling:
        raise ValueError(
            "Particle inference (marginal_particle_gibbs / particle_marginal_mh) does not "
            "support interval-summary observations; only point measurements are supported."
        )
    cache_key = (
        "particle_runtime_bundle",
        id(reparam),
        id(observation_support),
        declared_target.kind,
        _shape_dtype_signature(observations),
        _shape_dtype_signature(times),
    )

    def _build_runtime_bundle() -> dict[str, Any]:
        site_info = _discover_sites(
            model,
            observations,
            times,
            trace_key,
            _DummyLikelihoodBackend(),
            reparam=reparam,
        )
        unc_transform = build_unconstrained_site_transform(site_info)
        flat_example = unc_transform.flat_init
        unravel_fn = unc_transform.unconstrain_dict
        sample_resolver = _build_original_sample_resolver(
            site_info,
            model=model,
            observations=observations,
            times=times,
            reparam=reparam,
        )
        if sample_resolver is None:
            raise ValueError(
                "Particle inference only supports no reparameterization or "
                "AutoReparam with fixed centering."
            )

        prior_runtime = model.get_prior_runtime_bundle()
        runtime_registry = build_site_registry(model.spec, model.parameter_layout)
        manifest_chol_template = np.asarray(model.spec.manifest_chol_block.template)
        manifest_chol_offdiag = manifest_chol_template - np.diag(np.diag(manifest_chol_template))
        gaussian_measurement_block_is_diagonal = bool(np.allclose(manifest_chol_offdiag, 0.0))
        measurement_gibbs_gaussian_channel_mask = jnp.asarray(
            [
                gaussian_measurement_block_is_diagonal
                and dist == DistributionFamily.GAUSSIAN
                and link == LinkFunction.IDENTITY
                for dist, link in zip(model.spec.manifest_dists, manifest_links, strict=True)
            ],
            dtype=bool,
        )
        public_sites = _trace_public_sites(
            functools.partial(model.model, likelihood_backend=_DummyLikelihoodBackend()),
            observations,
            times,
        )

        def _constrain(z: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
            return unc_transform.constrain_dict(z), unc_transform.unconstrain_dict(z)

        log_prior_unc_fn = unc_transform.log_prior_unc

        def latent_context_runtime_fn(z: jnp.ndarray, runtime_times: jnp.ndarray) -> LatentContext:
            constrained, _ = _constrain(z)
            original_samples = sample_resolver(constrained)
            dynamics, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
                original_samples,
                model.spec,
                registry=runtime_registry,
                parameter_layout=model.parameter_layout,
            )
            time_intervals = (
                jnp.diff(runtime_times, prepend=runtime_times[0])
                .at[0]
                .set(jnp.asarray(MIN_DT, dtype=runtime_times.dtype))
            )
            transition_inputs = getattr(model, "transition_inputs", None)
            if transition_inputs is not None:
                transition_inputs = transition_inputs[: runtime_times.shape[0]]
            if declared_target.supports_affine_prefix_marginals:
                transitions = _build_context_discrete_transitions(
                    dynamics,
                    time_intervals,
                    init_mean=initial_state.mean,
                    transition_inputs=transition_inputs,
                )
                Ad, Qd = transitions.Ad, transitions.Qd
                cd_scan = (
                    jnp.zeros((Ad.shape[0], Ad.shape[1]), dtype=Ad.dtype)
                    if transitions.cd is None
                    else jnp.asarray(transitions.cd)
                )
            else:
                # Euler-Maruyama: the target discretizes from the vector field on
                # the fly, so the local-linear transitions are never read here.
                Ad = Qd = cd_scan = None
            return LatentContext(
                Ad=Ad,
                Qd=Qd,
                cd=cd_scan,
                vf_params=dynamics.vf_params,
                diffusion_cov=dynamics.diffusion_cov,
                input_effect=dynamics.input_effect,
                time_intervals=time_intervals,
                runtime_times=runtime_times,
                transition_inputs=transition_inputs,
                init_mean=initial_state.mean,
                init_cov=initial_state.cov,
                H=measurement_params.lambda_mat,
                d_meas=measurement_params.manifest_means,
                R=measurement_params.manifest_cov,
                extra_params=extra_params,
            )

        def _measurement_semantics_from_context(context: LatentContext):
            return compile_measurement_semantics(
                model.spec.manifest_dists,
                manifest_cov=context.R,
                extra_params=context.extra_params,
                manifest_links=manifest_links,
                observation_support=observation_support,
            )

        def observation_log_prob_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
        ) -> jnp.ndarray:
            obs_mask = ~jnp.isnan(runtime_observations)
            measurement_semantics = _measurement_semantics_from_context(context)
            obs_lp = trajectory_observation_log_prob(
                latent_trajectory,
                runtime_observations,
                obs_mask,
                context.H,
                context.d_meas,
                context.R,
                measurement_semantics.obs_kernel,
                measurement_semantics.mean_log_prob_fn,
                observation_support,
            )
            return jnp.asarray(obs_lp, dtype=latent_trajectory.dtype)

        def observation_increment_log_prob_from_context_runtime_fn(
            context: LatentContext,
            latent_state: jnp.ndarray,
            time_idx: jnp.ndarray,
            runtime_observations: jnp.ndarray,
        ) -> jnp.ndarray:
            measurement_semantics = _measurement_semantics_from_context(context)
            clean_observations = jnp.nan_to_num(runtime_observations, nan=0.0)
            obs_mask = ~jnp.isnan(runtime_observations)
            y_t = clean_observations[time_idx].astype(latent_state.dtype)
            mask_t = obs_mask[time_idx].astype(latent_state.dtype)
            obs_lp = measurement_semantics.obs_kernel.emission_fn(
                y_t,
                latent_state,
                context.H,
                context.d_meas,
                context.R,
                mask_t,
            )
            return jnp.asarray(obs_lp, dtype=latent_state.dtype)

        def observation_log_prob_per_t_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
        ) -> jnp.ndarray:
            obs_mask = ~jnp.isnan(runtime_observations)
            measurement_semantics = _measurement_semantics_from_context(context)
            per_t = trajectory_observation_log_probs(
                latent_trajectory,
                runtime_observations,
                obs_mask,
                context.H,
                context.d_meas,
                context.R,
                measurement_semantics.obs_kernel,
                measurement_semantics.mean_log_prob_fn,
                observation_support,
            )
            return jnp.asarray(per_t, dtype=latent_trajectory.dtype)

        def observation_log_prob_runtime_fn(
            z: jnp.ndarray,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
            runtime_times: jnp.ndarray,
        ) -> jnp.ndarray:
            context = latent_context_runtime_fn(z, runtime_times)
            return observation_log_prob_from_context_runtime_fn(
                context,
                latent_trajectory,
                runtime_observations,
            )

        observation_grad_runtime_fn = jax.grad(observation_log_prob_runtime_fn, argnums=1)
        observation_grad_from_context_runtime_fn = jax.grad(
            observation_log_prob_from_context_runtime_fn,
            argnums=1,
        )
        observation_log_prob_and_grad_from_context_runtime_fn = jax.value_and_grad(
            observation_log_prob_from_context_runtime_fn,
            argnums=1,
        )

        def prior_terms_from_context_fn(
            context: LatentContext,
        ) -> GaussianTrajectoryPriorTerms | None:
            if not declared_target.supports_affine_prefix_marginals:
                return None
            # Ad/Qd/cd are populated (non-None) exactly when the target supports
            # affine prefix marginals, which is the branch we are in; ty cannot
            # correlate the flag with the LatentContext fields' nullability.
            return build_gaussian_trajectory_prior_terms(
                context.Ad,  # ty: ignore[invalid-argument-type]
                context.Qd,  # ty: ignore[invalid-argument-type]
                context.cd,  # ty: ignore[invalid-argument-type]
                context.init_mean,
                context.init_cov,
                jitter=AUX_JITTER,
            )

        def trajectory_log_prob_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
            prior_terms: GaussianTrajectoryPriorTerms | None = None,
        ) -> jnp.ndarray:
            if prior_terms is None and declared_target.supports_affine_prefix_marginals:
                prior_terms = prior_terms_from_context_fn(context)
            prior_lp = declared_target.trajectory_prior_log_prob(
                context,
                latent_trajectory,
                prior_terms=prior_terms,
            )
            total = prior_lp + observation_log_prob_from_context_runtime_fn(
                context,
                latent_trajectory,
                runtime_observations,
            )
            return jnp.asarray(total, dtype=latent_trajectory.dtype)

        def trajectory_log_prob_runtime_fn(
            z: jnp.ndarray,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
            runtime_times: jnp.ndarray,
        ) -> jnp.ndarray:
            context = latent_context_runtime_fn(z, runtime_times)
            return trajectory_log_prob_from_context_runtime_fn(
                context,
                latent_trajectory,
                runtime_observations,
            )

        def complete_log_posterior_from_context_runtime_fn(
            z: jnp.ndarray,
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            trajectory_lp = trajectory_log_prob_from_context_runtime_fn(
                context,
                latent_trajectory,
                runtime_observations,
            )
            complete_lp = log_prior_unc_fn(z) + trajectory_lp
            return complete_lp, trajectory_lp

        def complete_log_posterior_runtime_fn(
            z: jnp.ndarray,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
            runtime_times: jnp.ndarray,
        ) -> jnp.ndarray:
            context = latent_context_runtime_fn(z, runtime_times)
            complete_lp, _ = complete_log_posterior_from_context_runtime_fn(
                z,
                context,
                latent_trajectory,
                runtime_observations,
            )
            return complete_lp

        def initial_latent_from_context_fn(context: LatentContext) -> jnp.ndarray:
            return declared_target.predictive_latent_init(context)

        def initial_latent_runtime_fn(
            z: jnp.ndarray,
            runtime_times: jnp.ndarray,
        ) -> jnp.ndarray:
            context = latent_context_runtime_fn(z, runtime_times)
            return initial_latent_from_context_fn(context)

        def public_latent_trajectory_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
            key: jnp.ndarray,
        ) -> jnp.ndarray:
            del context, runtime_observations, key
            return latent_trajectory

        def laplace_mode_to_runtime_latent_trajectory_fn(
            latent_mode: jnp.ndarray,
        ) -> jnp.ndarray:
            state = jnp.asarray(latent_mode)
            if state.shape[1] != model.spec.n_latent:
                raise ValueError(
                    "Laplace latent mode width does not match the SSM latent width; "
                    f"got {state.shape[1]}, expected {model.spec.n_latent}."
                )
            return state

        return {
            "dim": int(flat_example.shape[0]),
            "flat_example": flat_example,
            "site_info": site_info,
            "site_registry": runtime_registry,
            "prior_state": prior_runtime.prior_state,
            "unravel_fn": unravel_fn,
            "public_sites": public_sites,
            "manifest_links": tuple(manifest_links),
            "manifest_dists": tuple(model.spec.manifest_dists),
            "measurement_gibbs_gaussian_channel_mask": measurement_gibbs_gaussian_channel_mask,
            "trajectory_target": declared_target,
            "latent_transition_kind": declared_target.kind,
            "log_prior_unc_fn": log_prior_unc_fn,
            "latent_context_runtime_fn": latent_context_runtime_fn,
            "observation_log_prob_runtime_fn": observation_log_prob_runtime_fn,
            "observation_log_prob_from_context_runtime_fn": (
                observation_log_prob_from_context_runtime_fn
            ),
            "observation_log_prob_and_grad_from_context_runtime_fn": (
                observation_log_prob_and_grad_from_context_runtime_fn
            ),
            "observation_log_prob_per_t_from_context_runtime_fn": (
                observation_log_prob_per_t_from_context_runtime_fn
            ),
            "observation_increment_log_prob_from_context_runtime_fn": (
                observation_increment_log_prob_from_context_runtime_fn
            ),
            "observation_grad_runtime_fn": observation_grad_runtime_fn,
            "observation_grad_from_context_runtime_fn": observation_grad_from_context_runtime_fn,
            "trajectory_log_prob_runtime_fn": trajectory_log_prob_runtime_fn,
            "trajectory_log_prob_from_context_runtime_fn": (
                trajectory_log_prob_from_context_runtime_fn
            ),
            "prior_terms_from_context_fn": prior_terms_from_context_fn,
            "complete_log_posterior_from_context_runtime_fn": (
                complete_log_posterior_from_context_runtime_fn
            ),
            "complete_log_posterior_runtime_fn": complete_log_posterior_runtime_fn,
            "initial_latent_runtime_fn": initial_latent_runtime_fn,
            "initial_latent_from_context_fn": initial_latent_from_context_fn,
            "initial_latent_moments_from_context_fn": (declared_target.initial_moments),
            "transition_initial_log_prob_from_context_fn": (declared_target.initial_log_prob),
            "transition_log_prob_from_context_fn": declared_target.transition_log_prob,
            "transition_log_probs_for_pairs_from_context_fn": (
                declared_target.transition_log_probs_for_pairs
            ),
            "transition_pairwise_log_probs_from_context_fn": (
                declared_target.pairwise_transition_log_probs
            ),
            "transition_sample_from_context_fn": declared_target.sample_transition,
            "laplace_mode_to_runtime_latent_trajectory_fn": (
                laplace_mode_to_runtime_latent_trajectory_fn
            ),
            "public_latent_trajectory_runtime_fn": public_latent_trajectory_runtime_fn,
        }

    if hasattr(model, "get_cached_artifact"):
        runtime_bundle = model.get_cached_artifact(cache_key, _build_runtime_bundle)
    else:
        runtime_bundle = _build_runtime_bundle()

    runtime_observations = jnp.asarray(observations)
    runtime_times = jnp.asarray(times)

    def latent_context_fn(z: jnp.ndarray) -> LatentContext:
        return runtime_bundle["latent_context_runtime_fn"](z, runtime_times)

    def observation_log_prob_from_context_fn(
        context: LatentContext,
        latent_trajectory: jnp.ndarray,
    ) -> jnp.ndarray:
        return runtime_bundle["observation_log_prob_from_context_runtime_fn"](
            context,
            latent_trajectory,
            runtime_observations,
        )

    def observation_increment_log_prob_from_context_fn(
        context: LatentContext,
        latent_state: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        return runtime_bundle["observation_increment_log_prob_from_context_runtime_fn"](
            context,
            latent_state,
            time_idx,
            runtime_observations,
        )

    def observation_log_prob_per_t_from_context_fn(
        context: LatentContext,
        latent_trajectory: jnp.ndarray,
    ) -> jnp.ndarray:
        return runtime_bundle["observation_log_prob_per_t_from_context_runtime_fn"](
            context,
            latent_trajectory,
            runtime_observations,
        )

    def observation_log_prob_fn(z: jnp.ndarray, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        return runtime_bundle["observation_log_prob_runtime_fn"](
            z,
            latent_trajectory,
            runtime_observations,
            runtime_times,
        )

    def observation_grad_fn(z: jnp.ndarray, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        return runtime_bundle["observation_grad_runtime_fn"](
            z,
            latent_trajectory,
            runtime_observations,
            runtime_times,
        )

    def trajectory_log_prob_from_context_fn(
        context: LatentContext,
        latent_trajectory: jnp.ndarray,
        prior_terms: GaussianTrajectoryPriorTerms | None = None,
    ) -> jnp.ndarray:
        return runtime_bundle["trajectory_log_prob_from_context_runtime_fn"](
            context,
            latent_trajectory,
            runtime_observations,
            prior_terms=prior_terms,
        )

    def trajectory_log_prob_fn(z: jnp.ndarray, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        return runtime_bundle["trajectory_log_prob_runtime_fn"](
            z,
            latent_trajectory,
            runtime_observations,
            runtime_times,
        )

    def complete_log_posterior_from_context_fn(
        z: jnp.ndarray,
        context: LatentContext,
        latent_trajectory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return runtime_bundle["complete_log_posterior_from_context_runtime_fn"](
            z,
            context,
            latent_trajectory,
            runtime_observations,
        )

    def complete_log_posterior_fn(
        z: jnp.ndarray,
        latent_trajectory: jnp.ndarray,
    ) -> jnp.ndarray:
        return runtime_bundle["complete_log_posterior_runtime_fn"](
            z,
            latent_trajectory,
            runtime_observations,
            runtime_times,
        )

    def initial_latent_fn(z: jnp.ndarray) -> jnp.ndarray:
        return runtime_bundle["initial_latent_runtime_fn"](z, runtime_times)

    return {
        **runtime_bundle,
        "observations": runtime_observations,
        "times": runtime_times,
        "latent_context_fn": latent_context_fn,
        "observation_log_prob_fn": observation_log_prob_fn,
        "observation_log_prob_from_context_fn": observation_log_prob_from_context_fn,
        "observation_log_prob_and_grad_from_context_fn": (
            lambda context, latent_trajectory: runtime_bundle[
                "observation_log_prob_and_grad_from_context_runtime_fn"
            ](
                context,
                latent_trajectory,
                runtime_observations,
            )
        ),
        "observation_log_prob_per_t_from_context_fn": observation_log_prob_per_t_from_context_fn,
        "observation_increment_log_prob_from_context_fn": observation_increment_log_prob_from_context_fn,
        "observation_grad_fn": observation_grad_fn,
        "observation_grad_from_context_fn": (
            lambda context, latent_trajectory: runtime_bundle[
                "observation_grad_from_context_runtime_fn"
            ](
                context,
                latent_trajectory,
                runtime_observations,
            )
        ),
        "trajectory_log_prob_fn": trajectory_log_prob_fn,
        "trajectory_log_prob_from_context_fn": trajectory_log_prob_from_context_fn,
        "complete_log_posterior_from_context_fn": complete_log_posterior_from_context_fn,
        "complete_log_posterior_fn": complete_log_posterior_fn,
        "initial_latent_fn": initial_latent_fn,
    }
