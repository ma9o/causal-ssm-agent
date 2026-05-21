"""Auxiliary-Kalman latent MH proposal families and parameter kernels.

Implements two latent auxiliary-Kalman proposals from Corenflos & Sarkka
(2025, Sec 2.1-2.2):

* ``eq8``: the reparametrised augmentation
  ``u ~ N(x + (delta/2) grad_x f(x; theta), (delta/2) I)``. Under this
  augmentation the LGSSM proposal for ``x`` is independent of the current
  trajectory, so the forward and reverse smoothing densities share one filter.
* ``eq10_11``: the standard non-reparametrised auxiliary proposal
  ``u ~ N(x, (delta/2) I)`` with Kalman pseudo-observations
  ``u + (delta/2) grad_x f(x; theta)``. This matches the paper's main SSM
  construction and requires separate forward and reverse proposal filters
  because each direction is linearised at a different trajectory.

The parameter block updates the complete log-posterior at fixed ``x`` with
either MALA or NUTS. Both kernels report their own accept signal so the scale
adaptation in ``run_aux_kalman_mcmc`` can tune each scale against its own target.
"""

from __future__ import annotations

import functools
import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from blackjax.mcmc import nuts as blackjax_nuts

from nof1_causal_lab.artifacts import LinkFunction
from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.discretization import discretize_system_with_inputs_batched
from nof1_causal_lab.models.ssm.inference.parallel_kalman import (
    aux_filter_lgssm_lightweight,
    sample_lgssm_trajectory,
)
from nof1_causal_lab.models.ssm.inference.shared import _trace_public_sites
from nof1_causal_lab.models.ssm.inference.targets.affine import derive_affine_dynamics
from nof1_causal_lab.models.ssm.inference.targets.kernels import compile_measurement_semantics
from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
    GaussianTrajectoryPriorTerms,
    _build_linear_summary_accumulator_plan,
    _predictive_latent_init,
    build_gaussian_trajectory_prior_terms,
    trajectory_prior_log_prob_from_terms,
)
from nof1_causal_lab.models.ssm.inference.targets.linear_summary_augmentation import (
    build_linear_summary_augmented_system,
    row_observation_log_prob,
    row_observation_log_probs,
)
from nof1_causal_lab.models.ssm.inference.targets.spec_metadata import has_student_t_diffusion
from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
    get_support_kind_codes,
    trajectory_observation_log_prob,
    trajectory_observation_log_probs,
)
from nof1_causal_lab.models.ssm.inference.utils import (
    _assemble_likelihood_inputs,
    _build_original_sample_resolver,
    _discover_sites,
    _DummyLikelihoodBackend,
    build_unconstrained_site_transform,
)
from nof1_causal_lab.models.ssm.parameterization import build_site_registry

# Match laplace/shared.py's default jitter so the auxiliary-Kalman proposal and
# the target trajectory log-prob agree on the covariance being evaluated.
AUX_JITTER = 1e-6


class LatentContext(NamedTuple):
    Ad: jnp.ndarray
    Qd: jnp.ndarray
    cd: jnp.ndarray
    init_mean: jnp.ndarray
    init_cov: jnp.ndarray
    H: jnp.ndarray
    d_meas: jnp.ndarray
    R: jnp.ndarray
    extra_params: dict[str, jnp.ndarray] | None
    H_rows: jnp.ndarray | None
    d_rows: jnp.ndarray | None


def _shape_dtype_signature(array: jnp.ndarray) -> tuple[tuple[int, ...], str]:
    return tuple(array.shape), str(jnp.dtype(array.dtype))


def gaussian_log_prob_isotropic(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    variance: jnp.ndarray,
) -> jnp.ndarray:
    """Log-density of ``N(value; mean, variance * I)``, summed over all elements.

    ``variance`` may be a scalar (same variance across every ``(t, d)`` slot),
    or a ``(T,)`` per-time-step vector (same variance within a time slot but
    different across slots — the per-time δ_t case). Both forms reduce to
    the same total log-probability when ``variance`` is constant.
    """
    if value.ndim == 0:
        raise ValueError("gaussian_log_prob_isotropic requires at least 1-D value.")
    diff = value - mean
    variance_arr = jnp.asarray(variance, dtype=diff.dtype)
    if variance_arr.ndim == 0:
        flat = jnp.reshape(diff, (-1,))
        dim = flat.shape[0]
        return -0.5 * (
            dim * jnp.log(2.0 * jnp.pi * variance_arr) + jnp.sum(flat * flat) / variance_arr
        )
    # Per-time variance: ``value``/``mean`` are (T, D), variance is (T,).
    if diff.ndim < 2:
        raise ValueError(
            "Per-time-step variance requires value with shape (T, D) or larger; "
            f"got value.ndim={diff.ndim}."
        )
    if variance_arr.shape[0] != diff.shape[0]:
        raise ValueError(
            f"Per-time variance shape {variance_arr.shape} incompatible with "
            f"value leading dim {diff.shape[0]}."
        )
    # ``diff.shape[1:]`` is a Python tuple of static ints (JAX tracing always
    # keeps shapes concrete) — use plain Python arithmetic to avoid tracing.
    import math as _math

    per_time_dim = _math.prod(diff.shape[1:])
    ss_per_t = jnp.sum(jnp.reshape(diff, (diff.shape[0], -1)) ** 2, axis=-1)
    logvar_per_t = jnp.log(2.0 * jnp.pi * variance_arr)
    return -0.5 * jnp.sum(per_time_dim * logvar_per_t + ss_per_t / variance_arr)


def _trajectory_prior_log_prob_per_t_from_terms(
    latent_trajectory: jnp.ndarray,
    Ad: jnp.ndarray,
    cd: jnp.ndarray,
    prior_terms,
) -> jnp.ndarray:
    """Per-timestep ``(T,)`` Gaussian trajectory-prior log-prob.

    Slot ``t=0`` contains the initial log-density ``log p(z_0)`` and slot
    ``t>0`` contains the transition log-density ``log p(z_t | z_{t-1})``.
    Summing equals :func:`trajectory_prior_log_prob_from_terms`. Used by
    the per-t MH-ratio diagnostic only.
    """
    from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
        _coerce_transition_intercepts,
        _gaussian_log_prob_from_cholesky,
    )

    T = latent_trajectory.shape[0]
    cd = _coerce_transition_intercepts(
        cd,
        state_dim=int(Ad.shape[1]),
        dtype=jnp.result_type(latent_trajectory, Ad, cd),
    )
    init_ll = _gaussian_log_prob_from_cholesky(
        latent_trajectory[0],
        prior_terms.init_mean,
        prior_terms.init_chol,
        prior_terms.init_logdet,
    )
    if T == 1:
        return jnp.reshape(init_ll, (1,))
    transition_means = jax.vmap(lambda Ad_t, z_tm1, cd_t: Ad_t @ z_tm1 + cd_t)(
        Ad[1:], latent_trajectory[:-1], cd[1:]
    )
    transition_ll = jax.vmap(_gaussian_log_prob_from_cholesky)(
        latent_trajectory[1:],
        transition_means,
        prior_terms.transition_chol,
        prior_terms.transition_logdet,
    )
    return jnp.concatenate([jnp.reshape(init_ll, (1,)), transition_ll])


def gaussian_log_prob_isotropic_per_t(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    variance: jnp.ndarray,
) -> jnp.ndarray:
    """Per-timestep ``(T,)`` log-density of ``N(value; mean, variance * I)``.

    Like :func:`gaussian_log_prob_isotropic` but returns the per-t vector
    whose sum equals the scalar output. Used only by the per-t MH-ratio
    diagnostic; zero cost when the diagnostic flag is off.
    """
    if value.ndim < 2:
        raise ValueError("per-t isotropic density requires value of shape (T, D) or larger.")
    diff = value - mean
    import math as _math

    per_time_dim = _math.prod(diff.shape[1:])
    ss_per_t = jnp.sum(jnp.reshape(diff, (diff.shape[0], -1)) ** 2, axis=-1)
    variance_arr = jnp.asarray(variance, dtype=diff.dtype)
    if variance_arr.ndim == 0:
        logvar_per_t = jnp.log(2.0 * jnp.pi * variance_arr)
        return -0.5 * (per_time_dim * logvar_per_t + ss_per_t / variance_arr)
    if variance_arr.shape[0] != diff.shape[0]:
        raise ValueError(
            f"Per-time variance shape {variance_arr.shape} incompatible with "
            f"value leading dim {diff.shape[0]}."
        )
    logvar_per_t = jnp.log(2.0 * jnp.pi * variance_arr)
    return -0.5 * (per_time_dim * logvar_per_t + ss_per_t / variance_arr)


def _initial_latent_moments(context: LatentContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    init_pred_mean = context.Ad[0] @ context.init_mean + context.cd[0]
    init_pred_cov = symmetrize_with_jitter(
        context.Ad[0] @ context.init_cov @ context.Ad[0].T + context.Qd[0],
        jitter=AUX_JITTER,
    )
    return init_pred_mean, init_pred_cov


def _filter_auxiliary_lgssm_lightweight(
    context: LatentContext,
    pseudo_observations: jnp.ndarray,
    delta: jnp.ndarray,
    *,
    parallel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Auxiliary-LGSSM filter for the latent MH hot path."""
    state = aux_filter_lgssm_lightweight(
        init_mean=context.init_mean,
        init_cov=context.init_cov,
        Fs=context.Ad,
        Qs=context.Qd,
        bs=context.cd,
        pseudo_observations=pseudo_observations,
        aux_variance=0.5 * delta,
        jitter=AUX_JITTER,
        parallel=parallel,
    )
    return (
        state.filt_mean,
        state.filt_cov,
        state.loglik,
    )


def _sample_auxiliary_trajectory(
    key: jnp.ndarray,
    context: LatentContext,
    *,
    filt_means: jnp.ndarray,
    filt_covs: jnp.ndarray,
    parallel: bool = True,
) -> jnp.ndarray:
    return sample_lgssm_trajectory(
        key,
        filt_means,
        filt_covs,
        Fs=context.Ad[1:],
        Qs=context.Qd[1:],
        bs=context.cd[1:],
        jitter=AUX_JITTER,
        parallel=parallel,
    )


def _auxiliary_posterior_log_prob(
    latent_trajectory: jnp.ndarray,
    context: LatentContext,
    pseudo_observations: jnp.ndarray,
    *,
    delta: jnp.ndarray,
    log_evidence: jnp.ndarray,
    prior_terms: GaussianTrajectoryPriorTerms,
) -> jnp.ndarray:
    prior_lp = trajectory_prior_log_prob_from_terms(
        latent_trajectory,
        context.Ad,
        context.cd,
        prior_terms,
    )
    pseudo_lp = gaussian_log_prob_isotropic(
        pseudo_observations,
        latent_trajectory,
        0.5 * delta,
    )
    return prior_lp + pseudo_lp - log_evidence


def _shifted_auxiliary_pseudo_observations(
    u: jnp.ndarray,
    grad: jnp.ndarray,
    half_delta_bcast: jnp.ndarray,
) -> jnp.ndarray:
    """Return the eq-10/11 shifted pseudo-observations ``u + (delta/2) grad``."""
    return u + half_delta_bcast * grad


TULAC_H = float(os.environ.get("TULAC_H", "0.1"))
"""Fixed taming threshold for ``tame_gradient_tulac``.

Chosen so that "well-behaved" gradients (``|grad| ≪ 10``) are essentially
untouched while extreme gradients (``|grad| ≫ 10``) saturate at ``1/h = 10``.
A fixed ``h`` is critical: tying ``h`` to the step size ``δ/2`` would mean
the bounded pseudo-observation perturbation ``(δ/2) * T(grad)`` stays at ±1
per coordinate regardless of δ — so adaptation could shrink δ to the floor
without ever shrinking the proposal magnitude. With fixed ``h = 0.1`` the
bound is ``5 * δ`` per coord, which actually scales down with adaptation.

The env var ``TULAC_H`` overrides the default for sweep experiments."""


def tame_gradient_tulac(grad: jnp.ndarray) -> jnp.ndarray:
    """Coordinatewise tamed gradient (TULAc, Brosse et al. 2017): ``g_i / (1 + h * |g_i|)``.

    Recovers ``grad`` when ``h * |grad| ≪ 1`` and saturates each component at
    ``sign(g_i) / h`` when ``|g_i|`` is huge. With fixed ``h = TULAC_H``
    (currently 0.1), the resulting pseudo-observation perturbation
    ``(δ/2) * T(grad)`` is bounded by ``5 * δ`` per coordinate — proportional
    to the step size so adaptation can actually shrink it.

    Prevents superlinear blow-up from log-link observations (Poisson / NB /
    Gamma with ``exp`` link) from poisoning the MH ratio. Applied identically
    on forward and reverse passes, so the auxiliary-Kalman MH ratio remains
    correct — the proposal kernel is just a different (still valid) kernel.
    """
    return grad / (1.0 + TULAC_H * jnp.abs(grad))


def build_auxiliary_kalman_bundle(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    trace_key: jnp.ndarray,
    reparam,
) -> dict[str, Any]:
    """Assemble all static helpers needed by the auxiliary Kalman method."""
    if has_student_t_diffusion(model.spec):
        raise ValueError(
            "aux_kalman_mcmc with latent_kernel='kalman' currently requires Gaussian latent diffusion for every state."
        )
    observation_support = getattr(model, "observation_support", None)
    cache_key = (
        "aux_kalman_runtime_bundle",
        id(reparam),
        id(observation_support),
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
                "aux_kalman_mcmc with latent_kernel='kalman' only supports no reparameterization or "
                "AutoReparam with fixed centering."
            )

        runtime_registry = build_site_registry(model.spec, model.parameter_layout)
        manifest_links = model.spec.manifest_links or [
            LinkFunction.IDENTITY for _ in range(model.spec.n_manifest)
        ]
        support_kind_codes = (
            get_support_kind_codes(observation_support)
            if observation_support is not None
            else jnp.zeros((model.spec.n_manifest,), dtype=jnp.int64)
        )
        linear_summary_plan = _build_linear_summary_accumulator_plan(
            observation_support,
            model.spec.manifest_dists,
            manifest_links,
        )
        use_linear_summary_augmentation = (
            observation_support is not None
            and observation_support.requires_interval_summary_handling
        )
        if use_linear_summary_augmentation and linear_summary_plan is None:
            raise ValueError(
                "aux_kalman_mcmc with latent_kernel='kalman' only supports linear interval summaries "
                "(mean/sum with supported Gaussian or Student-t identity-link measurements)."
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
            dynamics, measurement_params, initial_state, extra_params = (
                _assemble_likelihood_inputs(
                    original_samples,
                    model.spec,
                    registry=runtime_registry,
                    parameter_layout=model.parameter_layout,
                )
            )
            affine_dynamics = derive_affine_dynamics(dynamics)
            time_intervals = jnp.diff(runtime_times, prepend=runtime_times[0]).at[0].set(MIN_DT)
            transition_inputs = getattr(model, "transition_inputs", None)
            if transition_inputs is not None:
                transition_inputs = transition_inputs[: runtime_times.shape[0]]
            Ad, Qd, cd = discretize_system_with_inputs_batched(
                affine_dynamics.drift,
                affine_dynamics.diffusion_cov,
                affine_dynamics.cint,
                affine_dynamics.input_effect,
                transition_inputs,
                time_intervals,
            )
            cd_scan = (
                jnp.zeros((Ad.shape[0], Ad.shape[1]), dtype=Ad.dtype)
                if cd is None
                else jnp.asarray(cd)
            )
            H_rows = None
            d_rows = None
            init_mean = initial_state.mean
            init_cov = initial_state.cov
            H = measurement_params.lambda_mat
            d_meas = measurement_params.manifest_means
            if use_linear_summary_augmentation:
                (
                    Ad,
                    Qd,
                    cd_scan,
                    init_mean,
                    init_cov,
                    H_rows,
                    d_rows,
                ) = build_linear_summary_augmented_system(
                    plan=linear_summary_plan,
                    time_intervals=time_intervals,
                    drift=affine_dynamics.drift,
                    diffusion_cov=affine_dynamics.diffusion_cov,
                    cint=affine_dynamics.cint
                    if affine_dynamics.cint is not None
                    else jnp.zeros(
                        (affine_dynamics.drift.shape[0],),
                        dtype=affine_dynamics.drift.dtype,
                    ),
                    H=measurement_params.lambda_mat,
                    d=measurement_params.manifest_means,
                    init_mean=initial_state.mean,
                    init_cov=initial_state.cov,
                    support_kind_codes=support_kind_codes,
                )
            return LatentContext(
                Ad=Ad,
                Qd=Qd,
                cd=cd_scan,
                init_mean=init_mean,
                init_cov=init_cov,
                H=H,
                d_meas=d_meas,
                R=measurement_params.manifest_cov,
                extra_params=extra_params,
                H_rows=H_rows,
                d_rows=d_rows,
            )

        def _measurement_semantics_from_context(context: LatentContext):
            return compile_measurement_semantics(
                model.spec.manifest_dists,
                manifest_cov=context.R,
                extra_params=context.extra_params,
                manifest_links=manifest_links,
                observation_support=None
                if use_linear_summary_augmentation
                else observation_support,
            )

        def observation_log_prob_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
        ) -> jnp.ndarray:
            obs_mask = ~jnp.isnan(runtime_observations)
            measurement_semantics = _measurement_semantics_from_context(context)
            if use_linear_summary_augmentation:
                assert context.H_rows is not None
                assert context.d_rows is not None
                obs_lp = row_observation_log_prob(
                    latent_trajectory,
                    runtime_observations,
                    obs_mask,
                    context.H_rows,
                    context.d_rows,
                    context.R,
                    measurement_semantics.obs_kernel,
                )
            else:
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
            if use_linear_summary_augmentation:
                assert context.H_rows is not None
                assert context.d_rows is not None
                H_t = context.H_rows[time_idx]
                d_t = context.d_rows[time_idx]
            else:
                H_t = context.H
                d_t = context.d_meas
            obs_lp = measurement_semantics.obs_kernel.emission_fn(
                y_t,
                latent_state,
                H_t,
                d_t,
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
            if use_linear_summary_augmentation:
                assert context.H_rows is not None
                assert context.d_rows is not None
                per_t = row_observation_log_probs(
                    latent_trajectory,
                    runtime_observations,
                    obs_mask,
                    context.H_rows,
                    context.d_rows,
                    context.R,
                    measurement_semantics.obs_kernel,
                )
            else:
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

        def prior_terms_from_context_fn(context: LatentContext) -> GaussianTrajectoryPriorTerms:
            return build_gaussian_trajectory_prior_terms(
                context.Ad,
                context.Qd,
                context.cd,
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
            if prior_terms is None:
                prior_terms = prior_terms_from_context_fn(context)
            prior_lp = trajectory_prior_log_prob_from_terms(
                latent_trajectory,
                context.Ad,
                context.cd,
                prior_terms,
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

        def complete_log_posterior_with_aux_runtime_fn(
            z: jnp.ndarray,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
            runtime_times: jnp.ndarray,
        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, LatentContext]]:
            context = latent_context_runtime_fn(z, runtime_times)
            complete_lp, trajectory_lp = complete_log_posterior_from_context_runtime_fn(
                z,
                context,
                latent_trajectory,
                runtime_observations,
            )
            return complete_lp, (trajectory_lp, context)

        def initial_latent_from_context_fn(context: LatentContext) -> jnp.ndarray:
            return _predictive_latent_init(context.Ad, context.cd, context.init_mean)

        def initial_latent_runtime_fn(
            z: jnp.ndarray,
            runtime_times: jnp.ndarray,
        ) -> jnp.ndarray:
            context = latent_context_runtime_fn(z, runtime_times)
            return initial_latent_from_context_fn(context)

        return {
            "dim": int(flat_example.shape[0]),
            "flat_example": flat_example,
            "site_info": site_info,
            "unravel_fn": unravel_fn,
            "public_sites": public_sites,
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
            "complete_log_posterior_with_aux_runtime_fn": (
                complete_log_posterior_with_aux_runtime_fn
            ),
            "initial_latent_runtime_fn": initial_latent_runtime_fn,
            "initial_latent_from_context_fn": initial_latent_from_context_fn,
            "initial_latent_moments_from_context_fn": _initial_latent_moments,
            "project_latent_trajectory_fn": (
                (lambda latent_trajectory: latent_trajectory[:, : model.spec.n_latent])
                if use_linear_summary_augmentation
                else (lambda latent_trajectory: latent_trajectory)
            ),
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

    def complete_log_posterior_with_aux_fn(
        z: jnp.ndarray,
        latent_trajectory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, LatentContext]]:
        return runtime_bundle["complete_log_posterior_with_aux_runtime_fn"](
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
        "complete_log_posterior_with_aux_fn": complete_log_posterior_with_aux_fn,
        "initial_latent_fn": initial_latent_fn,
    }


@functools.partial(jax.jit, static_argnames=("runtime_complete_log_posterior_fn",))
def _complete_log_posterior_grad_runtime(
    z: jnp.ndarray,
    latent_trajectory: jnp.ndarray,
    runtime_observations: jnp.ndarray,
    runtime_times: jnp.ndarray,
    *,
    runtime_complete_log_posterior_fn,
) -> jnp.ndarray:
    return jax.grad(runtime_complete_log_posterior_fn, argnums=0)(
        z,
        latent_trajectory,
        runtime_observations,
        runtime_times,
    )


@functools.partial(jax.jit, static_argnames=("runtime_complete_log_posterior_fn",))
def _complete_log_posterior_value_and_grad_runtime(
    z: jnp.ndarray,
    latent_trajectory: jnp.ndarray,
    runtime_observations: jnp.ndarray,
    runtime_times: jnp.ndarray,
    *,
    runtime_complete_log_posterior_fn,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jax.value_and_grad(runtime_complete_log_posterior_fn, argnums=0)(
        z,
        latent_trajectory,
        runtime_observations,
        runtime_times,
    )


def _prepare_latent_step_runtime(
    state,
    runtime_observations: jnp.ndarray,
    *,
    prior_terms_from_context_fn,
    log_prior_unc_fn,
    observation_grad_from_context_runtime_fn,
):
    x_curr = state.latent_trajectory
    context = state.latent_context
    latent_dtype = x_curr.dtype
    traj_dtype = state.trajectory_log_prob.dtype
    complete_dtype = state.complete_log_posterior.dtype
    delta_val = state.latent_delta

    if delta_val.ndim == 0:
        half_delta_bcast = 0.5 * delta_val
    else:
        half_delta_bcast = 0.5 * delta_val[:, None]
    half_delta_variance = 0.5 * delta_val

    prior_terms = prior_terms_from_context_fn(context)
    traj_curr = jnp.asarray(state.trajectory_log_prob, dtype=traj_dtype)
    log_prior_z = jnp.asarray(log_prior_unc_fn(state.position), dtype=complete_dtype)
    grad_curr = jnp.asarray(
        observation_grad_from_context_runtime_fn(context, x_curr, runtime_observations),
        dtype=latent_dtype,
    )
    return (
        x_curr,
        context,
        latent_dtype,
        traj_dtype,
        complete_dtype,
        delta_val,
        half_delta_bcast,
        half_delta_variance,
        prior_terms,
        traj_curr,
        log_prior_z,
        grad_curr,
    )


def _latent_mh_step_eq8_runtime(
    state,
    key: jnp.ndarray,
    runtime_observations: jnp.ndarray,
    *,
    prior_terms_from_context_fn,
    log_prior_unc_fn,
    observation_grad_from_context_runtime_fn,
    observation_log_prob_and_grad_from_context_runtime_fn,
    observation_log_prob_per_t_from_context_runtime_fn,
    parallel: bool,
    emit_per_t_log_alpha: bool,
):
    aux_key, sample_key, accept_key = random.split(key, 3)
    (
        x_curr,
        context,
        latent_dtype,
        traj_dtype,
        complete_dtype,
        delta_val,
        half_delta_bcast,
        half_delta_variance,
        prior_terms,
        traj_curr,
        log_prior_z,
        grad_curr,
    ) = _prepare_latent_step_runtime(
        state,
        runtime_observations,
        prior_terms_from_context_fn=prior_terms_from_context_fn,
        log_prior_unc_fn=log_prior_unc_fn,
        observation_grad_from_context_runtime_fn=observation_grad_from_context_runtime_fn,
    )

    # TULAc coordinatewise gradient taming (fixed h, see TULAC_H). Prevents
    # log-link observation gradients from poisoning the MH ratio. Applied to
    # both ``grad_curr`` and (later) ``grad_prop`` with the same threshold.
    grad_curr = tame_gradient_tulac(grad_curr)
    u = (
        x_curr
        + half_delta_bcast * grad_curr
        + jnp.sqrt(half_delta_bcast) * random.normal(aux_key, x_curr.shape, dtype=latent_dtype)
    )

    filt_means, filt_covs, loglik = _filter_auxiliary_lgssm_lightweight(
        context,
        u,
        delta_val,
        parallel=parallel,
    )
    x_prop = jnp.asarray(
        _sample_auxiliary_trajectory(
            sample_key,
            context,
            filt_means=filt_means,
            filt_covs=filt_covs,
            parallel=parallel,
        ),
        dtype=latent_dtype,
    )
    log_evidence = jnp.sum(loglik)

    q_fwd = jnp.asarray(
        _auxiliary_posterior_log_prob(
            x_prop,
            context,
            u,
            delta=delta_val,
            log_evidence=log_evidence,
            prior_terms=prior_terms,
        ),
        dtype=traj_dtype,
    )
    q_rev = jnp.asarray(
        _auxiliary_posterior_log_prob(
            x_curr,
            context,
            u,
            delta=delta_val,
            log_evidence=log_evidence,
            prior_terms=prior_terms,
        ),
        dtype=traj_dtype,
    )

    obs_prop, grad_prop = observation_log_prob_and_grad_from_context_runtime_fn(
        context,
        x_prop,
        runtime_observations,
    )
    grad_prop = jnp.asarray(grad_prop, dtype=latent_dtype)
    # TULAc taming on the proposal-side gradient (same fixed h as grad_curr).
    grad_prop = tame_gradient_tulac(grad_prop)
    prior_prop = trajectory_prior_log_prob_from_terms(
        x_prop,
        context.Ad,
        context.cd,
        prior_terms,
    )
    traj_prop = jnp.asarray(prior_prop + obs_prop, dtype=traj_dtype)

    log_alpha = traj_prop - traj_curr
    log_alpha = log_alpha + gaussian_log_prob_isotropic(
        u,
        x_prop + half_delta_bcast * grad_prop,
        half_delta_variance,
    )
    log_alpha = log_alpha - gaussian_log_prob_isotropic(
        u,
        x_curr + half_delta_bcast * grad_curr,
        half_delta_variance,
    )
    log_alpha = log_alpha + q_rev - q_fwd

    extras: dict[str, jnp.ndarray] = {}
    if emit_per_t_log_alpha:
        prior_per_t_prop = _trajectory_prior_log_prob_per_t_from_terms(
            x_prop,
            context.Ad,
            context.cd,
            prior_terms,
        )
        prior_per_t_curr = _trajectory_prior_log_prob_per_t_from_terms(
            x_curr,
            context.Ad,
            context.cd,
            prior_terms,
        )
        obs_per_t_prop = observation_log_prob_per_t_from_context_runtime_fn(
            context,
            x_prop,
            runtime_observations,
        )
        obs_per_t_curr = observation_log_prob_per_t_from_context_runtime_fn(
            context,
            x_curr,
            runtime_observations,
        )
        traj_per_t = (prior_per_t_prop + obs_per_t_prop) - (prior_per_t_curr + obs_per_t_curr)
        fwd_prop_per_t = gaussian_log_prob_isotropic_per_t(
            u,
            x_prop + half_delta_bcast * grad_prop,
            half_delta_variance,
        )
        fwd_curr_per_t = gaussian_log_prob_isotropic_per_t(
            u,
            x_curr + half_delta_bcast * grad_curr,
            half_delta_variance,
        )
        q_rev_per_t = _trajectory_prior_log_prob_per_t_from_terms(
            x_curr,
            context.Ad,
            context.cd,
            prior_terms,
        ) + gaussian_log_prob_isotropic_per_t(u, x_curr, half_delta_variance)
        q_fwd_per_t = _trajectory_prior_log_prob_per_t_from_terms(
            x_prop,
            context.Ad,
            context.cd,
            prior_terms,
        ) + gaussian_log_prob_isotropic_per_t(u, x_prop, half_delta_variance)
        log_alpha_per_t = (
            traj_per_t + (fwd_prop_per_t - fwd_curr_per_t) + (q_rev_per_t - q_fwd_per_t)
        )
        extras["log_alpha_per_t"] = log_alpha_per_t.astype(traj_dtype)
        extras["log_alpha_obs_per_t"] = (obs_per_t_prop - obs_per_t_curr).astype(traj_dtype)
        extras["log_alpha_fwd_minus_rev_per_t"] = (fwd_prop_per_t - fwd_curr_per_t).astype(
            traj_dtype
        )
        extras["log_alpha_q_per_t"] = (q_rev_per_t - q_fwd_per_t).astype(traj_dtype)

    accept_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
    accept = random.bernoulli(accept_key, accept_prob)
    next_traj = jnp.asarray(jnp.where(accept, x_prop, x_curr), dtype=latent_dtype)
    next_traj_lp = jnp.asarray(jnp.where(accept, traj_prop, traj_curr), dtype=traj_dtype)
    next_complete = log_prior_z + next_traj_lp.astype(complete_dtype)
    next_state = state._replace(
        latent_trajectory=next_traj,
        trajectory_log_prob=next_traj_lp,
        complete_log_posterior=next_complete,
    )
    extras["accepted"] = accept.astype(state.position.dtype)
    extras["log_alpha"] = log_alpha.astype(traj_dtype)
    return next_state, extras


def _latent_mh_step_eq10_11_runtime(
    state,
    key: jnp.ndarray,
    runtime_observations: jnp.ndarray,
    *,
    prior_terms_from_context_fn,
    log_prior_unc_fn,
    observation_grad_from_context_runtime_fn,
    observation_log_prob_and_grad_from_context_runtime_fn,
    observation_log_prob_per_t_from_context_runtime_fn=None,
    parallel: bool,
    emit_per_t_log_alpha: bool = False,
):
    aux_key, sample_key, accept_key = random.split(key, 3)
    (
        x_curr,
        context,
        latent_dtype,
        traj_dtype,
        complete_dtype,
        delta_val,
        half_delta_bcast,
        half_delta_variance,
        prior_terms,
        traj_curr,
        log_prior_z,
        grad_curr,
    ) = _prepare_latent_step_runtime(
        state,
        runtime_observations,
        prior_terms_from_context_fn=prior_terms_from_context_fn,
        log_prior_unc_fn=log_prior_unc_fn,
        observation_grad_from_context_runtime_fn=observation_grad_from_context_runtime_fn,
    )

    u = x_curr + jnp.sqrt(half_delta_bcast) * random.normal(
        aux_key,
        x_curr.shape,
        dtype=latent_dtype,
    )
    # TULAc coordinatewise gradient taming (fixed h, see TULAC_H). Bounds
    # the pseudo-observation perturbation per coordinate; prevents log-link
    # observation gradients from poisoning the MH ratio. Applied identically
    # to both forward and reverse passes, so MH remains valid.
    grad_curr_tamed = tame_gradient_tulac(grad_curr)
    pseudo_obs_fwd = _shifted_auxiliary_pseudo_observations(u, grad_curr_tamed, half_delta_bcast)
    filt_means_fwd, filt_covs_fwd, loglik_fwd = _filter_auxiliary_lgssm_lightweight(
        context,
        pseudo_obs_fwd,
        delta_val,
        parallel=parallel,
    )
    x_prop = jnp.asarray(
        _sample_auxiliary_trajectory(
            sample_key,
            context,
            filt_means=filt_means_fwd,
            filt_covs=filt_covs_fwd,
            parallel=parallel,
        ),
        dtype=latent_dtype,
    )
    obs_prop, grad_prop = observation_log_prob_and_grad_from_context_runtime_fn(
        context,
        x_prop,
        runtime_observations,
    )
    prior_prop = trajectory_prior_log_prob_from_terms(
        x_prop,
        context.Ad,
        context.cd,
        prior_terms,
    )
    traj_prop = jnp.asarray(prior_prop + obs_prop, dtype=traj_dtype)
    log_evidence_fwd = jnp.sum(loglik_fwd)
    q_fwd = jnp.asarray(
        _auxiliary_posterior_log_prob(
            x_prop,
            context,
            pseudo_obs_fwd,
            delta=delta_val,
            log_evidence=log_evidence_fwd,
            prior_terms=prior_terms,
        ),
        dtype=traj_dtype,
    )

    grad_prop = jnp.asarray(grad_prop, dtype=latent_dtype)
    grad_prop_tamed = tame_gradient_tulac(grad_prop)
    pseudo_obs_rev = _shifted_auxiliary_pseudo_observations(u, grad_prop_tamed, half_delta_bcast)
    _fm_rev, _fc_rev, loglik_rev = _filter_auxiliary_lgssm_lightweight(
        context,
        pseudo_obs_rev,
        delta_val,
        parallel=parallel,
    )
    log_evidence_rev = jnp.sum(loglik_rev)
    q_rev = jnp.asarray(
        _auxiliary_posterior_log_prob(
            x_curr,
            context,
            pseudo_obs_rev,
            delta=delta_val,
            log_evidence=log_evidence_rev,
            prior_terms=prior_terms,
        ),
        dtype=traj_dtype,
    )

    log_alpha = traj_prop - traj_curr
    log_alpha = log_alpha + gaussian_log_prob_isotropic(u, x_prop, half_delta_variance)
    log_alpha = log_alpha - gaussian_log_prob_isotropic(u, x_curr, half_delta_variance)
    log_alpha = log_alpha + q_rev - q_fwd

    extras: dict[str, jnp.ndarray] = {}
    if emit_per_t_log_alpha:
        # eq10_11 MH ratio after prior cancellation:
        #   log_alpha = sum_t [obs_prop_t - obs_curr_t]
        #             + sum_t [iso(u_t|x_prop_t) - iso(u_t|x_curr_t)]
        #             + sum_t [pseudo_lp_rev_t - pseudo_lp_fwd_t]
        #             + (log_evidence_fwd - log_evidence_rev)        # global
        # The first three terms decompose cleanly per-t; the global
        # log-evidence diff is folded into a single "log_alpha_global"
        # field so the probe can still reconstruct the total.
        if observation_log_prob_per_t_from_context_runtime_fn is None:
            raise ValueError(
                "emit_per_t_log_alpha=True requires "
                "observation_log_prob_per_t_from_context_runtime_fn."
            )
        obs_per_t_prop = observation_log_prob_per_t_from_context_runtime_fn(
            context, x_prop, runtime_observations
        )
        obs_per_t_curr = observation_log_prob_per_t_from_context_runtime_fn(
            context, x_curr, runtime_observations
        )
        iso_per_t_xprop = gaussian_log_prob_isotropic_per_t(u, x_prop, half_delta_variance)
        iso_per_t_xcurr = gaussian_log_prob_isotropic_per_t(u, x_curr, half_delta_variance)
        pseudo_lp_rev_per_t = gaussian_log_prob_isotropic_per_t(
            pseudo_obs_rev, x_curr, half_delta_variance
        )
        pseudo_lp_fwd_per_t = gaussian_log_prob_isotropic_per_t(
            pseudo_obs_fwd, x_prop, half_delta_variance
        )
        log_alpha_obs_per_t = (obs_per_t_prop - obs_per_t_curr).astype(traj_dtype)
        log_alpha_fwd_minus_rev_per_t = (iso_per_t_xprop - iso_per_t_xcurr).astype(traj_dtype)
        log_alpha_q_per_t = (pseudo_lp_rev_per_t - pseudo_lp_fwd_per_t).astype(traj_dtype)
        log_alpha_per_t = log_alpha_obs_per_t + log_alpha_fwd_minus_rev_per_t + log_alpha_q_per_t
        extras["log_alpha_obs_per_t"] = log_alpha_obs_per_t
        extras["log_alpha_fwd_minus_rev_per_t"] = log_alpha_fwd_minus_rev_per_t
        extras["log_alpha_q_per_t"] = log_alpha_q_per_t
        extras["log_alpha_per_t"] = log_alpha_per_t
        extras["log_alpha_global"] = jnp.asarray(
            log_evidence_fwd - log_evidence_rev, dtype=traj_dtype
        )

    accept_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
    accept = random.bernoulli(accept_key, accept_prob)
    next_traj = jnp.asarray(jnp.where(accept, x_prop, x_curr), dtype=latent_dtype)
    next_traj_lp = jnp.asarray(jnp.where(accept, traj_prop, traj_curr), dtype=traj_dtype)
    next_complete = log_prior_z + next_traj_lp.astype(complete_dtype)
    next_state = state._replace(
        latent_trajectory=next_traj,
        trajectory_log_prob=next_traj_lp,
        complete_log_posterior=next_complete,
    )
    extras["accepted"] = accept.astype(state.position.dtype)
    extras["log_alpha"] = log_alpha.astype(traj_dtype)
    return next_state, extras


def _parameter_mala_step_runtime(
    state,
    key: jnp.ndarray,
    runtime_observations: jnp.ndarray,
    runtime_times: jnp.ndarray,
    *,
    dim: int,
    runtime_complete_log_posterior_fn,
    runtime_latent_context_fn,
    log_prior_unc_fn,
    precond_chol: jnp.ndarray | None,
    preconditioner_mat: jnp.ndarray | None,
):
    if dim == 0:
        return state, {"accepted": jnp.asarray(1.0, dtype=state.latent_trajectory.dtype)}

    proposal_key, accept_key = random.split(key)
    complete_curr = state.complete_log_posterior
    grad_curr = _complete_log_posterior_grad_runtime(
        state.position,
        state.latent_trajectory,
        runtime_observations,
        runtime_times,
        runtime_complete_log_posterior_fn=runtime_complete_log_posterior_fn,
    )
    # T-MALA / MALTA-style coordinatewise truncated drift (Roberts & Stramer
    # 2002, Atchadé 2006). Same tame_gradient_tulac used on the latent side.
    # Bounds the parameter-side drift so a single superlinear gradient (e.g.
    # 1/sigma blowing up as sigma -> 0) can't push the proposal into a NaN
    # region. Applied symmetrically on forward (grad_curr) and reverse
    # (grad_prop) drifts, so the MH ratio is exact MCMC under the tamed
    # kernel.
    grad_curr = tame_gradient_tulac(grad_curr)
    h = state.param_step_size
    noise = random.normal(proposal_key, state.position.shape, dtype=state.position.dtype)
    if precond_chol is None or preconditioner_mat is None:
        mean_fwd = state.position + 0.5 * (h**2) * grad_curr
        proposal = mean_fwd + h * noise
    else:
        drift_curr = 0.5 * (h**2) * (preconditioner_mat @ grad_curr)
        mean_fwd = state.position + drift_curr
        proposal = mean_fwd + h * (precond_chol @ noise)
    complete_prop, grad_prop = _complete_log_posterior_value_and_grad_runtime(
        proposal,
        state.latent_trajectory,
        runtime_observations,
        runtime_times,
        runtime_complete_log_posterior_fn=runtime_complete_log_posterior_fn,
    )
    grad_prop = tame_gradient_tulac(grad_prop)
    if precond_chol is None or preconditioner_mat is None:
        mean_rev = proposal + 0.5 * (h**2) * grad_prop
        log_alpha = complete_prop - complete_curr
        log_alpha = log_alpha + gaussian_log_prob_isotropic(state.position, mean_rev, h**2)
        log_alpha = log_alpha - gaussian_log_prob_isotropic(proposal, mean_fwd, h**2)
    else:
        drift_prop = 0.5 * (h**2) * (preconditioner_mat @ grad_prop)
        mean_rev = proposal + drift_prop
        log_alpha = complete_prop - complete_curr
        log_alpha = log_alpha + _gaussian_log_prob_preconditioned(
            state.position,
            mean_rev,
            h,
            precond_chol,
        )
        log_alpha = log_alpha - _gaussian_log_prob_preconditioned(
            proposal,
            mean_fwd,
            h,
            precond_chol,
        )
    accept_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
    accept = random.bernoulli(accept_key, accept_prob)

    def _accept_branch(_):
        context_prop = runtime_latent_context_fn(proposal, runtime_times)
        log_prior_prop = log_prior_unc_fn(proposal)
        traj_prop = jnp.asarray(
            complete_prop - log_prior_prop,
            dtype=state.trajectory_log_prob.dtype,
        )
        return state._replace(
            position=proposal,
            latent_context=context_prop,
            trajectory_log_prob=traj_prop,
            complete_log_posterior=complete_prop,
        )

    next_state = jax.lax.cond(accept, _accept_branch, lambda _: state, operand=None)
    return next_state, {"accepted": accept.astype(state.position.dtype)}


_PARAMETER_NUTS_KERNEL = blackjax_nuts.build_kernel()


def _parameter_nuts_step_runtime(
    state,
    key: jnp.ndarray,
    runtime_observations: jnp.ndarray,
    runtime_times: jnp.ndarray,
    *,
    dim: int,
    runtime_complete_log_posterior_fn,
    runtime_latent_context_fn,
    log_prior_unc_fn,
    inverse_mass_matrix: jnp.ndarray,
    max_num_doublings: int,
):
    if dim == 0:
        return state, {
            "accepted": jnp.asarray(1.0, dtype=state.position.dtype),
            "accept_prob": jnp.asarray(1.0, dtype=state.position.dtype),
            "diverging": jnp.asarray(0.0, dtype=state.position.dtype),
            "num_steps": jnp.asarray(0.0, dtype=state.position.dtype),
            "energy": jnp.asarray(0.0, dtype=state.position.dtype),
        }

    def logdensity_fn(position):
        return runtime_complete_log_posterior_fn(
            position,
            state.latent_trajectory,
            runtime_observations,
            runtime_times,
        )

    hmc_state = blackjax_nuts.init(state.position, logdensity_fn)
    proposal_state, info = _PARAMETER_NUTS_KERNEL(
        key,
        hmc_state,
        logdensity_fn,
        state.param_step_size,
        inverse_mass_matrix,
        max_num_doublings,
    )
    context_next = runtime_latent_context_fn(proposal_state.position, runtime_times)
    log_prior_next = log_prior_unc_fn(proposal_state.position)
    trajectory_log_prob_next = jnp.asarray(
        proposal_state.logdensity - log_prior_next,
        dtype=state.trajectory_log_prob.dtype,
    )
    next_state = state._replace(
        position=proposal_state.position,
        latent_context=context_next,
        trajectory_log_prob=trajectory_log_prob_next,
        complete_log_posterior=proposal_state.logdensity,
    )
    dtype = state.position.dtype
    return next_state, {
        "accepted": jnp.asarray(info.acceptance_rate, dtype=dtype),
        "accept_prob": jnp.asarray(info.acceptance_rate, dtype=dtype),
        "diverging": jnp.asarray(info.is_divergent, dtype=dtype),
        "num_steps": jnp.asarray(info.num_integration_steps, dtype=dtype),
        "energy": jnp.asarray(info.energy, dtype=dtype),
    }


def build_auxiliary_kalman_latent_kernel(
    bundle: dict[str, Any],
    *,
    delta: float,
    target_accept: float,
    proposal_family: str = "eq8",
    parallel: bool = True,
    delta_profile: jnp.ndarray | None = None,
    emit_per_t_log_alpha: bool = False,
) -> dict[str, Any]:
    """Auxiliary-Kalman latent MH under the selected proposal family.

    ``proposal_family="eq8"`` uses the reparametrised auxiliary variable
    ``u ~ N(x + (delta/2) grad_x log g(y|x,theta), (delta/2) I)``. Conditional
    on ``u`` and ``theta``, the proposal LGSSM is independent of the current
    trajectory, so one filter pass suffices for both forward and reverse
    proposal densities.

    ``proposal_family="eq10_11"`` uses the paper's standard auxiliary sampler:
    ``u ~ N(x, (delta/2) I)``, then the proposal is the smoothing posterior of
    an LGSSM with pseudo-observations
    ``u + (delta/2) grad_x log g(y|x,theta)`` and observation variance
    ``(delta/2) I``. This requires separate forward and reverse proposal
    filters because the linearisation point changes from ``x`` to ``x*``.

    ``parallel`` selects between the Corenflos/Särkkä O(log T) associative
    scan and a plain O(T) sequential ``lax.scan`` filter/sampler for the
    auxiliary LGSSM proposal.

    ``delta_profile`` (optional) is a ``(T,)`` array of per-time-step step
    sizes δ_t used to stabilise the sampler under heterogeneously informative
    observations (Corenflos & Särkkä §4.4). When provided, the chain
    initialises ``state.latent_delta`` to this array and the filter uses a
    per-time auxiliary-observation noise ``(δ_t/2) I`` instead of a single
    scalar. The global accept/reject is unchanged, so this does not fix the
    ``O(1/T^{1/3})`` asymptotic collapse of the Kalman sampler (Remark 3.1) —
    it redistributes step-size budget across time so a few highly informative
    slots do not drive the whole trajectory accept probability down.
    """
    if proposal_family not in {"eq8", "eq10_11"}:
        raise ValueError(
            f"Unsupported aux-Kalman proposal family {proposal_family!r}. "
            "Supported: 'eq8' or 'eq10_11'."
        )

    def _prepare_latent_step(state):
        x_curr = state.latent_trajectory
        context = state.latent_context
        latent_dtype = x_curr.dtype
        traj_dtype = state.trajectory_log_prob.dtype
        complete_dtype = state.complete_log_posterior.dtype
        delta_val = state.latent_delta

        if delta_val.ndim == 0:
            half_delta_bcast = 0.5 * delta_val
        else:
            half_delta_bcast = 0.5 * delta_val[:, None]
        half_delta_variance = 0.5 * delta_val

        prior_terms = bundle["prior_terms_from_context_fn"](context)
        traj_curr = jnp.asarray(state.trajectory_log_prob, dtype=traj_dtype)
        log_prior_z = jnp.asarray(bundle["log_prior_unc_fn"](state.position), dtype=complete_dtype)
        grad_curr = jnp.asarray(
            bundle["observation_grad_from_context_fn"](context, x_curr),
            dtype=latent_dtype,
        )
        return (
            x_curr,
            context,
            latent_dtype,
            traj_dtype,
            complete_dtype,
            delta_val,
            half_delta_bcast,
            half_delta_variance,
            prior_terms,
            traj_curr,
            log_prior_z,
            grad_curr,
        )

    def _latent_mh_step_eq8(state, key: jnp.ndarray):
        aux_key, sample_key, accept_key = random.split(key, 3)
        (
            x_curr,
            context,
            latent_dtype,
            traj_dtype,
            complete_dtype,
            delta_val,
            half_delta_bcast,
            half_delta_variance,
            prior_terms,
            traj_curr,
            log_prior_z,
            grad_curr,
        ) = _prepare_latent_step(state)

        u = (
            x_curr
            + half_delta_bcast * grad_curr
            + jnp.sqrt(half_delta_bcast) * random.normal(aux_key, x_curr.shape, dtype=latent_dtype)
        )

        filt_means, filt_covs, loglik = _filter_auxiliary_lgssm_lightweight(
            context, u, delta_val, parallel=parallel
        )
        x_prop = jnp.asarray(
            _sample_auxiliary_trajectory(
                sample_key,
                context,
                filt_means=filt_means,
                filt_covs=filt_covs,
                parallel=parallel,
            ),
            dtype=latent_dtype,
        )
        log_evidence = jnp.sum(loglik)

        q_fwd = jnp.asarray(
            _auxiliary_posterior_log_prob(
                x_prop,
                context,
                u,
                delta=delta_val,
                log_evidence=log_evidence,
                prior_terms=prior_terms,
            ),
            dtype=traj_dtype,
        )
        q_rev = jnp.asarray(
            _auxiliary_posterior_log_prob(
                x_curr,
                context,
                u,
                delta=delta_val,
                log_evidence=log_evidence,
                prior_terms=prior_terms,
            ),
            dtype=traj_dtype,
        )

        obs_prop, grad_prop = bundle["observation_log_prob_and_grad_from_context_fn"](
            context,
            x_prop,
        )
        grad_prop = jnp.asarray(grad_prop, dtype=latent_dtype)
        prior_prop = trajectory_prior_log_prob_from_terms(
            x_prop,
            context.Ad,
            context.cd,
            prior_terms,
        )
        traj_prop = jnp.asarray(
            prior_prop + obs_prop,
            dtype=traj_dtype,
        )

        log_alpha = traj_prop - traj_curr
        log_alpha = log_alpha + gaussian_log_prob_isotropic(
            u, x_prop + half_delta_bcast * grad_prop, half_delta_variance
        )
        log_alpha = log_alpha - gaussian_log_prob_isotropic(
            u, x_curr + half_delta_bcast * grad_curr, half_delta_variance
        )
        log_alpha = log_alpha + q_rev - q_fwd

        extras: dict[str, jnp.ndarray] = {}
        if emit_per_t_log_alpha:
            # Per-t decomposition of the eq-8 MH ratio. For an LGSSM target
            # (prior_target == prior_surrogate), the prior contributions cancel
            # between (traj_prop - traj_curr) and (q_rev - q_fwd). For
            # non-Gaussian dynamics they don't; we keep all addends explicit.
            prior_per_t_prop = _trajectory_prior_log_prob_per_t_from_terms(
                x_prop, context.Ad, context.cd, prior_terms
            )
            prior_per_t_curr = _trajectory_prior_log_prob_per_t_from_terms(
                x_curr, context.Ad, context.cd, prior_terms
            )
            obs_per_t_prop = bundle["observation_log_prob_per_t_from_context_fn"](context, x_prop)
            obs_per_t_curr = bundle["observation_log_prob_per_t_from_context_fn"](context, x_curr)
            traj_per_t = (prior_per_t_prop + obs_per_t_prop) - (prior_per_t_curr + obs_per_t_curr)
            fwd_prop_per_t = gaussian_log_prob_isotropic_per_t(
                u, x_prop + half_delta_bcast * grad_prop, half_delta_variance
            )
            fwd_curr_per_t = gaussian_log_prob_isotropic_per_t(
                u, x_curr + half_delta_bcast * grad_curr, half_delta_variance
            )
            q_rev_per_t = _trajectory_prior_log_prob_per_t_from_terms(
                x_curr, context.Ad, context.cd, prior_terms
            ) + gaussian_log_prob_isotropic_per_t(u, x_curr, half_delta_variance)
            q_fwd_per_t = _trajectory_prior_log_prob_per_t_from_terms(
                x_prop, context.Ad, context.cd, prior_terms
            ) + gaussian_log_prob_isotropic_per_t(u, x_prop, half_delta_variance)
            log_alpha_per_t = (
                traj_per_t + (fwd_prop_per_t - fwd_curr_per_t) + (q_rev_per_t - q_fwd_per_t)
            )
            extras["log_alpha_per_t"] = log_alpha_per_t.astype(traj_dtype)
            extras["log_alpha_obs_per_t"] = (obs_per_t_prop - obs_per_t_curr).astype(traj_dtype)
            extras["log_alpha_fwd_minus_rev_per_t"] = (fwd_prop_per_t - fwd_curr_per_t).astype(
                traj_dtype
            )
            extras["log_alpha_q_per_t"] = (q_rev_per_t - q_fwd_per_t).astype(traj_dtype)

        accept_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
        accept = random.bernoulli(accept_key, accept_prob)
        next_traj = jnp.asarray(jnp.where(accept, x_prop, x_curr), dtype=latent_dtype)
        next_traj_lp = jnp.asarray(jnp.where(accept, traj_prop, traj_curr), dtype=traj_dtype)
        next_complete = log_prior_z + next_traj_lp.astype(complete_dtype)
        next_state = state._replace(
            latent_trajectory=next_traj,
            trajectory_log_prob=next_traj_lp,
            complete_log_posterior=next_complete,
        )
        extras["accepted"] = accept.astype(state.position.dtype)
        extras["log_alpha"] = log_alpha.astype(traj_dtype)
        return next_state, extras

    def _latent_mh_step_eq10_11(state, key: jnp.ndarray):
        aux_key, sample_key, accept_key = random.split(key, 3)
        (
            x_curr,
            context,
            latent_dtype,
            traj_dtype,
            complete_dtype,
            delta_val,
            half_delta_bcast,
            half_delta_variance,
            prior_terms,
            traj_curr,
            log_prior_z,
            grad_curr,
        ) = _prepare_latent_step(state)

        u = x_curr + jnp.sqrt(half_delta_bcast) * random.normal(
            aux_key, x_curr.shape, dtype=latent_dtype
        )
        pseudo_obs_fwd = _shifted_auxiliary_pseudo_observations(u, grad_curr, half_delta_bcast)
        filt_means_fwd, filt_covs_fwd, loglik_fwd = _filter_auxiliary_lgssm_lightweight(
            context, pseudo_obs_fwd, delta_val, parallel=parallel
        )
        x_prop = jnp.asarray(
            _sample_auxiliary_trajectory(
                sample_key,
                context,
                filt_means=filt_means_fwd,
                filt_covs=filt_covs_fwd,
                parallel=parallel,
            ),
            dtype=latent_dtype,
        )
        obs_prop, grad_prop = bundle["observation_log_prob_and_grad_from_context_fn"](
            context,
            x_prop,
        )
        prior_prop = trajectory_prior_log_prob_from_terms(
            x_prop,
            context.Ad,
            context.cd,
            prior_terms,
        )
        traj_prop = jnp.asarray(
            prior_prop + obs_prop,
            dtype=traj_dtype,
        )
        log_evidence_fwd = jnp.sum(loglik_fwd)
        q_fwd = jnp.asarray(
            _auxiliary_posterior_log_prob(
                x_prop,
                context,
                pseudo_obs_fwd,
                delta=delta_val,
                log_evidence=log_evidence_fwd,
                prior_terms=prior_terms,
            ),
            dtype=traj_dtype,
        )

        grad_prop = jnp.asarray(grad_prop, dtype=latent_dtype)
        pseudo_obs_rev = _shifted_auxiliary_pseudo_observations(u, grad_prop, half_delta_bcast)
        _fm_rev, _fc_rev, loglik_rev = _filter_auxiliary_lgssm_lightweight(
            context, pseudo_obs_rev, delta_val, parallel=parallel
        )
        log_evidence_rev = jnp.sum(loglik_rev)
        q_rev = jnp.asarray(
            _auxiliary_posterior_log_prob(
                x_curr,
                context,
                pseudo_obs_rev,
                delta=delta_val,
                log_evidence=log_evidence_rev,
                prior_terms=prior_terms,
            ),
            dtype=traj_dtype,
        )

        log_alpha = traj_prop - traj_curr
        log_alpha = log_alpha + gaussian_log_prob_isotropic(u, x_prop, half_delta_variance)
        log_alpha = log_alpha - gaussian_log_prob_isotropic(u, x_curr, half_delta_variance)
        log_alpha = log_alpha + q_rev - q_fwd

        accept_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
        accept = random.bernoulli(accept_key, accept_prob)
        next_traj = jnp.asarray(jnp.where(accept, x_prop, x_curr), dtype=latent_dtype)
        next_traj_lp = jnp.asarray(jnp.where(accept, traj_prop, traj_curr), dtype=traj_dtype)
        next_complete = log_prior_z + next_traj_lp.astype(complete_dtype)
        next_state = state._replace(
            latent_trajectory=next_traj,
            trajectory_log_prob=next_traj_lp,
            complete_log_posterior=next_complete,
        )
        return next_state, {"accepted": accept.astype(state.position.dtype)}

    latent_kernel = {
        "name": "kalman",
        "proposal_family": proposal_family,
        "scale_field": "latent_delta",
        "initial_scale": delta,
        "initial_scale_value": delta,
        "initial_scale_mode": "direct",
        "target_accept": target_accept,
        "step_fn": _latent_mh_step_eq8 if proposal_family == "eq8" else _latent_mh_step_eq10_11,
        "parallel": parallel,
    }
    if delta_profile is not None:
        profile_array = jnp.asarray(delta_profile)
        if profile_array.ndim != 1:
            raise ValueError(f"delta_profile must be 1-D (T,); got shape {profile_array.shape}.")
        latent_kernel["initial_scale_value"] = profile_array
    return latent_kernel


def _gaussian_log_prob_preconditioned(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    scale: jnp.ndarray,
    precond_chol: jnp.ndarray,
) -> jnp.ndarray:
    """Log-density of ``N(value; mean, scale^2 * M)`` with ``M = L L^T``.

    ``precond_chol`` is the lower-triangular Cholesky factor of the
    preconditioner ``M``. The covariance of the proposal is ``h^2 M`` so we
    pass ``scale = h`` and rely on ``precond_chol`` to supply the shape.
    """
    diff = jnp.reshape(value - mean, (-1,))
    dim = diff.shape[0]
    # Solve L^T z = diff / scale  =>  z = L^{-T} diff / scale  =>  Mahalanobis^2 = z^T z.
    whitened = jla.solve_triangular(precond_chol, diff, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(precond_chol)))  # log |L L^T|
    mahal = jnp.sum(whitened * whitened) / (scale * scale)
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + dim * 2.0 * jnp.log(scale) + logdet + mahal)


def build_mala_parameter_kernel(
    bundle: dict[str, Any],
    *,
    step_size: float,
    target_accept: float,
    preconditioner_chol: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """MALA update on the complete log-posterior at fixed latent trajectory.

    ``preconditioner_chol`` is an optional lower-triangular Cholesky factor of
    a positive-definite mass matrix ``M`` with shape ``(dim, dim)``. When
    provided the MALA drift and diffusion become
    ``proposal = theta + 0.5 h^2 M grad + h L Z`` (with ``L L^T = M``) — the
    classical preconditioned MALA of Roberts & Stramer (2002). ``M`` should
    approximate the posterior covariance so MALA mixes in the whitened space.
    """

    if preconditioner_chol is not None:
        precond_chol = jnp.asarray(preconditioner_chol)
        if precond_chol.ndim != 2 or precond_chol.shape[0] != precond_chol.shape[1]:
            raise ValueError(
                f"preconditioner_chol must be square 2-D, got shape {precond_chol.shape}."
            )
        if precond_chol.shape[0] != int(bundle["dim"]):
            raise ValueError(
                "preconditioner_chol side must equal bundle['dim']; "
                f"got {precond_chol.shape[0]} vs {bundle['dim']}."
            )
        preconditioner_mat = precond_chol @ precond_chol.T
    else:
        precond_chol = None
        preconditioner_mat = None

    runtime_complete_log_posterior_fn = bundle.get(
        "complete_log_posterior_runtime_fn",
        lambda z, latent_trajectory, _runtime_observations, _runtime_times: bundle[
            "complete_log_posterior_fn"
        ](z, latent_trajectory),
    )
    runtime_latent_context_fn = bundle.get(
        "latent_context_runtime_fn",
        lambda z, _runtime_times: bundle["latent_context_fn"](z),
    )
    runtime_observations = bundle["observations"]
    runtime_times = bundle["times"]

    def _parameter_mala_step(state, key: jnp.ndarray):
        next_state, info = _parameter_mala_step_runtime(
            state,
            key,
            runtime_observations,
            runtime_times,
            dim=int(bundle["dim"]),
            runtime_complete_log_posterior_fn=runtime_complete_log_posterior_fn,
            runtime_latent_context_fn=runtime_latent_context_fn,
            log_prior_unc_fn=bundle["log_prior_unc_fn"],
            precond_chol=precond_chol,
            preconditioner_mat=preconditioner_mat,
        )
        return next_state, info

    return {
        "name": "mala",
        "scale_field": "param_step_size",
        "initial_scale": step_size,
        "target_accept": target_accept,
        "step_fn": _parameter_mala_step,
        "preconditioned": precond_chol is not None,
        "preconditioner_chol": precond_chol,
        "preconditioner_mat": preconditioner_mat,
    }


def build_nuts_parameter_kernel(
    bundle: dict[str, Any],
    *,
    step_size: float,
    target_accept: float,
    max_num_doublings: int = 10,
    preconditioner_chol: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """NUTS update on the complete log-posterior at fixed latent trajectory.

    ``preconditioner_chol`` follows the same convention as MALA: a Cholesky
    factor of an approximate posterior covariance. BlackJAX expects an inverse
    mass matrix, which is the covariance-shaped object used for the dynamics.
    """
    if max_num_doublings < 1:
        raise ValueError("max_num_doublings must be >= 1.")

    if preconditioner_chol is not None:
        precond_chol = jnp.asarray(preconditioner_chol)
        if precond_chol.ndim != 2 or precond_chol.shape[0] != precond_chol.shape[1]:
            raise ValueError(
                f"preconditioner_chol must be square 2-D, got shape {precond_chol.shape}."
            )
        if precond_chol.shape[0] != int(bundle["dim"]):
            raise ValueError(
                "preconditioner_chol side must equal bundle['dim']; "
                f"got {precond_chol.shape[0]} vs {bundle['dim']}."
            )
        inverse_mass_matrix = precond_chol @ precond_chol.T
    else:
        precond_chol = None
        inverse_mass_matrix = jnp.ones((int(bundle["dim"]),), dtype=bundle["flat_example"].dtype)

    runtime_complete_log_posterior_fn = bundle.get(
        "complete_log_posterior_runtime_fn",
        lambda z, latent_trajectory, _runtime_observations, _runtime_times: bundle[
            "complete_log_posterior_fn"
        ](z, latent_trajectory),
    )
    runtime_latent_context_fn = bundle.get(
        "latent_context_runtime_fn",
        lambda z, _runtime_times: bundle["latent_context_fn"](z),
    )
    runtime_observations = bundle["observations"]
    runtime_times = bundle["times"]

    def _parameter_nuts_step(state, key: jnp.ndarray):
        next_state, info = _parameter_nuts_step_runtime(
            state,
            key,
            runtime_observations,
            runtime_times,
            dim=int(bundle["dim"]),
            runtime_complete_log_posterior_fn=runtime_complete_log_posterior_fn,
            runtime_latent_context_fn=runtime_latent_context_fn,
            log_prior_unc_fn=bundle["log_prior_unc_fn"],
            inverse_mass_matrix=inverse_mass_matrix,
            max_num_doublings=int(max_num_doublings),
        )
        return next_state, info

    return {
        "name": "nuts",
        "scale_field": "param_step_size",
        "initial_scale": step_size,
        "target_accept": target_accept,
        "step_fn": _parameter_nuts_step,
        "preconditioned": precond_chol is not None,
        "preconditioner_chol": precond_chol,
        "inverse_mass_matrix": inverse_mass_matrix,
        "max_num_doublings": int(max_num_doublings),
    }
