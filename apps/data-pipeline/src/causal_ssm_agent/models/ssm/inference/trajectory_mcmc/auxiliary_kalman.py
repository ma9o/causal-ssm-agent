"""Auxiliary-Kalman latent MH proposal families and MALA parameter kernel.

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

The parameter block stays as a MALA update on the complete log-posterior at
fixed ``x``. Both kernels report their own accept signal so the scale
adaptation in ``run_aux_gibbs`` can tune each scale against its own target.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
from jax.flatten_util import ravel_pytree

from causal_ssm_agent.artifacts import LinkFunction
from causal_ssm_agent.models.ssm.constants import MIN_DT
from causal_ssm_agent.models.ssm.covariance_utils import symmetrize_with_jitter
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference.parallel_kalman import (
    aux_filter_lgssm,
    sample_lgssm_trajectory,
)
from causal_ssm_agent.models.ssm.inference.shared import _trace_public_sites
from causal_ssm_agent.models.ssm.inference.targets.graph_analysis import has_student_t_diffusion
from causal_ssm_agent.models.ssm.inference.targets.kernels import compile_measurement_semantics
from causal_ssm_agent.models.ssm.inference.targets.laplace.shared import (
    GaussianTrajectoryPriorTerms,
    _build_gaussian_trajectory_prior_terms,
    _build_linear_summary_accumulator_plan,
    _predictive_latent_init,
    _trajectory_prior_log_prob_from_terms,
)
from causal_ssm_agent.models.ssm.inference.targets.linear_summary_augmentation import (
    build_linear_summary_augmented_system,
    row_observation_log_prob,
    row_observation_log_probs,
)
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    get_support_kind_codes,
    trajectory_observation_log_prob,
    trajectory_observation_log_probs,
)
from causal_ssm_agent.models.ssm.inference.utils import (
    _assemble_likelihood_inputs,
    _build_original_sample_resolver,
    _discover_sites,
    _DummyLikelihoodBackend,
)
from causal_ssm_agent.models.ssm.parameterization import build_site_registry

# Match laplace/shared.py's default jitter so the auxiliary-Kalman proposal and
# the target trajectory log-prob agree on the covariance being evaluated.
_AUX_JITTER = 1e-6


class _LatentContext(NamedTuple):
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


def _gaussian_log_prob_isotropic(
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
        raise ValueError("_gaussian_log_prob_isotropic requires at least 1-D value.")
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
    Summing equals :func:`_trajectory_prior_log_prob_from_terms`. Used by
    the per-t MH-ratio diagnostic only.
    """
    from causal_ssm_agent.models.ssm.inference.targets.laplace.shared import (
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


def _gaussian_log_prob_isotropic_per_t(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    variance: jnp.ndarray,
) -> jnp.ndarray:
    """Per-timestep ``(T,)`` log-density of ``N(value; mean, variance * I)``.

    Like :func:`_gaussian_log_prob_isotropic` but returns the per-t vector
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


def _select_tree(accepted: jnp.ndarray, proposal_tree, current_tree):
    """Elementwise ``where`` over a pytree, tolerating ``None`` leaves."""
    return jax.tree_util.tree_map(
        lambda proposal, current: (
            None if proposal is None else jnp.where(accepted, proposal, current)
        ),
        proposal_tree,
        current_tree,
        is_leaf=lambda leaf: leaf is None,
    )


def _initial_latent_moments(context: _LatentContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    init_pred_mean = context.Ad[0] @ context.init_mean + context.cd[0]
    init_pred_cov = symmetrize_with_jitter(
        context.Ad[0] @ context.init_cov @ context.Ad[0].T + context.Qd[0],
        jitter=_AUX_JITTER,
    )
    return init_pred_mean, init_pred_cov


def _filter_auxiliary_lgssm(
    context: _LatentContext,
    pseudo_observations: jnp.ndarray,
    delta: jnp.ndarray,
    *,
    parallel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Auxiliary-LGSSM Kalman filter backed by :mod:`parallel_kalman`.

    Works for every aux_gibbs path (point-in-time, linear-summary augmented
    block-tridiagonal Ad/Qd, and any other LGSSM whose ``context.Ad`` is
    block-decomposable): the observation matrix is always ``H = I`` because
    the auxiliary proposal observes the full (possibly augmented) state
    with isotropic noise ``(delta/2) I``. ``parallel`` toggles the
    Corenflos/Särkkä O(log T) associative scan against a plain O(T)
    sequential ``lax.scan`` filter.
    """
    state = aux_filter_lgssm(
        init_mean=context.init_mean,
        init_cov=context.init_cov,
        Fs=context.Ad,
        Qs=context.Qd,
        bs=context.cd,
        pseudo_observations=pseudo_observations,
        aux_variance=0.5 * delta,
        jitter=_AUX_JITTER,
        parallel=parallel,
    )
    return (
        state.pred_mean,
        state.pred_cov,
        state.filt_mean,
        state.filt_cov,
        state.loglik,
    )


def _sample_auxiliary_trajectory(
    key: jnp.ndarray,
    context: _LatentContext,
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
        jitter=_AUX_JITTER,
        parallel=parallel,
    )


def _auxiliary_posterior_log_prob(
    latent_trajectory: jnp.ndarray,
    context: _LatentContext,
    pseudo_observations: jnp.ndarray,
    *,
    delta: jnp.ndarray,
    log_evidence: jnp.ndarray,
    prior_terms: GaussianTrajectoryPriorTerms,
) -> jnp.ndarray:
    prior_lp = _trajectory_prior_log_prob_from_terms(
        latent_trajectory,
        context.Ad,
        context.cd,
        prior_terms,
    )
    pseudo_lp = _gaussian_log_prob_isotropic(
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
            "aux_gibbs with latent_kernel='kalman' currently requires Gaussian latent diffusion for every state."
        )

    site_info = _discover_sites(
        model,
        observations,
        times,
        trace_key,
        _DummyLikelihoodBackend(),
        reparam=reparam,
    )
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)

    transforms = {name: info["transform"] for name, info in site_info.items()}
    distributions = {name: info["distribution"] for name, info in site_info.items()}
    sample_resolver = _build_original_sample_resolver(
        site_info,
        model=model,
        observations=observations,
        times=times,
        reparam=reparam,
    )
    if sample_resolver is None:
        raise ValueError(
            "aux_gibbs with latent_kernel='kalman' only supports no reparameterization or AutoReparam with fixed centering."
        )

    runtime_registry = build_site_registry(model.spec, model.structure_runtime)
    obs_mask = ~jnp.isnan(observations)
    time_intervals = jnp.diff(times, prepend=times[0]).at[0].set(MIN_DT)
    observation_support = getattr(model, "observation_support", None)
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
        observation_support is not None and observation_support.requires_interval_summary_handling
    )
    if use_linear_summary_augmentation and linear_summary_plan is None:
        raise ValueError(
            "aux_gibbs with latent_kernel='kalman' only supports linear interval summaries "
            "(mean/sum with supported Gaussian or Student-t identity-link measurements)."
        )
    public_sites = _trace_public_sites(
        functools.partial(model.model, likelihood_backend=_DummyLikelihoodBackend()),
        observations,
        times,
    )

    def _constrain(z: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        unconstrained = unravel_fn(z)
        constrained = {name: transforms[name](unconstrained[name]) for name in unconstrained}
        return constrained, unconstrained

    def log_prior_unc_fn(z: jnp.ndarray) -> jnp.ndarray:
        constrained, unconstrained = _constrain(z)
        log_prior = jnp.array(0.0, dtype=observations.dtype)
        log_jacobian = jnp.array(0.0, dtype=observations.dtype)
        for name in unconstrained:
            log_prior = log_prior + jnp.sum(distributions[name].log_prob(constrained[name]))
            log_jacobian = log_jacobian + jnp.sum(
                transforms[name].log_abs_det_jacobian(unconstrained[name], constrained[name])
            )
        return log_prior + log_jacobian

    def latent_context_fn(z: jnp.ndarray) -> _LatentContext:
        constrained, _ = _constrain(z)
        original_samples = sample_resolver(constrained)
        ct_params, measurement_params, initial_state, extra_params = _assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
            structure_runtime=model.structure_runtime,
        )
        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift,
            ct_params.diffusion_cov,
            ct_params.cint,
            time_intervals,
        )
        cd_scan = (
            jnp.zeros((Ad.shape[0], Ad.shape[1]), dtype=Ad.dtype) if cd is None else jnp.asarray(cd)
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
                drift=ct_params.drift,
                diffusion_cov=ct_params.diffusion_cov,
                cint=ct_params.cint
                if ct_params.cint is not None
                else jnp.zeros((ct_params.drift.shape[0],), dtype=ct_params.drift.dtype),
                H=measurement_params.lambda_mat,
                d=measurement_params.manifest_means,
                init_mean=initial_state.mean,
                init_cov=initial_state.cov,
                support_kind_codes=support_kind_codes,
            )
        return _LatentContext(
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

    def _measurement_semantics_from_context(context: _LatentContext):
        return compile_measurement_semantics(
            model.spec.manifest_dists,
            manifest_cov=context.R,
            extra_params=context.extra_params,
            manifest_links=manifest_links,
            observation_support=None if use_linear_summary_augmentation else observation_support,
        )

    def observation_log_prob_from_context_fn(
        context: _LatentContext,
        latent_trajectory: jnp.ndarray,
    ) -> jnp.ndarray:
        measurement_semantics = _measurement_semantics_from_context(context)
        if use_linear_summary_augmentation:
            assert context.H_rows is not None
            assert context.d_rows is not None
            obs_lp = row_observation_log_prob(
                latent_trajectory,
                observations,
                obs_mask,
                context.H_rows,
                context.d_rows,
                context.R,
                measurement_semantics.obs_kernel,
            )
        else:
            obs_lp = trajectory_observation_log_prob(
                latent_trajectory,
                observations,
                obs_mask,
                context.H,
                context.d_meas,
                context.R,
                measurement_semantics.obs_kernel,
                measurement_semantics.mean_log_prob_fn,
                observation_support,
            )
        return jnp.asarray(obs_lp, dtype=latent_trajectory.dtype)

    clean_observations = jnp.nan_to_num(observations, nan=0.0)

    def observation_increment_log_prob_from_context_fn(
        context: _LatentContext,
        latent_state: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        measurement_semantics = _measurement_semantics_from_context(context)
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

    def observation_log_prob_per_t_from_context_fn(
        context: _LatentContext,
        latent_trajectory: jnp.ndarray,
    ) -> jnp.ndarray:
        """Per-timestep ``(T,)`` observation log-prob. Sum equals the scalar variant.

        Used only by the per-t MH-ratio diagnostic.
        """
        measurement_semantics = _measurement_semantics_from_context(context)
        if use_linear_summary_augmentation:
            assert context.H_rows is not None
            assert context.d_rows is not None
            per_t = row_observation_log_probs(
                latent_trajectory,
                observations,
                obs_mask,
                context.H_rows,
                context.d_rows,
                context.R,
                measurement_semantics.obs_kernel,
            )
        else:
            per_t = trajectory_observation_log_probs(
                latent_trajectory,
                observations,
                obs_mask,
                context.H,
                context.d_meas,
                context.R,
                measurement_semantics.obs_kernel,
                measurement_semantics.mean_log_prob_fn,
                observation_support,
            )
        return jnp.asarray(per_t, dtype=latent_trajectory.dtype)

    def observation_log_prob_fn(z: jnp.ndarray, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        context = latent_context_fn(z)
        return observation_log_prob_from_context_fn(context, latent_trajectory)

    observation_grad_fn = jax.grad(observation_log_prob_fn, argnums=1)
    observation_grad_from_context_fn = jax.grad(observation_log_prob_from_context_fn, argnums=1)

    def _prior_terms_from_context(context: _LatentContext) -> GaussianTrajectoryPriorTerms:
        return _build_gaussian_trajectory_prior_terms(
            context.Ad,
            context.Qd,
            context.cd,
            context.init_mean,
            context.init_cov,
            jitter=_AUX_JITTER,
        )

    def trajectory_log_prob_from_context_fn(
        context: _LatentContext,
        latent_trajectory: jnp.ndarray,
        prior_terms: GaussianTrajectoryPriorTerms | None = None,
    ) -> jnp.ndarray:
        if prior_terms is None:
            prior_terms = _prior_terms_from_context(context)
        prior_lp = _trajectory_prior_log_prob_from_terms(
            latent_trajectory,
            context.Ad,
            context.cd,
            prior_terms,
        )
        total = prior_lp + observation_log_prob_from_context_fn(context, latent_trajectory)
        return jnp.asarray(total, dtype=latent_trajectory.dtype)

    def trajectory_log_prob_fn(z: jnp.ndarray, latent_trajectory: jnp.ndarray) -> jnp.ndarray:
        context = latent_context_fn(z)
        return trajectory_log_prob_from_context_fn(context, latent_trajectory)

    def complete_log_posterior_from_context_fn(
        z: jnp.ndarray,
        context: _LatentContext,
        latent_trajectory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        trajectory_lp = trajectory_log_prob_from_context_fn(context, latent_trajectory)
        complete_lp = log_prior_unc_fn(z) + trajectory_lp
        return complete_lp, trajectory_lp

    def complete_log_posterior_with_aux_fn(
        z: jnp.ndarray,
        latent_trajectory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, _LatentContext]]:
        context = latent_context_fn(z)
        complete_lp, trajectory_lp = complete_log_posterior_from_context_fn(
            z,
            context,
            latent_trajectory,
        )
        return complete_lp, (trajectory_lp, context)

    def initial_latent_from_context_fn(context: _LatentContext) -> jnp.ndarray:
        return _predictive_latent_init(context.Ad, context.cd, context.init_mean)

    def initial_latent_fn(z: jnp.ndarray) -> jnp.ndarray:
        context = latent_context_fn(z)
        return initial_latent_from_context_fn(context)

    return {
        "dim": int(flat_example.shape[0]),
        "flat_example": flat_example,
        "site_info": site_info,
        "unravel_fn": unravel_fn,
        "public_sites": public_sites,
        "log_prior_unc_fn": log_prior_unc_fn,
        "latent_context_fn": latent_context_fn,
        "observation_log_prob_fn": observation_log_prob_fn,
        "observation_log_prob_from_context_fn": observation_log_prob_from_context_fn,
        "observation_log_prob_per_t_from_context_fn": (observation_log_prob_per_t_from_context_fn),
        "observation_increment_log_prob_from_context_fn": (
            observation_increment_log_prob_from_context_fn
        ),
        "observation_grad_fn": observation_grad_fn,
        "observation_grad_from_context_fn": observation_grad_from_context_fn,
        "trajectory_log_prob_fn": trajectory_log_prob_fn,
        "trajectory_log_prob_from_context_fn": trajectory_log_prob_from_context_fn,
        "prior_terms_from_context_fn": _prior_terms_from_context,
        "complete_log_posterior_from_context_fn": complete_log_posterior_from_context_fn,
        "complete_log_posterior_with_aux_fn": complete_log_posterior_with_aux_fn,
        "initial_latent_fn": initial_latent_fn,
        "initial_latent_from_context_fn": initial_latent_from_context_fn,
        "initial_latent_moments_from_context_fn": _initial_latent_moments,
        "project_latent_trajectory_fn": (
            (lambda latent_trajectory: latent_trajectory[:, : model.spec.n_latent])
            if use_linear_summary_augmentation
            else (lambda latent_trajectory: latent_trajectory)
        ),
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
        traj_curr = jnp.asarray(
            bundle["trajectory_log_prob_from_context_fn"](context, x_curr, prior_terms),
            dtype=traj_dtype,
        )
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

        _pm, _pc, filt_means, filt_covs, loglik = _filter_auxiliary_lgssm(
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

        grad_prop = jnp.asarray(
            bundle["observation_grad_from_context_fn"](context, x_prop),
            dtype=latent_dtype,
        )
        traj_prop = jnp.asarray(
            bundle["trajectory_log_prob_from_context_fn"](context, x_prop, prior_terms),
            dtype=traj_dtype,
        )

        log_alpha = traj_prop - traj_curr
        log_alpha = log_alpha + _gaussian_log_prob_isotropic(
            u, x_prop + half_delta_bcast * grad_prop, half_delta_variance
        )
        log_alpha = log_alpha - _gaussian_log_prob_isotropic(
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
            fwd_prop_per_t = _gaussian_log_prob_isotropic_per_t(
                u, x_prop + half_delta_bcast * grad_prop, half_delta_variance
            )
            fwd_curr_per_t = _gaussian_log_prob_isotropic_per_t(
                u, x_curr + half_delta_bcast * grad_curr, half_delta_variance
            )
            q_rev_per_t = _trajectory_prior_log_prob_per_t_from_terms(
                x_curr, context.Ad, context.cd, prior_terms
            ) + _gaussian_log_prob_isotropic_per_t(u, x_curr, half_delta_variance)
            q_fwd_per_t = _trajectory_prior_log_prob_per_t_from_terms(
                x_prop, context.Ad, context.cd, prior_terms
            ) + _gaussian_log_prob_isotropic_per_t(u, x_prop, half_delta_variance)
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
        _pm, _pc, filt_means_fwd, filt_covs_fwd, loglik_fwd = _filter_auxiliary_lgssm(
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
        traj_prop = jnp.asarray(
            bundle["trajectory_log_prob_from_context_fn"](context, x_prop, prior_terms),
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

        grad_prop = jnp.asarray(
            bundle["observation_grad_from_context_fn"](context, x_prop),
            dtype=latent_dtype,
        )
        pseudo_obs_rev = _shifted_auxiliary_pseudo_observations(u, grad_prop, half_delta_bcast)
        _pm_rev, _pc_rev, _fm_rev, _fc_rev, loglik_rev = _filter_auxiliary_lgssm(
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
        log_alpha = log_alpha + _gaussian_log_prob_isotropic(u, x_prop, half_delta_variance)
        log_alpha = log_alpha - _gaussian_log_prob_isotropic(u, x_curr, half_delta_variance)
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
        "target_accept": target_accept,
        "step_fn": _latent_mh_step_eq8 if proposal_family == "eq8" else _latent_mh_step_eq10_11,
        "parallel": parallel,
    }
    if delta_profile is not None:
        profile_array = jnp.asarray(delta_profile)
        if profile_array.ndim != 1:
            raise ValueError(f"delta_profile must be 1-D (T,); got shape {profile_array.shape}.")
        latent_kernel["initial_scale_from_latent_fn"] = lambda _latent_trajectory, dtype: (
            profile_array.astype(dtype)
        )
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

    complete_value_and_grad = jax.value_and_grad(
        bundle["complete_log_posterior_with_aux_fn"],
        argnums=0,
        has_aux=True,
    )
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

    def _parameter_mala_step(state, key: jnp.ndarray):
        if bundle["dim"] == 0:
            return state, {"accepted": jnp.asarray(1.0, dtype=state.latent_trajectory.dtype)}

        proposal_key, accept_key = random.split(key)
        (complete_curr, (traj_curr, _curr_context)), grad_curr = complete_value_and_grad(
            state.position, state.latent_trajectory
        )
        h = state.param_step_size
        noise = random.normal(proposal_key, state.position.shape, dtype=state.position.dtype)
        if precond_chol is None:
            mean_fwd = state.position + 0.5 * (h**2) * grad_curr
            proposal = mean_fwd + h * noise
        else:
            drift_curr = 0.5 * (h**2) * (preconditioner_mat @ grad_curr)
            mean_fwd = state.position + drift_curr
            proposal = mean_fwd + h * (precond_chol @ noise)
        (complete_prop, (traj_prop, context_prop)), grad_prop = complete_value_and_grad(
            proposal, state.latent_trajectory
        )
        if precond_chol is None:
            mean_rev = proposal + 0.5 * (h**2) * grad_prop
            log_alpha = complete_prop - complete_curr
            log_alpha = log_alpha + _gaussian_log_prob_isotropic(state.position, mean_rev, h**2)
            log_alpha = log_alpha - _gaussian_log_prob_isotropic(proposal, mean_fwd, h**2)
        else:
            drift_prop = 0.5 * (h**2) * (preconditioner_mat @ grad_prop)
            mean_rev = proposal + drift_prop
            log_alpha = complete_prop - complete_curr
            log_alpha = log_alpha + _gaussian_log_prob_preconditioned(
                state.position, mean_rev, h, precond_chol
            )
            log_alpha = log_alpha - _gaussian_log_prob_preconditioned(
                proposal, mean_fwd, h, precond_chol
            )
        accept_prob = jnp.exp(jnp.minimum(log_alpha, 0.0))
        accept = random.bernoulli(accept_key, accept_prob)
        next_context = _select_tree(accept, context_prop, state.latent_context)
        next_state = state._replace(
            position=jnp.where(accept, proposal, state.position),
            latent_context=next_context,
            trajectory_log_prob=jnp.where(accept, traj_prop, traj_curr),
            complete_log_posterior=jnp.where(accept, complete_prop, complete_curr),
        )
        return next_state, {"accepted": accept.astype(state.position.dtype)}

    return {
        "name": "mala",
        "scale_field": "param_step_size",
        "initial_scale": step_size,
        "target_accept": target_accept,
        "step_fn": _parameter_mala_step,
        "preconditioned": precond_chol is not None,
    }
