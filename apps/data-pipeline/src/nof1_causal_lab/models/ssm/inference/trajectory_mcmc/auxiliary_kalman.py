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

The production parameter block is ``hybrid_gibbs_nuts``: compiler-proved exact
Gibbs blocks are applied first, then a residual NUTS update handles every
remaining parameter. Exact blocks shrink the residual subspace without changing
the MCMC driver API.
"""

from __future__ import annotations

import functools
import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jsp_linalg
import numpy as np
from blackjax.mcmc import nuts as blackjax_nuts

from nof1_causal_lab.artifacts import DistributionFamily, LinkFunction
from nof1_causal_lab.distributions import (
    PriorDistributionFamily,
    get_positive_runtime_kind_from_index,
    get_real_runtime_family_index,
)
from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.dynamics.linearisation import infer_linearisation
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
from nof1_causal_lab.models.ssm.inference.targets.polya_gamma import (
    PolyaGammaObservationPlan,
    build_polya_gamma_observation_plan,
    initialize_polya_gamma_auxiliary_state,
    mask_polya_gamma_observations,
    normalize_polya_gamma_sampler,
    polya_gamma_increment_log_prob,
    polya_gamma_quadratic_log_prob,
    polya_gamma_quadratic_log_probs,
    polya_gamma_sufficient_statistics,
    refresh_polya_gamma_auxiliary_state,
)
from nof1_causal_lab.models.ssm.inference.targets.rao_blackwell import (
    RBPFPartitionSpec,
    build_gaussian_rbpf_observation_plan,
    build_rbpf_marginal_context,
    derive_rbpf_partition,
    mask_rbpf_observations,
    normalize_rbpf_mode,
    rbpf_initial_filter_update,
    rbpf_marginal_log_likelihood,
    rbpf_marginal_log_likelihoods,
    rbpf_step_filter_update,
    reduce_context_to_carried,
    sample_rbpf_marginal_trajectory,
    validate_rbpf_mode,
)
from nof1_causal_lab.models.ssm.inference.targets.spec_metadata import has_student_t_diffusion
from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
    get_support_kind_codes,
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
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass

# Match laplace/shared.py's default jitter so the auxiliary-Kalman proposal and
# the target trajectory log-prob agree on the covariance being evaluated.
AUX_JITTER = 1e-6


def _augment_rbpf_partition_for_linear_summaries(
    partition: RBPFPartitionSpec,
    *,
    n_latent: int,
    linear_summary_plan,
    loading_matrix: jnp.ndarray,
) -> RBPFPartitionSpec:
    """Extend an original-latent RBPF partition across accumulator states."""
    if linear_summary_plan is None:
        return partition

    loading = np.asarray(loading_matrix)
    carried = list(partition.carried_latent_indices)
    marginalized = list(partition.marginalized_latent_indices)
    carried_cols = np.asarray(partition.carried_latent_indices, dtype=np.int64)
    marginalized_cols = np.asarray(partition.marginalized_latent_indices, dtype=np.int64)
    for accumulator_idx, manifest_idx_raw in enumerate(
        np.asarray(linear_summary_plan.accumulator_manifest_indices, dtype=np.int64).tolist()
    ):
        manifest_idx = int(manifest_idx_raw)
        accumulator_state_idx = int(n_latent + accumulator_idx)
        uses_marginalized = bool(np.any(~np.isclose(loading[manifest_idx, marginalized_cols], 0.0)))
        uses_carried = bool(np.any(~np.isclose(loading[manifest_idx, carried_cols], 0.0)))
        if uses_marginalized:
            marginalized.append(accumulator_state_idx)
        elif uses_carried:
            carried.append(accumulator_state_idx)
        else:
            carried.append(accumulator_state_idx)
    return RBPFPartitionSpec(
        carried_latent_indices=tuple(carried),
        marginalized_latent_indices=tuple(marginalized),
    )


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
    rbpf_marginal_context: Any | None
    full_H: jnp.ndarray
    full_d_meas: jnp.ndarray
    full_H_rows: jnp.ndarray | None
    full_d_rows: jnp.ndarray | None


class _LinearPredictorContext(NamedTuple):
    H: jnp.ndarray
    d_meas: jnp.ndarray
    H_rows: jnp.ndarray | None
    d_rows: jnp.ndarray | None
    extra_params: dict[str, jnp.ndarray] | None


class _FlatSiteElement(NamedTuple):
    flat_index: int
    site_name: str
    element_index: int


class _MeasurementGaussianParameterBlock(NamedTuple):
    flat_indices: jnp.ndarray
    prior_mean: jnp.ndarray
    prior_precision: jnp.ndarray
    gaussian_channel_mask: jnp.ndarray
    pg_channel_mask: jnp.ndarray


def _flat_site_elements(flat_example: jnp.ndarray, unravel_fn) -> list[_FlatSiteElement]:
    """Map flat unconstrained coordinates back to ``(site, element)`` entries."""
    dim = int(flat_example.shape[0])
    elements: list[_FlatSiteElement] = []
    for flat_index in range(dim):
        onehot = np.zeros(dim, dtype=np.float64)
        onehot[flat_index] = 1.0
        unraveled = unravel_fn(jnp.asarray(onehot, dtype=flat_example.dtype))
        for site_name, value in unraveled.items():
            array = np.asarray(value)
            if array.shape == ():
                if abs(float(array)) > 1e-10:
                    elements.append(
                        _FlatSiteElement(
                            flat_index=flat_index,
                            site_name=site_name,
                            element_index=0,
                        )
                    )
                    break
                continue
            nonzero = np.argwhere(np.abs(array) > 1e-10)
            if nonzero.size:
                element_tuple = tuple(int(v) for v in nonzero[0])
                element_index = int(np.ravel_multi_index(element_tuple, array.shape))
                elements.append(
                    _FlatSiteElement(
                        flat_index=flat_index,
                        site_name=site_name,
                        element_index=element_index,
                    )
                )
                break
    return elements


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


def _validate_and_compute_exact_pg_max_integer_shape(
    model,
    observations: jnp.ndarray,
    manifest_links: list[LinkFunction],
    *,
    sampler: str,
    enabled: bool,
) -> int | None:
    normalized_sampler = normalize_polya_gamma_sampler(sampler)
    if not enabled or normalized_sampler != "devroye_integer":
        return None

    negative_binomial_mask = np.asarray(
        [
            dist == DistributionFamily.NEGATIVE_BINOMIAL and link == LinkFunction.LOG
            for dist, link in zip(model.spec.manifest_dists, manifest_links, strict=True)
        ],
        dtype=bool,
    )
    if not bool(np.any(negative_binomial_mask)):
        return 1

    observations_np = np.asarray(observations, dtype=np.float64)
    if observations_np.ndim != 2:
        raise ValueError(
            "polya_gamma_sampler='devroye_integer' expects observations with shape "
            f"(n_times, n_manifest); got {observations_np.shape}."
        )
    nb_observations = observations_np[:, negative_binomial_mask]
    active = np.isfinite(nb_observations)
    active_observations = nb_observations[active]
    if active_observations.size:
        if bool(np.any(active_observations < 0.0)):
            raise ValueError(
                "polya_gamma_sampler='devroye_integer' for negative-binomial channels "
                "requires nonnegative integer observations."
            )
        if not bool(np.allclose(active_observations, np.rint(active_observations))):
            raise ValueError(
                "polya_gamma_sampler='devroye_integer' for negative-binomial channels "
                "requires integer-valued observations."
            )
        max_count = int(np.max(np.rint(active_observations)))
    else:
        max_count = 0

    prior_params = model.get_prior_runtime_bundle().prior_state.get("obs_r")
    if prior_params is None:
        raise ValueError(
            "polya_gamma_sampler='devroye_integer' for negative-binomial channels "
            "requires an explicit Delta prior for obs_r."
        )
    obs_r_family = get_positive_runtime_kind_from_index(
        int(np.asarray(prior_params["family"]).reshape(-1)[0])
    )
    if obs_r_family != PriorDistributionFamily.DELTA:
        raise ValueError(
            "polya_gamma_sampler='devroye_integer' for negative-binomial channels requires "
            "obs_r to have a Delta prior, because the exact PG base density depends on obs_r."
        )
    obs_r_value = float(np.asarray(prior_params["value"], dtype=np.float64).reshape(-1)[0])
    if not np.isfinite(obs_r_value) or obs_r_value < 1.0:
        raise ValueError(
            "polya_gamma_sampler='devroye_integer' for negative-binomial channels requires "
            f"fixed obs_r >= 1; got {obs_r_value}."
        )
    if not bool(np.isclose(obs_r_value, round(obs_r_value))):
        raise ValueError(
            "polya_gamma_sampler='devroye_integer' for negative-binomial channels requires "
            f"integer-valued obs_r; got {obs_r_value}."
        )
    return max(1, max_count + round(obs_r_value))


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
    polya_gamma_num_terms: int = 64,
    polya_gamma_sampler: str = "truncated_sum",
    enable_polya_gamma: bool = True,
    rbpf_mode: str = "none",
    rbpf_marginalized_latent_indices: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    """Assemble all static helpers needed by the auxiliary Kalman method."""
    if has_student_t_diffusion(model.spec):
        raise ValueError(
            "aux_kalman_mcmc with latent_kernel='kalman' currently requires Gaussian latent diffusion for every state."
        )
    normalized_rbpf_mode = normalize_rbpf_mode(rbpf_mode)
    marginalized_indices = (
        None
        if rbpf_marginalized_latent_indices is None
        else tuple(int(idx) for idx in rbpf_marginalized_latent_indices)
    )
    if normalized_rbpf_mode == "none" and marginalized_indices:
        raise ValueError(
            "rbpf_mode='none' cannot be combined with rbpf_marginalized_latent_indices."
        )
    observation_support = getattr(model, "observation_support", None)
    manifest_links = model.spec.manifest_links or [
        LinkFunction.IDENTITY for _ in range(model.spec.n_manifest)
    ]
    normalized_polya_gamma_sampler = normalize_polya_gamma_sampler(polya_gamma_sampler)
    exact_pg_max_integer_shape = _validate_and_compute_exact_pg_max_integer_shape(
        model,
        observations,
        manifest_links,
        sampler=normalized_polya_gamma_sampler,
        enabled=bool(enable_polya_gamma),
    )
    cache_key = (
        "aux_kalman_runtime_bundle",
        id(reparam),
        id(observation_support),
        int(polya_gamma_num_terms),
        normalized_polya_gamma_sampler,
        exact_pg_max_integer_shape,
        bool(enable_polya_gamma),
        normalized_rbpf_mode,
        marginalized_indices,
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

        prior_runtime = model.get_prior_runtime_bundle()
        runtime_registry = build_site_registry(model.spec, model.parameter_layout)
        polya_gamma_plan = build_polya_gamma_observation_plan(
            model.spec.manifest_dists,
            manifest_links,
            num_terms=int(polya_gamma_num_terms),
            sampler=normalized_polya_gamma_sampler,
            enabled=bool(enable_polya_gamma),
            max_integer_shape=exact_pg_max_integer_shape,
        )
        rbpf_partition, rbpf_partition_diagnostics = derive_rbpf_partition(
            spec=model.spec,
            rbpf_mode=normalized_rbpf_mode,
            marginalized_latent_indices=marginalized_indices,
            manifest_links=manifest_links,
            polya_gamma_channel_mask=polya_gamma_plan.channel_mask,
        )
        rbpf_observation_plan = build_gaussian_rbpf_observation_plan(
            model.spec,
            rbpf_partition,
            manifest_links,
            polya_gamma_plan.channel_mask,
        )
        validate_rbpf_mode(normalized_rbpf_mode, rbpf_partition, rbpf_observation_plan)
        rbpf_active = not rbpf_partition.is_full_path
        rbpf_structure = rbpf_observation_plan.structure
        rbpf_has_polya_gamma_rows = bool(
            np.any(np.asarray(rbpf_observation_plan.polya_gamma_channel_mask))
        )
        residual_polya_gamma_mask = polya_gamma_plan.channel_mask & (
            ~rbpf_observation_plan.polya_gamma_channel_mask
        )
        residual_polya_gamma_plan = PolyaGammaObservationPlan(
            channel_mask=residual_polya_gamma_mask,
            bernoulli_channel_mask=polya_gamma_plan.bernoulli_channel_mask
            & residual_polya_gamma_mask,
            negative_binomial_channel_mask=polya_gamma_plan.negative_binomial_channel_mask
            & residual_polya_gamma_mask,
            num_terms=polya_gamma_plan.num_terms,
            sampler=polya_gamma_plan.sampler,
            enabled=bool(np.any(np.asarray(residual_polya_gamma_mask))),
            consumes_all_channels=bool(np.all(np.asarray(residual_polya_gamma_mask)))
            if residual_polya_gamma_mask.size
            else False,
            max_integer_shape=polya_gamma_plan.max_integer_shape,
        )
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
        rbpf_state_partition = rbpf_partition
        if rbpf_active and use_linear_summary_augmentation:
            loading_active = np.asarray(model.spec.lambda_block.free_support, dtype=bool) | (
                ~np.isclose(np.asarray(model.spec.lambda_block.template), 0.0)
            )
            rbpf_state_partition = _augment_rbpf_partition_for_linear_summaries(
                rbpf_partition,
                n_latent=model.spec.n_latent,
                linear_summary_plan=linear_summary_plan,
                loading_matrix=jnp.asarray(loading_active, dtype=jnp.float32),
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
            transitions = _build_context_discrete_transitions(
                dynamics,
                time_intervals,
                init_mean=initial_state.mean,
                transition_inputs=transition_inputs,
            )
            Ad, Qd, cd = transitions.Ad, transitions.Qd, transitions.cd
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
            full_H = H
            full_d_meas = d_meas
            full_H_rows = None
            full_d_rows = None
            if use_linear_summary_augmentation:
                affine_dynamics = derive_affine_dynamics(dynamics)
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
                full_H = H
                full_d_meas = d_meas
                full_H_rows = H_rows
                full_d_rows = d_rows
            rbpf_marginal_context = None
            if rbpf_active:
                rbpf_marginal_context = build_rbpf_marginal_context(
                    Ad=Ad,
                    Qd=Qd,
                    cd=cd_scan,
                    init_mean=init_mean,
                    init_cov=init_cov,
                    H=H,
                    H_rows=H_rows,
                    d_meas=d_meas,
                    d_meas_rows=d_rows,
                    R=measurement_params.manifest_cov,
                    partition=rbpf_state_partition,
                    observation_plan=rbpf_observation_plan,
                    extra_params=extra_params,
                )
                (
                    Ad,
                    Qd,
                    cd_scan,
                    init_mean,
                    init_cov,
                    H,
                ) = reduce_context_to_carried(
                    Ad=Ad,
                    Qd=Qd,
                    cd=cd_scan,
                    init_mean=init_mean,
                    init_cov=init_cov,
                    H=H,
                    partition=rbpf_state_partition,
                )
                if H_rows is not None:
                    H_rows = jnp.take(
                        H_rows,
                        jnp.asarray(rbpf_state_partition.carried_latent_indices, dtype=jnp.int32),
                        axis=2,
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
                rbpf_marginal_context=rbpf_marginal_context,
                full_H=full_H,
                full_d_meas=full_d_meas,
                full_H_rows=full_H_rows,
                full_d_rows=full_d_rows,
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

        def _full_linear_predictor_context(context: LatentContext) -> _LinearPredictorContext:
            return _LinearPredictorContext(
                H=context.full_H,
                d_meas=context.full_d_meas,
                H_rows=context.full_H_rows,
                d_rows=context.full_d_rows,
                extra_params=context.extra_params,
            )

        def _carried_linear_predictor_context(context: LatentContext) -> _LinearPredictorContext:
            return _LinearPredictorContext(
                H=context.H,
                d_meas=context.d_meas,
                H_rows=context.H_rows,
                d_rows=context.d_rows,
                extra_params=context.extra_params,
            )

        def _reconstruct_full_latent_trajectory(
            carried_trajectory: jnp.ndarray,
            marginalized_trajectory: jnp.ndarray,
        ) -> jnp.ndarray:
            if not rbpf_active:
                return carried_trajectory
            state_dim = (
                model.spec.n_latent + linear_summary_plan.n_accumulators
                if use_linear_summary_augmentation
                else model.spec.n_latent
            )
            full = jnp.zeros(
                (carried_trajectory.shape[0], state_dim),
                dtype=carried_trajectory.dtype,
            )
            carried_idx = jnp.asarray(rbpf_state_partition.carried_latent_indices, dtype=jnp.int32)
            marginalized_idx = jnp.asarray(
                rbpf_state_partition.marginalized_latent_indices,
                dtype=jnp.int32,
            )
            full = full.at[:, carried_idx].set(carried_trajectory)
            return full.at[:, marginalized_idx].set(marginalized_trajectory)

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

        def initial_observation_auxiliary_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            runtime_observations: jnp.ndarray,
        ):
            pg_context = _carried_linear_predictor_context(context)
            pg_latent_trajectory = latent_trajectory
            if rbpf_active and rbpf_has_polya_gamma_rows:
                marginalized_mean = jnp.broadcast_to(
                    context.rbpf_marginal_context.init_mean_m,
                    (
                        latent_trajectory.shape[0],
                        context.rbpf_marginal_context.init_mean_m.shape[0],
                    ),
                )
                pg_latent_trajectory = _reconstruct_full_latent_trajectory(
                    latent_trajectory,
                    marginalized_mean.astype(latent_trajectory.dtype),
                )
                pg_context = _full_linear_predictor_context(context)
            return initialize_polya_gamma_auxiliary_state(
                polya_gamma_plan,
                pg_context,
                pg_latent_trajectory,
                runtime_observations,
            )

        def refresh_observation_auxiliary_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            _observation_auxiliary,
            runtime_observations: jnp.ndarray,
            key: jnp.ndarray,
        ):
            if not polya_gamma_plan.enabled:
                return _observation_auxiliary
            pg_context = _carried_linear_predictor_context(context)
            pg_latent_trajectory = latent_trajectory
            if rbpf_active and rbpf_has_polya_gamma_rows:
                sample_key, pg_key = jax.random.split(key)
                marginalized_trajectory = sample_rbpf_marginal_trajectory(
                    sample_key,
                    context.rbpf_marginal_context,
                    runtime_observations,
                    _observation_auxiliary,
                    latent_trajectory,
                    jitter=AUX_JITTER,
                )
                pg_latent_trajectory = _reconstruct_full_latent_trajectory(
                    latent_trajectory,
                    marginalized_trajectory,
                )
                pg_context = _full_linear_predictor_context(context)
                key = pg_key
            return refresh_polya_gamma_auxiliary_state(
                key,
                polya_gamma_plan,
                pg_context,
                pg_latent_trajectory,
                runtime_observations,
            )

        def residual_observations_runtime_fn(runtime_observations: jnp.ndarray) -> jnp.ndarray:
            return mask_rbpf_observations(
                rbpf_observation_plan,
                mask_polya_gamma_observations(
                    residual_polya_gamma_plan,
                    runtime_observations,
                ),
            )

        def rbpf_marginal_log_prob_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            observation_auxiliary,
            runtime_observations: jnp.ndarray,
        ) -> jnp.ndarray:
            if rbpf_has_polya_gamma_rows and observation_auxiliary is None:
                raise ValueError(
                    "PG-conditioned RBPF likelihood requires a Polya-Gamma auxiliary state."
                )
            return rbpf_marginal_log_likelihood(
                context.rbpf_marginal_context,
                runtime_observations,
                observation_auxiliary,
                latent_trajectory,
                jitter=AUX_JITTER,
            )

        def rbpf_marginal_log_probs_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            observation_auxiliary,
            runtime_observations: jnp.ndarray,
        ) -> jnp.ndarray:
            if rbpf_has_polya_gamma_rows and observation_auxiliary is None:
                raise ValueError(
                    "PG-conditioned RBPF likelihood requires a Polya-Gamma auxiliary state."
                )
            return rbpf_marginal_log_likelihoods(
                context.rbpf_marginal_context,
                runtime_observations,
                observation_auxiliary,
                latent_trajectory,
                jitter=AUX_JITTER,
            )

        def observation_log_prob_conditioned_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            observation_auxiliary,
            runtime_observations: jnp.ndarray,
        ) -> jnp.ndarray:
            residual_lp = (
                jnp.asarray(0.0, dtype=latent_trajectory.dtype)
                if polya_gamma_plan.consumes_all_channels
                else observation_log_prob_from_context_runtime_fn(
                    context,
                    latent_trajectory,
                    residual_observations_runtime_fn(runtime_observations),
                )
            )
            pg_lp = polya_gamma_quadratic_log_prob(
                residual_polya_gamma_plan,
                observation_auxiliary,
                context,
                latent_trajectory,
            )
            rbpf_lp = rbpf_marginal_log_prob_from_context_runtime_fn(
                context,
                latent_trajectory,
                observation_auxiliary,
                runtime_observations,
            )
            return jnp.asarray(residual_lp + pg_lp + rbpf_lp, dtype=latent_trajectory.dtype)

        def observation_increment_log_prob_conditioned_from_context_runtime_fn(
            context: LatentContext,
            latent_state: jnp.ndarray,
            observation_auxiliary,
            time_idx: jnp.ndarray,
            runtime_observations: jnp.ndarray,
        ) -> jnp.ndarray:
            residual_lp = (
                jnp.asarray(0.0, dtype=latent_state.dtype)
                if polya_gamma_plan.consumes_all_channels
                else observation_increment_log_prob_from_context_runtime_fn(
                    context,
                    latent_state,
                    time_idx,
                    residual_observations_runtime_fn(runtime_observations),
                )
            )
            pg_lp = polya_gamma_increment_log_prob(
                residual_polya_gamma_plan,
                observation_auxiliary,
                context,
                latent_state,
                time_idx,
            )
            return jnp.asarray(residual_lp + pg_lp, dtype=latent_state.dtype)

        def observation_log_prob_per_t_conditioned_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            observation_auxiliary,
            runtime_observations: jnp.ndarray,
        ) -> jnp.ndarray:
            residual_per_t = (
                jnp.zeros((latent_trajectory.shape[0],), dtype=latent_trajectory.dtype)
                if polya_gamma_plan.consumes_all_channels
                else observation_log_prob_per_t_from_context_runtime_fn(
                    context,
                    latent_trajectory,
                    residual_observations_runtime_fn(runtime_observations),
                )
            )
            pg_per_t = polya_gamma_quadratic_log_probs(
                residual_polya_gamma_plan,
                observation_auxiliary,
                context,
                latent_trajectory,
            )
            rbpf_per_t = rbpf_marginal_log_probs_from_context_runtime_fn(
                context,
                latent_trajectory,
                observation_auxiliary,
                runtime_observations,
            )
            return jnp.asarray(
                residual_per_t + pg_per_t + rbpf_per_t, dtype=latent_trajectory.dtype
            )

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
        observation_grad_conditioned_from_context_runtime_fn = jax.grad(
            observation_log_prob_conditioned_from_context_runtime_fn,
            argnums=1,
        )
        observation_log_prob_and_grad_conditioned_from_context_runtime_fn = jax.value_and_grad(
            observation_log_prob_conditioned_from_context_runtime_fn,
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
                mask_rbpf_observations(rbpf_observation_plan, runtime_observations),
            )
            rbpf_lp = rbpf_marginal_log_prob_from_context_runtime_fn(
                context,
                latent_trajectory,
                None,
                runtime_observations,
            )
            return jnp.asarray(total + rbpf_lp, dtype=latent_trajectory.dtype)

        def trajectory_log_prob_conditioned_from_context_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            observation_auxiliary,
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
            obs_lp = observation_log_prob_conditioned_from_context_runtime_fn(
                context,
                latent_trajectory,
                observation_auxiliary,
                runtime_observations,
            )
            return jnp.asarray(prior_lp + obs_lp, dtype=latent_trajectory.dtype)

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

        def complete_log_posterior_conditioned_from_context_runtime_fn(
            z: jnp.ndarray,
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            observation_auxiliary,
            runtime_observations: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            trajectory_lp = trajectory_log_prob_conditioned_from_context_runtime_fn(
                context,
                latent_trajectory,
                observation_auxiliary,
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

        def complete_log_posterior_conditioned_runtime_fn(
            z: jnp.ndarray,
            latent_trajectory: jnp.ndarray,
            observation_auxiliary,
            runtime_observations: jnp.ndarray,
            runtime_times: jnp.ndarray,
        ) -> jnp.ndarray:
            context = latent_context_runtime_fn(z, runtime_times)
            complete_lp, _ = complete_log_posterior_conditioned_from_context_runtime_fn(
                z,
                context,
                latent_trajectory,
                observation_auxiliary,
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

        def public_latent_trajectory_runtime_fn(
            context: LatentContext,
            latent_trajectory: jnp.ndarray,
            observation_auxiliary,
            runtime_observations: jnp.ndarray,
            key: jnp.ndarray,
        ) -> jnp.ndarray:
            if rbpf_active:
                marginalized_trajectory = sample_rbpf_marginal_trajectory(
                    key,
                    context.rbpf_marginal_context,
                    runtime_observations,
                    observation_auxiliary,
                    latent_trajectory,
                    jitter=AUX_JITTER,
                )
                full_trajectory = _reconstruct_full_latent_trajectory(
                    latent_trajectory,
                    marginalized_trajectory,
                )
            else:
                del key, observation_auxiliary, runtime_observations
                full_trajectory = latent_trajectory
            if use_linear_summary_augmentation:
                return full_trajectory[:, : model.spec.n_latent]
            return full_trajectory

        def laplace_mode_to_runtime_latent_trajectory_fn(
            latent_mode: jnp.ndarray,
        ) -> jnp.ndarray:
            state = jnp.asarray(latent_mode)
            if use_linear_summary_augmentation:
                state_dim = model.spec.n_latent + linear_summary_plan.n_accumulators
                if state.shape[1] == model.spec.n_latent:
                    zeros = jnp.zeros(
                        (state.shape[0], linear_summary_plan.n_accumulators),
                        dtype=state.dtype,
                    )
                    state = jnp.concatenate([state, zeros], axis=1)
                if state.shape[1] != state_dim:
                    raise ValueError(
                        "Laplace latent mode width does not match the SSM latent width or "
                        f"linear-summary augmented width; got {state.shape[1]}, expected "
                        f"{model.spec.n_latent} or {state_dim}."
                    )
            elif state.shape[1] != model.spec.n_latent:
                raise ValueError(
                    "Laplace latent mode width does not match the SSM latent width; "
                    f"got {state.shape[1]}, expected {model.spec.n_latent}."
                )
            if rbpf_active:
                carried_idx = jnp.asarray(
                    rbpf_state_partition.carried_latent_indices,
                    dtype=jnp.int32,
                )
                return jnp.take(state, carried_idx, axis=1)
            return state

        return {
            "dim": int(flat_example.shape[0]),
            "flat_example": flat_example,
            "site_info": site_info,
            "site_registry": runtime_registry,
            "prior_state": prior_runtime.prior_state,
            "unravel_fn": unravel_fn,
            "public_sites": public_sites,
            "polya_gamma_plan": polya_gamma_plan,
            "residual_polya_gamma_plan": residual_polya_gamma_plan,
            "polya_gamma_enabled": bool(enable_polya_gamma),
            "polya_gamma_sampler": polya_gamma_plan.sampler,
            "polya_gamma_max_integer_shape": polya_gamma_plan.max_integer_shape,
            "manifest_links": tuple(manifest_links),
            "manifest_dists": tuple(model.spec.manifest_dists),
            "measurement_gibbs_gaussian_channel_mask": measurement_gibbs_gaussian_channel_mask,
            "linear_summary_augmented": bool(use_linear_summary_augmentation),
            "rbpf_partition": rbpf_partition,
            "rbpf_partition_diagnostics": rbpf_partition_diagnostics.asdict(),
            "rbpf_observation_plan": rbpf_observation_plan,
            "rbpf_enabled": rbpf_active,
            "rbpf_requested": bool(normalized_rbpf_mode != "none"),
            "rbpf_mode": normalized_rbpf_mode,
            "rbpf_structure": rbpf_structure,
            "rbpf_initial_filter_update_fn": rbpf_initial_filter_update,
            "rbpf_step_filter_update_fn": rbpf_step_filter_update,
            "log_prior_unc_fn": log_prior_unc_fn,
            "latent_context_runtime_fn": latent_context_runtime_fn,
            "initial_observation_auxiliary_from_context_runtime_fn": (
                initial_observation_auxiliary_from_context_runtime_fn
            ),
            "refresh_observation_auxiliary_from_context_runtime_fn": (
                refresh_observation_auxiliary_from_context_runtime_fn
            ),
            "observation_log_prob_runtime_fn": observation_log_prob_runtime_fn,
            "observation_log_prob_from_context_runtime_fn": (
                observation_log_prob_from_context_runtime_fn
            ),
            "observation_log_prob_and_grad_from_context_runtime_fn": (
                observation_log_prob_and_grad_from_context_runtime_fn
            ),
            "observation_log_prob_conditioned_from_context_runtime_fn": (
                observation_log_prob_conditioned_from_context_runtime_fn
            ),
            "observation_log_prob_and_grad_conditioned_from_context_runtime_fn": (
                observation_log_prob_and_grad_conditioned_from_context_runtime_fn
            ),
            "observation_log_prob_per_t_from_context_runtime_fn": (
                observation_log_prob_per_t_from_context_runtime_fn
            ),
            "observation_log_prob_per_t_conditioned_from_context_runtime_fn": (
                observation_log_prob_per_t_conditioned_from_context_runtime_fn
            ),
            "observation_increment_log_prob_from_context_runtime_fn": (
                observation_increment_log_prob_from_context_runtime_fn
            ),
            "observation_increment_log_prob_conditioned_from_context_runtime_fn": (
                observation_increment_log_prob_conditioned_from_context_runtime_fn
            ),
            "observation_grad_runtime_fn": observation_grad_runtime_fn,
            "observation_grad_from_context_runtime_fn": observation_grad_from_context_runtime_fn,
            "observation_grad_conditioned_from_context_runtime_fn": (
                observation_grad_conditioned_from_context_runtime_fn
            ),
            "trajectory_log_prob_runtime_fn": trajectory_log_prob_runtime_fn,
            "trajectory_log_prob_from_context_runtime_fn": (
                trajectory_log_prob_from_context_runtime_fn
            ),
            "trajectory_log_prob_conditioned_from_context_runtime_fn": (
                trajectory_log_prob_conditioned_from_context_runtime_fn
            ),
            "prior_terms_from_context_fn": prior_terms_from_context_fn,
            "complete_log_posterior_from_context_runtime_fn": (
                complete_log_posterior_from_context_runtime_fn
            ),
            "complete_log_posterior_conditioned_from_context_runtime_fn": (
                complete_log_posterior_conditioned_from_context_runtime_fn
            ),
            "complete_log_posterior_runtime_fn": complete_log_posterior_runtime_fn,
            "complete_log_posterior_conditioned_runtime_fn": (
                complete_log_posterior_conditioned_runtime_fn
            ),
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

    def initial_observation_auxiliary_from_context_fn(
        context: LatentContext,
        latent_trajectory: jnp.ndarray,
    ):
        return runtime_bundle["initial_observation_auxiliary_from_context_runtime_fn"](
            context,
            latent_trajectory,
            runtime_observations,
        )

    def refresh_observation_auxiliary_from_context_fn(
        context: LatentContext,
        latent_trajectory: jnp.ndarray,
        observation_auxiliary,
        key: jnp.ndarray,
    ):
        return runtime_bundle["refresh_observation_auxiliary_from_context_runtime_fn"](
            context,
            latent_trajectory,
            observation_auxiliary,
            runtime_observations,
            key,
        )

    def observation_log_prob_conditioned_from_context_fn(
        context: LatentContext,
        latent_trajectory: jnp.ndarray,
        observation_auxiliary,
    ) -> jnp.ndarray:
        return runtime_bundle["observation_log_prob_conditioned_from_context_runtime_fn"](
            context,
            latent_trajectory,
            observation_auxiliary,
            runtime_observations,
        )

    def observation_increment_log_prob_conditioned_from_context_fn(
        context: LatentContext,
        latent_state: jnp.ndarray,
        observation_auxiliary,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        return runtime_bundle["observation_increment_log_prob_conditioned_from_context_runtime_fn"](
            context,
            latent_state,
            observation_auxiliary,
            time_idx,
            runtime_observations,
        )

    def observation_log_prob_per_t_conditioned_from_context_fn(
        context: LatentContext,
        latent_trajectory: jnp.ndarray,
        observation_auxiliary,
    ) -> jnp.ndarray:
        return runtime_bundle["observation_log_prob_per_t_conditioned_from_context_runtime_fn"](
            context,
            latent_trajectory,
            observation_auxiliary,
            runtime_observations,
        )

    def trajectory_log_prob_conditioned_from_context_fn(
        context: LatentContext,
        latent_trajectory: jnp.ndarray,
        observation_auxiliary,
        prior_terms: GaussianTrajectoryPriorTerms | None = None,
    ) -> jnp.ndarray:
        return runtime_bundle["trajectory_log_prob_conditioned_from_context_runtime_fn"](
            context,
            latent_trajectory,
            observation_auxiliary,
            runtime_observations,
            prior_terms=prior_terms,
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
        "initial_observation_auxiliary_from_context_fn": (
            initial_observation_auxiliary_from_context_fn
        ),
        "refresh_observation_auxiliary_from_context_fn": (
            refresh_observation_auxiliary_from_context_fn
        ),
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
        "observation_log_prob_conditioned_from_context_fn": (
            observation_log_prob_conditioned_from_context_fn
        ),
        "observation_log_prob_and_grad_conditioned_from_context_fn": (
            lambda context, latent_trajectory, observation_auxiliary: runtime_bundle[
                "observation_log_prob_and_grad_conditioned_from_context_runtime_fn"
            ](
                context,
                latent_trajectory,
                observation_auxiliary,
                runtime_observations,
            )
        ),
        "observation_log_prob_per_t_from_context_fn": observation_log_prob_per_t_from_context_fn,
        "observation_log_prob_per_t_conditioned_from_context_fn": (
            observation_log_prob_per_t_conditioned_from_context_fn
        ),
        "observation_increment_log_prob_from_context_fn": observation_increment_log_prob_from_context_fn,
        "observation_increment_log_prob_conditioned_from_context_fn": (
            observation_increment_log_prob_conditioned_from_context_fn
        ),
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
        "observation_grad_conditioned_from_context_fn": (
            lambda context, latent_trajectory, observation_auxiliary: runtime_bundle[
                "observation_grad_conditioned_from_context_runtime_fn"
            ](
                context,
                latent_trajectory,
                observation_auxiliary,
                runtime_observations,
            )
        ),
        "trajectory_log_prob_fn": trajectory_log_prob_fn,
        "trajectory_log_prob_from_context_fn": trajectory_log_prob_from_context_fn,
        "trajectory_log_prob_conditioned_from_context_fn": (
            trajectory_log_prob_conditioned_from_context_fn
        ),
        "complete_log_posterior_from_context_fn": complete_log_posterior_from_context_fn,
        "complete_log_posterior_fn": complete_log_posterior_fn,
        "complete_log_posterior_with_aux_fn": complete_log_posterior_with_aux_fn,
        "initial_latent_fn": initial_latent_fn,
    }


@functools.partial(jax.jit, static_argnames=("runtime_complete_log_posterior_fn",))
def _complete_log_posterior_grad_runtime(
    z: jnp.ndarray,
    latent_trajectory: jnp.ndarray,
    observation_auxiliary,
    runtime_observations: jnp.ndarray,
    runtime_times: jnp.ndarray,
    *,
    runtime_complete_log_posterior_fn,
) -> jnp.ndarray:
    return jax.grad(runtime_complete_log_posterior_fn, argnums=0)(
        z,
        latent_trajectory,
        observation_auxiliary,
        runtime_observations,
        runtime_times,
    )


@functools.partial(jax.jit, static_argnames=("runtime_complete_log_posterior_fn",))
def _complete_log_posterior_value_and_grad_runtime(
    z: jnp.ndarray,
    latent_trajectory: jnp.ndarray,
    observation_auxiliary,
    runtime_observations: jnp.ndarray,
    runtime_times: jnp.ndarray,
    *,
    runtime_complete_log_posterior_fn,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jax.value_and_grad(runtime_complete_log_posterior_fn, argnums=0)(
        z,
        latent_trajectory,
        observation_auxiliary,
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
        observation_grad_from_context_runtime_fn(
            context,
            x_curr,
            state.observation_auxiliary,
            runtime_observations,
        ),
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
        state.observation_auxiliary,
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
            state.observation_auxiliary,
            runtime_observations,
        )
        obs_per_t_curr = observation_log_prob_per_t_from_context_runtime_fn(
            context,
            x_curr,
            state.observation_auxiliary,
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
        state.observation_auxiliary,
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
            context,
            x_prop,
            state.observation_auxiliary,
            runtime_observations,
        )
        obs_per_t_curr = observation_log_prob_per_t_from_context_runtime_fn(
            context,
            x_curr,
            state.observation_auxiliary,
            runtime_observations,
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


_PARAMETER_NUTS_KERNEL = blackjax_nuts.build_kernel()


def _build_measurement_gaussian_parameter_block(
    bundle: dict[str, Any],
) -> _MeasurementGaussianParameterBlock | None:
    """Plan exact Gaussian conditionals for PG/Gaussian measurement coefficients."""
    if bool(bundle.get("rbpf_enabled", False)):
        return None
    if bool(bundle.get("linear_summary_augmented", False)):
        return None

    registry = bundle.get("site_registry")
    prior_state = bundle.get("prior_state")
    if registry is None or prior_state is None:
        return None

    residual_pg_plan = bundle["residual_polya_gamma_plan"]
    pg_channel_mask_np = np.asarray(residual_pg_plan.channel_mask, dtype=bool)
    gaussian_channel_mask_np = np.asarray(
        bundle.get("measurement_gibbs_gaussian_channel_mask", np.zeros_like(pg_channel_mask_np)),
        dtype=bool,
    )
    exact_channel_mask = pg_channel_mask_np | gaussian_channel_mask_np
    if not bool(np.any(exact_channel_mask)):
        return None

    normal_family = get_real_runtime_family_index(PriorDistributionFamily.NORMAL)
    flat_elements = _flat_site_elements(bundle["flat_example"], bundle["unravel_fn"])
    flat_by_site_element = {
        (entry.site_name, entry.element_index): entry.flat_index for entry in flat_elements
    }

    flat_indices: list[int] = []
    prior_means: list[float] = []
    prior_precisions: list[float] = []

    for site in registry:
        flat_site_name = site.name
        if flat_site_name not in bundle["site_info"]:
            decentered_name = f"{site.name}_decentered"
            if decentered_name in bundle["site_info"]:
                flat_site_name = decentered_name
        if flat_site_name not in bundle["site_info"]:
            continue
        if site.support != SupportClass.REAL:
            continue
        if site.site_kind not in {SiteKind.LOADING, SiteKind.MANIFEST_MEANS}:
            continue
        if site.name not in prior_state:
            continue
        params = prior_state[site.name]
        family = np.broadcast_to(np.asarray(params["family"], dtype=np.int64), site.shape).reshape(
            -1
        )
        scale = np.broadcast_to(np.asarray(params["scale"], dtype=np.float64), site.shape).reshape(
            -1
        )
        flat_distribution = bundle["site_info"][flat_site_name]["distribution"]
        if not hasattr(flat_distribution, "loc") or not hasattr(flat_distribution, "scale"):
            continue
        flat_loc = np.broadcast_to(
            np.asarray(flat_distribution.loc, dtype=np.float64),
            site.shape,
        ).reshape(-1)
        flat_scale = np.broadcast_to(
            np.asarray(flat_distribution.scale, dtype=np.float64),
            site.shape,
        ).reshape(-1)
        positions = tuple(site.positions)
        for element_index, position in enumerate(positions):
            if int(family[element_index]) != normal_family:
                continue
            scale_value = float(scale[element_index])
            flat_scale_value = float(flat_scale[element_index])
            if not np.isfinite(scale_value) or scale_value <= 0.0:
                continue
            if not np.isfinite(flat_scale_value) or flat_scale_value <= 0.0:
                continue
            if site.site_kind == SiteKind.LOADING:
                manifest_idx = int(position[0])
            else:
                manifest_idx = int(position)
            if manifest_idx < 0 or manifest_idx >= exact_channel_mask.shape[0]:
                continue
            if not bool(exact_channel_mask[manifest_idx]):
                continue
            flat_index = flat_by_site_element.get((flat_site_name, element_index))
            if flat_index is None:
                continue
            flat_indices.append(flat_index)
            prior_means.append(float(flat_loc[element_index]))
            prior_precisions.append(1.0 / (flat_scale_value * flat_scale_value))

    if not flat_indices:
        return None
    if len(set(flat_indices)) != len(flat_indices):
        raise ValueError("Measurement Gibbs parameter block produced duplicate flat indices.")

    dtype = bundle["flat_example"].dtype
    return _MeasurementGaussianParameterBlock(
        flat_indices=jnp.asarray(flat_indices, dtype=jnp.int32),
        prior_mean=jnp.asarray(prior_means, dtype=dtype),
        prior_precision=jnp.asarray(prior_precisions, dtype=dtype),
        gaussian_channel_mask=jnp.asarray(gaussian_channel_mask_np, dtype=bool),
        pg_channel_mask=jnp.asarray(pg_channel_mask_np, dtype=bool),
    )


def _measurement_gaussian_parameter_gibbs_step(
    state,
    key: jnp.ndarray,
    runtime_observations: jnp.ndarray,
    runtime_times: jnp.ndarray,
    *,
    block: _MeasurementGaussianParameterBlock,
    pg_plan: PolyaGammaObservationPlan,
    runtime_latent_context_fn,
    trajectory_log_prob_conditioned_from_context_runtime_fn,
    log_prior_unc_fn,
):
    """Sample measurement coefficients from their exact Gaussian conditional."""
    dtype = state.position.dtype
    latent_trajectory = state.latent_trajectory
    context = state.latent_context
    eta = latent_trajectory @ context.H.T + context.d_meas
    eta_flat = jnp.reshape(eta, (-1,))
    beta_current = state.position[block.flat_indices]

    def _design_column(flat_index):
        position_plus = state.position.at[flat_index].add(1.0)
        context_plus = runtime_latent_context_fn(position_plus, runtime_times)
        eta_plus = latent_trajectory @ context_plus.H.T + context_plus.d_meas
        return jnp.reshape(eta_plus - eta, (-1,))

    design = jnp.swapaxes(jax.vmap(_design_column)(block.flat_indices), 0, 1)
    beta_linear_part = design @ beta_current
    eta_offset_flat = eta_flat - beta_linear_part

    precision = jnp.diag(block.prior_precision.astype(dtype))
    rhs = block.prior_precision.astype(dtype) * block.prior_mean.astype(dtype)

    _shape, kappa, linear_offset = polya_gamma_sufficient_statistics(
        pg_plan,
        context,
        runtime_observations,
        dtype,
    )
    active_pg = (
        state.observation_auxiliary.active_mask.astype(bool) & block.pg_channel_mask[None, :]
    )
    weight_pg = jnp.where(active_pg, state.observation_auxiliary.omega, 0.0)
    psi_offset_flat = eta_offset_flat + jnp.reshape(linear_offset, (-1,))
    rhs_pg = jnp.where(
        jnp.reshape(active_pg, (-1,)),
        jnp.reshape(kappa, (-1,)) - jnp.reshape(weight_pg, (-1,)) * psi_offset_flat,
        0.0,
    )
    weight_pg_flat = jnp.reshape(weight_pg, (-1,))
    precision = precision + design.T @ (weight_pg_flat[:, None] * design)
    rhs = rhs + design.T @ rhs_pg

    observed = jnp.nan_to_num(runtime_observations, nan=0.0).astype(dtype)
    active_gaussian = (~jnp.isnan(runtime_observations)) & block.gaussian_channel_mask[None, :]
    variance = jnp.maximum(jnp.diag(context.R).astype(dtype), jnp.asarray(AUX_JITTER, dtype))
    weight_gaussian = jnp.where(active_gaussian, 1.0 / variance[None, :], 0.0)
    rhs_gaussian = jnp.reshape(weight_gaussian, (-1,)) * (
        jnp.reshape(observed, (-1,)) - eta_offset_flat
    )
    precision = precision + design.T @ (jnp.reshape(weight_gaussian, (-1,))[:, None] * design)
    rhs = rhs + design.T @ rhs_gaussian

    precision = symmetrize_with_jitter(precision, jitter=AUX_JITTER)
    chol = jnp.linalg.cholesky(precision)
    mean = jsp_linalg.cho_solve((chol, True), rhs)
    noise = random.normal(key, mean.shape, dtype=dtype)
    sample = mean + jsp_linalg.solve_triangular(chol.T, noise, lower=False)
    next_position = state.position.at[block.flat_indices].set(sample)
    next_context = runtime_latent_context_fn(next_position, runtime_times)
    trajectory_lp = trajectory_log_prob_conditioned_from_context_runtime_fn(
        next_context,
        state.latent_trajectory,
        state.observation_auxiliary,
        runtime_observations,
    )
    complete_lp = log_prior_unc_fn(next_position) + trajectory_lp
    return state._replace(
        position=next_position,
        latent_context=next_context,
        trajectory_log_prob=trajectory_lp,
        complete_log_posterior=complete_lp,
    )


def _parameter_hybrid_gibbs_nuts_step_runtime(
    state,
    key: jnp.ndarray,
    runtime_observations: jnp.ndarray,
    runtime_times: jnp.ndarray,
    *,
    residual_indices: jnp.ndarray,
    residual_dim: int,
    gibbs_block_count: int,
    gibbs_step_fns,
    runtime_complete_log_posterior_fn,
    runtime_latent_context_fn,
    log_prior_unc_fn,
    inverse_mass_matrix: jnp.ndarray,
    max_num_doublings: int,
):
    """Apply exact parameter Gibbs blocks, then NUTS over residual coordinates."""
    dtype = state.position.dtype
    working_state = state
    if gibbs_block_count:
        step_keys = random.split(key, gibbs_block_count + 1)
        for block_idx, gibbs_step_fn in enumerate(gibbs_step_fns):
            working_state = gibbs_step_fn(working_state, step_keys[block_idx])
        key = step_keys[-1]

    if residual_dim == 0:
        return working_state, {
            "accepted": jnp.asarray(1.0, dtype=dtype),
            "accept_prob": jnp.asarray(1.0, dtype=dtype),
            "diverging": jnp.asarray(0.0, dtype=dtype),
            "num_steps": jnp.asarray(0.0, dtype=dtype),
            "energy": jnp.asarray(0.0, dtype=dtype),
            "gibbs_block_count": jnp.asarray(gibbs_block_count, dtype=dtype),
            "residual_dim": jnp.asarray(residual_dim, dtype=dtype),
        }

    def _assemble_position(residual_position: jnp.ndarray) -> jnp.ndarray:
        return working_state.position.at[residual_indices].set(residual_position)

    def logdensity_fn(residual_position: jnp.ndarray) -> jnp.ndarray:
        full_position = _assemble_position(residual_position)
        return runtime_complete_log_posterior_fn(
            full_position,
            working_state.latent_trajectory,
            working_state.observation_auxiliary,
            runtime_observations,
            runtime_times,
        )

    residual_position = working_state.position[residual_indices]
    hmc_state = blackjax_nuts.init(residual_position, logdensity_fn)
    proposal_state, info = _PARAMETER_NUTS_KERNEL(
        key,
        hmc_state,
        logdensity_fn,
        state.param_step_size,
        inverse_mass_matrix,
        max_num_doublings,
    )
    next_position = _assemble_position(proposal_state.position)
    context_next = runtime_latent_context_fn(next_position, runtime_times)
    log_prior_next = log_prior_unc_fn(next_position)
    trajectory_log_prob_next = jnp.asarray(
        proposal_state.logdensity - log_prior_next,
        dtype=working_state.trajectory_log_prob.dtype,
    )
    next_state = working_state._replace(
        position=next_position,
        latent_context=context_next,
        trajectory_log_prob=trajectory_log_prob_next,
        complete_log_posterior=proposal_state.logdensity,
    )
    return next_state, {
        "accepted": jnp.asarray(info.acceptance_rate, dtype=dtype),
        "accept_prob": jnp.asarray(info.acceptance_rate, dtype=dtype),
        "diverging": jnp.asarray(info.is_divergent, dtype=dtype),
        "num_steps": jnp.asarray(info.num_integration_steps, dtype=dtype),
        "energy": jnp.asarray(info.energy, dtype=dtype),
        "gibbs_block_count": jnp.asarray(gibbs_block_count, dtype=dtype),
        "residual_dim": jnp.asarray(residual_dim, dtype=dtype),
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


def build_hybrid_gibbs_nuts_parameter_kernel(
    bundle: dict[str, Any],
    *,
    step_size: float,
    target_accept: float,
    max_num_doublings: int = 10,
    preconditioner_chol: jnp.ndarray | None = None,
    residual_indices: jnp.ndarray | np.ndarray | tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    """Hybrid parameter update: exact Gibbs blocks plus residual NUTS.

    The exact Gibbs block surface is compiler-owned. The default residual set is
    the complement of all exact block coordinates.
    """
    if max_num_doublings < 1:
        raise ValueError("max_num_doublings must be >= 1.")

    dim = int(bundle["dim"])
    runtime_complete_log_posterior_fn = bundle["complete_log_posterior_conditioned_runtime_fn"]
    runtime_latent_context_fn = bundle.get(
        "latent_context_runtime_fn",
        lambda z, _runtime_times: bundle["latent_context_fn"](z),
    )
    trajectory_log_prob_conditioned_from_context_runtime_fn = bundle[
        "trajectory_log_prob_conditioned_from_context_runtime_fn"
    ]
    runtime_observations = bundle["observations"]
    runtime_times = bundle["times"]

    gibbs_step_fns = []
    gibbs_flat_indices: list[int] = []
    measurement_block = _build_measurement_gaussian_parameter_block(bundle)
    if measurement_block is not None:

        def _measurement_step(state, key: jnp.ndarray):
            return _measurement_gaussian_parameter_gibbs_step(
                state,
                key,
                runtime_observations,
                runtime_times,
                block=measurement_block,
                pg_plan=bundle["residual_polya_gamma_plan"],
                runtime_latent_context_fn=runtime_latent_context_fn,
                trajectory_log_prob_conditioned_from_context_runtime_fn=(
                    trajectory_log_prob_conditioned_from_context_runtime_fn
                ),
                log_prior_unc_fn=bundle["log_prior_unc_fn"],
            )

        gibbs_step_fns.append(_measurement_step)
        gibbs_flat_indices.extend(
            np.asarray(measurement_block.flat_indices, dtype=np.int32).tolist()
        )

    consumed_idx_np = np.asarray(sorted(set(gibbs_flat_indices)), dtype=np.int32)
    if residual_indices is None:
        residual_idx_np = np.setdiff1d(np.arange(dim, dtype=np.int32), consumed_idx_np)
    else:
        residual_idx_np = np.asarray(residual_indices, dtype=np.int32)
    if residual_idx_np.ndim != 1:
        raise ValueError(f"residual_indices must be 1-D; got shape {residual_idx_np.shape}.")
    if residual_idx_np.size:
        if int(residual_idx_np.min()) < 0 or int(residual_idx_np.max()) >= dim:
            raise ValueError(
                "residual_indices entries must be within the parameter dimension; "
                f"got min={int(residual_idx_np.min())}, max={int(residual_idx_np.max())}, dim={dim}."
            )
        if np.unique(residual_idx_np).size != residual_idx_np.size:
            raise ValueError("residual_indices must not contain duplicates.")
    if consumed_idx_np.size and np.intersect1d(residual_idx_np, consumed_idx_np).size:
        raise ValueError("residual_indices must be disjoint from exact Gibbs block indices.")
    residual_dim = int(residual_idx_np.size)
    residual_idx = jnp.asarray(residual_idx_np, dtype=jnp.int32)

    if preconditioner_chol is not None:
        precond_chol = jnp.asarray(preconditioner_chol)
        if precond_chol.ndim != 2 or precond_chol.shape[0] != precond_chol.shape[1]:
            raise ValueError(
                f"preconditioner_chol must be square 2-D, got shape {precond_chol.shape}."
            )
        if precond_chol.shape[0] != dim:
            raise ValueError(
                "preconditioner_chol side must equal bundle['dim']; "
                f"got {precond_chol.shape[0]} vs {dim}."
            )
        full_mass_matrix = precond_chol @ precond_chol.T
        inverse_mass_matrix = full_mass_matrix[jnp.ix_(residual_idx, residual_idx)]
    else:
        inverse_mass_matrix = jnp.ones((residual_dim,), dtype=bundle["flat_example"].dtype)

    gibbs_block_count = len(gibbs_step_fns)
    gibbs_step_fns = tuple(gibbs_step_fns)

    def _parameter_hybrid_gibbs_nuts_step(state, key: jnp.ndarray):
        return _parameter_hybrid_gibbs_nuts_step_runtime(
            state,
            key,
            runtime_observations,
            runtime_times,
            residual_indices=residual_idx,
            residual_dim=residual_dim,
            gibbs_block_count=gibbs_block_count,
            gibbs_step_fns=gibbs_step_fns,
            runtime_complete_log_posterior_fn=runtime_complete_log_posterior_fn,
            runtime_latent_context_fn=runtime_latent_context_fn,
            log_prior_unc_fn=bundle["log_prior_unc_fn"],
            inverse_mass_matrix=inverse_mass_matrix,
            max_num_doublings=int(max_num_doublings),
        )

    return {
        "name": "hybrid_gibbs_nuts",
        "scale_field": "param_step_size",
        "initial_scale": step_size,
        "target_accept": target_accept,
        "step_fn": _parameter_hybrid_gibbs_nuts_step,
        "preconditioned": preconditioner_chol is not None,
        "preconditioner_chol": preconditioner_chol,
        "inverse_mass_matrix": inverse_mass_matrix,
        "max_num_doublings": int(max_num_doublings),
        "gibbs_block_count": gibbs_block_count,
        "residual_dim": residual_dim,
        "residual_indices": residual_idx,
        "gibbs_indices": jnp.asarray(consumed_idx_np, dtype=jnp.int32),
    }
