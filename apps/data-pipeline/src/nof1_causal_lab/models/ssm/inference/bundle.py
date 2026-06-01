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
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts import DistributionFamily, LinkFunction
from nof1_causal_lab.distributions import (
    PriorDistributionFamily,
    get_positive_runtime_kind_from_index,
)
from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.dynamics.linearisation import infer_linearisation
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

# Match laplace/shared.py's default jitter so the auxiliary-Kalman proposal and
# the target trajectory log-prob agree on the covariance being evaluated.
AUX_JITTER = 1e-6

# Default coordinatewise saturation level for TULAc-style gradient taming.
DEFAULT_TULAC_GRAD_CLIP = 10.0


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


def _initial_latent_moments(context: LatentContext) -> tuple[jnp.ndarray, jnp.ndarray]:
    init_pred_mean = context.Ad[0] @ context.init_mean + context.cd[0]
    init_pred_cov = symmetrize_with_jitter(
        context.Ad[0] @ context.init_cov @ context.Ad[0].T + context.Qd[0],
        jitter=AUX_JITTER,
    )
    return init_pred_mean, init_pred_cov


def tame_gradient_tulac(
    grad: jnp.ndarray,
    *,
    grad_clip: float = DEFAULT_TULAC_GRAD_CLIP,
) -> jnp.ndarray:
    """Coordinatewise tamed gradient with saturation at ``grad_clip``.

    This is TULAc's ``g_i / (1 + h * |g_i|)`` with ``h = 1 / grad_clip``.
    It recovers ``grad`` when ``|grad| / grad_clip`` is small and saturates
    each component at ``sign(g_i) * grad_clip`` when ``|g_i|`` is huge.

    Prevents superlinear blow-up from log-link observations (Poisson / NB /
    Gamma with ``exp`` link) from poisoning the MH ratio. Applied identically
    on forward and reverse passes, so the auxiliary-Kalman MH ratio remains
    correct — the proposal kernel is just a different (still valid) kernel.
    """
    clip = jnp.asarray(grad_clip, dtype=grad.dtype)
    return grad / (1.0 + jnp.abs(grad) / clip)


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
            else jnp.zeros((model.spec.n_manifest,), dtype=jnp.int32)
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
