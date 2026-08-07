"""Trajectory-level observation likelihoods for support-aware measurement structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.execution.contracts import NUMERICAL_EPSILON

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.execution.emissions import MeanLogProbFn
    from nof1_causal_lab.models.ssm.inference.targets.kernels import ObservationKernel
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

_POINT_SUPPORT_CODE = 0
_INTERVAL_SUPPORT_CODE = 1
_SUPPORT_KIND_TO_CODE = {
    "point": _POINT_SUPPORT_CODE,
    "interval": _INTERVAL_SUPPORT_CODE,
}

_FIRST_OPERATOR_CODE = 0
_LAST_OPERATOR_CODE = 1
_SUM_OPERATOR_CODE = 2
_COUNT_OPERATOR_CODE = 3
_MEAN_OPERATOR_CODE = 4
_STD_OPERATOR_CODE = 5
_SUMMARY_OPERATOR_TO_CODE = {
    "first": _FIRST_OPERATOR_CODE,
    "last": _LAST_OPERATOR_CODE,
    "sum": _SUM_OPERATOR_CODE,
    "count": _COUNT_OPERATOR_CODE,
    "mean": _MEAN_OPERATOR_CODE,
    "std": _STD_OPERATOR_CODE,
}


@dataclass(frozen=True)
class ObservationOperator:
    """Compiled observation-window semantics used by likelihoods and predictive paths."""

    observation_support: ObservationSupportRuntime | None
    support_kind_codes: jnp.ndarray | None
    summary_operator_codes: jnp.ndarray | None
    interval_summary_indices: tuple[int, ...] = ()
    prev_coeffs: jnp.ndarray | None = None
    curr_coeffs: jnp.ndarray | None = None
    interval_weights: jnp.ndarray | None = None
    emission_slots: jnp.ndarray | None = None
    max_active_windows: int = 0
    n_manifest: int = 0

    @property
    def requires_interval_summary_handling(self) -> bool:
        return bool(self.support_kind_codes is not None and self.interval_summary_indices)

    def point_like_mask(self, dtype: jnp.dtype) -> jnp.ndarray:
        if self.support_kind_codes is None:
            raise ValueError("point_like_mask is undefined without observation support")
        return get_point_like_mask(self.support_kind_codes, dtype)

    def interval_summary_mask(self, dtype: jnp.dtype) -> jnp.ndarray:
        if self.support_kind_codes is None:
            raise ValueError("interval_summary_mask is undefined without observation support")
        return get_interval_summary_mask(self.support_kind_codes, dtype)

    def project_response_trajectory(
        self,
        response_trajectory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _project_response_trajectory_with_operator(response_trajectory, self)

    def empty_accumulators(
        self,
        dtype: jnp.dtype,
        leading_shape: tuple[int, ...] = (),
    ) -> jnp.ndarray:
        if not self.requires_interval_summary_handling:
            raise ValueError(
                "empty_accumulators is undefined without interval-summary observation support"
            )
        return jnp.zeros(
            (*leading_shape, self.n_manifest, self.max_active_windows),
            dtype=dtype,
        )


class SupportObservationSummary(NamedTuple):
    expected_mean: jnp.ndarray
    semantic_mask: jnp.ndarray
    emitted_interval_summary_mask: jnp.ndarray


class SupportObservationStepResult(NamedTuple):
    obs_sum: jnp.ndarray
    obs_sumsq: jnp.ndarray
    obs_weight: jnp.ndarray
    next_accum_sum: jnp.ndarray
    next_accum_sumsq: jnp.ndarray
    next_accum_weight: jnp.ndarray
    summary: SupportObservationSummary


def compile_observation_operator(
    observation_support: ObservationSupportRuntime | None = None,
) -> ObservationOperator:
    """Compile reusable observation-window semantics from runtime metadata."""
    if observation_support is None:
        return ObservationOperator(
            observation_support=observation_support,
            support_kind_codes=None,
            summary_operator_codes=None,
            interval_summary_indices=(),
            prev_coeffs=None,
            curr_coeffs=None,
            interval_weights=None,
            emission_slots=None,
            max_active_windows=0,
            n_manifest=0,
        )

    interval_summary_indices = tuple(
        idx for idx, kind in enumerate(observation_support.support_kinds) if kind == "interval"
    )
    return ObservationOperator(
        observation_support=observation_support,
        support_kind_codes=get_support_kind_codes(observation_support),
        summary_operator_codes=get_summary_operator_codes(observation_support),
        interval_summary_indices=interval_summary_indices,
        prev_coeffs=jnp.asarray(observation_support.interval_prev_coeffs),
        curr_coeffs=jnp.asarray(observation_support.interval_curr_coeffs),
        interval_weights=jnp.asarray(observation_support.interval_weights),
        emission_slots=jnp.asarray(observation_support.emission_slot_indices, dtype=jnp.int32),
        max_active_windows=observation_support.max_active_windows,
        n_manifest=len(observation_support.support_kinds),
    )


def get_support_kind_codes(observation_support: ObservationSupportRuntime) -> jnp.ndarray:
    """Map support kinds to integer codes aligned with manifest order."""
    return jnp.asarray(
        [
            _SUPPORT_KIND_TO_CODE.get(kind, _POINT_SUPPORT_CODE)
            for kind in observation_support.support_kinds
        ],
        dtype=jnp.int32,
    )


def get_summary_operator_codes(observation_support: ObservationSupportRuntime) -> jnp.ndarray:
    """Map summary operators to integer codes aligned with manifest order."""
    return jnp.asarray(
        [
            _SUMMARY_OPERATOR_TO_CODE.get(operator, _LAST_OPERATOR_CODE)
            for operator in observation_support.summary_operators
        ],
        dtype=jnp.int32,
    )


def get_point_like_mask(support_kind_codes: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    """Mask manifests that still use point-observation semantics."""
    return (support_kind_codes == _POINT_SUPPORT_CODE).astype(dtype)


def get_interval_summary_mask(support_kind_codes: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    """Mask manifests that require interval-summary handling."""
    return 1.0 - get_point_like_mask(support_kind_codes, dtype)


def accumulate_support_statistics(
    prev_response: jnp.ndarray,
    accum_sum: jnp.ndarray,
    accum_sumsq: jnp.ndarray,
    accum_weight: jnp.ndarray,
    response_t: jnp.ndarray,
    prev_coeff_t: jnp.ndarray,
    curr_coeff_t: jnp.ndarray,
    weight_t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advance interval-summary sufficient statistics by one model step."""
    prev_response_exp = jnp.expand_dims(prev_response, axis=-1)
    response_t_exp = jnp.expand_dims(response_t, axis=-1)
    obs_sum = accum_sum + prev_coeff_t * prev_response_exp + curr_coeff_t * response_t_exp
    obs_sumsq = (
        accum_sumsq + prev_coeff_t * (prev_response_exp**2) + curr_coeff_t * (response_t_exp**2)
    )
    obs_weight = accum_weight + weight_t
    return obs_sum, obs_sumsq, obs_weight


def select_emitted_support_statistics(
    obs_sum: jnp.ndarray,
    obs_sumsq: jnp.ndarray,
    obs_weight: jnp.ndarray,
    emission_slot_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Gather the active support slot referenced by each emitted observation."""
    safe_indices = jnp.clip(emission_slot_indices, 0, obs_sum.shape[-1] - 1)
    gather_idx = jnp.expand_dims(safe_indices, axis=-1)
    selected_sum = jnp.take_along_axis(obs_sum, gather_idx, axis=-1).squeeze(-1)
    selected_sumsq = jnp.take_along_axis(obs_sumsq, gather_idx, axis=-1).squeeze(-1)
    selected_weight = jnp.take_along_axis(obs_weight, gather_idx, axis=-1).squeeze(-1)
    valid = emission_slot_indices >= 0
    zeros = jnp.zeros_like(selected_sum)
    return (
        jnp.where(valid, selected_sum, zeros),
        jnp.where(valid, selected_sumsq, zeros),
        jnp.where(valid, selected_weight, zeros),
    )


def expected_observation_mean(
    response_t: jnp.ndarray,
    obs_sum: jnp.ndarray,
    obs_sumsq: jnp.ndarray,
    obs_weight: jnp.ndarray,
    summary_operator_codes: jnp.ndarray,
) -> jnp.ndarray:
    """Map accumulated response statistics to the expected observation mean."""
    safe_weight = jnp.maximum(obs_weight, NUMERICAL_EPSILON)
    window_mean = obs_sum / safe_weight
    window_var = jnp.maximum(obs_sumsq / safe_weight - window_mean**2, NUMERICAL_EPSILON)
    std_mean = jnp.sqrt(window_var)

    expected_mean = response_t
    expected_mean = jnp.where(
        jnp.logical_or(
            summary_operator_codes == _SUM_OPERATOR_CODE,
            summary_operator_codes == _COUNT_OPERATOR_CODE,
        ),
        obs_sum,
        expected_mean,
    )
    expected_mean = jnp.where(
        summary_operator_codes == _MEAN_OPERATOR_CODE, window_mean, expected_mean
    )
    return jnp.where(summary_operator_codes == _STD_OPERATOR_CODE, std_mean, expected_mean)


def reset_support_accumulators(
    obs_sum: jnp.ndarray,
    obs_sumsq: jnp.ndarray,
    obs_weight: jnp.ndarray,
    emission_slot_indices: jnp.ndarray,
    emit_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reset interval accumulators once an interval-summary observation has been emitted."""
    safe_indices = jnp.clip(emission_slot_indices, 0, obs_sum.shape[-1] - 1)
    slot_ids = jnp.arange(obs_sum.shape[-1], dtype=safe_indices.dtype)
    while slot_ids.ndim < obs_sum.ndim:
        slot_ids = slot_ids.reshape((1,) * (obs_sum.ndim - 1) + (-1,))
    reset_mask = jnp.expand_dims(emit_mask > 0.5, axis=-1) & (
        slot_ids == jnp.expand_dims(safe_indices, axis=-1)
    )
    zeros = jnp.zeros_like(obs_sum)
    next_sum = jnp.where(reset_mask, zeros, obs_sum)
    next_sumsq = jnp.where(reset_mask, zeros, obs_sumsq)
    next_weight = jnp.where(reset_mask, zeros, obs_weight)
    return next_sum, next_sumsq, next_weight


def summarize_support_observation(
    observation_operator: ObservationOperator,
    response_t: jnp.ndarray,
    obs_sum: jnp.ndarray,
    obs_sumsq: jnp.ndarray,
    obs_weight: jnp.ndarray,
    obs_mask_t: jnp.ndarray,
    emission_slot_indices: jnp.ndarray,
) -> SupportObservationSummary:
    """Summarize one support-aware observation emission at the current step."""
    if not observation_operator.requires_interval_summary_handling:
        raise ValueError(
            "summarize_support_observation requires interval-summary observation support"
        )
    assert observation_operator.summary_operator_codes is not None
    dtype = response_t.dtype
    point_like_mask = observation_operator.point_like_mask(dtype)
    interval_summary_mask = observation_operator.interval_summary_mask(dtype)
    selected_sum, selected_sumsq, selected_weight = select_emitted_support_statistics(
        obs_sum,
        obs_sumsq,
        obs_weight,
        emission_slot_indices,
    )
    expected_mean = expected_observation_mean(
        response_t,
        selected_sum,
        selected_sumsq,
        selected_weight,
        observation_operator.summary_operator_codes,
    )
    obs_mask_float = jnp.asarray(obs_mask_t, dtype=dtype)
    emitted_interval_summary_mask = (
        obs_mask_float * interval_summary_mask * (emission_slot_indices >= 0).astype(dtype)
    )
    semantic_mask = point_like_mask + emitted_interval_summary_mask
    return SupportObservationSummary(
        expected_mean=expected_mean,
        semantic_mask=semantic_mask,
        emitted_interval_summary_mask=emitted_interval_summary_mask,
    )


def advance_support_observation_state(
    observation_operator: ObservationOperator,
    response_prev: jnp.ndarray,
    accum_sum: jnp.ndarray,
    accum_sumsq: jnp.ndarray,
    accum_weight: jnp.ndarray,
    response_t: jnp.ndarray,
    obs_mask_t: jnp.ndarray,
    prev_coeff_t: jnp.ndarray,
    curr_coeff_t: jnp.ndarray,
    weight_t: jnp.ndarray,
    emission_slot_indices: jnp.ndarray,
) -> SupportObservationStepResult:
    """Advance support-aware accumulators and summarize the current emitted observation."""
    obs_sum, obs_sumsq, obs_weight = accumulate_support_statistics(
        response_prev,
        accum_sum,
        accum_sumsq,
        accum_weight,
        response_t,
        prev_coeff_t,
        curr_coeff_t,
        weight_t,
    )
    summary = summarize_support_observation(
        observation_operator,
        response_t,
        obs_sum,
        obs_sumsq,
        obs_weight,
        obs_mask_t,
        emission_slot_indices,
    )
    next_sum, next_sumsq, next_weight = reset_support_accumulators(
        obs_sum,
        obs_sumsq,
        obs_weight,
        emission_slot_indices,
        summary.emitted_interval_summary_mask,
    )
    return SupportObservationStepResult(
        obs_sum=obs_sum,
        obs_sumsq=obs_sumsq,
        obs_weight=obs_weight,
        next_accum_sum=next_sum,
        next_accum_sumsq=next_sumsq,
        next_accum_weight=next_weight,
        summary=summary,
    )


def _project_response_trajectory_with_operator(
    response_trajectory: jnp.ndarray,
    observation_operator: ObservationOperator,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project response-space trajectories into emitted observation means.

    Returns:
        expected_means:
            Response / aggregated-mean trajectory aligned to model time rows.
        semantic_emission_mask:
            Float mask aligned to ``expected_means``. Point-like manifests emit
            on every row; interval-summary manifests emit only on their anchor rows.
    """
    dtype = response_trajectory.dtype
    T, n_manifest = response_trajectory.shape

    if not observation_operator.requires_interval_summary_handling:
        return response_trajectory, jnp.ones((T, n_manifest), dtype=dtype)

    support = observation_operator.observation_support
    assert support is not None
    assert observation_operator.summary_operator_codes is not None
    point_like_mask = observation_operator.point_like_mask(dtype)
    interval_summary_mask = observation_operator.interval_summary_mask(dtype)

    emission_slots = jnp.asarray(support.emission_slot_indices, dtype=jnp.int32)
    semantic_mask_0 = point_like_mask + interval_summary_mask * (emission_slots[0] >= 0).astype(
        dtype
    )
    expected_0 = response_trajectory[0]

    if T == 1:
        return expected_0[None, :], semantic_mask_0[None, :]

    assert observation_operator.prev_coeffs is not None
    assert observation_operator.curr_coeffs is not None
    assert observation_operator.interval_weights is not None
    assert observation_operator.emission_slots is not None
    prev_coeffs = jnp.asarray(observation_operator.prev_coeffs, dtype=dtype)
    curr_coeffs = jnp.asarray(observation_operator.curr_coeffs, dtype=dtype)
    interval_weights = jnp.asarray(observation_operator.interval_weights, dtype=dtype)
    emission_slots = jnp.asarray(observation_operator.emission_slots, dtype=jnp.int32)
    zeros = observation_operator.empty_accumulators(dtype)
    full_obs_mask = jnp.ones((n_manifest,), dtype=dtype)

    def _scan_step(carry, inputs):
        response_prev, accum_sum, accum_sumsq, accum_weight = carry
        response_t, prev_coeff_t, curr_coeff_t, weight_t, emission_slots_t = inputs

        step_result = advance_support_observation_state(
            observation_operator,
            response_prev,
            accum_sum,
            accum_sumsq,
            accum_weight,
            response_t,
            full_obs_mask,
            prev_coeff_t,
            curr_coeff_t,
            weight_t,
            emission_slots_t,
        )
        return (
            response_t,
            step_result.next_accum_sum,
            step_result.next_accum_sumsq,
            step_result.next_accum_weight,
        ), (
            step_result.summary.expected_mean,
            step_result.summary.semantic_mask,
        )

    _, (expected_rest, semantic_mask_rest) = jax.lax.scan(
        _scan_step,
        (response_trajectory[0], zeros, zeros, zeros),
        (
            response_trajectory[1:],
            prev_coeffs[1:],
            curr_coeffs[1:],
            interval_weights[1:],
            emission_slots[1:],
        ),
    )
    return (
        jnp.concatenate([expected_0[None, :], expected_rest], axis=0),
        jnp.concatenate([semantic_mask_0[None, :], semantic_mask_rest], axis=0),
    )


def project_response_trajectory(
    response_trajectory: jnp.ndarray,
    observation_support: ObservationSupportRuntime | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project response-space trajectories into emitted observation means."""
    return _project_response_trajectory_with_operator(
        response_trajectory,
        compile_observation_operator(observation_support),
    )


def trajectory_observation_log_probs(
    latent_trajectory: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray | None,
    H: jnp.ndarray,
    d_meas: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel: ObservationKernel,
    mean_log_prob_fn: MeanLogProbFn | None = None,
    observation_support: ObservationSupportRuntime | None = None,
) -> jnp.ndarray:
    """Return per-timestep log-likelihood contributions for a latent trajectory."""
    if obs_mask is None:
        obs_mask = ~jnp.isnan(observations)

    clean_obs = jnp.nan_to_num(observations, nan=0.0)
    mask_float = obs_mask.astype(jnp.float32)
    observation_operator = compile_observation_operator(observation_support)

    if not observation_operator.requires_interval_summary_handling:
        return jax.vmap(
            lambda y_t, z_t, mask_t: obs_kernel.log_prob_fn(
                y_t,
                H @ z_t + d_meas,
                R,
                mask_t,
            )
        )(clean_obs, latent_trajectory, mask_float)

    if mean_log_prob_fn is None:
        raise ValueError("mean_log_prob_fn is required for interval-summary observation semantics")

    point_like_mask = observation_operator.point_like_mask(mask_float.dtype)
    interval_summary_mask = observation_operator.interval_summary_mask(mask_float.dtype)
    responses = jax.vmap(lambda z_t: obs_kernel.response_fn(H @ z_t + d_meas))(latent_trajectory)
    expected_means, semantic_mask = observation_operator.project_response_trajectory(responses)

    point_ll = jax.vmap(
        lambda y_t, z_t, mask_t: obs_kernel.log_prob_fn(
            y_t,
            H @ z_t + d_meas,
            R,
            mask_t * point_like_mask,
        )
    )(clean_obs, latent_trajectory, mask_float)
    interval_summary_ll = jax.vmap(
        lambda y_t, mean_t, mask_t, semantic_t: mean_log_prob_fn(
            y_t,
            mean_t,
            R,
            mask_t * interval_summary_mask * semantic_t,
        )
    )(clean_obs, expected_means, mask_float, semantic_mask)
    return point_ll + interval_summary_ll


def trajectory_observation_log_prob(
    latent_trajectory: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray | None,
    H: jnp.ndarray,
    d_meas: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel: ObservationKernel,
    mean_log_prob_fn: MeanLogProbFn | None = None,
    observation_support: ObservationSupportRuntime | None = None,
) -> jnp.ndarray:
    """Return the total observation log-likelihood for a latent trajectory."""
    return jnp.sum(
        trajectory_observation_log_probs(
            latent_trajectory,
            observations,
            obs_mask,
            H,
            d_meas,
            R,
            obs_kernel,
            mean_log_prob_fn,
            observation_support,
        )
    )


def row_observation_log_probs(
    latent_trajectory: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel,
) -> jnp.ndarray:
    """Per-row ``(T,)`` point-observation log-prob for per-row observation operators.

    Sum over the leading axis equals :func:`row_observation_log_prob`.
    """
    clean_obs = jnp.nan_to_num(observations, nan=0.0)
    obs_mask_float = obs_mask.astype(latent_trajectory.dtype)
    return jax.vmap(
        lambda y_t, z_t, mask_t, H_t, d_t: obs_kernel.log_prob_fn(
            y_t,
            H_t @ z_t + d_t,
            R,
            mask_t,
        )
    )(clean_obs, latent_trajectory, obs_mask_float, H_rows, d_rows)


def row_observation_log_prob(
    latent_trajectory: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel,
) -> jnp.ndarray:
    """Return the point-observation log-probability for per-row observation operators."""
    return jnp.sum(
        row_observation_log_probs(
            latent_trajectory, observations, obs_mask, H_rows, d_rows, R, obs_kernel
        )
    )
