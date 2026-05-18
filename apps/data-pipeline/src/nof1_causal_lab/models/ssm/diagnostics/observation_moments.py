"""Observation-space moment projection helpers for parametric diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
from jax import lax

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize
from nof1_causal_lab.models.ssm.discretization import discretize_system_with_inputs_batched
from nof1_causal_lab.models.ssm.inference.targets.base import (
    NUMERICAL_EPSILON,
)
from nof1_causal_lab.models.ssm.likelihood_extra_params import (
    assemble_sampled_extra_params,
)
from nof1_causal_lab.models.ssm.parameterization import assemble_deterministics_from_registry

from .results import OutputSensitivityUnsupportedError

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.structure_runtime import SSMStructureRuntime


def _assemble_sensitivity_measurement_state(
    z_flat,
    unravel_fn,
    transforms,
    spec,
    *,
    structure_runtime: SSMStructureRuntime,
    registry,
):
    """Assemble deterministic matrices and observation hyperparameters for one draw."""
    unc_dict = unravel_fn(z_flat)
    con_dict = {name: transforms[name](unc_dict[name]) for name in unc_dict}

    batched = {name: value[None, ...] for name, value in con_dict.items()}
    det = assemble_deterministics_from_registry(
        batched,
        spec,
        registry,
        structure_runtime=structure_runtime,
    )
    det = {name: value[0] for name, value in det.items()}

    extra_params = assemble_sampled_extra_params(spec, con_dict)
    return det, extra_params


def _response_latent_variance_diag(
    eta_mean: jnp.ndarray,
    eta_cov: jnp.ndarray,
    response_mean: jnp.ndarray,
    *,
    obs_kernel,
    manifest_dists,
    manifest_links,
) -> jnp.ndarray:
    """Approximate latent-induced variance on the observation-mean scale."""
    unsupported_response_families = {
        DistributionFamily.ORDERED_LOGISTIC,
        DistributionFamily.CATEGORICAL,
    }
    if not any(dist in unsupported_response_families for dist in manifest_dists):
        eta_var_diag = jnp.maximum(jnp.diag(eta_cov), 0.0)
        deriv = []
        for idx, link in enumerate(manifest_links):
            eta_j = eta_mean[idx]
            mean_j = response_mean[idx]
            if link == LinkFunction.IDENTITY:
                deriv.append(jnp.asarray(1.0, dtype=eta_mean.dtype))
            elif link == LinkFunction.LOG:
                deriv.append(mean_j)
            elif link == LinkFunction.LOGIT:
                deriv.append(mean_j * (1.0 - mean_j))
            elif link == LinkFunction.PROBIT:
                deriv.append(jstats.norm.pdf(eta_j))
            elif link == LinkFunction.INVERSE:
                deriv.append(
                    jnp.where(eta_j > 0.0, -1.0 / (eta_j**2), jnp.nan).astype(eta_mean.dtype)
                )
            else:
                raise OutputSensitivityUnsupportedError(
                    f"output sensitivity does not support link={link.value!r}"
                )
        deriv_vec = jnp.stack(deriv)
        return jnp.square(deriv_vec) * eta_var_diag

    response_jacobian = jax.jacfwd(obs_kernel.response_fn)(eta_mean)
    response_cov = response_jacobian @ eta_cov @ response_jacobian.T
    return jnp.maximum(jnp.diag(response_cov), 0.0)


def _response_latent_covariance(
    eta_mean: jnp.ndarray,
    eta_cov: jnp.ndarray,
    *,
    obs_kernel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Approximate response-scale covariance and Jacobian at one predictor mean."""
    response_jacobian = jax.jacfwd(obs_kernel.response_fn)(eta_mean)
    response_cov = response_jacobian @ eta_cov @ response_jacobian.T
    response_cov = symmetrize(response_cov)
    diag_idx = jnp.diag_indices(response_cov.shape[0])
    response_cov = response_cov.at[diag_idx].set(jnp.maximum(jnp.diag(response_cov), 0.0))
    return response_cov, response_jacobian


def _observation_noise_variance_arguments(
    eta_mean: jnp.ndarray,
    response_mean: jnp.ndarray,
    *,
    manifest_dists,
) -> jnp.ndarray:
    """Build the per-channel input vector expected by ``obs_kernel.variance_fn``."""
    variance_args = []
    for idx, dist in enumerate(manifest_dists):
        if dist in {
            DistributionFamily.ORDERED_LOGISTIC,
            DistributionFamily.CATEGORICAL,
        }:
            variance_args.append(eta_mean[idx])
        else:
            variance_args.append(response_mean[idx])
    return jnp.stack(variance_args)


def _observation_noise_covariance(
    variance_args: jnp.ndarray,
    *,
    obs_kernel,
) -> jnp.ndarray:
    """Return one same-row observation-noise covariance matrix."""
    observation_noise_cov = symmetrize(obs_kernel.variance_fn(variance_args))
    diag_idx = jnp.diag_indices(observation_noise_cov.shape[0])
    return observation_noise_cov.at[diag_idx].set(
        jnp.maximum(jnp.diag(observation_noise_cov), NUMERICAL_EPSILON)
    )


def _extra_param_at(
    extra_params: dict,
    key: str,
    index: int,
    default: float,
) -> jnp.ndarray:
    """Return one scalar observation hyperparameter, broadcasting shared values."""
    value = extra_params.get(key, default)
    value_arr = jnp.asarray(value)
    if value_arr.ndim == 0:
        return value_arr
    return value_arr[index]


def _select_support_slot(stat: jnp.ndarray, emission_slot_indices: jnp.ndarray) -> jnp.ndarray:
    """Gather one accumulator statistic from the active interval slot."""
    safe_indices = jnp.clip(emission_slot_indices, 0, stat.shape[-1] - 1)
    selected = jnp.take_along_axis(
        stat,
        jnp.expand_dims(safe_indices, axis=-1),
        axis=-1,
    ).squeeze(-1)
    valid = emission_slot_indices >= 0
    return jnp.where(valid, selected, jnp.zeros_like(selected))


def _reset_support_stat(
    stat: jnp.ndarray,
    emission_slot_indices: jnp.ndarray,
    emit_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Reset one accumulator statistic after an interval-summary emission."""
    safe_indices = jnp.clip(emission_slot_indices, 0, stat.shape[-1] - 1)
    slot_ids = jnp.arange(stat.shape[-1], dtype=safe_indices.dtype)
    while slot_ids.ndim < stat.ndim:
        slot_ids = slot_ids.reshape((1,) * (stat.ndim - 1) + (-1,))
    reset_mask = jnp.expand_dims(emit_mask > 0.5, axis=-1) & (
        slot_ids == jnp.expand_dims(safe_indices, axis=-1)
    )
    return jnp.where(reset_mask, jnp.zeros_like(stat), stat)


def _project_response_moments(
    response_means: jnp.ndarray,
    response_vars: jnp.ndarray,
    observation_operator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project response-space first/second moments into emitted observation moments."""
    from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
        _COUNT_OPERATOR_CODE,
        _MEAN_OPERATOR_CODE,
        _STD_OPERATOR_CODE,
        _SUM_OPERATOR_CODE,
    )

    dtype = response_means.dtype
    if not observation_operator.requires_interval_summary_handling:
        return (
            response_means,
            response_vars,
            jnp.ones_like(response_means, dtype=dtype),
        )

    support = observation_operator.observation_support
    assert support is not None
    assert observation_operator.summary_operator_codes is not None
    assert observation_operator.prev_coeffs is not None
    assert observation_operator.curr_coeffs is not None
    assert observation_operator.interval_weights is not None
    assert observation_operator.emission_slots is not None

    n_timepoints, n_manifest = response_means.shape
    point_like_mask = observation_operator.point_like_mask(dtype)
    interval_summary_mask = observation_operator.interval_summary_mask(dtype)
    emission_slots = jnp.asarray(observation_operator.emission_slots, dtype=jnp.int64)
    summary_codes = observation_operator.summary_operator_codes
    semantic_mask_0 = point_like_mask + interval_summary_mask * (emission_slots[0] >= 0).astype(
        dtype
    )

    if n_timepoints == 1:
        return response_means, response_vars, semantic_mask_0[None, :]

    prev_coeffs = jnp.asarray(observation_operator.prev_coeffs, dtype=dtype)
    curr_coeffs = jnp.asarray(observation_operator.curr_coeffs, dtype=dtype)
    interval_weights = jnp.asarray(observation_operator.interval_weights, dtype=dtype)
    zeros = observation_operator.empty_accumulators(dtype)
    full_obs_mask = jnp.ones((n_manifest,), dtype=dtype)

    def _second_stats(mean_t: jnp.ndarray, var_t: jnp.ndarray):
        second_mean = mean_t**2 + var_t
        second_var = jnp.maximum(2.0 * var_t**2 + 4.0 * mean_t**2 * var_t, 0.0)
        cov = 2.0 * mean_t * var_t
        return second_mean, second_var, cov

    weight_zeros = observation_operator.empty_accumulators(dtype)

    def _scan_step_with_weight(carry, inputs):
        (
            response_prev,
            response_var_prev,
            accum_sum_mean,
            accum_sum_var,
            accum_sumsq_mean,
            accum_sumsq_var,
            accum_sum_sumsq_cov,
            accum_weight,
        ) = carry
        response_t, response_var_t, prev_coeff_t, curr_coeff_t, weight_t, emission_slots_t = inputs

        response_prev_exp = jnp.expand_dims(response_prev, axis=-1)
        response_prev_var_exp = jnp.expand_dims(response_var_prev, axis=-1)
        response_t_exp = jnp.expand_dims(response_t, axis=-1)
        response_t_var_exp = jnp.expand_dims(response_var_t, axis=-1)

        prev_second_mean, prev_second_var, prev_cov = _second_stats(
            response_prev,
            response_var_prev,
        )
        curr_second_mean, curr_second_var, curr_cov = _second_stats(response_t, response_var_t)

        obs_sum_mean = (
            accum_sum_mean + prev_coeff_t * response_prev_exp + curr_coeff_t * response_t_exp
        )
        obs_sum_var = (
            accum_sum_var
            + prev_coeff_t**2 * response_prev_var_exp
            + curr_coeff_t**2 * response_t_var_exp
        )
        obs_sumsq_mean = (
            accum_sumsq_mean
            + prev_coeff_t * jnp.expand_dims(prev_second_mean, axis=-1)
            + curr_coeff_t * jnp.expand_dims(curr_second_mean, axis=-1)
        )
        obs_sumsq_var = (
            accum_sumsq_var
            + prev_coeff_t**2 * jnp.expand_dims(prev_second_var, axis=-1)
            + curr_coeff_t**2 * jnp.expand_dims(curr_second_var, axis=-1)
        )
        obs_sum_sumsq_cov = (
            accum_sum_sumsq_cov
            + prev_coeff_t**2 * jnp.expand_dims(prev_cov, axis=-1)
            + curr_coeff_t**2 * jnp.expand_dims(curr_cov, axis=-1)
        )
        obs_weight = accum_weight + weight_t

        selected_sum_mean = _select_support_slot(obs_sum_mean, emission_slots_t)
        selected_sum_var = _select_support_slot(obs_sum_var, emission_slots_t)
        selected_sumsq_mean = _select_support_slot(obs_sumsq_mean, emission_slots_t)
        selected_sumsq_var = _select_support_slot(obs_sumsq_var, emission_slots_t)
        selected_sum_sumsq_cov = _select_support_slot(obs_sum_sumsq_cov, emission_slots_t)
        selected_weight = _select_support_slot(obs_weight, emission_slots_t)
        safe_weight = jnp.maximum(selected_weight, NUMERICAL_EPSILON)
        window_mean = selected_sum_mean / safe_weight
        window_mean_var = selected_sum_var / (safe_weight**2)
        window_second_mean = selected_sumsq_mean / safe_weight
        std_arg = jnp.maximum(window_second_mean - window_mean**2, NUMERICAL_EPSILON)
        std_mean = jnp.sqrt(std_arg)
        d_std_d_sum = -window_mean / (std_mean * safe_weight)
        d_std_d_sumsq = 1.0 / (2.0 * std_mean * safe_weight)
        std_var = (
            d_std_d_sum**2 * selected_sum_var
            + d_std_d_sumsq**2 * selected_sumsq_var
            + 2.0 * d_std_d_sum * d_std_d_sumsq * selected_sum_sumsq_cov
        )
        std_var = jnp.maximum(std_var, 0.0)

        expected_mean = response_t
        latent_var = response_var_t
        sum_like = jnp.logical_or(
            summary_codes == _SUM_OPERATOR_CODE,
            summary_codes == _COUNT_OPERATOR_CODE,
        )
        expected_mean = jnp.where(sum_like, selected_sum_mean, expected_mean)
        latent_var = jnp.where(sum_like, selected_sum_var, latent_var)
        expected_mean = jnp.where(summary_codes == _MEAN_OPERATOR_CODE, window_mean, expected_mean)
        latent_var = jnp.where(summary_codes == _MEAN_OPERATOR_CODE, window_mean_var, latent_var)
        expected_mean = jnp.where(summary_codes == _STD_OPERATOR_CODE, std_mean, expected_mean)
        latent_var = jnp.where(summary_codes == _STD_OPERATOR_CODE, std_var, latent_var)

        emitted_interval_summary_mask = (
            full_obs_mask * interval_summary_mask * (emission_slots_t >= 0).astype(dtype)
        )
        semantic_mask = point_like_mask + emitted_interval_summary_mask

        next_sum_mean = _reset_support_stat(
            obs_sum_mean,
            emission_slots_t,
            emitted_interval_summary_mask,
        )
        next_sum_var = _reset_support_stat(
            obs_sum_var,
            emission_slots_t,
            emitted_interval_summary_mask,
        )
        next_sumsq_mean = _reset_support_stat(
            obs_sumsq_mean,
            emission_slots_t,
            emitted_interval_summary_mask,
        )
        next_sumsq_var = _reset_support_stat(
            obs_sumsq_var,
            emission_slots_t,
            emitted_interval_summary_mask,
        )
        next_sum_sumsq_cov = _reset_support_stat(
            obs_sum_sumsq_cov,
            emission_slots_t,
            emitted_interval_summary_mask,
        )
        next_weight = _reset_support_stat(
            obs_weight,
            emission_slots_t,
            emitted_interval_summary_mask,
        )

        return (
            response_t,
            response_var_t,
            next_sum_mean,
            next_sum_var,
            next_sumsq_mean,
            next_sumsq_var,
            next_sum_sumsq_cov,
            next_weight,
        ), (
            expected_mean,
            latent_var,
            semantic_mask,
        )

    _, (expected_rest, latent_var_rest, semantic_mask_rest) = lax.scan(
        _scan_step_with_weight,
        (
            response_means[0],
            response_vars[0],
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            weight_zeros,
        ),
        (
            response_means[1:],
            response_vars[1:],
            prev_coeffs[1:],
            curr_coeffs[1:],
            interval_weights[1:],
            emission_slots[1:],
        ),
    )

    return (
        jnp.concatenate([response_means[0][None, :], expected_rest], axis=0),
        jnp.concatenate([response_vars[0][None, :], latent_var_rest], axis=0),
        jnp.concatenate([semantic_mask_0[None, :], semantic_mask_rest], axis=0),
    )


def _response_second_moment_loading(
    response_means: jnp.ndarray,
    response_state_loading: jnp.ndarray,
) -> jnp.ndarray:
    """Linearized loading for squared-response deviations."""
    return 2.0 * jnp.expand_dims(response_means, axis=-1) * response_state_loading


def _support_accumulator_response_map(coeffs: jnp.ndarray) -> jnp.ndarray:
    """Map one response vector into flattened manifest-slot accumulators."""
    eye = jnp.eye(coeffs.shape[0], dtype=coeffs.dtype)
    return (jnp.expand_dims(coeffs, axis=-1) * jnp.expand_dims(eye, axis=1)).reshape(
        coeffs.shape[0] * coeffs.shape[1],
        coeffs.shape[0],
    )


def _support_selection_matrix(
    emission_slot_indices: jnp.ndarray,
    *,
    n_slots: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Select one active slot per manifest from flattened accumulator state."""
    eye = jnp.eye(emission_slot_indices.shape[0], dtype=dtype)
    safe_indices = jnp.clip(emission_slot_indices, 0, n_slots - 1)
    slot_one_hot = (
        jax.nn.one_hot(safe_indices, n_slots, dtype=dtype)
        * (emission_slot_indices >= 0).astype(dtype)[:, None]
    )
    return (jnp.expand_dims(eye, axis=2) * jnp.expand_dims(slot_one_hot, axis=1)).reshape(
        emission_slot_indices.shape[0],
        emission_slot_indices.shape[0] * n_slots,
    )


def _point_response_covariances(
    response_state_loadings: jnp.ndarray,
    state_covs: jnp.ndarray,
    transitions: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return same-row and lag-1 response covariances without building T x T blocks."""
    same_covs = jax.vmap(lambda loading_t, state_cov_t: loading_t @ state_cov_t @ loading_t.T)(
        response_state_loadings,
        state_covs,
    )
    if response_state_loadings.shape[0] <= 1:
        return same_covs, jnp.zeros(
            (0, response_state_loadings.shape[1], response_state_loadings.shape[1]),
            dtype=response_state_loadings.dtype,
        )

    lag1_state_covs = jax.vmap(lambda transition_t, state_cov_prev: transition_t @ state_cov_prev)(
        transitions,
        state_covs[:-1],
    )
    lag1_covs = jax.vmap(
        lambda loading_t, lag1_state_cov_t, loading_prev: (
            loading_t @ lag1_state_cov_t @ loading_prev.T
        )
    )(response_state_loadings[1:], lag1_state_covs, response_state_loadings[:-1])
    return same_covs, lag1_covs


def _support_response_covariances(
    response_means: jnp.ndarray,
    response_state_loadings: jnp.ndarray,
    *,
    t0_cov: jnp.ndarray,
    transitions: jnp.ndarray,
    process_covs: jnp.ndarray,
    observation_operator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project support-aware same-row and lag-1 covariances with a local scan."""
    from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
        _COUNT_OPERATOR_CODE,
        _MEAN_OPERATOR_CODE,
        _STD_OPERATOR_CODE,
        _SUM_OPERATOR_CODE,
    )

    emitted_means, semantic_mask = observation_operator.project_response_trajectory(response_means)
    n_timepoints, n_manifest = response_means.shape
    dtype = response_means.dtype

    same_cov_0 = response_state_loadings[0] @ t0_cov @ response_state_loadings[0].T
    if n_timepoints <= 1:
        same_pair_mask = jnp.expand_dims(semantic_mask, axis=2) * jnp.expand_dims(
            semantic_mask,
            axis=1,
        )
        return (
            emitted_means,
            same_cov_0[None, :, :] * same_pair_mask,
            jnp.zeros((0, n_manifest, n_manifest), dtype=dtype),
            semantic_mask,
        )

    assert observation_operator.summary_operator_codes is not None
    assert observation_operator.prev_coeffs is not None
    assert observation_operator.curr_coeffs is not None
    assert observation_operator.interval_weights is not None
    assert observation_operator.emission_slots is not None

    interval_mask = observation_operator.interval_summary_mask(dtype)
    summary_codes = observation_operator.summary_operator_codes
    transformed_interval_mask = interval_mask * (
        (summary_codes == _SUM_OPERATOR_CODE)
        | (summary_codes == _COUNT_OPERATOR_CODE)
        | (summary_codes == _MEAN_OPERATOR_CODE)
        | (summary_codes == _STD_OPERATOR_CODE)
    ).astype(dtype)
    direct_response_mask = 1.0 - transformed_interval_mask
    direct_response_diag = jnp.diag(direct_response_mask)

    n_latent = int(t0_cov.shape[0])
    n_slots = observation_operator.max_active_windows
    accum_dim = n_manifest * n_slots

    second_loadings = jax.vmap(_response_second_moment_loading)(
        response_means,
        response_state_loadings,
    )
    prev_coeffs = jnp.asarray(observation_operator.prev_coeffs, dtype=dtype)
    curr_coeffs = jnp.asarray(observation_operator.curr_coeffs, dtype=dtype)
    interval_weights = jnp.asarray(observation_operator.interval_weights, dtype=dtype)
    emission_slots = jnp.asarray(observation_operator.emission_slots, dtype=jnp.int64)
    eye_latent = jnp.eye(n_latent, dtype=dtype)
    eye_accum = jnp.eye(accum_dim, dtype=dtype)
    zeros_accum = observation_operator.empty_accumulators(dtype)
    zeros_accum_cov = jnp.zeros((accum_dim, accum_dim), dtype=dtype)
    zeros_latent_accum = jnp.zeros((n_latent, accum_dim), dtype=dtype)
    p_aug_0 = jnp.block(
        [
            [t0_cov, zeros_latent_accum, zeros_latent_accum],
            [zeros_latent_accum.T, zeros_accum_cov, zeros_accum_cov],
            [zeros_latent_accum.T, zeros_accum_cov, zeros_accum_cov],
        ]
    )
    y0 = jnp.concatenate(
        [
            response_state_loadings[0],
            jnp.zeros((n_manifest, accum_dim), dtype=dtype),
            jnp.zeros((n_manifest, accum_dim), dtype=dtype),
        ],
        axis=1,
    )
    cross_prev_0 = p_aug_0 @ y0.T

    def _scan_step(carry, inputs):
        accum_sum_mean_prev, accum_sumsq_mean_prev, accum_weight_prev, p_aug_prev, cross_prev = (
            carry
        )
        (
            response_mean_prev,
            response_mean_t,
            response_loading_prev,
            response_loading_t,
            second_loading_prev,
            second_loading_t,
            transition_t,
            process_cov_t,
            prev_coeff_t,
            curr_coeff_t,
            weight_t,
            emission_slots_t,
        ) = inputs

        obs_sum_mean = (
            accum_sum_mean_prev
            + prev_coeff_t * jnp.expand_dims(response_mean_prev, axis=-1)
            + curr_coeff_t * jnp.expand_dims(response_mean_t, axis=-1)
        )
        obs_sumsq_mean = (
            accum_sumsq_mean_prev
            + prev_coeff_t * jnp.expand_dims(response_mean_prev**2, axis=-1)
            + curr_coeff_t * jnp.expand_dims(response_mean_t**2, axis=-1)
        )
        obs_weight = accum_weight_prev + weight_t

        selected_sum_mean = _select_support_slot(obs_sum_mean, emission_slots_t)
        selected_sumsq_mean = _select_support_slot(obs_sumsq_mean, emission_slots_t)
        selected_weight = _select_support_slot(obs_weight, emission_slots_t)
        safe_weight = jnp.maximum(selected_weight, NUMERICAL_EPSILON)
        window_mean = selected_sum_mean / safe_weight
        window_second_mean = selected_sumsq_mean / safe_weight
        std_arg = jnp.maximum(window_second_mean - window_mean**2, NUMERICAL_EPSILON)
        std_mean = jnp.sqrt(std_arg)
        d_std_d_sum = -window_mean / (std_mean * safe_weight)
        d_std_d_sumsq = 1.0 / (2.0 * std_mean * safe_weight)

        alpha_sum = jnp.where(
            (summary_codes == _SUM_OPERATOR_CODE) | (summary_codes == _COUNT_OPERATOR_CODE),
            1.0,
            0.0,
        )
        alpha_sum = jnp.where(summary_codes == _MEAN_OPERATOR_CODE, 1.0 / safe_weight, alpha_sum)
        alpha_sum = jnp.where(summary_codes == _STD_OPERATOR_CODE, d_std_d_sum, alpha_sum)
        alpha_sumsq = jnp.where(summary_codes == _STD_OPERATOR_CODE, d_std_d_sumsq, 0.0)

        select_matrix = _support_selection_matrix(
            emission_slots_t,
            n_slots=n_slots,
            dtype=dtype,
        )
        prev_response_map = _support_accumulator_response_map(prev_coeff_t)
        curr_response_map = _support_accumulator_response_map(curr_coeff_t)
        curr_response_to_prev_x = response_loading_t @ transition_t
        curr_second_to_prev_x = second_loading_t @ transition_t
        sum_prev_x = (
            prev_response_map @ response_loading_prev + curr_response_map @ curr_response_to_prev_x
        )
        sumsq_prev_x = (
            prev_response_map @ second_loading_prev + curr_response_map @ curr_second_to_prev_x
        )
        sum_noise = curr_response_map @ response_loading_t
        sumsq_noise = curr_response_map @ second_loading_t

        emitted_interval_summary_mask = interval_mask * (emission_slots_t >= 0).astype(dtype)
        keep_mask = _reset_support_stat(
            jnp.ones((n_manifest, n_slots), dtype=dtype),
            emission_slots_t,
            emitted_interval_summary_mask,
        ).reshape(-1)
        keep_rows = jnp.expand_dims(keep_mask, axis=-1)
        identity_kept = keep_rows * eye_accum
        sum_select = jnp.expand_dims(alpha_sum, axis=1) * select_matrix
        sumsq_select = jnp.expand_dims(alpha_sumsq, axis=1) * select_matrix

        y_prev_x = (
            direct_response_diag @ curr_response_to_prev_x
            + sum_select @ sum_prev_x
            + sumsq_select @ sumsq_prev_x
        )
        y_noise = (
            direct_response_diag @ response_loading_t
            + sum_select @ sum_noise
            + sumsq_select @ sumsq_noise
        )
        y_t = jnp.concatenate([y_prev_x, sum_select, sumsq_select], axis=1)

        f_t = jnp.block(
            [
                [
                    transition_t,
                    jnp.zeros((n_latent, accum_dim), dtype=dtype),
                    jnp.zeros((n_latent, accum_dim), dtype=dtype),
                ],
                [
                    keep_rows * sum_prev_x,
                    identity_kept,
                    jnp.zeros((accum_dim, accum_dim), dtype=dtype),
                ],
                [
                    keep_rows * sumsq_prev_x,
                    jnp.zeros((accum_dim, accum_dim), dtype=dtype),
                    identity_kept,
                ],
            ]
        )
        g_t = jnp.concatenate(
            [
                eye_latent,
                keep_rows * sum_noise,
                keep_rows * sumsq_noise,
            ],
            axis=0,
        )

        same_cov_t = y_t @ p_aug_prev @ y_t.T + y_noise @ process_cov_t @ y_noise.T
        lag1_cov_t = y_t @ cross_prev
        p_aug_t = f_t @ p_aug_prev @ f_t.T + g_t @ process_cov_t @ g_t.T
        cross_t = f_t @ p_aug_prev @ y_t.T + g_t @ process_cov_t @ y_noise.T

        next_accum_sum_mean = _reset_support_stat(
            obs_sum_mean,
            emission_slots_t,
            emitted_interval_summary_mask,
        )
        next_accum_sumsq_mean = _reset_support_stat(
            obs_sumsq_mean,
            emission_slots_t,
            emitted_interval_summary_mask,
        )
        next_accum_weight = _reset_support_stat(
            obs_weight,
            emission_slots_t,
            emitted_interval_summary_mask,
        )

        return (
            next_accum_sum_mean,
            next_accum_sumsq_mean,
            next_accum_weight,
            p_aug_t,
            cross_t,
        ), (
            same_cov_t,
            lag1_cov_t,
        )

    _, (same_covs_rest, lag1_covs) = lax.scan(
        _scan_step,
        (
            zeros_accum,
            zeros_accum,
            zeros_accum,
            p_aug_0,
            cross_prev_0,
        ),
        (
            response_means[:-1],
            response_means[1:],
            response_state_loadings[:-1],
            response_state_loadings[1:],
            second_loadings[:-1],
            second_loadings[1:],
            transitions,
            process_covs,
            prev_coeffs[1:],
            curr_coeffs[1:],
            interval_weights[1:],
            emission_slots[1:],
        ),
    )

    same_covs = jnp.concatenate([same_cov_0[None, :, :], same_covs_rest], axis=0)
    same_pair_mask = jnp.expand_dims(semantic_mask, axis=2) * jnp.expand_dims(
        semantic_mask,
        axis=1,
    )
    lag1_pair_mask = jnp.expand_dims(semantic_mask[1:], axis=2) * jnp.expand_dims(
        semantic_mask[:-1],
        axis=1,
    )
    return (
        emitted_means,
        same_covs * same_pair_mask,
        lag1_covs * lag1_pair_mask,
        semantic_mask,
    )


def _build_sensitivity_measurement_semantics(
    spec,
    *,
    manifest_cov: jnp.ndarray,
    extra_params: dict,
    observation_support,
):
    """Compile measurement semantics for the observation-space sensitivity map."""
    from nof1_causal_lab.models.ssm.inference.targets.kernels import compile_measurement_semantics

    return compile_measurement_semantics(
        manifest_dists=spec.manifest_dists,
        manifest_cov=manifest_cov,
        extra_params=extra_params or None,
        manifest_links=spec.manifest_links,
        observation_support=observation_support,
    )


def _flatten_time_block_covariance(cov_blocks: jnp.ndarray) -> jnp.ndarray:
    """Flatten ``(T, T, M, M)`` covariance blocks to ``(T*M, T*M)`` order."""
    return cov_blocks.transpose(0, 2, 1, 3).reshape(
        cov_blocks.shape[0] * cov_blocks.shape[2],
        cov_blocks.shape[1] * cov_blocks.shape[3],
    )


def _unflatten_time_block_covariance(
    cov_flat: jnp.ndarray,
    *,
    n_timepoints: int,
    n_manifest: int,
) -> jnp.ndarray:
    """Restore ``(T, T, M, M)`` covariance blocks from flattened form."""
    return cov_flat.reshape(n_timepoints, n_manifest, n_timepoints, n_manifest).transpose(
        0,
        2,
        1,
        3,
    )


def _build_state_cross_covariance_blocks(
    state_covs: jnp.ndarray,
    transitions: jnp.ndarray,
) -> jnp.ndarray:
    """Build all pairwise latent-state covariance blocks from one-step transitions."""
    n_timepoints = int(state_covs.shape[0])
    blocks: list[list[jnp.ndarray | None]] = [[None] * n_timepoints for _ in range(n_timepoints)]

    for time_idx in range(n_timepoints):
        blocks[time_idx][time_idx] = state_covs[time_idx]

    for time_idx in range(1, n_timepoints):
        transition = transitions[time_idx - 1]
        prev_row = blocks[time_idx - 1]
        for past_idx in range(time_idx):
            prev_block = prev_row[past_idx]
            assert prev_block is not None
            block = transition @ prev_block
            blocks[time_idx][past_idx] = block
            blocks[past_idx][time_idx] = block.T

    filled_blocks: list[list[jnp.ndarray]] = []
    for row in blocks:
        filled_row: list[jnp.ndarray] = []
        for block in row:
            assert block is not None
            filled_row.append(block)
        filled_blocks.append(filled_row)

    return jnp.stack(
        [jnp.stack(row, axis=0) for row in filled_blocks],
        axis=0,
    )


def _project_response_covariance_blocks(
    response_means: jnp.ndarray,
    response_cov_blocks: jnp.ndarray,
    observation_operator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project response-trajectory covariance through the support operator."""
    emitted_means, semantic_mask = observation_operator.project_response_trajectory(response_means)
    n_timepoints, n_manifest = response_means.shape

    if observation_operator.requires_interval_summary_handling:
        response_cov_flat = _flatten_time_block_covariance(response_cov_blocks)

        def _project_flat(response_flat: jnp.ndarray) -> jnp.ndarray:
            projected, _ = observation_operator.project_response_trajectory(
                response_flat.reshape(n_timepoints, n_manifest)
            )
            return projected.reshape(-1)

        emission_jacobian = jax.jacfwd(_project_flat)(response_means.reshape(-1))
        emitted_cov_flat = emission_jacobian @ response_cov_flat @ emission_jacobian.T
        emitted_cov_blocks = _unflatten_time_block_covariance(
            emitted_cov_flat,
            n_timepoints=n_timepoints,
            n_manifest=n_manifest,
        )
    else:
        emitted_cov_blocks = response_cov_blocks

    same_covs = jnp.stack(
        [emitted_cov_blocks[time_idx, time_idx] for time_idx in range(n_timepoints)],
        axis=0,
    )
    if n_timepoints > 1:
        lag1_covs = jnp.stack(
            [emitted_cov_blocks[time_idx, time_idx - 1] for time_idx in range(1, n_timepoints)],
            axis=0,
        )
    else:
        lag1_covs = jnp.zeros((0, n_manifest, n_manifest), dtype=response_means.dtype)

    same_pair_mask = jnp.expand_dims(semantic_mask, axis=2) * jnp.expand_dims(semantic_mask, axis=1)
    same_covs = same_covs * same_pair_mask
    if n_timepoints > 1:
        lag1_pair_mask = jnp.expand_dims(semantic_mask[1:], axis=2) * jnp.expand_dims(
            semantic_mask[:-1],
            axis=1,
        )
        lag1_covs = lag1_covs * lag1_pair_mask

    return emitted_means, same_covs, lag1_covs, semantic_mask


def _predict_observation_components(
    det: dict[str, jnp.ndarray],
    extra_params: dict,
    spec,
    times: jnp.ndarray,
    *,
    structure_runtime: SSMStructureRuntime,
    observation_support=None,
    transition_inputs: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Predict emitted-observation mean, covariance, lagged covariance, noise scale, and mask."""
    n_latent, n_manifest = spec.n_latent, spec.n_manifest

    drift = det.get("drift", jnp.zeros((n_latent, n_latent)))
    diffusion_chol = det.get("diffusion", jnp.eye(n_latent))
    diffusion_cov = diffusion_chol @ diffusion_chol.T
    t0_means = det.get("t0_means", jnp.zeros(n_latent))
    t0_cov = det.get("t0_cov", jnp.eye(n_latent))
    manifest_cov = det.get("manifest_cov", jnp.eye(n_manifest))
    manifest_means = det.get("manifest_means", jnp.zeros(n_manifest))

    lambda_val = det.get("lambda", structure_runtime.lambda_template)
    input_effect = det.get("input_effect")

    cint = det.get("cint", jnp.zeros(n_latent))
    measurement_semantics = _build_sensitivity_measurement_semantics(
        spec,
        manifest_cov=manifest_cov,
        extra_params=extra_params,
        observation_support=observation_support,
    )
    obs_kernel = measurement_semantics.obs_kernel
    observation_operator = measurement_semantics.observation_operator

    def _point_moments(x_mean: jnp.ndarray, state_cov: jnp.ndarray):
        eta_mean = lambda_val @ x_mean + manifest_means
        eta_cov = lambda_val @ state_cov @ lambda_val.T
        response_mean = obs_kernel.response_fn(eta_mean)
        _, response_jacobian = _response_latent_covariance(
            eta_mean,
            eta_cov,
            obs_kernel=obs_kernel,
        )
        response_state_loading = response_jacobian @ lambda_val
        point_noise_variance_args = _observation_noise_variance_arguments(
            eta_mean,
            response_mean,
            manifest_dists=measurement_semantics.manifest_dists,
        )
        return (
            eta_mean,
            response_mean,
            response_state_loading,
            point_noise_variance_args,
        )

    dt_array = jnp.diff(times)
    interval_inputs = None
    if transition_inputs is not None:
        interval_inputs = jnp.asarray(transition_inputs)[: times.shape[0]][1:]
    Ad, Qd, cd = discretize_system_with_inputs_batched(
        drift,
        diffusion_cov,
        cint,
        input_effect,
        interval_inputs,
        dt_array,
    )
    carry_dtype = jnp.result_type(t0_means, t0_cov)
    Ad = Ad.astype(carry_dtype)
    Qd = Qd.astype(carry_dtype)
    if cd is not None:
        cd = cd.astype(carry_dtype)
    t0_means = t0_means.astype(carry_dtype)
    t0_cov = t0_cov.astype(carry_dtype)

    (
        _eta_mean_0,
        response_mean_0,
        response_state_loading_0,
        point_noise_variance_args_0,
    ) = _point_moments(t0_means, t0_cov)

    def scan_fn(carry, inputs):
        x_m, P = carry
        Ad_t, Qd_t, cd_t = inputs

        x_m_next = Ad_t @ x_m + cd_t
        P_next = Ad_t @ P @ Ad_t.T + Qd_t
        (
            eta_mean_next,
            response_mean_next,
            response_state_loading_next,
            point_noise_variance_args_next,
        ) = _point_moments(x_m_next, P_next)
        return (x_m_next, P_next), (
            eta_mean_next,
            response_mean_next,
            response_state_loading_next,
            P_next,
            point_noise_variance_args_next,
        )

    (
        _,
        (
            _eta_means_rest,
            response_means_rest,
            response_state_loadings_rest,
            state_covs_rest,
            point_noise_variance_args_rest,
        ),
    ) = lax.scan(
        scan_fn,
        (t0_means, t0_cov),
        (Ad, Qd, cd),
    )

    response_means = jnp.concatenate([response_mean_0[None, :], response_means_rest], axis=0)
    response_state_loadings = jnp.concatenate(
        [response_state_loading_0[None, :, :], response_state_loadings_rest],
        axis=0,
    )
    state_covs = jnp.concatenate([t0_cov[None, :, :], state_covs_rest], axis=0)
    point_noise_variance_args = jnp.concatenate(
        [point_noise_variance_args_0[None, :], point_noise_variance_args_rest],
        axis=0,
    )

    if observation_operator.requires_interval_summary_handling:
        emitted_means, emitted_same_covs, emitted_lag1_covs, semantic_mask = (
            _support_response_covariances(
                response_means,
                response_state_loadings,
                t0_cov=t0_cov,
                transitions=Ad,
                process_covs=Qd,
                observation_operator=observation_operator,
            )
        )
    else:
        emitted_means = response_means
        emitted_same_covs, emitted_lag1_covs = _point_response_covariances(
            response_state_loadings,
            state_covs,
            Ad,
        )
        semantic_mask = jnp.ones_like(emitted_means, dtype=emitted_means.dtype)

    if observation_operator.requires_interval_summary_handling:
        point_like_mask = jnp.broadcast_to(
            observation_operator.point_like_mask(emitted_means.dtype),
            emitted_means.shape,
        )
        interval_emission_mask = jnp.maximum(semantic_mask - point_like_mask, 0.0)
        emitted_noise_variance_args = (
            point_noise_variance_args * point_like_mask + emitted_means * interval_emission_mask
        )
    else:
        emitted_noise_variance_args = point_noise_variance_args
        semantic_mask = jnp.ones_like(emitted_means, dtype=emitted_means.dtype)
    emitted_obs_noise_covs = jax.vmap(
        lambda variance_args_t: _observation_noise_covariance(
            variance_args_t,
            obs_kernel=obs_kernel,
        )
    )(emitted_noise_variance_args)
    same_pair_mask = jnp.expand_dims(semantic_mask, axis=2) * jnp.expand_dims(semantic_mask, axis=1)
    emitted_same_covs = emitted_same_covs + emitted_obs_noise_covs * same_pair_mask
    emitted_obs_noise_vars = jnp.diagonal(emitted_obs_noise_covs, axis1=1, axis2=2) * semantic_mask
    emitted_obs_noise_sd = jnp.sqrt(jnp.maximum(emitted_obs_noise_vars, NUMERICAL_EPSILON))

    return (
        emitted_means,
        emitted_same_covs,
        emitted_lag1_covs,
        emitted_obs_noise_sd,
        semantic_mask,
    )


def _flatten_lower_triangular(mats: jnp.ndarray) -> jnp.ndarray:
    """Flatten the lower triangle of a stack of symmetric matrices."""
    tri_i, tri_j = np.tril_indices(int(mats.shape[-1]))
    return mats[:, tri_i, tri_j].reshape(-1)


def _flatten_observation_moment_summary(
    means: jnp.ndarray,
    same_covs: jnp.ndarray,
    lag1_covs: jnp.ndarray,
) -> jnp.ndarray:
    """Flatten emitted-observation moments into one feature vector."""
    mean_features = means.reshape(-1)
    same_cov_features = _flatten_lower_triangular(same_covs)
    lag_cov_features = lag1_covs.reshape(-1)
    return jnp.concatenate([mean_features, same_cov_features, lag_cov_features])


def _moment_summary_row_scales(obs_noise_sd: jnp.ndarray) -> jnp.ndarray:
    """Diagonal-FIM noise-only row scaling for the moment summary.

    Returns per-feature scales aligned with ``_flatten_observation_moment_summary``: for
    a mean feature at manifest ``m`` the scale is ``sigma_obs[m]``; for a (same-time or
    lag-1) covariance feature at manifests ``(m, n)`` the scale is the product
    ``sigma_obs[m] * sigma_obs[n]``. Squaring these gives a diagonal approximation to
    ``Var(g_i)`` under Gaussian observations. The approximation is exact for mean
    features and matches only the noise-only term ``sigma_m^2 * sigma_n^2`` of the
    sampling variance for covariance features; it understates ``Var(g_i)`` when
    manifest means are large relative to ``sigma_obs``. Inter-moment correlations are
    ignored.
    """
    mean_scales = obs_noise_sd.reshape(-1)
    same_cov_scales = _flatten_lower_triangular(
        jnp.expand_dims(obs_noise_sd, axis=2) * jnp.expand_dims(obs_noise_sd, axis=1)
    )
    if obs_noise_sd.shape[0] <= 1:
        lag_cov_scales = jnp.zeros((0,), dtype=obs_noise_sd.dtype)
    else:
        lag_cov_scales = (
            jnp.expand_dims(obs_noise_sd[1:], axis=2) * jnp.expand_dims(obs_noise_sd[:-1], axis=1)
        ).reshape(-1)
    return jnp.concatenate([mean_scales, same_cov_scales, lag_cov_scales])


def _predict_observation_moments(
    z_flat,
    unravel_fn,
    transforms,
    spec,
    times,
    *,
    structure_runtime: SSMStructureRuntime,
    observation_support=None,
    registry,
    transition_inputs: jnp.ndarray | None = None,
):
    """Predicted observation-space moment summary from unconstrained params."""
    det, extra_params = _assemble_sensitivity_measurement_state(
        z_flat,
        unravel_fn,
        transforms,
        spec,
        structure_runtime=structure_runtime,
        registry=registry,
    )
    emitted_means, emitted_same_covs, emitted_lag1_covs, _emitted_obs_noise_sd, _semantic_mask = (
        _predict_observation_components(
            det,
            extra_params,
            spec,
            times,
            structure_runtime=structure_runtime,
            observation_support=observation_support,
            transition_inputs=transition_inputs,
        )
    )
    return _flatten_observation_moment_summary(
        emitted_means,
        emitted_same_covs,
        emitted_lag1_covs,
    )


def _predict_observation_row_scales(
    z_flat,
    unravel_fn,
    transforms,
    spec,
    times,
    *,
    structure_runtime: SSMStructureRuntime,
    observation_support=None,
    registry,
    transition_inputs: jnp.ndarray | None = None,
):
    """Predicted observation-scale normalizers aligned to the moment summary."""
    det, extra_params = _assemble_sensitivity_measurement_state(
        z_flat,
        unravel_fn,
        transforms,
        spec,
        structure_runtime=structure_runtime,
        registry=registry,
    )
    _emitted_means, _emitted_same_covs, _emitted_lag1_covs, emitted_obs_noise_sd, _semantic_mask = (
        _predict_observation_components(
            det,
            extra_params,
            spec,
            times,
            structure_runtime=structure_runtime,
            observation_support=observation_support,
            transition_inputs=transition_inputs,
        )
    )
    return _moment_summary_row_scales(emitted_obs_noise_sd)
