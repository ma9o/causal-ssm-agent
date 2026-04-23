"""Exact augmented-state rewrites for linear interval-summary observations."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.models.ssm.discretization import discretize_linear_system_exact_batched
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    advance_support_observation_state,
    compile_observation_operator,
)


def build_linear_summary_augmented_system(
    *,
    plan: Any,
    time_intervals: jnp.ndarray,
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    support_kind_codes: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Build augmented dynamics and per-row observation operators for linear summaries."""
    dtype = jnp.result_type(
        time_intervals,
        drift,
        diffusion_cov,
        cint,
        H,
        d,
        init_mean,
        init_cov,
    )
    time_intervals = jnp.asarray(time_intervals, dtype=dtype)
    drift = jnp.asarray(drift, dtype=dtype)
    diffusion_cov = jnp.asarray(diffusion_cov, dtype=dtype)
    cint = jnp.asarray(cint, dtype=dtype)
    H = jnp.asarray(H, dtype=dtype)
    d = jnp.asarray(d, dtype=dtype)
    init_mean = jnp.asarray(init_mean, dtype=dtype)
    init_cov = jnp.asarray(init_cov, dtype=dtype)

    T = int(time_intervals.shape[0])
    n_latent = int(drift.shape[0])
    n_manifest = int(H.shape[0])
    n_accumulators = plan.n_accumulators
    augmented_dim = n_latent + n_accumulators

    drift_aug = jnp.zeros((augmented_dim, augmented_dim), dtype=dtype)
    drift_aug = drift_aug.at[:n_latent, :n_latent].set(drift)
    if n_accumulators > 0:
        drift_aug = drift_aug.at[n_latent:, :n_latent].set(H[plan.accumulator_manifest_indices])

    diffusion_aug = jnp.zeros((augmented_dim, augmented_dim), dtype=dtype)
    diffusion_aug = diffusion_aug.at[:n_latent, :n_latent].set(diffusion_cov)

    cint_aug = jnp.zeros((augmented_dim,), dtype=dtype)
    cint_aug = cint_aug.at[:n_latent].set(cint)
    if n_accumulators > 0:
        cint_aug = cint_aug.at[n_latent:].set(d[plan.accumulator_manifest_indices])

    Ad_aug, Qd_aug, cd_aug = discretize_linear_system_exact_batched(
        drift_aug,
        diffusion_aug,
        cint_aug,
        time_intervals,
    )
    if cd_aug is None:
        cd_aug = jnp.zeros((T, augmented_dim), dtype=dtype)
    else:
        cd_aug = jnp.asarray(cd_aug, dtype=dtype)
        if cd_aug.ndim == 1:
            cd_aug = cd_aug[:, None]

    init_mean_aug = jnp.concatenate(
        [init_mean, jnp.zeros((n_accumulators,), dtype=dtype)],
        axis=0,
    )
    init_cov_aug = jnp.zeros((augmented_dim, augmented_dim), dtype=dtype)
    init_cov_aug = init_cov_aug.at[:n_latent, :n_latent].set(init_cov)

    H_rows = jnp.zeros((T, n_manifest, augmented_dim), dtype=dtype)
    d_rows = jnp.zeros((T, n_manifest), dtype=dtype)

    point_manifest_indices = np.flatnonzero(np.asarray(support_kind_codes) == 0)
    if point_manifest_indices.size > 0:
        point_idx = jnp.asarray(point_manifest_indices, dtype=jnp.int64)
        H_rows = H_rows.at[:, point_idx, :n_latent].set(
            jnp.broadcast_to(H[point_idx], (T, point_idx.shape[0], n_latent))
        )
        d_rows = d_rows.at[:, point_idx].set(
            jnp.broadcast_to(d[point_idx], (T, point_idx.shape[0]))
        )

    emission_indices = np.asarray(plan.row_emission_accumulator_indices)
    emission_scales = np.asarray(plan.row_emission_scales, dtype=np.float64)
    for time_idx in range(T):
        for manifest_idx in range(n_manifest):
            accumulator_idx = int(emission_indices[time_idx, manifest_idx])
            if accumulator_idx < 0:
                continue
            H_rows = H_rows.at[time_idx, manifest_idx, n_latent + accumulator_idx].set(
                jnp.asarray(emission_scales[time_idx, manifest_idx], dtype=dtype)
            )

    reset_scales = jnp.ones((T, augmented_dim), dtype=dtype)
    if n_accumulators > 0:
        reset_scales = reset_scales.at[:, n_latent:].set(1.0 - plan.row_reset_mask.astype(dtype))
    if T > 1:
        Ad_aug = Ad_aug.at[1:].set(Ad_aug[1:] * reset_scales[:-1, None, :])

    return Ad_aug, Qd_aug, cd_aug, init_mean_aug, init_cov_aug, H_rows, d_rows


def row_observation_log_probs(
    latent_trajectory: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel,
) -> jnp.ndarray:
    """Per-row ``(T,)`` point-observation log-prob for per-row operators.

    Sum over the leading axis equals :func:`row_observation_log_prob`.
    Used by the per-t MH-ratio diagnostic.
    """
    clean_obs = jnp.nan_to_num(observations, nan=0.0)
    obs_mask_float = obs_mask.astype(latent_trajectory.dtype)
    return jax.vmap(
        lambda y_t, z_t, mask_t, H_t, d_t: obs_kernel.emission_fn(
            y_t,
            z_t,
            H_t,
            d_t,
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


def lift_linear_summary_observation_trajectory(
    latent_trajectory: jnp.ndarray,
    *,
    H: jnp.ndarray,
    d: jnp.ndarray,
    plan: Any,
    observation_support,
) -> jnp.ndarray:
    """Embed a base latent trajectory into the augmented observation-state coordinates.

    This helper reproduces the exact emitted interval-summary statistics used by
    the support-aware observation likelihood. It is intended for correctness
    checks of the observation rewrite, not for prior evaluation.
    """
    operator = compile_observation_operator(observation_support)
    if not operator.requires_interval_summary_handling:
        return latent_trajectory

    dtype = latent_trajectory.dtype
    response_trajectory = jax.vmap(lambda z_t: H @ z_t + d)(latent_trajectory)
    n_time = int(latent_trajectory.shape[0])

    emission_slots = np.asarray(observation_support.emission_slot_indices, dtype=np.int64)
    emission_indices = np.asarray(plan.row_emission_accumulator_indices, dtype=np.int64)
    accumulator_slots = np.full((plan.n_accumulators,), -1, dtype=np.int64)
    for time_idx in range(n_time):
        for manifest_idx in range(emission_indices.shape[1]):
            accumulator_idx = int(emission_indices[time_idx, manifest_idx])
            if accumulator_idx < 0:
                continue
            slot_idx = int(emission_slots[time_idx, manifest_idx])
            if accumulator_slots[accumulator_idx] < 0:
                accumulator_slots[accumulator_idx] = slot_idx
    if np.any(accumulator_slots < 0):
        raise ValueError("Linear summary augmentation could not recover accumulator slots.")

    accumulator_manifest_indices = np.asarray(plan.accumulator_manifest_indices, dtype=np.int64)
    accumulator_slots_jnp = jnp.asarray(accumulator_slots, dtype=jnp.int64)
    accumulator_manifest_jnp = jnp.asarray(accumulator_manifest_indices, dtype=jnp.int64)

    def _gather_state(accum_sum: jnp.ndarray) -> jnp.ndarray:
        return accum_sum[accumulator_manifest_jnp, accumulator_slots_jnp]

    aug_state_0 = jnp.concatenate(
        [latent_trajectory[0], jnp.zeros((plan.n_accumulators,), dtype=dtype)],
        axis=0,
    )
    if n_time == 1:
        return aug_state_0[None, :]

    assert operator.prev_coeffs is not None
    assert operator.curr_coeffs is not None
    assert operator.interval_weights is not None
    assert operator.emission_slots is not None
    accum0 = operator.empty_accumulators(dtype)
    full_obs_mask = jnp.ones((operator.n_manifest,), dtype=dtype)
    prev_coeffs = jnp.asarray(operator.prev_coeffs, dtype=dtype)
    curr_coeffs = jnp.asarray(operator.curr_coeffs, dtype=dtype)
    interval_weights = jnp.asarray(operator.interval_weights, dtype=dtype)
    emission_slots_jax = jnp.asarray(operator.emission_slots, dtype=jnp.int64)

    def _scan_step(carry, inputs):
        response_prev, accum_sum, accum_sumsq, accum_weight = carry
        response_t, prev_coeff_t, curr_coeff_t, weight_t, emission_slots_t, latent_t = inputs
        step_result = advance_support_observation_state(
            operator,
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
        aug_t = jnp.concatenate(
            [latent_t, _gather_state(step_result.obs_sum)],
            axis=0,
        )
        return (
            response_t,
            step_result.next_accum_sum,
            step_result.next_accum_sumsq,
            step_result.next_accum_weight,
        ), aug_t

    _, aug_rest = jax.lax.scan(
        _scan_step,
        (response_trajectory[0], accum0, accum0, accum0),
        (
            response_trajectory[1:],
            prev_coeffs[1:],
            curr_coeffs[1:],
            interval_weights[1:],
            emission_slots_jax[1:],
            latent_trajectory[1:],
        ),
    )
    return jnp.concatenate([aug_state_0[None, :], aug_rest], axis=0)
