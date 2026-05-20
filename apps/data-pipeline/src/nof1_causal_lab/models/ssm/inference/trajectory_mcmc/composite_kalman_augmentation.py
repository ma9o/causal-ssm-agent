"""Interval-summary augmentation for the composite (non-linear) Kalman path.

For interval-summary observations ``y_k = ∫_{t_{k-1}}^{t_k} (H · x_s + d) ds``
the existing dense-linear path augments the state with an accumulator
latent ``S_i`` per summary manifest (``dS_i/dt = H_i · x``) and
discretises the block-triangular augmented system via Van Loan expm.
After each observation, the accumulator is reset to zero — encoded by
zeroing the accumulator column of the discrete transition matrix at the
next step (see ``inference/targets/linear_summary_augmentation.py``).

This module ports that augmentation to the composite (per-step
linearisation) regime:

1. Linearise the composite vector field at each ``x_traj[t-1]`` (using
   the original non-augmented latents).
2. For each step, build the block-triangular augmented drift from the
   *local* linearisation and the (time-invariant) summary operator.
3. Discretise the per-step augmented system via Van Loan expm.
4. Apply ``reset_scales`` to zero accumulator columns of ``A_d`` for
   transitions following an observation.

The returned ``LatentContext`` has augmented-state dimension
``n_latent + n_accumulators``. ``H_rows`` and ``d_rows`` are per-time
observation matrices: point-in-time manifests read off the original
latents, interval-summary manifests read off the accumulator slots
with the appropriate scale (``1`` for sum, ``1/Δt`` for mean).

For a pure ``DenseLinear`` field with no intervention the per-step
local linearisation equals the dense ``A`` exactly, so this matches
``build_linear_summary_augmented_system`` output bit-for-bit (verified
in tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.discretization import discretize_linear_system_exact
from nof1_causal_lab.models.ssm.dynamics import (
    Intervention,
    VectorFieldArgs,
)

from .auxiliary_kalman import LatentContext

if TYPE_CHECKING:
    from jax import Array

    from nof1_causal_lab.models.ssm.dynamics import CompositeVectorField


@dataclass(frozen=True)
class SimpleAccumulatorPlan:
    """Minimal plan for one-accumulator-per-interval-manifest augmentation.

    A simpler analogue of the existing ``LinearSummaryAccumulatorPlan``
    in ``targets/laplace/shared.py``. Each interval-summary manifest
    has exactly one accumulator that's reset to zero after every
    observation; the emission scale is ``1`` (sum) or ``1/Δt`` (mean).
    For models with shared-support summaries or multi-slot accumulators
    the full plan from ``_build_linear_summary_accumulator_plan`` should
    be used instead.

    Fields:
        n_accumulators: Number of accumulator latents (= number of
            interval-summary manifests).
        accumulator_manifest_indices: ``(n_accumulators,)`` — manifest
            index each accumulator backs. ``accumulator_manifest_indices[i] = m``
            means accumulator ``i`` accumulates ``H[m] · x + d[m]``.
        row_emission_accumulator_indices: ``(T, n_manifest)`` int array —
            which accumulator each manifest reads from at each step
            (``-1`` for point-in-time manifests).
        row_emission_scales: ``(T, n_manifest)`` float — scale applied
            when reading the accumulator value into the manifest
            prediction (``1`` for sum, ``1/Δt`` for mean).
        row_reset_mask: ``(T, n_accumulators)`` float — ``1`` if the
            accumulator should be reset to zero at the start of the
            *next* transition (i.e., the accumulator column of
            ``Ad_aug[t+1]`` is zeroed). The simple case sets this to
            all-ones (always reset after observation).
    """

    n_accumulators: int
    accumulator_manifest_indices: Array
    row_emission_accumulator_indices: Array
    row_emission_scales: Array
    row_reset_mask: Array


def build_simple_accumulator_plan(
    interval_manifest_indices: list[int],
    n_manifest: int,
    T: int,
    *,
    operator_kind: str = "sum",
    time_intervals: Array | None = None,
    dtype=jnp.float32,
) -> SimpleAccumulatorPlan:
    """Construct a one-accumulator-per-interval-manifest plan.

    Args:
        interval_manifest_indices: indices of manifests whose
            observations are interval summaries (sum or mean over the
            interval). The remaining manifests are treated as
            point-in-time observations.
        n_manifest: total number of manifests.
        T: number of observation time points.
        operator_kind: ``"sum"`` or ``"mean"``. For ``"mean"`` the
            ``time_intervals`` argument must be supplied.
        time_intervals: ``(T,)`` per-step interval lengths Δt. Required
            for ``"mean"``.
        dtype: float dtype for scale arrays.
    """
    n_accumulators = len(interval_manifest_indices)
    accumulator_manifest_indices = jnp.asarray(
        interval_manifest_indices, dtype=jnp.int64
    )

    row_emission_indices = -jnp.ones((T, n_manifest), dtype=jnp.int64)
    for accumulator_idx, manifest_idx in enumerate(interval_manifest_indices):
        row_emission_indices = row_emission_indices.at[:, manifest_idx].set(accumulator_idx)

    scales = jnp.ones((T, n_manifest), dtype=dtype)
    if operator_kind == "mean":
        if time_intervals is None:
            raise ValueError("operator_kind='mean' requires time_intervals.")
        ti = jnp.asarray(time_intervals, dtype=dtype)
        inv_ti = 1.0 / jnp.maximum(ti, jnp.asarray(MIN_DT, dtype=dtype))
        for manifest_idx in interval_manifest_indices:
            scales = scales.at[:, manifest_idx].set(inv_ti)
    elif operator_kind != "sum":
        raise ValueError(f"Unknown operator_kind: {operator_kind!r}")

    row_reset_mask = jnp.ones((T, n_accumulators), dtype=dtype)

    return SimpleAccumulatorPlan(
        n_accumulators=n_accumulators,
        accumulator_manifest_indices=accumulator_manifest_indices,
        row_emission_accumulator_indices=row_emission_indices,
        row_emission_scales=scales,
        row_reset_mask=row_reset_mask,
    )


def composite_latent_context_at_trajectory_augmented(
    *,
    vector_field: CompositeVectorField,
    vf_params: tuple[dict[str, Array], ...],
    x_traj: Array,
    init_mean: Array,
    init_cov: Array,
    diffusion_cov: Array,
    runtime_times: Array,
    H: Array,
    d_meas: Array,
    R: Array,
    plan: SimpleAccumulatorPlan,
    support_kind_codes: Array,
    transition_inputs: Array | None = None,
    input_effect: Array | None = None,
) -> LatentContext:
    """Composite per-step linearisation + linear-summary state augmentation.

    The trajectory ``x_traj`` is in the *original* (non-augmented)
    coordinates, with shape ``(T, n_latent)``. The returned
    ``LatentContext`` carries the augmented state of dimension
    ``n_latent + n_accumulators``; ``init_mean``/``init_cov`` are
    extended with zeros for accumulator slots.

    Args:
        vector_field: Original (non-augmented) composite drift.
        vf_params: Parameter tuple matching ``vector_field.components``.
        x_traj: ``(T, n_latent)`` trajectory in original latents.
        init_mean: ``(n_latent,)`` initial mean of original latents.
        init_cov: ``(n_latent, n_latent)`` initial covariance.
        diffusion_cov: ``(n_latent, n_latent)`` SDE diffusion ``G·G'``.
        runtime_times: ``(T,)`` observation times.
        H, d_meas, R: Original observation model — ``y = H · x + d_meas``
            (point-in-time) or ``y_k = ∫ (H · x + d_meas) ds``
            (interval, with the manifests selected by
            ``plan.accumulator_manifest_indices``).
        plan: Accumulator plan; see ``build_simple_accumulator_plan``.
        support_kind_codes: ``(n_manifest,)`` int array; ``0`` =
            point-in-time, ``1`` = interval.

    Returns:
        ``LatentContext`` with augmented per-step ``(Ad, Qd, cd)``,
        augmented ``(init_mean, init_cov)``, and per-time
        ``(H_rows, d_rows)`` that read the right slot for each
        manifest.
    """
    dtype = jnp.result_type(
        runtime_times, init_mean, init_cov, diffusion_cov, H, d_meas, R
    )
    n_latent = vector_field.n_latent
    n_accumulators = plan.n_accumulators
    augmented_dim = n_latent + n_accumulators
    T = int(runtime_times.shape[0])
    n_manifest = int(H.shape[0])

    time_intervals = (
        jnp.diff(runtime_times, prepend=runtime_times[0])
        .at[0]
        .set(MIN_DT)
        .astype(dtype)
    )

    # 1. Per-step linearisation of the composite vector field.
    args = VectorFieldArgs(params=vf_params, intervention=Intervention.none())
    if x_traj.shape[0] >= T:
        x_lin_tail = x_traj[: T - 1]
    else:
        x_lin_tail = jnp.broadcast_to(x_traj[-1], (T - 1, x_traj.shape[-1]))
    x_lin_batch = jnp.concatenate([init_mean[None, :], x_lin_tail], axis=0)

    def _linearize(x_lin: Array) -> tuple[Array, Array]:
        return vector_field.linearize(x_lin, args)

    A_local_batch, b_local_batch = jax.vmap(_linearize)(x_lin_batch)

    # Per-step covariate forcing on the original (non-accumulator) latents.
    if transition_inputs is not None and input_effect is not None:
        forcing_batch = (
            jnp.asarray(transition_inputs, dtype=dtype)
            @ jnp.asarray(input_effect, dtype=dtype).T
        )
        b_local_batch = b_local_batch + forcing_batch

    # 2. Per-step augmentation + Van Loan discretisation.
    H = jnp.asarray(H, dtype=dtype)
    d_meas = jnp.asarray(d_meas, dtype=dtype)
    diffusion_cov = jnp.asarray(diffusion_cov, dtype=dtype)

    H_accum = (
        H[plan.accumulator_manifest_indices]
        if n_accumulators > 0
        else jnp.zeros((0, n_latent), dtype=dtype)
    )
    d_accum = (
        d_meas[plan.accumulator_manifest_indices]
        if n_accumulators > 0
        else jnp.zeros((0,), dtype=dtype)
    )

    diffusion_aug = jnp.zeros((augmented_dim, augmented_dim), dtype=dtype)
    diffusion_aug = diffusion_aug.at[:n_latent, :n_latent].set(diffusion_cov)

    def _augment_and_discretise(
        A_local: Array, b_local: Array, dt: Array
    ) -> tuple[Array, Array, Array]:
        drift_aug = jnp.zeros((augmented_dim, augmented_dim), dtype=dtype)
        drift_aug = drift_aug.at[:n_latent, :n_latent].set(A_local.astype(dtype))
        if n_accumulators > 0:
            drift_aug = drift_aug.at[n_latent:, :n_latent].set(H_accum)

        cint_aug = jnp.zeros((augmented_dim,), dtype=dtype)
        cint_aug = cint_aug.at[:n_latent].set(b_local.astype(dtype))
        if n_accumulators > 0:
            cint_aug = cint_aug.at[n_latent:].set(d_accum)

        Ad, Qd, cd = discretize_linear_system_exact(drift_aug, diffusion_aug, cint_aug, dt)
        if cd is None:
            cd = jnp.zeros((augmented_dim,), dtype=dtype)
        return Ad, Qd, cd

    Ad_aug, Qd_aug, cd_aug = jax.vmap(_augment_and_discretise)(
        A_local_batch, b_local_batch, time_intervals
    )

    # 3. Reset accumulator columns after observations.
    reset_scales = jnp.ones((T, augmented_dim), dtype=dtype)
    if n_accumulators > 0:
        reset_scales = reset_scales.at[:, n_latent:].set(
            1.0 - plan.row_reset_mask.astype(dtype)
        )
    if T > 1:
        Ad_aug = Ad_aug.at[1:].set(Ad_aug[1:] * reset_scales[:-1, None, :])

    # 4. Augmented initial distribution.
    init_mean_aug = jnp.concatenate(
        [init_mean.astype(dtype), jnp.zeros((n_accumulators,), dtype=dtype)], axis=0
    )
    init_cov_aug = jnp.zeros((augmented_dim, augmented_dim), dtype=dtype)
    init_cov_aug = init_cov_aug.at[:n_latent, :n_latent].set(init_cov.astype(dtype))

    # 5. Per-time observation matrices.
    H_rows = jnp.zeros((T, n_manifest, augmented_dim), dtype=dtype)
    d_rows = jnp.zeros((T, n_manifest), dtype=dtype)

    support_kind_codes_arr = jnp.asarray(support_kind_codes)
    point_mask = support_kind_codes_arr == 0

    # Point manifests read off the original latents (the first n_latent cols of H_rows).
    H_point_broadcast = jnp.broadcast_to(
        jnp.where(point_mask[:, None], H, jnp.zeros_like(H)),
        (T, n_manifest, n_latent),
    )
    d_point_broadcast = jnp.broadcast_to(
        jnp.where(point_mask, d_meas, jnp.zeros_like(d_meas)),
        (T, n_manifest),
    )
    H_rows = H_rows.at[:, :, :n_latent].set(H_point_broadcast)
    d_rows = d_rows.at[:].set(d_point_broadcast)

    # Interval manifests read off the accumulator slot for their step,
    # scaled by row_emission_scales.
    if n_accumulators > 0:
        emission_indices = jnp.asarray(plan.row_emission_accumulator_indices, dtype=jnp.int64)
        emission_scales = jnp.asarray(plan.row_emission_scales, dtype=dtype)
        accumulator_valid = emission_indices >= 0  # (T, n_manifest)
        accumulator_idx_clip = jnp.clip(emission_indices, 0, n_accumulators - 1)
        accumulator_col = n_latent + accumulator_idx_clip  # (T, n_manifest)

        t_idx = jnp.arange(T)[:, None]
        m_idx = jnp.arange(n_manifest)[None, :]
        H_rows = H_rows.at[t_idx, m_idx, accumulator_col].set(
            jnp.where(accumulator_valid, emission_scales, 0.0)
        )

    return LatentContext(
        Ad=Ad_aug,
        Qd=Qd_aug,
        cd=cd_aug,
        init_mean=init_mean_aug,
        init_cov=init_cov_aug,
        H=H,
        d_meas=d_meas,
        R=R,
        extra_params=None,
        H_rows=H_rows,
        d_rows=d_rows,
    )
