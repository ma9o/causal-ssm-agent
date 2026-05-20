"""Tests for ``composite_latent_context_at_trajectory_augmented``.

Three layers:

1. **Linear-case parity**: for a pure ``DenseLinear`` field with no
   intervention, the composite augmentation must bit-match the existing
   dense ``build_linear_summary_augmented_system`` output. This is the
   primary correctness check — it pins the implementation against the
   pre-existing tested path.

2. **Non-linear shape correctness**: a Hill / Multiplicative field
   produces finite augmented matrices of the expected shape, with
   per-step ``Ad`` varying across the trajectory (proving the
   linearisation is per-step).

3. **Reset semantics**: accumulator columns of ``Ad`` for steps after
   an observation are zeroed (so the accumulator restarts from zero
   each interval).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.counterfactual import linear_vector_field
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeVectorField,
    DiagonalDecay,
    HillEdge,
)
from nof1_causal_lab.models.ssm.inference.targets.linear_summary_augmentation import (
    build_linear_summary_augmented_system,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.composite_kalman_augmentation import (
    build_simple_accumulator_plan,
    composite_latent_context_at_trajectory_augmented,
)

# =============================================================================
# Linear-case parity with the dense path
# =============================================================================


class TestAugmentationLinearParity:
    """For a single ``DenseLinear`` vector field the composite path's
    per-step linearisation collapses to the dense ``A`` exactly, so the
    augmented context must match ``build_linear_summary_augmented_system``
    output bit-for-bit (up to expm numerical noise)."""

    def test_matches_dense_path_for_sum_summary(self):
        # 2-latent system: latent 0 has decay -1, latent 1 has decay -0.5
        # with cross-coupling 0.3 from 0 to 1.
        A = jnp.array([[-1.0, 0.0], [0.3, -0.5]])
        cint = jnp.array([0.0, 0.1])
        diffusion_cov = jnp.eye(2) * 0.02

        # 2 manifests: manifest 0 is point-in-time on latent 1, manifest 1
        # is interval-sum on latent 0.
        H = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        d_meas = jnp.array([0.0, 0.0])
        R = jnp.eye(2) * 0.05
        init_mean = jnp.array([0.5, 0.2])
        init_cov = jnp.eye(2) * 0.1

        T = 5
        runtime_times = jnp.linspace(0.5, 2.5, T)
        time_intervals = jnp.diff(runtime_times, prepend=runtime_times[0]).at[0].set(MIN_DT)

        interval_manifests = [1]  # manifest 1 is the interval-sum
        plan = build_simple_accumulator_plan(
            interval_manifests,
            n_manifest=2,
            T=T,
            operator_kind="sum",
            dtype=jnp.float32,
        )
        support_kind_codes = jnp.array([0, 1], dtype=jnp.int64)

        # x_traj is irrelevant for a linear vector field — pick anything.
        x_traj = jnp.zeros((T, 2))

        composite_ctx = composite_latent_context_at_trajectory_augmented(
            vector_field=linear_vector_field(n_latent=2),
            vf_params=({"drift": A, "cint": cint},),
            x_traj=x_traj,
            init_mean=init_mean,
            init_cov=init_cov,
            diffusion_cov=diffusion_cov,
            runtime_times=runtime_times,
            H=H,
            d_meas=d_meas,
            R=R,
            plan=plan,
            support_kind_codes=support_kind_codes,
        )

        # Reference: existing dense augmentation
        ref = build_linear_summary_augmented_system(
            plan=plan,
            time_intervals=time_intervals,
            drift=A,
            diffusion_cov=diffusion_cov,
            cint=cint,
            H=H,
            d=d_meas,
            init_mean=init_mean,
            init_cov=init_cov,
            support_kind_codes=support_kind_codes,
        )
        Ad_ref, Qd_ref, cd_ref, init_mean_ref, init_cov_ref, H_rows_ref, d_rows_ref = ref

        assert jnp.allclose(composite_ctx.Ad, Ad_ref, atol=1e-5)
        assert jnp.allclose(composite_ctx.Qd, Qd_ref, atol=1e-5)
        # cd_ref shape may be (T, augmented_dim) or (T, augmented_dim, 1)
        cd_ref_2d = cd_ref.squeeze(-1) if cd_ref.ndim == 3 else cd_ref
        assert jnp.allclose(composite_ctx.cd, cd_ref_2d, atol=1e-5)
        assert jnp.allclose(composite_ctx.init_mean, init_mean_ref)
        assert jnp.allclose(composite_ctx.init_cov, init_cov_ref)
        assert jnp.allclose(composite_ctx.H_rows, H_rows_ref, atol=1e-5)
        assert jnp.allclose(composite_ctx.d_rows, d_rows_ref, atol=1e-5)

    def test_matches_dense_path_for_mean_summary(self):
        """Same parity check with operator_kind='mean' — accumulator
        emission scaled by 1/Δt."""
        A = jnp.array([[-0.7]])
        cint = jnp.array([0.05])
        diffusion_cov = jnp.eye(1) * 0.01

        H = jnp.array([[1.0]])
        d_meas = jnp.array([0.0])
        R = jnp.array([[0.02]])
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1) * 0.05

        T = 4
        runtime_times = jnp.linspace(1.0, 4.0, T)
        time_intervals = jnp.diff(runtime_times, prepend=runtime_times[0]).at[0].set(MIN_DT)

        interval_manifests = [0]
        plan = build_simple_accumulator_plan(
            interval_manifests,
            n_manifest=1,
            T=T,
            operator_kind="mean",
            time_intervals=time_intervals,
            dtype=jnp.float32,
        )
        support_kind_codes = jnp.array([1], dtype=jnp.int64)

        x_traj = jnp.zeros((T, 1))

        composite_ctx = composite_latent_context_at_trajectory_augmented(
            vector_field=linear_vector_field(n_latent=1),
            vf_params=({"drift": A, "cint": cint},),
            x_traj=x_traj,
            init_mean=init_mean,
            init_cov=init_cov,
            diffusion_cov=diffusion_cov,
            runtime_times=runtime_times,
            H=H,
            d_meas=d_meas,
            R=R,
            plan=plan,
            support_kind_codes=support_kind_codes,
        )

        ref = build_linear_summary_augmented_system(
            plan=plan,
            time_intervals=time_intervals,
            drift=A,
            diffusion_cov=diffusion_cov,
            cint=cint,
            H=H,
            d=d_meas,
            init_mean=init_mean,
            init_cov=init_cov,
            support_kind_codes=support_kind_codes,
        )
        Ad_ref, Qd_ref, cd_ref, _init_mean_ref, _init_cov_ref, H_rows_ref, _d_rows_ref = ref

        assert jnp.allclose(composite_ctx.Ad, Ad_ref, atol=1e-5)
        assert jnp.allclose(composite_ctx.Qd, Qd_ref, atol=1e-5)
        cd_ref_2d = cd_ref.squeeze(-1) if cd_ref.ndim == 3 else cd_ref
        assert jnp.allclose(composite_ctx.cd, cd_ref_2d, atol=1e-5)
        assert jnp.allclose(composite_ctx.H_rows, H_rows_ref, atol=1e-5)


# =============================================================================
# Reset semantics
# =============================================================================


class TestAugmentationResetSemantics:
    """The accumulator column of ``Ad_aug`` for transitions *after* the
    first one must be zero — encoding the reset of the accumulator
    after each observation."""

    def test_accumulator_column_zeroed_post_observation(self):
        A = jnp.array([[-1.0]])
        cint = jnp.array([0.0])
        diffusion_cov = jnp.eye(1) * 0.01
        H = jnp.array([[1.0]])
        d_meas = jnp.array([0.0])
        R = jnp.array([[0.05]])
        init_mean = jnp.array([0.5])
        init_cov = jnp.eye(1) * 0.1
        T = 3
        runtime_times = jnp.linspace(1.0, 3.0, T)

        plan = build_simple_accumulator_plan(
            interval_manifest_indices=[0],
            n_manifest=1,
            T=T,
            operator_kind="sum",
            dtype=jnp.float32,
        )
        support_kind_codes = jnp.array([1], dtype=jnp.int64)

        ctx = composite_latent_context_at_trajectory_augmented(
            vector_field=linear_vector_field(n_latent=1),
            vf_params=({"drift": A, "cint": cint},),
            x_traj=jnp.zeros((T, 1)),
            init_mean=init_mean,
            init_cov=init_cov,
            diffusion_cov=diffusion_cov,
            runtime_times=runtime_times,
            H=H,
            d_meas=d_meas,
            R=R,
            plan=plan,
            support_kind_codes=support_kind_codes,
        )
        # Augmented dim = 1 latent + 1 accumulator = 2.
        # Ad[t][:, n_latent:] should be all-zero for t >= 1
        # (accumulator column is zeroed after observation).
        for t in range(1, T):
            assert jnp.allclose(ctx.Ad[t][:, 1:], 0.0, atol=1e-6), (
                f"Ad at step {t} should have zero accumulator column"
            )


# =============================================================================
# Non-linear (Hill) shape + finiteness
# =============================================================================


class TestAugmentationNonLinear:
    """For a Hill vector field, the augmented context must be finite,
    correctly shaped, and have per-step-varying ``Ad`` (proving the
    linearisation is at the current trajectory)."""

    def test_hill_augmentation_finite_with_varying_jacobian(self):
        vf = CompositeVectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                HillEdge(source=0, target=1),
            ),
        )
        vf_params = (
            {"decay": jnp.array([0.5, 0.3])},
            {
                "Emax": jnp.asarray(1.5),
                "EC50": jnp.asarray(1.0),
                "n": jnp.asarray(2.0),
            },
        )
        diffusion_cov = jnp.eye(2) * 0.02
        H = jnp.array([[0.0, 1.0]])
        d_meas = jnp.array([0.0])
        R = jnp.array([[0.05]])
        init_mean = jnp.array([1.5, 0.0])
        init_cov = jnp.eye(2) * 0.1

        T = 5
        runtime_times = jnp.linspace(0.5, 2.5, T)
        # Trajectory varies — the linearisation point of latent 0 changes
        # → Hill derivative changes → Ad changes.
        x_traj = jnp.stack(
            [jnp.linspace(1.0, 0.3, T), jnp.linspace(0.2, 1.4, T)], axis=1
        )

        plan = build_simple_accumulator_plan(
            interval_manifest_indices=[0],
            n_manifest=1,
            T=T,
            operator_kind="sum",
            dtype=jnp.float32,
        )
        support_kind_codes = jnp.array([1], dtype=jnp.int64)

        ctx = composite_latent_context_at_trajectory_augmented(
            vector_field=vf,
            vf_params=vf_params,
            x_traj=x_traj,
            init_mean=init_mean,
            init_cov=init_cov,
            diffusion_cov=diffusion_cov,
            runtime_times=runtime_times,
            H=H,
            d_meas=d_meas,
            R=R,
            plan=plan,
            support_kind_codes=support_kind_codes,
        )
        assert ctx.Ad.shape == (T, 3, 3)  # 2 latents + 1 accumulator
        assert ctx.Qd.shape == (T, 3, 3)
        assert ctx.cd.shape == (T, 3)
        assert ctx.init_mean.shape == (3,)
        assert ctx.init_cov.shape == (3, 3)
        assert ctx.H_rows.shape == (T, 1, 3)
        assert bool(jnp.all(jnp.isfinite(ctx.Ad)))
        assert bool(jnp.all(jnp.isfinite(ctx.Qd)))
        assert bool(jnp.all(jnp.isfinite(ctx.cd)))
        # First two steps' top-left blocks differ (Hill Jacobian depends on x).
        assert not jnp.allclose(ctx.Ad[0, :2, :2], ctx.Ad[-1, :2, :2], atol=1e-3), (
            "Hill linearisation should vary across trajectory"
        )

    def test_hill_augmentation_h_rows_read_accumulator(self):
        """The observation row for the interval manifest must read from
        the accumulator slot (column n_latent), not from any latent."""
        vf = CompositeVectorField(
            n_latent=2,
            components=(DiagonalDecay(), HillEdge(source=0, target=1)),
        )
        vf_params = (
            {"decay": jnp.array([0.5, 0.3])},
            {
                "Emax": jnp.asarray(1.5),
                "EC50": jnp.asarray(1.0),
                "n": jnp.asarray(2.0),
            },
        )
        T = 3
        runtime_times = jnp.linspace(1.0, 3.0, T)
        x_traj = jnp.tile(jnp.array([1.0, 0.5]), (T, 1))

        plan = build_simple_accumulator_plan(
            interval_manifest_indices=[0],
            n_manifest=1,
            T=T,
            operator_kind="sum",
            dtype=jnp.float32,
        )
        support_kind_codes = jnp.array([1], dtype=jnp.int64)

        ctx = composite_latent_context_at_trajectory_augmented(
            vector_field=vf,
            vf_params=vf_params,
            x_traj=x_traj,
            init_mean=jnp.array([1.0, 0.0]),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.02,
            runtime_times=runtime_times,
            H=jnp.array([[1.0, 0.0]]),  # interval-sum of latent 0
            d_meas=jnp.array([0.0]),
            R=jnp.array([[0.05]]),
            plan=plan,
            support_kind_codes=support_kind_codes,
        )
        # H_rows[t, 0, :2] should be zero (no point-in-time read on latents).
        # H_rows[t, 0, 2] should be 1.0 (sum operator).
        for t in range(T):
            assert jnp.allclose(ctx.H_rows[t, 0, :2], 0.0)
            assert float(ctx.H_rows[t, 0, 2]) == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Plan factory
# =============================================================================


class TestSimpleAccumulatorPlan:
    def test_sum_plan_shape_and_values(self):
        plan = build_simple_accumulator_plan(
            interval_manifest_indices=[1, 3],
            n_manifest=5,
            T=4,
            operator_kind="sum",
            dtype=jnp.float32,
        )
        assert plan.n_accumulators == 2
        assert plan.accumulator_manifest_indices.tolist() == [1, 3]
        # Each interval manifest reads from its own accumulator across all T.
        assert int(plan.row_emission_accumulator_indices[0, 1]) == 0
        assert int(plan.row_emission_accumulator_indices[0, 3]) == 1
        # Point-in-time manifests get -1.
        assert int(plan.row_emission_accumulator_indices[0, 0]) == -1
        # Sum operator → scales are 1.
        assert plan.row_emission_scales[0, 1] == pytest.approx(1.0)
        # Always reset.
        assert bool(jnp.all(plan.row_reset_mask == 1.0))

    def test_mean_plan_scales_by_inverse_dt(self):
        time_intervals = jnp.array([1.0, 0.5, 0.25])
        plan = build_simple_accumulator_plan(
            interval_manifest_indices=[0],
            n_manifest=1,
            T=3,
            operator_kind="mean",
            time_intervals=time_intervals,
            dtype=jnp.float32,
        )
        assert float(plan.row_emission_scales[0, 0]) == pytest.approx(1.0)
        assert float(plan.row_emission_scales[1, 0]) == pytest.approx(2.0)
        assert float(plan.row_emission_scales[2, 0]) == pytest.approx(4.0)

    def test_mean_plan_requires_time_intervals(self):
        with pytest.raises(ValueError, match="requires time_intervals"):
            build_simple_accumulator_plan(
                interval_manifest_indices=[0],
                n_manifest=1,
                T=3,
                operator_kind="mean",
            )
