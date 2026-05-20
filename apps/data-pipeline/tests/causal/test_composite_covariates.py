"""Tests for ``transition_inputs`` / ``input_effect`` in the composite
context builders.

Verifies:

- **Linear-case parity**: composite context with a ``DenseLinear`` field
  and covariate forcing matches the existing dense
  ``discretize_system_with_inputs_batched`` output.
- **Hill case + covariates**: per-step augmented intercept stays
  finite; the covariate contribution appears in ``cd``.
- **Default behaviour unchanged**: omitting the covariate kwargs gives
  the same result as the pre-covariate API.
- **Augmented context** propagates covariates correctly to the
  ``n_latent`` portion of the augmented intercept (accumulator slots
  unaffected by direct covariate forcing).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jla

from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.counterfactual import linear_vector_field
from nof1_causal_lab.models.ssm.discretization import (
    discretize_system_with_inputs_batched,
)
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeVectorField,
    DiagonalDecay,
    HillEdge,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.composite_kalman import (
    composite_latent_context_at_trajectory,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.composite_kalman_augmentation import (
    build_simple_accumulator_plan,
    composite_latent_context_at_trajectory_augmented,
)


class TestCompositeCovariatesLinearParity:
    """A ``DenseLinear`` composite field with covariate forcing must
    produce the same per-step ``(Ad, Qd, cd)`` as the existing dense
    ``discretize_system_with_inputs_batched``."""

    def test_matches_dense_inputs_batched(self):
        A = jnp.array([[-1.0, 0.3], [0.5, -1.5]])
        cint = jnp.array([0.1, 0.0])
        diffusion_cov = jnp.eye(2) * 0.02

        # Two known inputs (e.g., dose + season).
        input_effect = jnp.array([[1.0, 0.0], [0.0, 0.5]])  # (n_latent, n_inputs)

        T = 5
        runtime_times = jnp.linspace(1.0, 5.0, T)
        time_intervals = jnp.diff(runtime_times, prepend=runtime_times[0]).at[0].set(MIN_DT)

        # Time-varying covariate values.
        transition_inputs = jnp.array(
            [
                [1.0, 0.0],
                [1.2, 0.1],
                [0.8, 0.2],
                [0.5, 0.3],
                [0.0, 0.2],
            ]
        )

        # Dense reference.
        Ad_ref, Qd_ref, cd_ref = discretize_system_with_inputs_batched(
            A, diffusion_cov, cint, input_effect, transition_inputs, time_intervals
        )

        # Composite path.
        ctx = composite_latent_context_at_trajectory(
            vector_field=linear_vector_field(n_latent=2),
            vf_params=({"drift": A, "cint": cint},),
            x_traj=jnp.zeros((T, 2)),  # irrelevant for DenseLinear
            init_mean=jnp.zeros(2),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=diffusion_cov,
            runtime_times=runtime_times,
            H=jnp.eye(2),
            d_meas=jnp.zeros(2),
            R=jnp.eye(2) * 0.05,
            transition_inputs=transition_inputs,
            input_effect=input_effect,
        )

        cd_ref_2d = cd_ref.squeeze(-1) if cd_ref.ndim == 3 else cd_ref
        assert jnp.allclose(ctx.Ad, Ad_ref, atol=1e-5)
        assert jnp.allclose(ctx.Qd, Qd_ref, atol=1e-5)
        assert jnp.allclose(ctx.cd, cd_ref_2d, atol=1e-5)

    def test_no_covariate_kwargs_unchanged(self):
        """Default API (no covariates) must produce identical output to
        passing zero forcing — backward compat sanity."""
        A = jnp.array([[-1.0]])
        cint = jnp.array([0.2])
        diffusion_cov = jnp.eye(1) * 0.01
        T = 4
        runtime_times = jnp.linspace(1.0, 4.0, T)

        vf = linear_vector_field(n_latent=1)
        params = ({"drift": A, "cint": cint},)

        ctx_no = composite_latent_context_at_trajectory(
            vector_field=vf,
            vf_params=params,
            x_traj=jnp.zeros((T, 1)),
            init_mean=jnp.zeros(1),
            init_cov=jnp.eye(1) * 0.1,
            diffusion_cov=diffusion_cov,
            runtime_times=runtime_times,
            H=jnp.eye(1),
            d_meas=jnp.zeros(1),
            R=jnp.eye(1) * 0.05,
        )
        ctx_zero = composite_latent_context_at_trajectory(
            vector_field=vf,
            vf_params=params,
            x_traj=jnp.zeros((T, 1)),
            init_mean=jnp.zeros(1),
            init_cov=jnp.eye(1) * 0.1,
            diffusion_cov=diffusion_cov,
            runtime_times=runtime_times,
            H=jnp.eye(1),
            d_meas=jnp.zeros(1),
            R=jnp.eye(1) * 0.05,
            transition_inputs=jnp.zeros((T, 1)),
            input_effect=jnp.zeros((1, 1)),
        )
        assert jnp.allclose(ctx_no.Ad, ctx_zero.Ad, atol=1e-7)
        assert jnp.allclose(ctx_no.cd, ctx_zero.cd, atol=1e-7)


class TestCompositeCovariatesHill:
    """Hill vector field + covariate forcing — finite outputs, per-step
    Ad varies (Hill jacobian), and cd reflects covariate contribution."""

    def test_hill_with_covariates_finite(self):
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
        T = 4
        runtime_times = jnp.linspace(0.5, 2.0, T)
        x_traj = jnp.array([[1.0, 0.3], [1.1, 0.5], [1.2, 0.6], [1.2, 0.7]])
        transition_inputs = jnp.array([[1.0], [0.8], [0.5], [0.3]])
        input_effect = jnp.array([[1.0], [0.0]])  # input drives latent 0 only

        ctx = composite_latent_context_at_trajectory(
            vector_field=vf,
            vf_params=vf_params,
            x_traj=x_traj,
            init_mean=jnp.array([1.0, 0.0]),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.02,
            runtime_times=runtime_times,
            H=jnp.array([[0.0, 1.0]]),
            d_meas=jnp.array([0.0]),
            R=jnp.array([[0.05]]),
            transition_inputs=transition_inputs,
            input_effect=input_effect,
        )
        assert ctx.Ad.shape == (T, 2, 2)
        assert bool(jnp.all(jnp.isfinite(ctx.Ad)))
        assert bool(jnp.all(jnp.isfinite(ctx.Qd)))
        assert bool(jnp.all(jnp.isfinite(ctx.cd)))
        # Per-step Ad differs across the trajectory (Hill jacobian varies).
        assert not jnp.allclose(ctx.Ad[0], ctx.Ad[-1], atol=1e-3)

    def test_covariates_change_intercept(self):
        """Covariate forcing must shift cd in the direction of
        input_effect; with input forcing turned off, cd should match
        the no-covariate path."""
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
        T = 3
        runtime_times = jnp.linspace(1.0, 3.0, T)
        x_traj = jnp.tile(jnp.array([1.0, 0.5]), (T, 1))
        input_effect = jnp.array([[2.0], [0.0]])  # large input on latent 0

        ctx_no_input = composite_latent_context_at_trajectory(
            vector_field=vf,
            vf_params=vf_params,
            x_traj=x_traj,
            init_mean=jnp.array([1.0, 0.0]),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.02,
            runtime_times=runtime_times,
            H=jnp.array([[0.0, 1.0]]),
            d_meas=jnp.array([0.0]),
            R=jnp.array([[0.05]]),
        )
        ctx_with_input = composite_latent_context_at_trajectory(
            vector_field=vf,
            vf_params=vf_params,
            x_traj=x_traj,
            init_mean=jnp.array([1.0, 0.0]),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.02,
            runtime_times=runtime_times,
            H=jnp.array([[0.0, 1.0]]),
            d_meas=jnp.array([0.0]),
            R=jnp.array([[0.05]]),
            transition_inputs=jnp.ones((T, 1)),
            input_effect=input_effect,
        )
        # cd should differ — covariates contribute to the intercept.
        assert not jnp.allclose(ctx_no_input.cd, ctx_with_input.cd, atol=1e-4)
        # Ad/Qd should be identical — the covariate is constant forcing,
        # not affecting the linearization of the vector field's drift.
        assert jnp.allclose(ctx_no_input.Ad, ctx_with_input.Ad, atol=1e-6)
        assert jnp.allclose(ctx_no_input.Qd, ctx_with_input.Qd, atol=1e-6)


class TestAugmentedCovariates:
    """Covariate forcing also flows through the augmented (interval-
    summary) context."""

    def test_augmented_with_covariates_finite(self):
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
        T = 4
        runtime_times = jnp.linspace(1.0, 4.0, T)

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
            x_traj=jnp.tile(jnp.array([1.0, 0.5]), (T, 1)),
            init_mean=jnp.array([1.0, 0.0]),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.02,
            runtime_times=runtime_times,
            H=jnp.array([[1.0, 0.0]]),
            d_meas=jnp.array([0.0]),
            R=jnp.array([[0.05]]),
            plan=plan,
            support_kind_codes=support_kind_codes,
            transition_inputs=jnp.ones((T, 1)) * 0.5,
            input_effect=jnp.array([[1.0], [0.0]]),
        )
        # Augmented dim = 2 + 1.
        assert ctx.Ad.shape == (T, 3, 3)
        assert bool(jnp.all(jnp.isfinite(ctx.Ad)))
        assert bool(jnp.all(jnp.isfinite(ctx.cd)))

    def test_augmented_covariates_match_linear_dense_with_inputs(self):
        """For DenseLinear + augmentation + covariates, the augmented
        composite path must produce the same Ad/Qd/cd as augmenting
        the dense system manually (Van Loan over the same block-
        triangular augmented drift with forcing added to b_local)."""
        # Single latent, single interval-summary manifest with input forcing.
        A = jnp.array([[-0.5]])
        cint = jnp.array([0.1])
        diffusion_cov = jnp.eye(1) * 0.01
        H = jnp.array([[1.0]])
        d_meas = jnp.array([0.0])
        R = jnp.array([[0.05]])

        T = 3
        runtime_times = jnp.linspace(1.0, 3.0, T)
        time_intervals = jnp.diff(runtime_times, prepend=runtime_times[0]).at[0].set(MIN_DT)

        input_effect = jnp.array([[0.7]])
        transition_inputs = jnp.array([[1.0], [0.5], [0.2]])

        plan = build_simple_accumulator_plan(
            interval_manifest_indices=[0],
            n_manifest=1,
            T=T,
            operator_kind="sum",
            dtype=jnp.float32,
        )

        ctx = composite_latent_context_at_trajectory_augmented(
            vector_field=linear_vector_field(n_latent=1),
            vf_params=({"drift": A, "cint": cint},),
            x_traj=jnp.zeros((T, 1)),
            init_mean=jnp.zeros(1),
            init_cov=jnp.eye(1) * 0.1,
            diffusion_cov=diffusion_cov,
            runtime_times=runtime_times,
            H=H,
            d_meas=d_meas,
            R=R,
            plan=plan,
            support_kind_codes=jnp.array([1], dtype=jnp.int64),
            transition_inputs=transition_inputs,
            input_effect=input_effect,
        )

        # Hand-compute reference per step.
        for t in range(T):
            forcing_t = float(transition_inputs[t, 0] * input_effect[0, 0])
            b_t = float(cint[0]) + forcing_t
            dt_t = float(time_intervals[t])
            drift_aug = jnp.array([[float(A[0, 0]), 0.0], [float(H[0, 0]), 0.0]])
            cint_aug = jnp.array([b_t, float(d_meas[0])])
            Ad_expected = jla.expm(drift_aug * dt_t)
            # Discrete intercept via solve of A·c_d = (exp(A·dt) − I)·c — but
            # A is singular here so use Van Loan.
            n_aug = 2
            big = jnp.zeros((n_aug + 1, n_aug + 1))
            big = big.at[:n_aug, :n_aug].set(drift_aug)
            big = big.at[:n_aug, n_aug].set(cint_aug)
            expBig = jla.expm(big * dt_t)
            cd_expected = expBig[:n_aug, n_aug]

            # For t >= 1 the accumulator column of Ad is reset to zero.
            if t >= 1:
                Ad_expected = Ad_expected.at[:, 1].set(0.0)
            assert jnp.allclose(ctx.Ad[t], Ad_expected, atol=1e-5), (
                f"Ad mismatch at step {t}"
            )
            assert jnp.allclose(ctx.cd[t], cd_expected, atol=1e-5), (
                f"cd mismatch at step {t}"
            )
