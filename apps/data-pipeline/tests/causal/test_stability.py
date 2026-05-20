"""Tests for ``check_jacobian_stability`` on composite vector fields."""

from __future__ import annotations

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.counterfactual import linear_vector_field
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeVectorField,
    DiagonalDecay,
    HillEdge,
    Intercept,
    LinearEdge,
    MultiplicativeEdge,
    check_jacobian_stability,
)


class TestStabilityLinear:
    def test_stable_linear_system(self):
        """Dense linear with all-negative-diagonal A: stable everywhere."""
        A = jnp.array([[-1.0, 0.2], [0.3, -0.7]])
        c = jnp.array([0.0, 0.0])
        vf = linear_vector_field(n_latent=2)
        report = check_jacobian_stability(
            vf,
            ({"drift": A, "cint": c},),
            x_lin=jnp.array([0.5, -0.3]),
        )
        assert report.is_stable
        assert report.max_real_part < 0.0
        assert report.eigenvalues.shape == (2,)

    def test_unstable_linear_system(self):
        """Diagonal A with positive entry: unstable."""
        A = jnp.array([[+0.3, 0.0], [0.0, -1.0]])
        c = jnp.array([0.0, 0.0])
        vf = linear_vector_field(n_latent=2)
        report = check_jacobian_stability(
            vf,
            ({"drift": A, "cint": c},),
            x_lin=jnp.zeros(2),
        )
        assert not report.is_stable
        assert report.max_real_part > 0.0

    def test_threshold_kwarg(self):
        """Threshold lets the caller require a margin."""
        A = jnp.array([[-0.01]])  # barely stable
        c = jnp.array([0.0])
        vf = linear_vector_field(n_latent=1)
        # Default threshold (0.0): stable
        report = check_jacobian_stability(
            vf,
            ({"drift": A, "cint": c},),
            x_lin=jnp.zeros(1),
        )
        assert report.is_stable
        # Tighter threshold (-0.1): no longer stable enough
        report_tight = check_jacobian_stability(
            vf,
            ({"drift": A, "cint": c},),
            x_lin=jnp.zeros(1),
            threshold=-0.1,
        )
        assert not report_tight.is_stable


class TestStabilityNonLinear:
    def test_hill_chain_stable_at_baseline(self):
        """Hill chain with positive decays: stable Jacobian at baseline."""
        vf = CompositeVectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                Intercept(),
                HillEdge(source=0, target=1),
            ),
        )
        params = (
            {"decay": jnp.array([0.5, 0.5])},
            {"cint": jnp.array([0.0, 0.0])},
            {
                "Emax": jnp.asarray(1.5),
                "EC50": jnp.asarray(1.0),
                "n": jnp.asarray(2.0),
            },
        )
        report = check_jacobian_stability(vf, params, x_lin=jnp.array([1.0, 0.5]))
        # Hill contributes ∂(Emax · x^n / (EC50^n + x^n)) / ∂x at the target's
        # row; with positive decay on both latents the diagonal still dominates.
        assert report.is_stable
        assert report.max_real_part < 0.0

    def test_multiplicative_chain_stable_at_typical_state(self):
        vf = CompositeVectorField(
            n_latent=3,
            components=(
                DiagonalDecay(),
                MultiplicativeEdge(source_a=0, source_b=1, target=2),
                LinearEdge(source=2, target=0),
            ),
        )
        params = (
            {"decay": jnp.array([1.0, 1.0, 1.0])},
            {"weight": jnp.asarray(0.1)},
            {"weight": jnp.asarray(0.05)},
        )
        report = check_jacobian_stability(
            vf, params, x_lin=jnp.array([1.0, 1.0, 0.5])
        )
        assert report.is_stable

    def test_no_decay_unstable_with_self_feedback(self):
        """Without decay on a latent with positive self-feedback (via
        Hill in this case), the Jacobian has a positive eigenvalue."""
        # A latent with zero decay + a strong Hill self-feedback (target=source=0)
        # makes its Jacobian row positive at the Hill curve's steep region.
        vf = CompositeVectorField(
            n_latent=1,
            components=(HillEdge(source=0, target=0),),
        )
        params = (
            {
                "Emax": jnp.asarray(10.0),
                "EC50": jnp.asarray(0.5),
                "n": jnp.asarray(4.0),
            },
        )
        report = check_jacobian_stability(
            vf, params, x_lin=jnp.array([0.5])
        )
        # At x=EC50 with steep Hill slope, no decay → positive feedback → unstable.
        assert not report.is_stable
        assert report.max_real_part > 0.0


class TestStabilityReportFields:
    def test_eigenvalues_match_jacobian_directly(self):
        A = jnp.array([[-1.5, 0.3], [0.2, -0.8]])
        c = jnp.array([0.0, 0.0])
        vf = linear_vector_field(n_latent=2)
        report = check_jacobian_stability(
            vf, ({"drift": A, "cint": c},), x_lin=jnp.zeros(2)
        )
        # For a DenseLinear field, the Jacobian is exactly A.
        expected_eigs = jnp.linalg.eigvals(A)
        assert jnp.allclose(
            jnp.sort(jnp.real(report.eigenvalues)),
            jnp.sort(jnp.real(expected_eigs)),
            atol=1e-6,
        )

    def test_linearization_point_preserved(self):
        vf = linear_vector_field(n_latent=2)
        x_lin = jnp.array([0.7, -1.3])
        report = check_jacobian_stability(
            vf, ({"drift": -jnp.eye(2), "cint": jnp.zeros(2)},), x_lin=x_lin
        )
        assert jnp.allclose(report.linearization_point, x_lin)
