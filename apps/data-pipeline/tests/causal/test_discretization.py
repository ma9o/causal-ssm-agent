"""Tests for ``discretize_at_state`` and ``CompositeVectorField.linearize``.

The discretization machinery here is the bridge from the new
``CompositeVectorField`` to the existing CT→DT expm path used by Stage
5b's filter. Three layers of checking:

1. ``linearize`` matches analytic Jacobians for each primitive in
   isolation (LinearEdge, HillEdge, MultiplicativeEdge, DenseLinear).
2. For a pure ``DenseLinear`` field, ``discretize_at_state`` reproduces
   the existing ``discretize_linear_system_exact`` output exactly — no
   precision lost in the new code path.
3. SSRI chain at baseline steady state: the Jacobian structure has the
   right sparsity and signs, and discretizing over a small ``dt``
   matches a Diffrax integration of the linearised system.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jla
import pytest

from nof1_causal_lab.models.ssm.counterfactual import linear_vector_field
from nof1_causal_lab.models.ssm.discretization import (
    discretize_at_state,
    discretize_linear_system_exact,
)
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeVectorField,
    DenseLinear,
    DiagonalDecay,
    HillEdge,
    Intercept,
    Intervention,
    LinearEdge,
    MultiplicativeEdge,
    VectorFieldArgs,
    simulate,
)

# =============================================================================
# Per-primitive linearization checks
# =============================================================================


class TestLinearizePrimitives:
    def test_dense_linear_recovers_matrix(self):
        """``linearize`` on a single ``DenseLinear`` component must return
        ``(A, c)`` exactly — autodiff through ``A @ x + c`` reproduces ``A``."""
        A = jnp.array([[-1.0, 0.5], [0.3, -2.0]])
        c = jnp.array([0.1, -0.2])
        vf = linear_vector_field(n_latent=2)
        args = VectorFieldArgs(
            params=({"drift": A, "cint": c},), intervention=Intervention.none()
        )
        x_lin = jnp.array([0.7, -0.4])
        A_loc, b_loc = vf.linearize(x_lin, args)
        assert jnp.allclose(A_loc, A, atol=1e-6)
        assert jnp.allclose(b_loc, c, atol=1e-6)

    def test_hill_jacobian_matches_analytic(self):
        """At ``x = EC50``, ``dHill/dx = Emax · n / (4 · EC50)``. The
        Jacobian entry for the source must match this."""
        Emax, EC50, n = 2.0, 1.0, 2.0
        vf = CompositeVectorField(
            n_latent=2,
            components=(HillEdge(source=0, target=1),),
        )
        params = (
            {
                "Emax": jnp.asarray(Emax),
                "EC50": jnp.asarray(EC50),
                "n": jnp.asarray(n),
            },
        )
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        x_lin = jnp.array([EC50, 0.0])  # source at EC50
        A_loc, _ = vf.linearize(x_lin, args)
        expected_slope = Emax * n / (4.0 * EC50)
        assert float(A_loc[1, 0]) == pytest.approx(expected_slope, abs=1e-4)
        # Other entries: target's effect on itself is 0 (no decay/feedback);
        # source has no self-influence either.
        assert float(A_loc[0, 0]) == pytest.approx(0.0, abs=1e-6)
        assert float(A_loc[1, 1]) == pytest.approx(0.0, abs=1e-6)

    def test_multiplicative_jacobian_off_diagonals(self):
        """For ``f(η) = w · η_a · η_b`` at ``(a₀, b₀)``: ``∂f/∂η_a = w · b₀``,
        ``∂f/∂η_b = w · a₀``."""
        w = 0.5
        a0, b0 = 3.0, 4.0
        vf = CompositeVectorField(
            n_latent=3,
            components=(MultiplicativeEdge(source_a=0, source_b=1, target=2),),
        )
        params = ({"weight": jnp.asarray(w)},)
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        x_lin = jnp.array([a0, b0, 0.0])
        A_loc, b_loc = vf.linearize(x_lin, args)
        assert float(A_loc[2, 0]) == pytest.approx(w * b0, abs=1e-6)
        assert float(A_loc[2, 1]) == pytest.approx(w * a0, abs=1e-6)
        # Intercept: f(x_lin) - A · x_lin = w·a·b - (w·b·a + w·a·b) = -w·a·b
        f_at_x = w * a0 * b0
        expected_b = jnp.array([0.0, 0.0, f_at_x - (w * b0 * a0 + w * a0 * b0)])
        assert jnp.allclose(b_loc, expected_b, atol=1e-6)


# =============================================================================
# discretize_at_state parity for DenseLinear case
# =============================================================================


class TestDiscretizeDenseLinearParity:
    """For a single ``DenseLinear`` component, ``discretize_at_state``
    must produce the same matrices as the existing
    ``discretize_linear_system_exact``, regardless of ``x_lin``."""

    def test_matches_linear_path_exactly(self):
        A = jnp.array(
            [
                [-1.5, 0.3, 0.1],
                [0.4, -2.0, 0.2],
                [0.1, 0.5, -1.8],
            ]
        )
        c = jnp.array([1.0, -0.5, 0.2])
        diffusion_cov = jnp.eye(3) * 0.1
        dt = 0.5

        A_d_ref, Q_d_ref, c_d_ref = discretize_linear_system_exact(A, diffusion_cov, c, dt)

        vf = linear_vector_field(n_latent=3)
        args = VectorFieldArgs(
            params=({"drift": A, "cint": c},), intervention=Intervention.none()
        )
        # x_lin is irrelevant for a linear field — pick something non-trivial
        x_lin = jnp.array([1.7, -0.3, 0.5])
        A_d, Q_d, c_d = discretize_at_state(vf, x_lin, args, diffusion_cov, dt)

        assert jnp.allclose(A_d, A_d_ref, atol=1e-6)
        assert jnp.allclose(Q_d, Q_d_ref, atol=1e-6)
        assert jnp.allclose(c_d, c_d_ref, atol=1e-6)

    def test_zero_intercept(self):
        A = -jnp.eye(2)
        diffusion_cov = jnp.eye(2) * 0.05
        vf = linear_vector_field(n_latent=2)
        args = VectorFieldArgs(
            params=({"drift": A, "cint": jnp.zeros(2)},),
            intervention=Intervention.none(),
        )
        x_lin = jnp.array([0.0, 0.0])
        A_d, _, c_d = discretize_at_state(vf, x_lin, args, diffusion_cov, dt=0.2)
        assert jnp.allclose(A_d, jla.expm(A * 0.2), atol=1e-6)
        assert jnp.allclose(c_d, 0.0, atol=1e-6)


# =============================================================================
# Non-linear SSRI chain: linearization structure + Diffrax cross-check
# =============================================================================


class TestSSRIChainLinearization:
    """Build the full SSRI chain, compute the Jacobian at baseline steady
    state, and verify the structure matches expectations."""

    DOSE = 0
    ADHERENCE = 1
    C_P = 2
    C_E = 3
    AFFECTIVE = 4

    K_P = 1.0
    K_E0 = 0.1
    DECAY_AFF = 1.0
    EMAX = 2.0
    EC50_VAL = 1.0
    N_HILL = 2.0

    def _build(self):
        vf = CompositeVectorField(
            n_latent=5,
            components=(
                DiagonalDecay(),
                Intercept(),
                MultiplicativeEdge(
                    source_a=self.DOSE, source_b=self.ADHERENCE, target=self.C_P
                ),
                LinearEdge(source=self.C_P, target=self.C_E),
                HillEdge(source=self.C_E, target=self.AFFECTIVE),
            ),
        )
        params = (
            {"decay": jnp.array([1.0, 1.0, self.K_P, self.K_E0, self.DECAY_AFF])},
            {"cint": jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])},
            {"weight": jnp.asarray(self.K_P)},
            {"weight": jnp.asarray(self.K_E0)},
            {
                "Emax": jnp.asarray(self.EMAX),
                "EC50": jnp.asarray(self.EC50_VAL),
                "n": jnp.asarray(self.N_HILL),
            },
        )
        return vf, params

    def test_jacobian_at_baseline_has_expected_structure(self):
        vf, params = self._build()
        baseline = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        A_loc, b_loc = vf.linearize(baseline, args)

        # Diagonal: each latent's natural decay
        assert float(A_loc[self.DOSE, self.DOSE]) == pytest.approx(-1.0, abs=1e-6)
        assert float(A_loc[self.ADHERENCE, self.ADHERENCE]) == pytest.approx(-1.0, abs=1e-6)
        assert float(A_loc[self.C_P, self.C_P]) == pytest.approx(-self.K_P, abs=1e-6)
        assert float(A_loc[self.C_E, self.C_E]) == pytest.approx(-self.K_E0, abs=1e-6)
        assert float(A_loc[self.AFFECTIVE, self.AFFECTIVE]) == pytest.approx(
            -self.DECAY_AFF, abs=1e-6
        )

        # Multiplicative edge at baseline (dose=1, adherence=1):
        #   ∂(k_p · dose · adherence)/∂dose      = k_p · 1 = k_p
        #   ∂(k_p · dose · adherence)/∂adherence = k_p · 1 = k_p
        assert float(A_loc[self.C_P, self.DOSE]) == pytest.approx(self.K_P, abs=1e-6)
        assert float(A_loc[self.C_P, self.ADHERENCE]) == pytest.approx(self.K_P, abs=1e-6)

        # LinearEdge C_P → C_E with weight k_e0
        assert float(A_loc[self.C_E, self.C_P]) == pytest.approx(self.K_E0, abs=1e-6)

        # HillEdge C_E → AFFECTIVE at C_E=1=EC50: slope = Emax·n/(4·EC50)
        expected_hill_slope = self.EMAX * self.N_HILL / (4.0 * self.EC50_VAL)
        assert float(A_loc[self.AFFECTIVE, self.C_E]) == pytest.approx(
            expected_hill_slope, abs=1e-4
        )

        # b_loc: drift at baseline minus A · baseline. At baseline the
        # natural drift is zero (steady state), so b_loc = -A · baseline.
        assert jnp.allclose(b_loc, -A_loc @ baseline, atol=1e-5)

    def test_discrete_step_matches_diffrax_for_linearized_system(self):
        """Discretizing at ``x_lin`` over a small ``dt`` and applying once
        must match a Diffrax integration of the linearized vector field
        from the same point. The linearization is the same; this is a
        sanity check that the expm discretization and the Diffrax
        integrator agree on the linear case."""
        vf, params = self._build()
        x_lin = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        diffusion_cov = jnp.zeros((5, 5))  # deterministic for the check
        dt = 0.1

        A_d, _, b_d = discretize_at_state(vf, x_lin, args, diffusion_cov, dt)
        # Step from the linearization point itself
        next_state_discrete = A_d @ x_lin + b_d

        # Linearized system as a fresh DenseLinear vector field
        A_loc, b_loc = vf.linearize(x_lin, args)
        lin_vf = CompositeVectorField(n_latent=5, components=(DenseLinear(),))
        lin_args = VectorFieldArgs(
            params=({"drift": A_loc, "cint": b_loc},),
            intervention=Intervention.none(),
        )
        time_grid = jnp.array([0.0, dt])
        traj = simulate(
            lin_vf,
            lin_args.params,
            Intervention.none(),
            x_lin,
            time_grid,
        )
        next_state_diffrax = traj[-1]

        assert jnp.allclose(next_state_discrete, next_state_diffrax, atol=1e-4)


# =============================================================================
# Cuthbert moments-filter callback: end-to-end run
# =============================================================================


def _run_moments_filter_via_callback(
    vf, vf_params, diffusion_cov, init_mean, init_cov, H, R, ys, dts, jitter=1e-6
):
    """Run cuthbert.gaussian.moments.build_filter via make_filter_dynamics_callback.

    Returns the cumulative log-normalizing-constant at the final step
    (i.e., marginal log-likelihood of the observations under the model).
    """
    from cuthbert.filtering import filter as cuthbert_filter
    from cuthbert.gaussian.moments import build_filter

    from nof1_causal_lab.models.ssm.discretization import make_filter_dynamics_callback

    T = ys.shape[0]
    n = init_mean.shape[0]
    n_m = ys.shape[-1]
    dtype = init_mean.dtype

    chol_P0 = jnp.linalg.cholesky(init_cov + jitter * jnp.eye(n, dtype=dtype))
    chol_R = jnp.linalg.cholesky(R + jitter * jnp.eye(n_m, dtype=dtype))

    def _prepend_init(steps):
        head = jnp.zeros((1, *steps.shape[1:]), dtype=steps.dtype)
        return jnp.concatenate([head, steps], axis=0)

    model_inputs = {
        "m0": jnp.broadcast_to(init_mean, (T + 1, n)),
        "chol_P0": jnp.broadcast_to(chol_P0, (T + 1, n, n)),
        "dt": _prepend_init(jnp.asarray(dts, dtype=dtype)[:, None]).squeeze(-1),
        "H": _prepend_init(jnp.broadcast_to(H, (T, n_m, n))),
        "d": _prepend_init(jnp.zeros((T, n_m), dtype=dtype)),
        "chol_R": _prepend_init(jnp.broadcast_to(chol_R, (T, n_m, n_m))),
        "y": _prepend_init(jnp.asarray(ys, dtype=dtype)),
    }

    def get_init_params(model_inputs):
        return model_inputs["m0"], model_inputs["chol_P0"]

    get_dynamics_params = make_filter_dynamics_callback(
        vf, vf_params, diffusion_cov=diffusion_cov, jitter=jitter
    )

    def get_observation_params(state, model_inputs):
        H_t = model_inputs["H"]
        d_t = model_inputs["d"]
        chol_R_t = model_inputs["chol_R"]
        y_t = model_inputs["y"]

        def obs_fn(x):
            return H_t @ x + d_t, chol_R_t

        return obs_fn, state.mean, y_t

    filter_obj = build_filter(
        get_init_params=get_init_params,
        get_dynamics_params=get_dynamics_params,
        get_observation_params=get_observation_params,
        associative=False,
    )
    filter_states = cuthbert_filter(filter_obj, model_inputs)
    return float(filter_states.log_normalizing_constant[-1])


class TestCuthbertCallbackIntegration:
    """End-to-end: run cuthbert's moments-based filter with our callback
    on a linear system (where the linearization is exact) and a
    non-linear system (where it's the EKF approximation). Both must
    return finite log-likelihoods, and the linear case must match a
    hand-computed reference within numerical tolerance."""

    def _generate_linear_obs(self, key, A, c, GG_cov, init_mean, H, R, dts):
        """Forward-simulate a linear-Gaussian SSM and return noisy obs."""
        import jax.random as jr

        from nof1_causal_lab.models.ssm.dynamics import (
            Intervention,
            simulate,
        )

        vf = linear_vector_field(n_latent=A.shape[0])
        params = ({"drift": A, "cint": c},)
        time_grid = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dts)])
        traj = simulate(vf, params, Intervention.none(), init_mean, time_grid)
        signal = traj[1:] @ H.T
        noise = jr.normal(key, signal.shape, dtype=signal.dtype) * jnp.sqrt(R[0, 0])
        return signal + noise

    def test_filter_runs_and_returns_finite_loglik_linear(self):
        A = jnp.array([[-1.0, 0.0], [0.5, -1.5]])
        c = jnp.zeros(2)
        GG = jnp.eye(2) * 0.05
        init_mean = jnp.array([0.0, 0.0])
        init_cov = jnp.eye(2) * 0.5
        H = jnp.array([[1.0, 0.0]])
        R = jnp.array([[0.1]])
        T = 8
        dts = jnp.full(T, 0.5)

        import jax.random as jr

        ys = self._generate_linear_obs(jr.PRNGKey(0), A, c, GG, init_mean, H, R, dts)

        ll = _run_moments_filter_via_callback(
            linear_vector_field(n_latent=2),
            ({"drift": A, "cint": c},),
            GG,
            init_mean,
            init_cov,
            H,
            R,
            ys,
            dts,
        )
        assert jnp.isfinite(ll), f"linear filter log-likelihood is not finite: {ll}"

    def test_filter_runs_and_returns_finite_loglik_hill(self):
        """Non-linear (HillEdge) system: filter must complete and produce
        a finite log-likelihood, proving the callback path handles state-
        dependent linearization correctly inside cuthbert's scan."""
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeVectorField,
            DiagonalDecay,
            HillEdge,
        )

        vf = CompositeVectorField(
            n_latent=2,
            components=(DiagonalDecay(), HillEdge(source=0, target=1)),
        )
        vf_params = (
            {"decay": jnp.array([0.5, 0.5])},
            {
                "Emax": jnp.asarray(2.0),
                "EC50": jnp.asarray(1.0),
                "n": jnp.asarray(2.0),
            },
        )
        GG = jnp.eye(2) * 0.05
        init_mean = jnp.array([1.0, 0.5])
        init_cov = jnp.eye(2) * 0.5
        H = jnp.array([[0.0, 1.0]])  # observe the Hill target
        R = jnp.array([[0.05]])
        T = 6
        dts = jnp.full(T, 0.5)

        import jax.random as jr

        ys = (
            jnp.array([0.6, 0.55, 0.5, 0.45, 0.42, 0.4])[:, None]
            + jr.normal(jr.PRNGKey(1), (T, 1)) * 0.05
        )

        ll = _run_moments_filter_via_callback(
            vf, vf_params, GG, init_mean, init_cov, H, R, ys, dts
        )
        assert jnp.isfinite(ll), f"Hill filter log-likelihood is not finite: {ll}"

    def test_filter_matches_dense_path_for_linear(self):
        """For a linear DenseLinear system the callback's per-step
        linearization recovers A exactly; the marginal log-likelihood
        must match a reference filter built directly from the
        pre-discretized matrices."""
        import jax.random as jr
        from cuthbert.filtering import filter as cuthbert_filter
        from cuthbert.gaussian.moments import build_filter

        A = jnp.array([[-1.0, 0.0], [0.5, -1.5]])
        c = jnp.zeros(2)
        GG = jnp.eye(2) * 0.05
        init_mean = jnp.array([0.0, 0.0])
        init_cov = jnp.eye(2) * 0.5
        H = jnp.array([[1.0, 0.0]])
        R = jnp.array([[0.1]])
        T = 6
        dts = jnp.full(T, 0.4)
        ys = self._generate_linear_obs(jr.PRNGKey(2), A, c, GG, init_mean, H, R, dts)

        # Reference: build cuthbert filter directly with pre-discretized matrices
        from nof1_causal_lab.models.ssm.discretization import (
            discretize_linear_system_exact,
        )

        jitter = 1e-6
        n = 2
        chol_P0 = jnp.linalg.cholesky(init_cov + jitter * jnp.eye(n))
        chol_R = jnp.linalg.cholesky(R + jitter * jnp.eye(1))

        # Discretize once (time-invariant)
        A_d, Q_d, b_d = discretize_linear_system_exact(A, GG, c, dts[0])
        chol_Q = jnp.linalg.cholesky(Q_d + jitter * jnp.eye(n))

        def _prepend_init(steps):
            head = jnp.zeros((1, *steps.shape[1:]), dtype=steps.dtype)
            return jnp.concatenate([head, steps], axis=0)

        ref_inputs = {
            "m0": jnp.broadcast_to(init_mean, (T + 1, n)),
            "chol_P0": jnp.broadcast_to(chol_P0, (T + 1, n, n)),
            "F": _prepend_init(jnp.broadcast_to(A_d, (T, n, n))),
            "c": _prepend_init(jnp.broadcast_to(b_d, (T, n))),
            "chol_Q": _prepend_init(jnp.broadcast_to(chol_Q, (T, n, n))),
            "H": _prepend_init(jnp.broadcast_to(H, (T, 1, n))),
            "d": _prepend_init(jnp.zeros((T, 1))),
            "chol_R": _prepend_init(jnp.broadcast_to(chol_R, (T, 1, 1))),
            "y": _prepend_init(ys),
        }

        def ref_init(mi):
            return mi["m0"], mi["chol_P0"]

        def ref_dynamics(state, mi):
            F_t, c_t, chol_Q_t = mi["F"], mi["c"], mi["chol_Q"]

            def dynamics_fn(x):
                return F_t @ x + c_t, chol_Q_t

            return dynamics_fn, state.mean

        def ref_obs(state, mi):
            H_t, d_t, chol_R_t, y_t = mi["H"], mi["d"], mi["chol_R"], mi["y"]

            def obs_fn(x):
                return H_t @ x + d_t, chol_R_t

            return obs_fn, state.mean, y_t

        ref_filter = build_filter(
            get_init_params=ref_init,
            get_dynamics_params=ref_dynamics,
            get_observation_params=ref_obs,
            associative=False,
        )
        ref_states = cuthbert_filter(ref_filter, ref_inputs)
        ref_ll = float(ref_states.log_normalizing_constant[-1])

        # New: via callback
        new_ll = _run_moments_filter_via_callback(
            linear_vector_field(n_latent=2),
            ({"drift": A, "cint": c},),
            GG,
            init_mean,
            init_cov,
            H,
            R,
            ys,
            dts,
        )

        assert jnp.isfinite(ref_ll), f"reference log-likelihood not finite: {ref_ll}"
        assert jnp.isfinite(new_ll), f"callback log-likelihood not finite: {new_ll}"
        assert new_ll == pytest.approx(ref_ll, abs=1e-4), (
            f"callback log-likelihood {new_ll} != reference {ref_ll}"
        )
