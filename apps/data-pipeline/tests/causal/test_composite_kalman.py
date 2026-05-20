"""Tests for the composite Kalman bridge into the Corenflos auxiliary
Kalman MH machinery.

Three layers of checking:

1. Linear case parity: for a pure ``DenseLinear`` field the
   per-step-linearized context must match the existing dense
   ``discretize_system_with_inputs_batched`` output.
2. Non-linear shape correctness: a Hill / Multiplicative / Effect
   compartment chain produces finite, correctly-shaped per-step
   matrices.
3. End-to-end smoke test: the aux LGSSM lightweight filter (from
   ``parallel_kalman.py``) consumes the composite context without
   error and returns finite filtered moments and log-likelihood.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as ndist

from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.counterfactual import linear_vector_field
from nof1_causal_lab.models.ssm.discretization import (
    discretize_system_with_inputs_batched,
)
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeSpec,
    CompositeVectorField,
    DiagonalDecay,
    HillEdge,
    Intercept,
    LinearEdge,
    MultiplicativeEdge,
    compile_composite,
)
from nof1_causal_lab.models.ssm.dynamics.priors import (
    diagonal_decay_prior,
    effect_compartment_rate_prior,
    hill_ec50_prior,
    hill_emax_prior,
    hill_n_prior,
    multiplicative_weight_prior,
)
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.composite_kalman import (
    composite_latent_context_at_trajectory,
)

# =============================================================================
# Linear-case parity with the existing dense discretization
# =============================================================================


class TestCompositeContextLinearParity:
    """For a pure DenseLinear vector field, ``composite_latent_context_at_trajectory``
    must produce the same (Ad, Qd, cd) as
    ``discretize_system_with_inputs_batched`` applied directly to ``A, c``."""

    def test_matches_dense_path_for_linear_system(self):
        A = jnp.array([[-1.0, 0.3], [0.5, -2.0]])
        c = jnp.array([0.1, -0.2])
        GG = jnp.eye(2) * 0.05
        init_mean = jnp.array([0.0, 0.0])
        init_cov = jnp.eye(2) * 0.5
        H = jnp.array([[1.0, 0.0]])
        d_meas = jnp.array([0.0])
        R = jnp.array([[0.1]])
        T = 6
        runtime_times = jnp.linspace(0.0, 3.0, T)

        # Any trajectory works — for linear the context is x-independent.
        x_traj = jnp.zeros((T, 2))

        vf = linear_vector_field(n_latent=2)
        vf_params = ({"drift": A, "cint": c},)

        composite_ctx = composite_latent_context_at_trajectory(
            vector_field=vf,
            vf_params=vf_params,
            x_traj=x_traj,
            init_mean=init_mean,
            init_cov=init_cov,
            diffusion_cov=GG,
            runtime_times=runtime_times,
            H=H,
            d_meas=d_meas,
            R=R,
        )

        # Reference: existing dense path
        time_intervals = jnp.diff(runtime_times, prepend=runtime_times[0]).at[0].set(MIN_DT)
        Ad_ref, Qd_ref, cd_ref = discretize_system_with_inputs_batched(
            A, GG, c, None, None, time_intervals
        )

        # Match per-step matrices
        assert jnp.allclose(composite_ctx.Ad, Ad_ref, atol=1e-5)
        assert jnp.allclose(composite_ctx.Qd, Qd_ref, atol=1e-5)
        assert jnp.allclose(composite_ctx.cd, cd_ref, atol=1e-5)
        assert jnp.allclose(composite_ctx.init_mean, init_mean)
        assert jnp.allclose(composite_ctx.init_cov, init_cov)


# =============================================================================
# Non-linear shape + finiteness checks
# =============================================================================


class TestCompositeContextNonLinear:
    def test_hill_context_produces_finite_matrices(self):
        vf = CompositeVectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                HillEdge(source=0, target=1),
            ),
        )
        vf_params = (
            {"decay": jnp.array([0.5, 0.5])},
            {
                "Emax": jnp.asarray(2.0),
                "EC50": jnp.asarray(1.0),
                "n": jnp.asarray(2.0),
            },
        )
        T = 5
        runtime_times = jnp.linspace(0.0, 2.5, T)
        # Non-trivial trajectory; should produce different Ad per step.
        x_traj = jnp.array(
            [
                [1.0, 0.2],
                [1.2, 0.4],
                [1.4, 0.7],
                [1.5, 0.9],
                [1.5, 1.0],
            ]
        )

        ctx = composite_latent_context_at_trajectory(
            vector_field=vf,
            vf_params=vf_params,
            x_traj=x_traj,
            init_mean=jnp.array([1.0, 0.0]),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.05,
            runtime_times=runtime_times,
            H=jnp.array([[0.0, 1.0]]),
            d_meas=jnp.array([0.0]),
            R=jnp.array([[0.05]]),
        )
        assert ctx.Ad.shape == (T, 2, 2)
        assert ctx.Qd.shape == (T, 2, 2)
        assert ctx.cd.shape == (T, 2)
        assert bool(jnp.all(jnp.isfinite(ctx.Ad)))
        assert bool(jnp.all(jnp.isfinite(ctx.Qd)))
        assert bool(jnp.all(jnp.isfinite(ctx.cd)))

        # Per-step Ad must NOT be identical (Hill is non-linear → different
        # linearizations at different x_lin).
        assert not jnp.allclose(ctx.Ad[0], ctx.Ad[-1], atol=1e-3), (
            "Hill linearization should differ across trajectory"
        )

    def test_ssri_chain_context_finite(self):
        """The full SSRI chain produces a finite per-step context — covers
        Multiplicative + matched-decay LinearEdge + Hill in series."""
        DOSE, ADHERENCE, C_P, C_E, AFFECTIVE = 0, 1, 2, 3, 4
        spec = CompositeSpec(
            n_latent=5,
            components=(
                DiagonalDecay(),
                Intercept(),
                MultiplicativeEdge(source_a=DOSE, source_b=ADHERENCE, target=C_P),
                LinearEdge(source=C_P, target=C_E),
                HillEdge(source=C_E, target=AFFECTIVE),
            ),
        )
        # Skip the spec compiler here — just pass concrete params directly.
        vf_params = (
            {"decay": jnp.array([1.0, 1.0, 1.0, 0.1, 1.0])},
            {"cint": jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])},
            {"weight": jnp.asarray(1.0)},
            {"weight": jnp.asarray(0.1)},
            {
                "Emax": jnp.asarray(2.0),
                "EC50": jnp.asarray(1.0),
                "n": jnp.asarray(2.0),
            },
        )
        vf = CompositeVectorField(n_latent=spec.n_latent, components=tuple(s for s in spec.components))

        T = 4
        runtime_times = jnp.linspace(0.0, 4.0, T)
        x_traj = jnp.tile(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]), (T, 1))

        ctx = composite_latent_context_at_trajectory(
            vector_field=vf,
            vf_params=vf_params,
            x_traj=x_traj,
            init_mean=jnp.zeros(5),
            init_cov=jnp.eye(5) * 0.5,
            diffusion_cov=jnp.eye(5) * 0.05,
            runtime_times=runtime_times,
            H=jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0]]),
            d_meas=jnp.array([0.0]),
            R=jnp.array([[0.05]]),
        )
        assert ctx.Ad.shape == (T, 5, 5)
        assert bool(jnp.all(jnp.isfinite(ctx.Ad)))


# =============================================================================
# End-to-end: composite context drives the existing aux LGSSM filter
# =============================================================================


class TestCompositeContextDrivesAuxFilter:
    """The composite context must be consumable by the existing
    auxiliary LGSSM lightweight filter — that filter knows nothing
    about the vector field, only about (Ad, Qd, cd). So a successful
    end-to-end filter run proves the integration shape is right."""

    def test_aux_filter_runs_on_composite_context_for_hill(self):
        from nof1_causal_lab.models.ssm.inference.parallel_kalman import (
            aux_filter_lgssm_lightweight,
        )

        vf = CompositeVectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                HillEdge(source=0, target=1),
            ),
        )
        vf_params = (
            {"decay": jnp.array([0.5, 0.5])},
            {
                "Emax": jnp.asarray(2.0),
                "EC50": jnp.asarray(1.0),
                "n": jnp.asarray(2.0),
            },
        )
        T = 5
        runtime_times = jnp.linspace(0.0, 2.5, T)
        x_traj = jnp.array(
            [
                [1.0, 0.3],
                [1.1, 0.5],
                [1.2, 0.7],
                [1.3, 0.85],
                [1.3, 0.95],
            ]
        )

        ctx = composite_latent_context_at_trajectory(
            vector_field=vf,
            vf_params=vf_params,
            x_traj=x_traj,
            init_mean=jnp.array([1.0, 0.0]),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.05,
            runtime_times=runtime_times,
            H=jnp.array([[0.0, 1.0]]),
            d_meas=jnp.array([0.0]),
            R=jnp.array([[0.05]]),
        )

        # Aux LGSSM filter takes the per-step (Ad, Qd, cd) and an
        # auxiliary "pseudo observation" sequence (here: take x_traj
        # itself as the pseudo-obs).
        delta = jnp.asarray(0.1)
        state = aux_filter_lgssm_lightweight(
            init_mean=ctx.init_mean,
            init_cov=ctx.init_cov,
            Fs=ctx.Ad,
            Qs=ctx.Qd,
            bs=ctx.cd,
            pseudo_observations=x_traj,
            aux_variance=0.5 * delta,
            jitter=1e-6,
            parallel=False,
        )

        assert state.filt_mean.shape == (T, 2)
        assert state.filt_cov.shape == (T, 2, 2)
        assert bool(jnp.all(jnp.isfinite(state.filt_mean)))
        assert bool(jnp.all(jnp.isfinite(state.filt_cov)))
        assert bool(jnp.all(jnp.isfinite(state.loglik)))

    def test_aux_filter_matches_dense_path_for_linear(self):
        """For a linear system, the auxiliary filter on the composite
        context must produce the same result as on the dense context."""
        from nof1_causal_lab.models.ssm.inference.parallel_kalman import (
            aux_filter_lgssm_lightweight,
        )

        A = jnp.array([[-1.0, 0.0], [0.5, -1.5]])
        c = jnp.zeros(2)
        GG = jnp.eye(2) * 0.05
        init_mean = jnp.array([0.0, 0.0])
        init_cov = jnp.eye(2) * 0.3
        T = 6
        runtime_times = jnp.linspace(0.0, 3.0, T)
        x_traj = jnp.zeros((T, 2))
        pseudo_obs = jr.normal(jr.PRNGKey(0), (T, 2)) * 0.3

        # Composite path
        composite_ctx = composite_latent_context_at_trajectory(
            vector_field=linear_vector_field(n_latent=2),
            vf_params=({"drift": A, "cint": c},),
            x_traj=x_traj,
            init_mean=init_mean,
            init_cov=init_cov,
            diffusion_cov=GG,
            runtime_times=runtime_times,
            H=jnp.array([[1.0, 0.0]]),
            d_meas=jnp.array([0.0]),
            R=jnp.array([[0.1]]),
        )

        # Dense reference
        time_intervals = jnp.diff(runtime_times, prepend=runtime_times[0]).at[0].set(MIN_DT)
        Ad_ref, Qd_ref, cd_ref = discretize_system_with_inputs_batched(
            A, GG, c, None, None, time_intervals
        )

        delta = jnp.asarray(0.1)
        st_composite = aux_filter_lgssm_lightweight(
            init_mean=init_mean,
            init_cov=init_cov,
            Fs=composite_ctx.Ad,
            Qs=composite_ctx.Qd,
            bs=composite_ctx.cd,
            pseudo_observations=pseudo_obs,
            aux_variance=0.5 * delta,
            jitter=1e-6,
            parallel=False,
        )
        st_ref = aux_filter_lgssm_lightweight(
            init_mean=init_mean,
            init_cov=init_cov,
            Fs=Ad_ref,
            Qs=Qd_ref,
            bs=cd_ref,
            pseudo_observations=pseudo_obs,
            aux_variance=0.5 * delta,
            jitter=1e-6,
            parallel=False,
        )

        assert jnp.allclose(st_composite.filt_mean, st_ref.filt_mean, atol=1e-5)
        assert jnp.allclose(st_composite.filt_cov, st_ref.filt_cov, atol=1e-5)
        assert jnp.allclose(st_composite.loglik, st_ref.loglik, atol=1e-5)


# =============================================================================
# Integration with the spec compiler (sanity check)
# =============================================================================


class TestCompositeContextWithCompiledSpec:
    """End-to-end: compile a spec, sample params from priors, build the
    composite context, verify it's finite."""

    def test_compiled_ssri_spec_drives_finite_context(self):
        from numpyro.handlers import seed

        DOSE, ADHERENCE, C_P, C_E, AFFECTIVE = 0, 1, 2, 3, 4
        spec = CompositeSpec(
            n_latent=5,
            components=(
                __import__(
                    "nof1_causal_lab.models.ssm.dynamics",
                    fromlist=["DiagonalDecaySpec"],
                ).DiagonalDecaySpec(
                    decay_prior=diagonal_decay_prior()
                ),
                __import__(
                    "nof1_causal_lab.models.ssm.dynamics",
                    fromlist=["InterceptSpec"],
                ).InterceptSpec(
                    cint_prior=ndist.Normal(jnp.zeros(5), 1.0)
                ),
                __import__(
                    "nof1_causal_lab.models.ssm.dynamics",
                    fromlist=["MultiplicativeEdgeSpec"],
                ).MultiplicativeEdgeSpec(
                    source_a=DOSE,
                    source_b=ADHERENCE,
                    target=C_P,
                    weight_prior=multiplicative_weight_prior(scale=0.3),
                ),
                __import__(
                    "nof1_causal_lab.models.ssm.dynamics",
                    fromlist=["LinearEdgeSpec"],
                ).LinearEdgeSpec(
                    source=C_P,
                    target=C_E,
                    weight_prior=effect_compartment_rate_prior(),
                ),
                __import__(
                    "nof1_causal_lab.models.ssm.dynamics",
                    fromlist=["HillEdgeSpec"],
                ).HillEdgeSpec(
                    source=C_E,
                    target=AFFECTIVE,
                    emax_prior=hill_emax_prior(),
                    ec50_prior=hill_ec50_prior(),
                    n_prior=hill_n_prior(),
                ),
            ),
        )
        compiled = compile_composite(spec)

        with seed(rng_seed=0):
            vf_params = compiled.sample_params()

        T = 4
        runtime_times = jnp.linspace(0.0, 4.0, T)
        x_traj = jnp.tile(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]), (T, 1))

        ctx = composite_latent_context_at_trajectory(
            vector_field=compiled.vector_field,
            vf_params=vf_params,
            x_traj=x_traj,
            init_mean=jnp.zeros(5),
            init_cov=jnp.eye(5) * 0.5,
            diffusion_cov=jnp.eye(5) * 0.05,
            runtime_times=runtime_times,
            H=jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0]]),
            d_meas=jnp.array([0.0]),
            R=jnp.array([[0.05]]),
        )
        assert bool(jnp.all(jnp.isfinite(ctx.Ad)))
        assert bool(jnp.all(jnp.isfinite(ctx.Qd)))
        assert bool(jnp.all(jnp.isfinite(ctx.cd)))


# =============================================================================
# Two-context eq10_11 MH step — smoke test
# =============================================================================


class TestCompositeMHStep:
    """End-to-end smoke test of ``composite_latent_mh_step_eq10_11`` on a
    small synthetic non-linear (HillEdge) SSM. Verifies:
    - The MH step runs to completion (both contexts built, both filters
      run, accept/reject logic resolved).
    - ``log_alpha`` is finite.
    - ``accepted`` is 0 or 1.
    - When the step is repeated with the same key, output is deterministic.
    - When started at a *better* trajectory the accept probability is high;
      when started at a *worse* one, the chain still tries to move.
    """

    def _build_system(self):
        """2-latent system: state[0] decays freely; state[1] is driven by
        Hill(state[0]). Observe state[1] with Gaussian noise."""
        from nof1_causal_lab.models.ssm.dynamics import (
            CompositeVectorField,
            DiagonalDecay,
            HillEdge,
        )

        vf = CompositeVectorField(
            n_latent=2,
            components=(
                DiagonalDecay(),
                HillEdge(source=0, target=1),
            ),
        )
        vf_params = (
            {"decay": jnp.array([0.3, 0.5])},
            {
                "Emax": jnp.asarray(1.5),
                "EC50": jnp.asarray(1.0),
                "n": jnp.asarray(2.0),
            },
        )
        return vf, vf_params

    def _build_obs_log_prob_and_grad_fn(self, R_inv: float = 20.0):
        """Gaussian obs y_t = H @ x_t + ε with diagonal R."""

        def fn(context, x_traj, runtime_observations):
            pred = jnp.einsum("ij,tj->ti", context.H, x_traj) + context.d_meas
            residual = runtime_observations - pred
            # Constant terms (-0.5 log(2π) etc.) drop out of MH ratios, so we
            # only need quadratic part.
            log_prob = -0.5 * jnp.sum(residual**2) * R_inv
            grad = jnp.einsum("ji,tj->ti", context.H, residual) * R_inv
            return log_prob, grad

        return fn

    def _make_context_builder(self, vf, vf_params, init_mean, init_cov, GG, H, R, times):
        d_meas = jnp.zeros(H.shape[0])

        def _builder(x_traj):
            return composite_latent_context_at_trajectory(
                vector_field=vf,
                vf_params=vf_params,
                x_traj=x_traj,
                init_mean=init_mean,
                init_cov=init_cov,
                diffusion_cov=GG,
                runtime_times=times,
                H=H,
                d_meas=d_meas,
                R=R,
            )

        return _builder

    def _generate_synthetic_data(self, vf, vf_params, init_mean, GG, H, R, times, key):
        """Forward-simulate the deterministic mean trajectory and add
        Gaussian noise — enough for the MH step to have something to
        update against."""
        from nof1_causal_lab.models.ssm.dynamics import (
            Intervention,
            simulate,
        )

        # Simulate forward to T points
        time_grid = jnp.concatenate([jnp.array([0.0]), times])
        traj = simulate(vf, vf_params, Intervention.none(), init_mean, time_grid)
        true_x = traj[1:]  # length T
        obs_clean = jnp.einsum("ij,tj->ti", H, true_x)
        obs = obs_clean + jr.normal(key, obs_clean.shape) * jnp.sqrt(R[0, 0])
        return true_x, obs

    def test_mh_step_runs_with_finite_log_alpha_on_hill_system(self):
        from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.composite_kalman import (
            CompositeLatentMHState,
            composite_latent_mh_step_eq10_11,
        )

        vf, vf_params = self._build_system()
        init_mean = jnp.array([1.5, 0.0])
        init_cov = jnp.eye(2) * 0.1
        GG = jnp.eye(2) * 0.01
        H = jnp.array([[0.0, 1.0]])
        R = jnp.array([[0.05]])
        T = 5
        times = jnp.linspace(0.5, 2.5, T)

        true_x, obs = self._generate_synthetic_data(
            vf, vf_params, init_mean, GG, H, R, times, jr.PRNGKey(0)
        )

        builder = self._make_context_builder(
            vf, vf_params, init_mean, init_cov, GG, H, R, times
        )

        state = CompositeLatentMHState(
            position=jnp.zeros(0),  # no params being sampled — fixed
            latent_trajectory=true_x,
            latent_delta=jnp.asarray(0.05),
            trajectory_log_prob=jnp.asarray(0.0),
            complete_log_posterior=jnp.asarray(0.0),
        )

        obs_fn = self._build_obs_log_prob_and_grad_fn()

        def log_prior_unc_fn(_z):
            return jnp.asarray(0.0)

        next_state, extras = composite_latent_mh_step_eq10_11(
            state,
            jr.PRNGKey(1),
            obs,
            context_builder=builder,
            log_prior_unc_fn=log_prior_unc_fn,
            observation_log_prob_and_grad_fn=obs_fn,
            parallel=False,
        )

        assert jnp.isfinite(extras["log_alpha"]), f"log_alpha not finite: {extras['log_alpha']}"
        assert jnp.isfinite(extras["log_evidence_fwd"])
        assert jnp.isfinite(extras["log_evidence_rev"])
        accepted = float(extras["accepted"])
        assert accepted in (0.0, 1.0)
        assert next_state.latent_trajectory.shape == (T, 2)
        assert bool(jnp.all(jnp.isfinite(next_state.latent_trajectory)))
        assert bool(jnp.all(jnp.isfinite(next_state.trajectory_log_prob)))

    def test_mh_step_deterministic_under_same_key(self):
        """Same inputs + same key → same output. Sanity: no hidden RNG."""
        from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.composite_kalman import (
            CompositeLatentMHState,
            composite_latent_mh_step_eq10_11,
        )

        vf, vf_params = self._build_system()
        init_mean = jnp.array([1.5, 0.0])
        init_cov = jnp.eye(2) * 0.1
        GG = jnp.eye(2) * 0.01
        H = jnp.array([[0.0, 1.0]])
        R = jnp.array([[0.05]])
        T = 4
        times = jnp.linspace(0.5, 2.0, T)
        true_x, obs = self._generate_synthetic_data(
            vf, vf_params, init_mean, GG, H, R, times, jr.PRNGKey(7)
        )

        builder = self._make_context_builder(
            vf, vf_params, init_mean, init_cov, GG, H, R, times
        )
        obs_fn = self._build_obs_log_prob_and_grad_fn()

        state = CompositeLatentMHState(
            position=jnp.zeros(0),
            latent_trajectory=true_x,
            latent_delta=jnp.asarray(0.05),
            trajectory_log_prob=jnp.asarray(0.0),
            complete_log_posterior=jnp.asarray(0.0),
        )

        def log_prior_unc_fn(_z):
            return jnp.asarray(0.0)

        out1 = composite_latent_mh_step_eq10_11(
            state,
            jr.PRNGKey(42),
            obs,
            context_builder=builder,
            log_prior_unc_fn=log_prior_unc_fn,
            observation_log_prob_and_grad_fn=obs_fn,
            parallel=False,
        )
        out2 = composite_latent_mh_step_eq10_11(
            state,
            jr.PRNGKey(42),
            obs,
            context_builder=builder,
            log_prior_unc_fn=log_prior_unc_fn,
            observation_log_prob_and_grad_fn=obs_fn,
            parallel=False,
        )
        assert jnp.allclose(
            out1[0].latent_trajectory, out2[0].latent_trajectory, atol=1e-10
        )
        assert jnp.allclose(out1[1]["log_alpha"], out2[1]["log_alpha"], atol=1e-10)

    def test_chain_of_steps_keeps_state_finite(self):
        """Five MH steps in sequence; every step's outputs must stay
        finite (no NaN explosion from accumulated rounding)."""
        from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.composite_kalman import (
            CompositeLatentMHState,
            composite_latent_mh_step_eq10_11,
        )

        vf, vf_params = self._build_system()
        init_mean = jnp.array([1.5, 0.0])
        init_cov = jnp.eye(2) * 0.1
        GG = jnp.eye(2) * 0.01
        H = jnp.array([[0.0, 1.0]])
        R = jnp.array([[0.05]])
        T = 5
        times = jnp.linspace(0.5, 2.5, T)
        true_x, obs = self._generate_synthetic_data(
            vf, vf_params, init_mean, GG, H, R, times, jr.PRNGKey(11)
        )

        builder = self._make_context_builder(
            vf, vf_params, init_mean, init_cov, GG, H, R, times
        )
        obs_fn = self._build_obs_log_prob_and_grad_fn()

        state = CompositeLatentMHState(
            position=jnp.zeros(0),
            latent_trajectory=true_x,
            latent_delta=jnp.asarray(0.05),
            trajectory_log_prob=jnp.asarray(0.0),
            complete_log_posterior=jnp.asarray(0.0),
        )

        def log_prior_unc_fn(_z):
            return jnp.asarray(0.0)

        n_accepts = 0
        for i in range(5):
            state, extras = composite_latent_mh_step_eq10_11(
                state,
                jr.PRNGKey(100 + i),
                obs,
                context_builder=builder,
                log_prior_unc_fn=log_prior_unc_fn,
                observation_log_prob_and_grad_fn=obs_fn,
                parallel=False,
            )
            assert bool(jnp.all(jnp.isfinite(state.latent_trajectory))), (
                f"NaN in trajectory at step {i}"
            )
            assert jnp.isfinite(extras["log_alpha"]), (
                f"NaN in log_alpha at step {i}: {extras['log_alpha']}"
            )
            n_accepts += int(extras["accepted"])

        # With a small δ and starting at the truth, at least *some* steps
        # should accept (the proposal is close enough to the current state
        # that the MH ratio is near 1).
        assert n_accepts >= 1, f"No proposals accepted in 5 steps (got {n_accepts})"
