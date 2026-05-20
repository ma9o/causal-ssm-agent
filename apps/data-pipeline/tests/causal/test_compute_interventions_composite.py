"""Tests for the composite-spec counterpart of ``compute_interventions``.

The linear path returns ``compute_interventions`` results keyed by
``(drift, cint)`` posterior samples. Composite specs produce different
posterior shapes (per-component param tuples), so a sibling function
exists. This test pins the integration:

- ``compute_interventions_composite`` runs on a Hill chain.
- The returned dicts match the linear-path schema (``treatment``,
  ``posterior_draws``, optional ``temporal``).
- Sign of the steady-state effect matches the deterministic prediction
  for a single posterior draw at the true parameters.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as ndist

from nof1_causal_lab.models.ssm.counterfactual import compute_interventions_composite
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    compile_composite,
)


class TestApproximateAbductedStateCompositeEks:
    """EKS-based composite abduction: per-parameter conditional means
    via forward Kalman filter + RTS smoother on the linearised LGSSM.
    Statistically cleaner than the trajectory-marginal estimator when
    the MCMC has not fully converged."""

    def test_eks_abduction_on_hill_chain(self):
        import numpy as np

        from nof1_causal_lab.artifacts.model_spec import (
            DistributionFamily,
            LinkFunction,
        )
        from nof1_causal_lab.models.ssm.counterfactual import (
            approximate_abducted_state_composite_eks,
        )
        from nof1_causal_lab.models.ssm.dynamics import (
            runtime_from_composite,
        )
        from nof1_causal_lab.models.ssm.inference.targets.kernels import (
            build_observation_kernel,
        )

        spec = CompositeSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.1)),
                HillEdgeSpec(
                    source=0,
                    target=1,
                    emax_prior=ndist.LogNormal(0.0, 0.1),
                    ec50_prior=ndist.LogNormal(0.0, 0.1),
                    n_prior=ndist.TruncatedNormal(
                        loc=2.0, scale=0.1, low=1.5, high=2.5
                    ),
                ),
            ),
        )
        compiled = compile_composite(spec)
        H = jnp.array([[0.0, 1.0]])
        d_meas = jnp.array([0.0])
        R = jnp.array([[0.02]])
        kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=np.asarray(R),
        )
        canonical = runtime_from_composite(
            compiled,
            init_mean=jnp.array([1.5, 0.0]),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.005,
            H=H,
            d_meas=d_meas,
            R=R,
            obs_kernel=kernel,
        )
        param_samples = [
            (
                {"decay": jnp.array([0.5, 0.5])},
                {
                    "Emax": jnp.asarray(1.5),
                    "EC50": jnp.asarray(1.0),
                    "n": jnp.asarray(2.0),
                },
            )
            for _ in range(3)
        ]
        T = 5
        runtime_times = jnp.linspace(0.5, 2.5, T)
        observations = jnp.zeros((T, 1))  # synthetic — just exercises the smoother

        abducted = approximate_abducted_state_composite_eks(
            canonical=canonical,
            param_samples=param_samples,
            runtime_times=runtime_times,
            observations=observations,
            evidence_end_idx=2,
        )
        assert abducted["method"] == "composite_eks"
        assert abducted["state"].shape == (2,)
        assert bool(jnp.all(jnp.isfinite(abducted["state"])))

    def test_eks_works_with_beta_observation_via_ekf(self):
        """Round 22 — EKF observation linearisation lets EKS work for
        non-Gaussian observation families. Beta/logit observations are
        handled via local response-function linearisation at the
        predicted mean."""
        import numpy as np

        from nof1_causal_lab.artifacts.model_spec import (
            DistributionFamily,
            LinkFunction,
        )
        from nof1_causal_lab.models.ssm.counterfactual import (
            approximate_abducted_state_composite_eks,
        )
        from nof1_causal_lab.models.ssm.dynamics import (
            runtime_from_composite,
        )
        from nof1_causal_lab.models.ssm.inference.targets.kernels import (
            build_observation_kernel,
        )

        spec = CompositeSpec(
            n_latent=1,
            components=(DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.1)),),
        )
        compiled = compile_composite(spec)
        beta_kernel = build_observation_kernel(
            DistributionFamily.BETA,
            LinkFunction.LOGIT,
            extra_params={"obs_concentration": jnp.asarray([50.0])},
            manifest_cov=np.asarray(jnp.eye(1) * 0.05),
        )
        canonical = runtime_from_composite(
            compiled,
            init_mean=jnp.zeros(1),
            init_cov=jnp.eye(1) * 0.1,
            diffusion_cov=jnp.eye(1) * 0.01,
            H=jnp.array([[1.0]]),
            d_meas=jnp.zeros(1),
            R=jnp.eye(1) * 0.05,
            obs_kernel=beta_kernel,
        )
        # Beta observations live in (0, 1); use mid-range values
        obs = jnp.array([[0.4], [0.5], [0.6]])
        abducted = approximate_abducted_state_composite_eks(
            canonical=canonical,
            param_samples=[({"decay": jnp.array([0.5])},)],
            runtime_times=jnp.linspace(0.0, 1.0, 3),
            observations=obs,
            evidence_end_idx=1,
        )
        assert abducted["method"] == "composite_eks"
        assert abducted["state"].shape == (1,)
        assert bool(jnp.all(jnp.isfinite(abducted["state"])))


class TestApproximateAbductedStateCompositeIeks:
    """IEKS-based composite abduction: iterates EKS re-linearising at
    the smoothed trajectory until convergence. Quality upgrade over the
    single-pass EKS for highly non-linear systems."""

    def test_ieks_converges_on_hill_chain(self):
        import numpy as np

        from nof1_causal_lab.artifacts.model_spec import (
            DistributionFamily,
            LinkFunction,
        )
        from nof1_causal_lab.models.ssm.counterfactual import (
            approximate_abducted_state_composite_ieks,
        )
        from nof1_causal_lab.models.ssm.dynamics import (
            runtime_from_composite,
        )
        from nof1_causal_lab.models.ssm.inference.targets.kernels import (
            build_observation_kernel,
        )

        spec = CompositeSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.1)),
                HillEdgeSpec(
                    source=0,
                    target=1,
                    emax_prior=ndist.LogNormal(0.0, 0.1),
                    ec50_prior=ndist.LogNormal(0.0, 0.1),
                    n_prior=ndist.TruncatedNormal(
                        loc=2.0, scale=0.1, low=1.5, high=2.5
                    ),
                ),
            ),
        )
        compiled = compile_composite(spec)
        H = jnp.array([[0.0, 1.0]])
        d_meas = jnp.array([0.0])
        R = jnp.array([[0.02]])
        kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=np.asarray(R),
        )
        canonical = runtime_from_composite(
            compiled,
            init_mean=jnp.array([1.5, 0.0]),
            init_cov=jnp.eye(2) * 0.1,
            diffusion_cov=jnp.eye(2) * 0.005,
            H=H,
            d_meas=d_meas,
            R=R,
            obs_kernel=kernel,
        )
        param_samples = [
            (
                {"decay": jnp.array([0.5, 0.5])},
                {
                    "Emax": jnp.asarray(1.5),
                    "EC50": jnp.asarray(1.0),
                    "n": jnp.asarray(2.0),
                },
            )
            for _ in range(2)
        ]
        T = 5
        runtime_times = jnp.linspace(0.5, 2.5, T)
        observations = jnp.full((T, 1), 0.5)  # synthetic observation around midpoint

        abducted = approximate_abducted_state_composite_ieks(
            canonical=canonical,
            param_samples=param_samples,
            runtime_times=runtime_times,
            observations=observations,
            evidence_end_idx=2,
            n_iters=4,
            tol=1e-4,
        )
        assert abducted["method"] == "composite_ieks"
        assert abducted["state"].shape == (2,)
        assert bool(jnp.all(jnp.isfinite(abducted["state"])))
        # Each draw used at least 1 iteration; n_iters_per_draw exposed.
        assert len(abducted["n_iters_per_draw"]) == 2
        assert all(1 <= n <= 4 for n in abducted["n_iters_per_draw"])

    def test_ieks_n_iters_zero_falls_back_to_initial_xlin(self):
        """If ``n_iters=0`` (no iterations), IEKS must still return a
        finite result — falling back to the initial linearisation point.
        Defensive check on the iteration loop boundary."""
        import numpy as np

        from nof1_causal_lab.artifacts.model_spec import (
            DistributionFamily,
            LinkFunction,
        )
        from nof1_causal_lab.models.ssm.counterfactual import (
            approximate_abducted_state_composite_ieks,
        )
        from nof1_causal_lab.models.ssm.dynamics import (
            runtime_from_composite,
        )
        from nof1_causal_lab.models.ssm.inference.targets.kernels import (
            build_observation_kernel,
        )

        spec = CompositeSpec(
            n_latent=1, components=(DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.1)),)
        )
        compiled = compile_composite(spec)
        kernel = build_observation_kernel(
            DistributionFamily.GAUSSIAN,
            LinkFunction.IDENTITY,
            manifest_cov=np.asarray(jnp.eye(1) * 0.05),
        )
        canonical = runtime_from_composite(
            compiled,
            init_mean=jnp.array([1.0]),
            init_cov=jnp.eye(1) * 0.1,
            diffusion_cov=jnp.eye(1) * 0.01,
            H=jnp.array([[1.0]]),
            d_meas=jnp.zeros(1),
            R=jnp.eye(1) * 0.05,
            obs_kernel=kernel,
        )
        abducted = approximate_abducted_state_composite_ieks(
            canonical=canonical,
            param_samples=[({"decay": jnp.array([0.5])},)],
            runtime_times=jnp.linspace(0.0, 1.0, 3),
            observations=jnp.zeros((3, 1)),
            evidence_end_idx=1,
            n_iters=0,
        )
        # n_iters=0 means the loop didn't run; abducted state is the
        # initial linearisation point (init_mean broadcast). Shape OK
        # and the loop boundary doesn't crash.
        assert abducted["state"].shape == (1,)
        assert abducted["n_iters_per_draw"] == [0]


class TestApproximateAbductedStateComposite:
    """Composite rung-3 abduction: the trajectory-marginal estimator
    over the composite MCMC's smoothing-posterior samples."""

    def test_returns_mean_of_trajectory_samples_at_evidence_end(self):
        from types import SimpleNamespace

        from nof1_causal_lab.models.ssm.counterfactual import (
            approximate_abducted_state_composite,
        )

        # 3 draws of a T=4 trajectory with 2 latents
        trajectory_samples = jnp.array([
            [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[0.0, 0.0], [1.5, 2.5], [3.5, 4.5], [5.5, 6.5]],
            [[0.0, 0.0], [0.5, 1.5], [2.5, 3.5], [4.5, 5.5]],
        ])
        fake_result = SimpleNamespace(
            diagnostics={"trajectory_samples": trajectory_samples},
        )
        abducted = approximate_abducted_state_composite(fake_result, evidence_end_idx=2)
        assert jnp.allclose(abducted["state"], jnp.array([3.0, 4.0]))
        assert abducted["method"] == "composite_trajectory_marginal"
        assert abducted["warning"] is None

    def test_out_of_bounds_raises(self):
        from types import SimpleNamespace

        import pytest

        from nof1_causal_lab.models.ssm.counterfactual import (
            approximate_abducted_state_composite,
        )

        fake_result = SimpleNamespace(
            diagnostics={"trajectory_samples": jnp.zeros((3, 4, 2))},
        )
        with pytest.raises(ValueError, match="out of bounds"):
            approximate_abducted_state_composite(fake_result, evidence_end_idx=5)


class TestComputeInterventionsComposite:
    def test_runs_on_hill_chain(self):
        """A 2-latent Hill chain: state[0] decays freely, state[1] is
        Hill-driven by state[0]. Treating state[0] should produce a
        positive steady-state shift on state[1]."""
        spec = CompositeSpec(
            n_latent=2,
            components=(
                DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.1)),
                HillEdgeSpec(
                    source=0,
                    target=1,
                    emax_prior=ndist.LogNormal(0.0, 0.1),
                    ec50_prior=ndist.LogNormal(0.0, 0.1),
                    n_prior=ndist.TruncatedNormal(
                        loc=2.0, scale=0.1, low=1.5, high=2.5
                    ),
                ),
            ),
        )
        compiled = compile_composite(spec)
        n_draws = 5
        param_samples = [
            (
                {"decay": jnp.array([0.5, 0.5])},
                {
                    "Emax": jnp.asarray(1.5),
                    "EC50": jnp.asarray(1.0),
                    "n": jnp.asarray(2.0),
                },
            )
            for _ in range(n_draws)
        ]
        results = compute_interventions_composite(
            param_samples=param_samples,
            vector_field=compiled.vector_field,
            treatments=["src"],
            outcome="tgt",
            latent_names=["src", "tgt"],
            shift_size=0.5,
        )
        assert len(results) == 1
        entry = results[0]
        assert entry["treatment"] == "src"
        assert "posterior_draws" in entry
        assert len(entry["posterior_draws"]) == n_draws
        # Increasing 'src' should increase 'tgt' via Hill → positive draws
        assert all(d > 0 for d in entry["posterior_draws"])

    def test_skips_unknown_treatment(self):
        spec = CompositeSpec(
            n_latent=1, components=(DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.1)),)
        )
        compiled = compile_composite(spec)
        param_samples = [({"decay": jnp.array([0.5])},)]
        results = compute_interventions_composite(
            param_samples=param_samples,
            vector_field=compiled.vector_field,
            treatments=["nonexistent"],
            outcome="x",
            latent_names=["x"],
        )
        assert len(results) == 1
        assert results[0] == {"treatment": "nonexistent"}

    def test_empty_param_samples_returns_skeletons(self):
        """Composite path must not crash on an empty posterior."""
        spec = CompositeSpec(
            n_latent=1, components=(DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.1)),)
        )
        compiled = compile_composite(spec)
        results = compute_interventions_composite(
            param_samples=[],
            vector_field=compiled.vector_field,
            treatments=["x"],
            outcome="x",
            latent_names=["x"],
        )
        assert results == [{"treatment": "x"}]
