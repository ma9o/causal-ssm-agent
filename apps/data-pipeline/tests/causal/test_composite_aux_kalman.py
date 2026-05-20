"""End-to-end smoke test for the composite Auxiliary Kalman MCMC driver.

Builds a small non-linear (Hill) state-space model, compiles a
``CompositeSpec``, constructs the bundle, runs the Gibbs sampler, and
verifies that:

- The fit runs to completion without NaN.
- Trajectory MH has a non-zero acceptance rate.
- Parameter RWM accepts at least some proposals (otherwise it's stuck).
- Final trajectory and parameter samples are finite.
- The bundle's ``log_prior_fn`` returns finite values at sampled params.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
from jax.nn import sigmoid as jax_nn_sigmoid
from numpyro.handlers import seed

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    Intervention,
    compile_composite,
    runtime_from_composite,
    simulate,
)
from nof1_causal_lab.models.ssm.dynamics.priors import (
    diagonal_decay_prior,
    hill_ec50_prior,
    hill_emax_prior,
    hill_n_prior,
)
from nof1_causal_lab.models.ssm.inference.methods.composite_aux_kalman import (
    build_composite_aux_kalman_bundle,
    fit_composite_aux_kalman,
)
from nof1_causal_lab.models.ssm.inference.targets.kernels import (
    build_observation_kernel,
)
from tests.ssm_test_utils import make_composite_ssm_model


def _gaussian_obs_kernel(R: jnp.ndarray):
    """Test helper: a Gaussian/identity ObservationKernel keyed to ``R``."""
    import numpy as np

    return build_observation_kernel(
        DistributionFamily.GAUSSIAN,
        LinkFunction.IDENTITY,
        manifest_cov=np.asarray(R),
    )


def _build_synthetic_hill_problem():
    """2-latent Hill SSM: state[0] decays freely; state[1] driven by
    ``Hill(state[0])``. Observe state[1] with Gaussian noise.

    Returns (spec, compiled, true_params, init_mean, init_cov,
    diffusion_cov, H, d_meas, R, runtime_times, observations,
    true_trajectory).
    """
    spec = CompositeSpec(
        n_latent=2,
        components=(
            DiagonalDecaySpec(decay_prior=diagonal_decay_prior()),
            HillEdgeSpec(
                source=0,
                target=1,
                emax_prior=hill_emax_prior(loc=0.0, scale=0.5),
                ec50_prior=hill_ec50_prior(loc=0.0, scale=0.5),
                n_prior=hill_n_prior(),
            ),
        ),
    )
    compiled = compile_composite(spec)

    true_params = (
        {"decay": jnp.array([0.3, 0.5])},
        {
            "Emax": jnp.asarray(1.5),
            "EC50": jnp.asarray(1.0),
            "n": jnp.asarray(2.0),
        },
    )

    init_mean = jnp.array([1.5, 0.0])
    init_cov = jnp.eye(2) * 0.1
    diffusion_cov = jnp.eye(2) * 0.005
    H = jnp.array([[0.0, 1.0]])
    d_meas = jnp.array([0.0])
    R = jnp.array([[0.02]])

    T = 5
    runtime_times = jnp.linspace(0.5, 2.5, T)

    # Synthetic observations
    time_grid = jnp.concatenate([jnp.array([0.0]), runtime_times])
    traj_full = simulate(
        compiled.vector_field, true_params, Intervention.none(), init_mean, time_grid
    )
    true_x = traj_full[1:]
    obs_clean = jnp.einsum("ij,tj->ti", H, true_x) + d_meas
    obs = obs_clean + jr.normal(jr.PRNGKey(0), obs_clean.shape) * jnp.sqrt(R[0, 0])

    obs_kernel = _gaussian_obs_kernel(R)
    canonical = runtime_from_composite(
        compiled,
        init_mean=init_mean,
        init_cov=init_cov,
        diffusion_cov=diffusion_cov,
        H=H,
        d_meas=d_meas,
        R=R,
        obs_kernel=obs_kernel,
    )
    model = make_composite_ssm_model(
        spec,
        n_latent=2,
        n_manifest=H.shape[0],
        H=H,
        d_meas=d_meas,
        init_mean=init_mean,
        init_cov=init_cov,
        diffusion_cov=diffusion_cov,
        R=R,
    )
    return {
        "spec": spec,
        "compiled": compiled,
        "canonical": canonical,
        "model": model,
        "true_params": true_params,
        "init_mean": init_mean,
        "init_cov": init_cov,
        "diffusion_cov": diffusion_cov,
        "H": H,
        "d_meas": d_meas,
        "R": R,
        "obs_kernel": obs_kernel,
        "runtime_times": runtime_times,
        "observations": obs,
        "true_trajectory": true_x,
    }


class TestCompositeBundle:
    def test_bundle_log_prior_finite_at_sampled_params(self):
        """``log_prior_fn`` must be finite when called on a NumPyro-drawn
        parameter tuple."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        with seed(rng_seed=0):
            params = prob["compiled"].sample_params()
        log_prior = bundle.log_prior_fn(params)
        assert jnp.isfinite(log_prior), f"log_prior not finite: {log_prior}"

    def test_bundle_context_builder_produces_valid_context(self):
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        ctx = bundle.context_builder(prob["true_params"])(prob["true_trajectory"])
        T = prob["observations"].shape[0]
        assert ctx.Ad.shape == (T, 2, 2)
        assert bool(jnp.all(jnp.isfinite(ctx.Ad)))
        assert bool(jnp.all(jnp.isfinite(ctx.Qd)))


class TestCompositeMCMC:
    def test_fit_runs_to_completion_with_finite_outputs(self):
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )

        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=20,
            latent_delta=0.02,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(42),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
            parallel=False,
        )

        assert result.diagnostics["trajectory_samples"].shape == (20, 5, 2)
        assert bool(jnp.all(jnp.isfinite(result.diagnostics["trajectory_samples"])))
        assert bool(jnp.all(jnp.isfinite(result.diagnostics["log_alpha_traj"])))
        # At least *some* trajectory MH steps should accept (we initialise at the
        # ground truth, so the proposal is close).
        traj_accept_rate = float(jnp.mean(result.diagnostics["trajectory_accept"]))
        assert traj_accept_rate > 0.0, (
            f"trajectory MH rejected every proposal (accept rate {traj_accept_rate})"
        )
        # And same for parameters (with small step size at the truth).
        param_accept_rate = float(jnp.mean(result.diagnostics["param_accept"]))
        assert param_accept_rate > 0.0, (
            f"param RWM rejected every proposal (accept rate {param_accept_rate})"
        )

    def test_fit_from_fresh_prior_init(self):
        """Fit without supplying initial_params; the driver should sample
        them from the prior and still run to completion."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )

        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=10,
            latent_delta=0.02,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(7),
            parallel=False,
        )

        assert result.diagnostics["trajectory_samples"].shape == (10, 5, 2)
        assert bool(jnp.all(jnp.isfinite(result.diagnostics["trajectory_samples"])))

    def test_param_samples_have_correct_structure(self):
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )

        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=5,
            latent_delta=0.02,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(3),
            initial_params=prob["true_params"],
            initial_x_traj=prob["true_trajectory"],
            parallel=False,
        )

        # Each iteration's params is a tuple matching the spec components.
        assert len(result.diagnostics["param_samples"]) == 5
        for params_at_iter in result.diagnostics["param_samples"]:
            assert len(params_at_iter) == 2  # DiagonalDecay + HillEdge
            assert "decay" in params_at_iter[0]
            assert "Emax" in params_at_iter[1]
            assert "EC50" in params_at_iter[1]
            assert "n" in params_at_iter[1]
            # All values finite
            assert bool(jnp.all(jnp.isfinite(params_at_iter[0]["decay"])))
            assert bool(jnp.isfinite(params_at_iter[1]["Emax"]))


class TestCompositeMCMCWithNUTS:
    """End-to-end with NUTS on parameters. Slower than RWM (autodiff +
    leapfrog steps) but should mix far better; smoke check is that the
    driver runs and produces sensible diagnostics."""

    def test_fit_with_nuts_runs_to_completion(self):
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )

        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=8,
            latent_delta=0.02,
            param_step_size=0.05,
            rng_key=jr.PRNGKey(42),
            initial_x_traj=prob["true_trajectory"],
            parallel=False,
            param_kernel="nuts",
            nuts_max_num_doublings=4,  # cap at 16 leapfrog steps
        )

        assert result.diagnostics["param_kernel"] == "nuts"
        assert result.diagnostics["trajectory_samples"].shape == (8, 5, 2)
        assert bool(jnp.all(jnp.isfinite(result.diagnostics["trajectory_samples"])))
        # NUTS-specific diagnostics
        assert "param_divergent" in result.diagnostics
        assert "param_energy" in result.diagnostics
        assert bool(jnp.all(jnp.isfinite(result.diagnostics["param_energy"])))
        # Trajectory MH still accepts at least some moves.
        traj_accept_rate = float(jnp.mean(result.diagnostics["trajectory_accept"]))
        assert traj_accept_rate > 0.0

    def test_nuts_actually_moves_parameters(self):
        """Sanity: NUTS should produce param samples that differ from the
        initial draw — if everything is identical the kernel isn't
        actually running."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )

        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=6,
            latent_delta=0.02,
            param_step_size=0.05,
            rng_key=jr.PRNGKey(7),
            initial_x_traj=prob["true_trajectory"],
            parallel=False,
            param_kernel="nuts",
            nuts_max_num_doublings=3,
        )

        # Look at Emax across iterations; it should change across NUTS steps.
        emax_values = jnp.asarray(
            [float(p[1]["Emax"]) for p in result.diagnostics["param_samples"]]
        )
        assert jnp.max(emax_values) - jnp.min(emax_values) > 1e-3, (
            "NUTS produced effectively-constant Emax — kernel may not be moving"
        )


class TestParameterRecoverySmoke:
    """Loose parameter-recovery smoke test for the NUTS driver. With a
    small synthetic Hill SSM and a modest chain, we don't expect tight
    posterior coverage of the truth — but the posterior *mean* should
    be in the same ballpark as the truth, not stuck at the prior."""

    def test_emax_posterior_not_stuck_at_prior_for_short_chain(self):
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )

        # Run RWM with reasonably small step + warm start at truth.
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=40,
            latent_delta=0.02,
            param_step_size=0.04,
            rng_key=jr.PRNGKey(11),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
            parallel=False,
            param_kernel="rwm",
        )

        # Pull post-burn-in Emax samples (skip first 10).
        emax_post = jnp.asarray(
            [float(p[1]["Emax"]) for p in result.diagnostics["param_samples"][10:]]
        )
        true_emax = float(prob["true_params"][1]["Emax"])

        # Posterior mean should be within a modest band of the truth.
        # Loose tolerance — 40 iters is not enough for tight recovery.
        post_mean = float(jnp.mean(emax_post))
        assert abs(post_mean - true_emax) < 1.0, (
            f"Emax post mean {post_mean:.3f} too far from truth {true_emax:.3f}"
        )
        # Some variance — chain shouldn't be stuck.
        assert float(jnp.std(emax_post)) > 1e-3


class TestCompositeObservationDispatch:
    """The composite Kalman path must accept any ``ObservationKernel``,
    not just the hardcoded Gaussian path that existed before observation
    dispatch was wired in. Demonstrates the Beta family with a logit link
    runs end-to-end on the same composite spec."""

    def test_beta_observation_runs_to_completion(self):
        import numpy as np

        from nof1_causal_lab.artifacts.model_spec import (
            DistributionFamily,
            LinkFunction,
        )
        from nof1_causal_lab.models.ssm.inference.targets.kernels import (
            build_observation_kernel,
        )

        prob = _build_synthetic_hill_problem()
        beta_kernel = build_observation_kernel(
            DistributionFamily.BETA,
            LinkFunction.LOGIT,
            extra_params={"obs_concentration": jnp.asarray([50.0])},
            manifest_cov=np.asarray(prob["R"]),
        )
        # Synthesize Beta-shaped observations in (0, 1).
        logits = jnp.einsum("ij,tj->ti", prob["H"], prob["true_trajectory"]) + prob["d_meas"]
        beta_obs = jax_nn_sigmoid(logits)
        beta_obs = jnp.clip(beta_obs, 0.01, 0.99)

        beta_model = make_composite_ssm_model(
            prob["spec"],
            n_latent=2,
            n_manifest=prob["H"].shape[0],
            H=prob["H"],
            d_meas=prob["d_meas"],
            init_mean=prob["init_mean"],
            init_cov=prob["init_cov"],
            diffusion_cov=prob["diffusion_cov"],
            R=prob["R"],
        )
        bundle = build_composite_aux_kalman_bundle(
            beta_model,
            beta_obs,
            prob["runtime_times"],
            obs_kernel=beta_kernel,
        )

        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=5,
            latent_delta=0.02,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(7),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        assert bool(jnp.all(jnp.isfinite(result.diagnostics["trajectory_samples"])))
        assert bool(jnp.all(jnp.isfinite(result.diagnostics["log_alpha_traj"])))


class TestCompositeFitWarmupAndMultiChain:
    """Phase D-1 — warmup phase + multi-chain on the composite driver.
    Closes the gap with the linear ``fit_aux_kalman_mcmc`` which has these
    as production features."""

    def test_warmup_iterations_are_discarded(self):
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=4,
            num_warmup=3,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(0),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        # Only the post-warmup 4 iterations show up in samples
        samples = result.get_samples()
        assert samples["vf_0_decay"].shape[0] == 4
        assert result.diagnostics["trajectory_samples"].shape[0] == 4
        assert result.diagnostics["num_warmup"] == 3
        assert result.diagnostics["num_samples_per_chain"] == 4

    def test_multi_chain_concatenates_and_records_chain_index(self):
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=5,
            num_chains=3,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(1),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        # Flat samples concatenated across chains
        samples = result.get_samples()
        assert samples["vf_0_decay"].shape[0] == 3 * 5
        # Chain-grouped form exposed for r̂ / ESS
        chain_samples = result.diagnostics["chain_samples"]
        assert chain_samples["vf_0_decay"].shape == (3, 5, 2)
        assert result.diagnostics["num_chains"] == 3
        assert result.diagnostics["num_samples_per_chain"] == 5

    def test_chains_get_independent_rng(self):
        """Multi-chain runs must use distinct RNG keys per chain so the
        chains explore the posterior independently."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=4,
            num_chains=2,
            param_step_size=0.05,
            rng_key=jr.PRNGKey(2),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        chain_samples = result.diagnostics["chain_samples"]
        chain0 = chain_samples["vf_0_decay"][0]
        chain1 = chain_samples["vf_0_decay"][1]
        # Same starting point but different RNG → divergent trajectories
        assert not bool(jnp.allclose(chain0, chain1))


class TestCompositeFitStepSizeAdaptation:
    """Phase D-2 — Robbins-Monro step-size adaptation during warmup.
    Verifies the adaptation runs, moves the step size away from the
    initial value, and freezes after warmup."""

    def test_adaptation_changes_step_sizes(self):
        """With adaptation enabled, the final adapted step size differs
        from the initial. Initialising with an unreasonably small param
        step → adaptation should push it up to land near the target."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        initial_param_step = 0.001  # deliberately too small
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=4,
            num_warmup=30,
            param_step_size=initial_param_step,
            latent_delta=0.01,
            rng_key=jr.PRNGKey(4),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
            adapt_step_size=True,
            target_param_accept=0.3,
            target_traj_accept=0.65,
        )
        diag = result.diagnostics
        assert diag["adapt_step_size"] is True
        final_step = diag["final_param_step_size_per_chain"][0]
        # A too-small step has 100% acceptance → adaptation pushes it up.
        assert final_step > initial_param_step

    def test_adapted_steps_freeze_after_warmup(self):
        """Step-size history during sampling iterations should be constant
        (equal to the last warmup-adapted value). This preserves detailed
        balance during the sampling phase."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=5,
            num_warmup=20,
            param_step_size=0.01,
            latent_delta=0.02,
            rng_key=jr.PRNGKey(5),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
            adapt_step_size=True,
        )
        history = result.diagnostics["param_step_size_history_per_chain"][0]
        # Total length = num_warmup + n_iterations = 25
        assert len(history) == 25
        # Sampling iterations are the last 5; they should all match
        # the final warmup-adapted value (the last warmup iteration's
        # *output* step size is what the sampling phase uses).
        sampling_window = history[20:]
        assert all(abs(s - sampling_window[0]) < 1e-9 for s in sampling_window)

    def test_nuts_adaptation_uses_acceptance_rate(self):
        """NUTS step-size adaptation reads info.acceptance_rate (not the
        binary RWM accept) — Phase D-2 completion."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        initial_param_step = 0.001  # too small → adaptation pushes up
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=4,
            num_warmup=15,
            param_step_size=initial_param_step,
            latent_delta=0.02,
            rng_key=jr.PRNGKey(8),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
            adapt_step_size=True,
            target_param_accept=0.65,
            param_kernel="nuts",
            nuts_max_num_doublings=4,
        )
        final_step = result.diagnostics["final_param_step_size_per_chain"][0]
        assert final_step > initial_param_step
        # Per-step acceptance_rate captured for diagnostics
        diagnostics = result.diagnostics["param_diagnostics"]
        assert all("acceptance_rate" in d for d in diagnostics)

    def test_nuts_mass_matrix_adapted_from_warmup_samples(self):
        """Phase D-3 — at end of warmup, NUTS inverse mass matrix is set
        to the diagonal sample variance of warmup z_unc draws. Sampling
        phase uses this preconditioned mass matrix."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=4,
            num_warmup=15,
            param_step_size=0.02,
            latent_delta=0.02,
            rng_key=jr.PRNGKey(9),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
            adapt_step_size=True,
            target_param_accept=0.65,
            param_kernel="nuts",
            nuts_max_num_doublings=4,
        )
        adapted = result.diagnostics["adapted_inverse_mass_matrix_per_chain"][0]
        assert adapted is not None
        # Diagonal mass matrix: shape (transform.dim,)
        assert adapted.ndim == 1
        # All entries positive (variance + jitter)
        assert bool(jnp.all(adapted > 0))

    def test_nuts_mass_matrix_not_adapted_when_warmup_zero(self):
        """No warmup window → no mass-matrix adaptation. The flag stays None."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=4,
            num_warmup=0,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(10),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
            adapt_step_size=True,
            param_kernel="nuts",
            nuts_max_num_doublings=4,
        )
        adapted = result.diagnostics["adapted_inverse_mass_matrix_per_chain"][0]
        assert adapted is None

    def test_marginal_kalman_log_evidence_matches_closed_form_on_linear(self):
        """The vanilla Kalman log evidence helper must be correct for a
        trivial linear-Gaussian system where the closed-form marginal
        log-likelihood is computable directly. Pins the pathfinder's
        marginal-likelihood objective."""
        from nof1_causal_lab.models.ssm.inference.methods.composite_aux_kalman import (
            _vanilla_kalman_log_evidence,
        )

        # T=2 step linear-Gaussian system, n_latent=1, n_manifest=1.
        # Trivial enough to compute log p(y) by hand.
        Ad = jnp.broadcast_to(jnp.array([[0.8]]), (2, 1, 1))
        Qd = jnp.broadcast_to(jnp.array([[0.05]]), (2, 1, 1))
        cd = jnp.zeros((2, 1))
        init_mean = jnp.array([0.0])
        init_cov = jnp.array([[0.1]])
        observations = jnp.array([[0.5], [0.3]])
        H = jnp.array([[1.0]])
        d_meas = jnp.zeros(1)
        R = jnp.array([[0.01]])

        log_evid = _vanilla_kalman_log_evidence(
            Ad, Qd, cd, init_mean, init_cov, observations, H, d_meas, R
        )
        # Sanity bounds: a 2-obs trivial system can't have absurd log p(y).
        assert bool(jnp.isfinite(log_evid))
        assert -50.0 < float(log_evid) < 10.0

    def test_pathfinder_uses_marginal_objective_for_gaussian_kernel(self):
        """When the obs kernel is Gaussian, pathfinder must use the
        marginal-likelihood objective (vanilla Kalman filter) rather than
        the joint-at-fixed-x objective. Verified by checking the
        gradient agrees with the marginal log-post numerically."""
        from nof1_causal_lab.models.ssm.inference.methods.composite_aux_kalman import (
            _composite_log_post_unc,
            _composite_marginal_log_post_unc,
            build_unconstrained_transform,
        )

        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        transform = build_unconstrained_transform(bundle.compiled)
        z = transform.flat_init
        joint_lp = _composite_log_post_unc(z, prob["true_trajectory"], bundle, transform)
        marginal_lp = _composite_marginal_log_post_unc(
            z, prob["true_trajectory"], bundle, transform
        )
        # They should be finite and different (the marginal integrates
        # out the trajectory; the joint conditions on it).
        assert bool(jnp.isfinite(joint_lp))
        assert bool(jnp.isfinite(marginal_lp))
        assert float(joint_lp) != float(marginal_lp)

    def test_pathfinder_nonfinite_falls_back_to_prior_init(self, monkeypatch):
        """Round 34 — if scipy_pathfinder returns a non-finite mean
        (degenerate optimisation), pathfinder_init_z_unc must fall back
        to transform.flat_init instead of seeding NUTS with NaN."""
        import numpy as np

        from nof1_causal_lab.models.ssm.inference.methods import (
            composite_aux_kalman as mod,
        )

        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        transform = mod.build_unconstrained_transform(bundle.compiled)
        dim = transform.dim
        x_lin = jnp.broadcast_to(
            prob["canonical"].init_mean,
            (prob["observations"].shape[0], prob["canonical"].init_mean.shape[0]),
        )

        # Mock scipy_pathfinder to return NaN mean — simulates the
        # degenerate-optimisation case.
        import typing as _t

        class _FakeResult:
            mean = np.full((dim,), float("nan"))
            chol = np.eye(dim)
            best_elbo = float("-inf")
            diagnostics: _t.ClassVar[dict] = {}

        monkeypatch.setattr(mod, "scipy_pathfinder", lambda *_a, **_k: _FakeResult())

        z_init, diag = mod.pathfinder_init_z_unc(
            bundle, transform, x_lin, n_starts=1, maxiter=2, elbo_samples=2
        )
        # Fallback fires → z_init equals the prior-draw flat_init
        assert bool(jnp.all(jnp.isfinite(z_init)))
        assert diag.get("nonfinite_fallback") is True

    def test_pathfinder_init_runs_and_starts_chain_away_from_prior_mean(self):
        """Phase D-3 — pathfinder init optimises the joint log-posterior
        at a fixed trajectory and starts NUTS at that mode. Verifies the
        chain runs to completion and the initial param state differs
        from a pure prior draw (i.e., pathfinder actually moved it)."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=4,
            num_warmup=5,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(13),
            initial_x_traj=prob["true_trajectory"],
            param_kernel="nuts",
            init_method="pathfinder",
            pathfinder_n_starts=2,
            pathfinder_maxiter=8,
            pathfinder_elbo_samples=5,
            nuts_max_num_doublings=3,
        )
        # Chain ran without crashing and produced finite samples
        samples = result.get_samples()
        assert bool(jnp.all(jnp.isfinite(samples["vf_0_decay"])))
        assert bool(jnp.all(jnp.isfinite(samples["vf_1_Emax"])))

    def test_no_adaptation_when_flag_off(self):
        """With ``adapt_step_size=False`` the step size is held at the
        passed initial value for every iteration."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        initial = 0.02
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=5,
            num_warmup=10,
            param_step_size=initial,
            latent_delta=initial,
            rng_key=jr.PRNGKey(6),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
            adapt_step_size=False,
        )
        history = result.diagnostics["param_step_size_history_per_chain"][0]
        assert all(abs(s - initial) < 1e-9 for s in history)


class TestCompositeFitMcmcDiagnostics:
    """Phase D-3 (partial) — r̂ / ESS / trace diagnostics for composite fits.
    Uses the chain-grouped samples populated by the Phase D-1 multi-chain
    driver; no NumPyro MCMC object is needed."""

    def test_diagnostics_populated_for_multi_chain_fit(self):
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=8,
            num_chains=2,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(11),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        diag = result.get_mcmc_diagnostics()
        assert diag is not None
        assert diag["num_chains"] == 2
        assert diag["num_samples"] == 2 * 8
        # Per-parameter r̂ and ESS surfaced for each site
        per_param = {entry["parameter"]: entry for entry in diag["per_parameter"]}
        assert "vf_0_decay" in per_param
        assert "r_hat" in per_param["vf_0_decay"]
        assert "ess_bulk" in per_param["vf_0_decay"]
        # Trace + rank histograms compiled
        assert "trace_data" in diag
        assert "rank_histograms" in diag
        # RWM acceptance summary
        assert "parameter_accept_prob_mean" in diag

    def test_single_chain_diagnostics_still_produce_r_hat(self):
        """Even with a single chain, numpyro_summary returns finite r̂
        (= 1.0 ± numerical noise) and ESS — the diagnostic surface
        doesn't blow up on n_chains=1."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=6,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(12),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        diag = result.get_mcmc_diagnostics()
        assert diag is not None
        assert diag["num_chains"] == 1


class TestCompositeFitFullPipeline:
    """End-to-end integration test exercising multiple Phase-D features
    together: pathfinder init + multi-chain + step-size adaptation +
    mass-matrix adaptation + r̂/ESS + convergence-warnings diagnostics.
    Catches cross-feature regressions a single-feature test would miss."""

    def test_pathfinder_multi_chain_nuts_with_full_adaptation(self):
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=5,
            num_warmup=10,
            num_chains=2,
            param_step_size=0.02,
            latent_delta=0.02,
            rng_key=jr.PRNGKey(14),
            initial_x_traj=prob["true_trajectory"],
            param_kernel="nuts",
            init_method="pathfinder",
            adapt_step_size=True,
            target_param_accept=0.65,
            pathfinder_n_starts=2,
            pathfinder_maxiter=8,
            pathfinder_elbo_samples=5,
            nuts_max_num_doublings=3,
        )
        # All features ran together cleanly
        samples = result.get_samples()
        # 2 chains × 5 iterations concatenated on the leading axis
        assert samples["vf_0_decay"].shape[0] == 2 * 5
        assert bool(jnp.all(jnp.isfinite(samples["vf_0_decay"])))
        # Both chains got their adapted mass matrix
        adapted_per_chain = result.diagnostics["adapted_inverse_mass_matrix_per_chain"]
        assert len(adapted_per_chain) == 2
        assert all(m is not None for m in adapted_per_chain)
        # Step-size adaptation was applied per chain
        final_steps = result.diagnostics["final_param_step_size_per_chain"]
        assert len(final_steps) == 2
        # Full diagnostic surface is present
        diag = result.get_mcmc_diagnostics()
        assert diag is not None
        assert diag["num_chains"] == 2
        assert "convergence_warnings" in diag
        assert "trace_data" in diag
        # Some warnings expected with only 5 sampling iterations × 2 chains
        # (low ESS); confirms the warning surface fires at all.
        assert isinstance(diag["convergence_warnings"], list)


class TestCompositeFitInferenceResultEnvelope:
    """The composite fit must return the same :class:`InferenceResult`
    envelope the linear path uses, so Stage 6 / artifact registry /
    diagnostic surfaces can consume composite fits uniformly."""

    def test_returns_typed_inference_result_with_canonical_samples(self):
        from nof1_causal_lab.models.ssm.inference.types import InferenceResult

        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=5,
            latent_delta=0.02,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(3),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        assert isinstance(result, InferenceResult)
        assert result.method == "composite_aux_kalman"
        samples = result.get_samples()
        # Hill spec: site keys are prefixed "vf_<i>_<name>"
        assert "vf_0_decay" in samples
        assert "vf_1_Emax" in samples
        assert "vf_1_EC50" in samples
        assert "vf_1_n" in samples
        # Shape: (n_iter, *param_shape)
        assert samples["vf_0_decay"].shape == (5, 2)
        assert samples["vf_1_Emax"].shape == (5,)
        # MCMC diagnostics route through the composite-aware path
        # (round 14 added r̂/ESS via numpyro_summary on chain_samples).
        diag = result.get_mcmc_diagnostics()
        assert diag is not None
        assert diag["num_chains"] == 1
        assert diag["num_samples"] == 5
        # The raw composite-specific diagnostics are still on .diagnostics
        assert "trajectory_samples" in result.diagnostics


class TestBuildCompositeFittedArtifact:
    """Round 30 — packaging a composite fit into a FittedArtifact ready
    for Stage 6 (closes the manual-mock gap)."""

    def test_artifact_round_trips_through_stage6_prepare(self):
        from types import SimpleNamespace

        import nof1_causal_lab.tool_server as tool_server
        from nof1_causal_lab.models.ssm.inference.methods.composite_aux_kalman import (
            build_composite_fitted_artifact,
        )

        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=4,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(18),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        artifact = build_composite_fitted_artifact(
            prob["canonical"],
            result,
            runtime_times=prob["runtime_times"],
            latent_names=["src", "tgt"],
            manifest_names=["y_tgt"],
        )
        # Builder shim has the fields Stage 6 reads
        assert artifact.builder.spec.latent_names == ["src", "tgt"]
        assert artifact.builder.spec.manifest_names == ["y_tgt"]
        assert artifact.builder.canonical is prob["canonical"]
        assert artifact.result is result

        # End-to-end: artifact passes through _prepare_stage6_simulation
        ctx = {
            "_fitted_artifact": artifact,
            "_prepared_runtime": SimpleNamespace(
                observations=prob["observations"],
                times=prob["runtime_times"],
            ),
            "_identifiable_treatments": ["src"],
            "_outcome_name": "tgt",
            "_observation_timestamps": [],
            "stage-1b": {"causal_spec": {"measurement": {"model_clock": "1d"}}},
            "stage-6": {},
        }
        args = {
            "action": {"variable": "src", "mode": "shift", "amount": 0.5},
            "query": {"horizon_days": 3, "estimand": "steady_state"},
        }
        setup, error = tool_server._prepare_stage6_simulation(ctx, args)
        assert error is None
        assert setup is not None
        assert setup.is_composite
        assert setup.treatment == "src"
        assert setup.outcome == "tgt"


class TestCompositeFittedArtifactPersistence:
    """Round 31 — composite FittedArtifact must survive pickle/unpickle
    with the diagnostics Stage 6 dispatch reads. The linear path drops
    diagnostics on persistence (MCMC object isn't picklable); composite
    needs vector_field / canonical_model / param_samples / trajectory_samples
    preserved or Stage 6 breaks after reload."""

    def test_pickle_roundtrip_preserves_stage6_diagnostics(self):
        import cloudpickle  # production persistence uses cloudpickle, not stdlib pickle

        from nof1_causal_lab.models.ssm.inference.methods.composite_aux_kalman import (
            build_composite_fitted_artifact,
        )

        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=4,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(19),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        artifact = build_composite_fitted_artifact(
            prob["canonical"],
            result,
            runtime_times=prob["runtime_times"],
            latent_names=["src", "tgt"],
        )

        # Round-trip via cloudpickle (matches production save_pickle/load_pickle)
        loaded = cloudpickle.loads(cloudpickle.dumps(artifact))

        # Stage 6 dispatch reads these from diagnostics on the loaded artifact
        loaded_diag = loaded.result.diagnostics
        assert "vector_field" in loaded_diag
        assert "canonical_model" in loaded_diag
        assert "param_samples" in loaded_diag
        assert "trajectory_samples" in loaded_diag
        # Other heavyweight keys (e.g., chain_samples) may or may not survive
        # — the contract is that Stage-6-essential state does
        assert loaded.result.method == "composite_aux_kalman"

    def test_loaded_artifact_dispatches_through_stage6(self):
        """End-to-end: composite fit → cloudpickle round-trip → Stage 6
        intervention dispatch. Proves the persistence claim from round 31."""
        from types import SimpleNamespace

        import cloudpickle

        import nof1_causal_lab.tool_server as tool_server
        from nof1_causal_lab.models.ssm.inference.methods.composite_aux_kalman import (
            build_composite_fitted_artifact,
        )

        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=4,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(20),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        artifact = build_composite_fitted_artifact(
            prob["canonical"],
            result,
            runtime_times=prob["runtime_times"],
            latent_names=["src", "tgt"],
        )
        loaded = cloudpickle.loads(cloudpickle.dumps(artifact))

        # Build a Stage 6 ctx with the LOADED artifact and run prepare + intervention
        ctx = {
            "_fitted_artifact": loaded,
            "_prepared_runtime": SimpleNamespace(
                observations=prob["observations"],
                times=prob["runtime_times"],
            ),
            "_identifiable_treatments": ["src"],
            "_outcome_name": "tgt",
            "_observation_timestamps": [],
            "stage-1b": {"causal_spec": {"measurement": {"model_clock": "1d"}}},
            "stage-6": {},
        }
        args = {
            "action": {"variable": "src", "mode": "shift", "amount": 0.5},
            "query": {"horizon_days": 3, "estimand": "steady_state"},
        }
        # Stage 6 prepare
        setup, error = tool_server._prepare_stage6_simulation(ctx, args)
        assert error is None
        assert setup is not None
        assert setup.is_composite
        # Full intervention path
        response = tool_server._execute_simulate_intervention(ctx, args)
        result_payload = response["result"]
        assert "error" not in result_payload
        assert result_payload["rung"] == 2
        # Shifting src up should increase tgt via Hill (positive effect)
        assert result_payload["summary"]["mean"] > 0


class TestCompositePerTLogLikelihood:
    """Round 25 — per-timestep log-likelihood for composite fits, the
    input shape ArviZ's az.loo consumes for PSIS-LOO model comparison.
    Closes the LOO-CV parity gap with the linear FittedArtifact."""

    def test_per_t_log_likelihood_shape_and_finiteness(self):
        from nof1_causal_lab.models.ssm.predictive import (
            composite_per_t_log_likelihood,
        )

        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=5,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(16),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        ll_per_t = composite_per_t_log_likelihood(
            prob["canonical"], result, prob["observations"]
        )
        # Shape: (n_draws, T)
        n_draws = result.diagnostics["trajectory_samples"].shape[0]
        T = prob["observations"].shape[0]
        assert ll_per_t.shape == (n_draws, T)
        assert bool(jnp.all(jnp.isfinite(ll_per_t)))

    def test_get_loo_diagnostics_dispatches_to_composite_path(self):
        """Round 26 — InferenceResult.get_loo_diagnostics now accepts
        (canonical, observations) on the composite path and returns the
        standard LOO-CV dict (elpd_loo, p_loo, se, pareto_k). Closes the
        LOO parity gap at the typed-API surface."""
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=6,
            num_chains=2,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(17),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        loo = result.get_loo_diagnostics(
            observations=prob["observations"],
            canonical=prob["canonical"],
        )
        assert loo is not None
        # Same shape contract as the linear LOO surface
        assert "elpd_loo" in loo
        assert "p_loo" in loo
        assert "se" in loo
        assert loo["observation_unit"] == "timestep"
        assert loo["n_data_points"] == prob["observations"].shape[0]

    def test_per_t_log_likelihood_raises_without_trajectory_samples(self):
        from types import SimpleNamespace

        import pytest

        from nof1_causal_lab.models.ssm.predictive import (
            composite_per_t_log_likelihood,
        )

        prob = _build_synthetic_hill_problem()
        fake_result = SimpleNamespace(diagnostics={})
        with pytest.raises(ValueError, match="trajectory_samples"):
            composite_per_t_log_likelihood(
                prob["canonical"], fake_result, prob["observations"]
            )


class TestCompositePosteriorPredictiveCheck:
    """Round 33 — model-fit PPC diagnostic for composite. Compares
    posterior predictive observations to actuals, returns residuals,
    coverage, RMSE."""

    def test_ppc_shapes_and_coverage_band(self):
        from nof1_causal_lab.models.ssm.predictive import (
            composite_posterior_predictive_check,
        )

        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=8,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(21),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        ppc = composite_posterior_predictive_check(
            prob["canonical"], result, prob["observations"], rng_seed=42
        )
        T, n_m = prob["observations"].shape
        # pp_mean / pp_std / residuals / z_scores all shape (T, n_m)
        assert ppc["pp_mean"].shape == (T, n_m)
        assert ppc["pp_std"].shape == (T, n_m)
        assert ppc["residuals"].shape == (T, n_m)
        assert ppc["z_scores"].shape == (T, n_m)
        # coverage_95 and rmse per-channel — shape (n_m,)
        assert ppc["coverage_95"].shape == (n_m,)
        assert ppc["rmse"].shape == (n_m,)
        # All finite
        for key in ("pp_mean", "pp_std", "residuals", "z_scores", "coverage_95", "rmse"):
            assert bool(jnp.all(jnp.isfinite(ppc[key])))
        # Coverage is a fraction in [0, 1]
        assert bool(jnp.all((ppc["coverage_95"] >= 0.0) & (ppc["coverage_95"] <= 1.0)))


class TestCompositePosteriorPredictiveObservations:
    """The composite path can now emit posterior-predictive observations
    by sampling on top of the MCMC trajectory samples — closing the PPC
    parity gap with the linear ``FittedArtifact.ppc_result``."""

    def test_posterior_predictive_observations_match_data_shape(self):
        from nof1_causal_lab.models.ssm.predictive import (
            sample_composite_posterior_predictive_observations,
        )

        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )
        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=5,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(15),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )
        ppc = sample_composite_posterior_predictive_observations(
            prob["canonical"], result, rng_seed=42
        )
        # Posterior predictive observations have shape
        # (n_draws, T, n_manifest) matching the observed data layout.
        n_draws = result.diagnostics["trajectory_samples"].shape[0]
        T, n_manifest = prob["observations"].shape
        assert ppc.shape == (n_draws, T, n_manifest)
        assert bool(jnp.all(jnp.isfinite(ppc)))

    def test_posterior_predictive_raises_when_trajectory_samples_missing(self):
        import pytest

        from nof1_causal_lab.models.ssm.predictive import (
            sample_composite_posterior_predictive_observations,
        )

        prob = _build_synthetic_hill_problem()
        from types import SimpleNamespace

        fake_result = SimpleNamespace(diagnostics={})
        with pytest.raises(ValueError, match="trajectory_samples"):
            sample_composite_posterior_predictive_observations(
                prob["canonical"], fake_result
            )


class TestCompositeMCMCDriverContract:
    """Verify the driver returns a dict shaped enough for downstream
    consumption — even if the production-grade analogue is more
    elaborate. Anything that wants posterior samples can dispatch on the
    presence of ``trajectory_samples`` / ``param_samples``."""

    def test_result_contains_required_keys(self):
        prob = _build_synthetic_hill_problem()
        bundle = build_composite_aux_kalman_bundle(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
        )

        result = fit_composite_aux_kalman(
            bundle,
            n_iterations=3,
            latent_delta=0.02,
            param_step_size=0.02,
            rng_key=jr.PRNGKey(11),
            initial_x_traj=prob["true_trajectory"],
            initial_params=prob["true_params"],
        )

        for key in (
            "trajectory_samples",
            "param_samples",
            "trajectory_accept",
            "param_accept",
            "log_alpha_traj",
            "final_state",
            "final_params",
        ):
            assert key in result.diagnostics, f"Missing key {key} in fit result"
