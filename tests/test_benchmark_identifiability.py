"""Identifiability gate for benchmark problems.

Every RecoveryProblem in benchmarks/problems/ must satisfy necessary conditions
for parametric identifiability before being used in recovery benchmarks.

Two tiers of checks:

1. **Analytical (hard gate)** -- fast, reliable necessary conditions:
   - Observability: n_manifest >= n_latent
   - Lambda rank: loading matrix has full column rank
   - Drift stability: all eigenvalues have negative real parts

2. **Numerical diagnostics (informational)** -- profile likelihood + SBC:
   - Profile likelihood: per-parameter identifiability classification
   - SBC: posterior calibration validation with data-dependent test quantities
   - Results are printed for human review.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.problems import ALL_PROBLEMS

# -- Tier 1: Analytical necessary conditions (hard gate) ------------------


@pytest.mark.parametrize("problem_name", ALL_PROBLEMS.keys())
class TestBenchmarkSpecIdentifiability:
    """Analytical identifiability checks -- fast, deterministic, reliable."""

    def test_observability(self, problem_name):
        """n_manifest >= n_latent (necessary for full-rank observation)."""
        problem = ALL_PROBLEMS[problem_name]
        assert problem.n_manifest >= problem.n_latent, (
            f"'{problem_name}': n_manifest={problem.n_manifest} < n_latent={problem.n_latent}. "
            "Cannot observe all latent states."
        )

    def test_lambda_rank(self, problem_name):
        """Loading matrix must have full column rank (no rotation indeterminacy)."""
        problem = ALL_PROBLEMS[problem_name]
        rank = int(np.linalg.matrix_rank(np.array(problem.true_lambda)))
        assert rank >= problem.n_latent, (
            f"'{problem_name}': lambda rank={rank} < n_latent={problem.n_latent}. "
            "Loading matrix is rank-deficient -- latent states are not distinguishable."
        )

    def test_drift_stability(self, problem_name):
        """All drift eigenvalues must have negative real parts (stationary process)."""
        problem = ALL_PROBLEMS[problem_name]
        eigvals = np.linalg.eigvals(np.array(problem.true_drift))
        max_real = max(e.real for e in eigvals)
        assert max_real < 0, (
            f"'{problem_name}': max Re(eigenvalue)={max_real:.4f} >= 0. "
            f"Drift is not stable. Eigenvalues: {eigvals}"
        )

    def test_manifest_noise_positive(self, problem_name):
        """Measurement noise variances must be strictly positive."""
        problem = ALL_PROBLEMS[problem_name]
        min_var = float(jnp.min(problem.true_mvar_diag))
        assert min_var > 0, (
            f"'{problem_name}': min manifest variance={min_var} <= 0. "
            "Zero measurement noise creates a singular observation model."
        )

    def test_diffusion_positive(self, problem_name):
        """Process noise (diffusion) must be strictly positive."""
        problem = ALL_PROBLEMS[problem_name]
        min_diff = float(jnp.min(problem.true_diff_diag))
        assert min_diff > 0, (
            f"'{problem_name}': min diffusion SD={min_diff} <= 0. "
            "Zero diffusion makes latent dynamics deterministic and unidentifiable."
        )

    def test_t_rule(self, problem_name):
        """T-rule: free params must not exceed available moment conditions."""
        from causal_ssm_agent.utils.parametric_id import check_t_rule

        problem = ALL_PROBLEMS[problem_name]
        # Use T=100 (same as profile likelihood benchmark)
        result = check_t_rule(problem.spec, T=100)
        assert result.satisfies, (
            f"'{problem_name}': t-rule violated — {result.n_free_params} free params "
            f"> {result.n_moments} moment conditions. "
            f"Breakdown: {result.param_counts}"
        )


# -- Tier 2a: Profile likelihood diagnostic (informational) ---------------


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.parametrize("problem_name", ALL_PROBLEMS.keys())
def test_benchmark_profile_likelihood(problem_name):
    """Profile likelihood diagnostic on benchmark problems.

    Profiles estimand parameters (drift off-diagonals) and prints report.
    Asserts: no drift off-diagonal is structurally unidentifiable.
    """
    from causal_ssm_agent.models.ssm import SSMModel
    from causal_ssm_agent.utils.parametric_id import profile_likelihood

    problem = ALL_PROBLEMS[problem_name]
    model = SSMModel(problem.spec, priors=problem.priors, n_particles=50, likelihood="kalman")

    T = 100
    obs, times, _ = problem.simulate(T, seed=42)

    result = profile_likelihood(
        model=model,
        observations=obs,
        times=times,
        profile_params=["drift_offdiag_pop"],
        n_grid=15,
        seed=42,
    )

    print(f"\n--- Profile likelihood: {problem_name} ---")
    result.print_report()

    summary = result.summary()
    for name, cls in summary.items():
        assert cls != "structurally_unidentifiable", (
            f"'{problem_name}': drift param {name} is structurally unidentifiable"
        )


# -- Tier 2b: SBC diagnostic (informational) ------------------------------


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize("problem_name", ALL_PROBLEMS.keys())
def test_benchmark_sbc(problem_name):
    """SBC calibration check on benchmark problems.

    Runs SBC with n_sbc=10 replicates and laplace_em fitting.
    Asserts: chi-squared p-value > 0.01 for all params.
    """
    from causal_ssm_agent.models.ssm import SSMModel
    from causal_ssm_agent.utils.parametric_id import sbc_check

    problem = ALL_PROBLEMS[problem_name]
    model = SSMModel(problem.spec, priors=problem.priors, n_particles=50, likelihood="kalman")

    result = sbc_check(
        model,
        T=80,
        dt=0.5,
        n_sbc=10,
        method="laplace_em",
        seed=42,
    )

    print(f"\n--- SBC: {problem_name} ---")
    result.print_report()

    summary = result.summary()
    for name, info in summary.items():
        assert info["p_value"] > 0.01, (
            f"'{problem_name}': SBC failed for {name} (p={info['p_value']:.4f})"
        )
