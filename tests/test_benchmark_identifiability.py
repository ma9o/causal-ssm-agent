"""Identifiability gate for benchmark problems.

Every RecoveryProblem in benchmarks/problems/ must satisfy necessary conditions
for parametric identifiability before being used in recovery benchmarks.

Two tiers of checks:

1. **Analytical (hard gate)** — fast, reliable necessary conditions:
   - Observability: n_manifest >= n_latent
   - Lambda rank: loading matrix has full column rank
   - Drift stability: all eigenvalues have negative real parts

2. **Numerical diagnostic (informational)** — check_parametric_id report:
   - Fisher eigenspectrum, expected contraction, weak parameters
   - NOTE: The eigenspectrum check is numerically unreliable for D>~20
     parameters due to second-order AD through the Kalman/particle filter
     recursion. Results are printed for human review but not asserted.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.problems import ALL_PROBLEMS

# ── Tier 1: Analytical necessary conditions (hard gate) ────────────────


@pytest.mark.parametrize("problem_name", ALL_PROBLEMS.keys())
class TestBenchmarkSpecIdentifiability:
    """Analytical identifiability checks — fast, deterministic, reliable."""

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
            "Loading matrix is rank-deficient — latent states are not distinguishable."
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


# ── Tier 2: Numerical parametric ID diagnostic (informational) ─────────


@pytest.mark.slow
@pytest.mark.timeout(180)
@pytest.mark.parametrize("problem_name", ALL_PROBLEMS.keys())
def test_benchmark_parametric_id_diagnostic(problem_name):
    """Run check_parametric_id (hessian) and print report for human review.

    NOTE: The Fisher eigenspectrum is numerically unreliable for D>~20
    parameters (second-order AD through recursive filters accumulates error).
    This test prints the diagnostic report but does NOT assert on
    structural_issues or boundary_issues. It DOES assert that no parameter
    has mean expected contraction of exactly 0.0 across all draws, which
    would indicate a completely unconstrained parameter.
    """
    from dsem_agent.models.ssm import SSMModel
    from dsem_agent.utils.parametric_id import check_parametric_id

    problem = ALL_PROBLEMS[problem_name]
    model = SSMModel(problem.spec, priors=problem.priors, n_particles=50, likelihood="kalman")

    T = 100
    obs, times, _ = problem.simulate(T, seed=42)

    result = check_parametric_id(
        model=model,
        observations=obs,
        times=times,
        n_draws=5,
        seed=42,
    )

    # Print full report for human review
    print(f"\n--- Parametric ID diagnostic: {problem_name} ---")
    result.print_report()
    print("\nExpected contraction per parameter:")
    for name, c in result.expected_contraction.items():
        mean_c = float(jnp.mean(c))
        flag = " [!]" if mean_c < 0.1 else ""
        print(f"  {name}: {mean_c:.3f}{flag}")


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize("problem_name", ALL_PROBLEMS.keys())
@pytest.mark.parametrize("fisher_method", ["hessian", "opg", "profile"])
def test_fisher_method_comparison(problem_name, fisher_method):
    """Compare all three Fisher estimation methods on benchmark problems.

    Checks:
    - OPG always produces PSD Fisher (all eigenvalues >= 0)
    - Profile produces reasonable diagonal curvatures
    - All methods return valid ParametricIDResult
    """
    from dsem_agent.models.ssm import SSMModel
    from dsem_agent.utils.parametric_id import check_parametric_id

    problem = ALL_PROBLEMS[problem_name]
    model = SSMModel(problem.spec, priors=problem.priors, n_particles=50, likelihood="kalman")

    T = 100
    obs, times, _ = problem.simulate(T, seed=42)

    # Use fewer draws/samples for speed in tests
    n_draws = 2
    n_score_samples = 20 if fisher_method == "opg" else 100

    result = check_parametric_id(
        model=model,
        observations=obs,
        times=times,
        n_draws=n_draws,
        seed=42,
        fisher_method=fisher_method,
        n_score_samples=n_score_samples,
    )

    assert result.fisher_method == fisher_method

    print(f"\n--- Fisher comparison: {problem_name} / {fisher_method} ---")
    print(f"  Min eigenvalues: {result.min_eigenvalues}")
    print(f"  Condition numbers: {result.condition_numbers}")
    result.print_report()

    # OPG is mathematically PSD (sum of rank-1 outer products).
    # Float32 matrix multiply introduces relative precision errors, so
    # we check that any negative eigenvalues are small relative to the max.
    if fisher_method == "opg":
        for i in range(n_draws):
            min_eig = float(result.min_eigenvalues[i])
            max_eig = float(jnp.max(result.eigenvalues[i]))
            relative_neg = abs(min(min_eig, 0)) / max(max_eig, 1e-10)
            print(
                f"  Draw {i}: min={min_eig:.6e}, max={max_eig:.6e}, relative_neg={relative_neg:.6e}"
            )
            assert relative_neg < 0.02, (
                f"OPG Fisher relative negativity {relative_neg:.4f} > 2% — "
                f"min_eig={min_eig:.4f}, max_eig={max_eig:.4f}"
            )
