"""Fast structural identifiability checks for benchmark problem definitions.

The stochastic numerical diagnostics for profile likelihood and SBC live in
the dedicated parametric-identifiability tests. This file stays deterministic
and CI-friendly so benchmark fixtures fail quickly when their structure regresses.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.problems import ALL_PROBLEMS

# -- Tier 1: Analytical necessary conditions -------------------------------


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
