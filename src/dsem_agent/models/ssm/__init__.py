"""State-Space Models (SSM) in NumPyro.

This module implements hierarchical Bayesian state-space models with:
- Continuous-time dynamics via stochastic differential equations
- Automatic CT→DT discretization for irregular time intervals
- Multiple inference backends (Kalman, UKF, Particle filter)
"""

from dsem_agent.models.ssm.core import cholesky_of_diffusion, ensure_stability
from dsem_agent.models.ssm.discretization import (
    compute_asymptotic_diffusion,
    compute_discrete_cint,
    compute_discrete_diffusion,
    discretize_system,
    matrix_fraction_decomposition,
    solve_lyapunov,
)
from dsem_agent.models.ssm.kalman import kalman_filter, kalman_log_likelihood
from dsem_agent.models.ssm.model import (
    NoiseFamily,
    SSMModel,
    SSMPriors,
    SSMSpec,
    build_ssm_model,
)

__all__ = [
    # Core utilities
    "cholesky_of_diffusion",
    "ensure_stability",
    # Discretization
    "solve_lyapunov",
    "compute_asymptotic_diffusion",
    "compute_discrete_diffusion",
    "compute_discrete_cint",
    "discretize_system",
    "matrix_fraction_decomposition",
    # Kalman filter
    "kalman_filter",
    "kalman_log_likelihood",
    # Model
    "SSMModel",
    "SSMPriors",
    "SSMSpec",
    "NoiseFamily",
    "build_ssm_model",
]
