"""State-Space Models (SSM) in NumPyro.

This module implements Bayesian state-space models with:
- Continuous-time dynamics via stochastic differential equations
- Automatic CT→DT discretization for irregular time intervals
- Multiple inference backends: SVI (default), NUTS, NUTS-DA, Hess-MC², PGAS,
  Tempered SMC, Laplace-EM, Structured VI, Differentiable PF
"""

from causal_ssm_agent.models.ssm.discretization import (
    compute_asymptotic_diffusion,
    compute_discrete_cint,
    compute_discrete_diffusion,
    discretize_system,
    discretize_system_batched,
    solve_lyapunov,
)
from causal_ssm_agent.models.ssm.inference import InferenceResult, fit
from causal_ssm_agent.models.ssm.model import (
    NoiseFamily,
    SSMModel,
    SSMPriors,
    SSMSpec,
)

__all__ = [
    # Discretization
    "solve_lyapunov",
    "compute_asymptotic_diffusion",
    "compute_discrete_diffusion",
    "compute_discrete_cint",
    "discretize_system",
    "discretize_system_batched",
    # Model
    "SSMModel",
    "SSMPriors",
    "SSMSpec",
    "NoiseFamily",
    # Inference
    "InferenceResult",
    "fit",
]
