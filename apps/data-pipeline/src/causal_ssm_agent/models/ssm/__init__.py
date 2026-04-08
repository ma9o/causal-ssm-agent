"""State-Space Models (SSM) in NumPyro.

This module implements Bayesian state-space models with:
- Continuous-time dynamics via stochastic differential equations
- Automatic CT→DT discretization for irregular time intervals
- Multiple inference backends: SVI (default), NUTS, NUTS-DA, Hess-MC², PGAS,
  Tempered SMC, Laplace-EM, Structured VI, Differentiable PF
- Automatic reparameterization via AutoReparam
"""

from causal_ssm_agent.models.ssm.autoreparam import AutoReparam, MinimalReparam, Strategy
from causal_ssm_agent.models.ssm.discretization import (
    compute_asymptotic_diffusion,
    compute_discrete_cint,
    compute_discrete_diffusion,
    discretize_system,
    discretize_system_batched,
    solve_lyapunov,
)
from causal_ssm_agent.models.ssm.inference import InferenceMethod, InferenceResult, fit
from causal_ssm_agent.models.ssm.model import (
    DistributionFamily,
    SSMModel,
    SSMPriors,
    SSMSpec,
    full_cholesky_mask,
    full_diagonal_mask,
    full_drift_offdiag_mask,
    full_vector_mask,
    strict_lower_triangle_mask,
    zero_diagonal_mask,
    zero_loading_mask,
    zero_square_mask,
    zero_vector_mask,
)
from causal_ssm_agent.models.ssm.structure_runtime import SSMStructureRuntime

__all__ = [
    # Discretization
    "solve_lyapunov",
    "SSMStructureRuntime",
    "compute_asymptotic_diffusion",
    "compute_discrete_diffusion",
    "compute_discrete_cint",
    "discretize_system",
    "discretize_system_batched",
    # Model
    "SSMModel",
    "SSMPriors",
    "SSMSpec",
    "DistributionFamily",
    "full_cholesky_mask",
    "full_diagonal_mask",
    "full_drift_offdiag_mask",
    "full_vector_mask",
    "strict_lower_triangle_mask",
    "zero_diagonal_mask",
    "zero_loading_mask",
    "zero_square_mask",
    "zero_vector_mask",
    # Inference
    "InferenceMethod",
    "InferenceResult",
    "fit",
    # Reparameterization
    "AutoReparam",
    "MinimalReparam",
    "Strategy",
]
