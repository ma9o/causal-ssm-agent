"""State-Space Models (SSM) in NumPyro.

This module implements Bayesian state-space models with:
- Continuous-time dynamics via stochastic differential equations
- Automatic CT→DT discretization for irregular time intervals
- Public inference backends: auxiliary Kalman MCMC, Particle-mGRAD, and
  marginalized Particle Gibbs
- Automatic reparameterization via AutoReparam
"""

from nof1_causal_lab.models.ssm.autoreparam import AutoReparam, MinimalReparam, Strategy
from nof1_causal_lab.models.ssm.discretization import (
    compute_asymptotic_diffusion,
    compute_discrete_cint,
    compute_discrete_diffusion,
    discretize_linear_system_exact,
    discretize_linear_system_exact_batched,
    discretize_system,
    discretize_system_batched,
    discretize_system_with_inputs_batched,
    solve_lyapunov,
)
from nof1_causal_lab.models.ssm.inference import InferenceMethod, InferenceResult, fit
from nof1_causal_lab.models.ssm.model import (
    SSMModel,
    SSMSpec,
)
from nof1_causal_lab.models.ssm.parameter_layout import SSMParameterLayout
from nof1_causal_lab.models.ssm.priors import PriorRegistry, PriorSpec
from nof1_causal_lab.models.ssm.transition_kinds import (
    LATENT_TRANSITION_EULER_MARUYAMA,
    LATENT_TRANSITION_LOCAL_LINEAR_GAUSSIAN,
)

__all__ = [
    # Discretization
    "solve_lyapunov",
    "SSMParameterLayout",
    "compute_asymptotic_diffusion",
    "compute_discrete_diffusion",
    "compute_discrete_cint",
    "discretize_linear_system_exact",
    "discretize_linear_system_exact_batched",
    "discretize_system",
    "discretize_system_batched",
    "discretize_system_with_inputs_batched",
    # Model
    "LATENT_TRANSITION_EULER_MARUYAMA",
    "LATENT_TRANSITION_LOCAL_LINEAR_GAUSSIAN",
    "SSMModel",
    "SSMSpec",
    "PriorRegistry",
    "PriorSpec",
    # Inference
    "InferenceMethod",
    "InferenceResult",
    "fit",
    # Reparameterization
    "AutoReparam",
    "MinimalReparam",
    "Strategy",
]
