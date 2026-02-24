"""Likelihood computation backends for state-space models.

Each backend implements compute_log_likelihood() to integrate out latent states
and return p(y|θ) for use in NumPyro via numpyro.factor().

Available backends:
- kalman: Exact Kalman filter for linear Gaussian SSMs (fastest, no particles)
- particle: Universal backend via differentiable bootstrap PF (cuthbert SMC)
- composed: Two-level RB — exact Kalman on decoupled Gaussian block + PF on rest
"""

from causal_ssm_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    LikelihoodBackend,
    MeasurementParams,
)
from causal_ssm_agent.models.likelihoods.composed import ComposedLikelihood
from causal_ssm_agent.models.likelihoods.emissions import get_emission_fn
from causal_ssm_agent.models.likelihoods.graph_analysis import RBPartition, analyze_first_pass_rb
from causal_ssm_agent.models.likelihoods.kalman import KalmanLikelihood
from causal_ssm_agent.models.likelihoods.particle import ParticleLikelihood

__all__ = [
    "CTParams",
    "InitialStateParams",
    "MeasurementParams",
    "LikelihoodBackend",
    "KalmanLikelihood",
    "ParticleLikelihood",
    "ComposedLikelihood",
    "RBPartition",
    "analyze_first_pass_rb",
    "get_emission_fn",
]
