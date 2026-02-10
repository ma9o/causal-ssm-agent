"""Likelihood computation backends for state-space models.

Each backend implements compute_log_likelihood() to integrate out latent states
and return p(y|θ) for use in NumPyro via numpyro.factor().

Available backends:
- kalman: Exact Kalman filter for linear Gaussian SSMs (fastest, no particles)
- particle: Universal backend via differentiable bootstrap PF (cuthbert SMC)
"""

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    LikelihoodBackend,
    MeasurementParams,
)
from dsem_agent.models.likelihoods.kalman import KalmanLikelihood
from dsem_agent.models.likelihoods.particle import ParticleLikelihood

__all__ = [
    "CTParams",
    "InitialStateParams",
    "MeasurementParams",
    "LikelihoodBackend",
    "KalmanLikelihood",
    "ParticleLikelihood",
]
