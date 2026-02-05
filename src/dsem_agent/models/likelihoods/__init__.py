"""Likelihood computation backends for state-space models.

Each backend implements compute_log_likelihood() to integrate out latent states
and return p(y|θ) for use in NumPyro via numpyro.factor().

Available backends:
- kalman: Exact inference for linear-Gaussian models (dynamax)
- ukf: Unscented Kalman filter for mildly nonlinear models (dynamax)
- particle: Bootstrap particle filter for general models (cuthbert)
"""

from dsem_agent.models.likelihoods.base import (
    CTParams,
    DTParams,
    LikelihoodBackend,
    MeasurementParams,
)
from dsem_agent.models.likelihoods.kalman import KalmanLikelihood
from dsem_agent.models.likelihoods.particle import ParticleLikelihood
from dsem_agent.models.likelihoods.ukf import UKFLikelihood

__all__ = [
    "CTParams",
    "DTParams",
    "MeasurementParams",
    "LikelihoodBackend",
    "KalmanLikelihood",
    "UKFLikelihood",
    "ParticleLikelihood",
]
