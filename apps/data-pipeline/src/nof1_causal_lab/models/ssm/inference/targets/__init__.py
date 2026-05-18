"""Likelihood computation backends for state-space models.

Each backend implements compute_log_likelihood() to integrate out latent states
and return p(y|θ) for use in NumPyro via numpyro.factor().

Available backends:
- kalman: Exact Kalman filter for linear Gaussian SSMs (fastest, no particles)
- particle: Universal backend via differentiable bootstrap PF (cuthbert SMC)
- composed: Two-level RB — exact Kalman on decoupled Gaussian block + PF on rest
"""

from nof1_causal_lab.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    LikelihoodBackend,
    MeasurementParams,
)
from nof1_causal_lab.models.ssm.inference.targets.composed import ComposedLikelihood
from nof1_causal_lab.models.ssm.inference.targets.graph_analysis import (
    RBPartition,
    analyze_first_pass_rb,
)
from nof1_causal_lab.models.ssm.inference.targets.kalman import KalmanLikelihood
from nof1_causal_lab.models.ssm.inference.targets.observation_dispatch import get_emission_fn
from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleLikelihood

__all__ = [
    "CTParams",
    "ComposedLikelihood",
    "InitialStateParams",
    "KalmanLikelihood",
    "LikelihoodBackend",
    "MeasurementParams",
    "ParticleLikelihood",
    "RBPartition",
    "analyze_first_pass_rb",
    "get_emission_fn",
]
