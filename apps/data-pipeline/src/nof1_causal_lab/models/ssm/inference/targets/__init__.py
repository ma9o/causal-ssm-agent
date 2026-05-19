"""Likelihood computation backends for state-space models.

Each backend implements compute_log_likelihood() to integrate out latent states
and return p(y|θ) for use in NumPyro via numpyro.factor().

Production SSM fitting uses the IEKS/Laplace marginal likelihood path.
"""

from nof1_causal_lab.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    LikelihoodBackend,
    MeasurementParams,
)
from nof1_causal_lab.models.ssm.inference.targets.observation_dispatch import get_emission_fn

__all__ = [
    "CTParams",
    "InitialStateParams",
    "LikelihoodBackend",
    "MeasurementParams",
    "get_emission_fn",
]
