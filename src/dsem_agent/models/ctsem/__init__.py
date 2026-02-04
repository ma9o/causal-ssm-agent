"""Continuous Time Structural Equation Modeling (CT-SEM) in NumPyro.

This module implements hierarchical Bayesian CT-SEM following:
- Driver, Oud, Voelkle (2017) JSS paper
- Driver & Voelkle (2018) Psych Methods

The CT-SEM model uses stochastic differential equations to model
continuous-time dynamics, allowing for irregularly spaced observations.
"""

from dsem_agent.models.ctsem.core import (
    compute_asymptotic_diffusion,
    compute_discrete_cint,
    compute_discrete_diffusion,
    discretize_system,
    matrix_fraction_decomposition,
    solve_lyapunov,
)
from dsem_agent.models.ctsem.kalman import kalman_filter, kalman_log_likelihood
from dsem_agent.models.ctsem.model import CTSEMModel, CTSEMPriors, CTSEMSpec

__all__ = [
    # Core utilities
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
    "CTSEMModel",
    "CTSEMPriors",
    "CTSEMSpec",
]
