"""Orchestrator module for causal model specification.

Two-stage approach following Anderson & Gerbing (1988):
1. Latent Model - theoretical constructs + causal edges (theory-driven)
2. Measurement Model - observed indicators that reflect constructs (data-driven)
"""

from .agents import (
    build_dsem_model,
    propose_latent_model,
    propose_measurement_model,
)
from .schemas import (
    CausalEdge,
    Construct,
    DSEMModel,
    Indicator,
    LatentModel,
    MeasurementModel,
)

__all__ = [
    # Agents
    "propose_latent_model",
    "propose_measurement_model",
    "build_dsem_model",
    # Schemas - Latent
    "Construct",
    "CausalEdge",
    "LatentModel",
    # Schemas - Measurement
    "Indicator",
    "MeasurementModel",
    # Schemas - Combined
    "DSEMModel",
]
