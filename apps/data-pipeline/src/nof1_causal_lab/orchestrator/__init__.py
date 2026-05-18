"""Orchestrator module for causal model specification.

Two-stage approach following Anderson & Gerbing (1988):
1. Latent Model - theoretical constructs + causal edges (theory-driven)
2. Measurement Model - observed indicators that reflect constructs (data-driven)
"""

from causal_ssm_agent.artifacts import (
    CausalEdge,
    CausalSpec,
    Construct,
    Indicator,
    LatentModel,
    MeasurementModel,
)

from .agents import (
    propose_latent_model,
    propose_measurement_model,
)

__all__ = [
    # Agents
    "propose_latent_model",
    "propose_measurement_model",
    # Schemas - Latent
    "Construct",
    "CausalEdge",
    "LatentModel",
    # Schemas - Measurement
    "Indicator",
    "MeasurementModel",
    # Schemas - Combined
    "CausalSpec",
]
