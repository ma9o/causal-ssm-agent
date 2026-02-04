"""Prompts for the orchestrator LLM agents.

Two-stage approach following Anderson & Gerbing (1988):
1. Latent Model - theoretical constructs + causal edges, NO DATA
2. Measurement Model - operationalize constructs into indicators, WITH DATA

Stage 4: Model specification proposal by orchestrator.
"""

from . import latent_model, measurement_model, model_proposal

__all__ = [
    "latent_model",
    "measurement_model",
    "model_proposal",
]
