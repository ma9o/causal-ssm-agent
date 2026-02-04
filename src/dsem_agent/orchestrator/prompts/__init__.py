"""Prompts for the orchestrator LLM agents.

Two-stage approach following Anderson & Gerbing (1988):
1. Latent Model - theoretical constructs + causal edges, NO DATA
2. Measurement Model - operationalize constructs into indicators, WITH DATA

Stage 4: GLMM specification proposal by orchestrator.
"""

from . import glmm_proposal, latent_model, measurement_model

__all__ = [
    "glmm_proposal",
    "latent_model",
    "measurement_model",
]
