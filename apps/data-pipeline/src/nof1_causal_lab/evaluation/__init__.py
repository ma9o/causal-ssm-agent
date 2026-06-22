"""Capability-evaluation registry: ``(scenario x stage)`` cells across the pipeline.

See :mod:`~nof1_causal_lab.evaluation.contracts` for the Scenario / StageRunner /
StageScorer protocols and :mod:`~nof1_causal_lab.evaluation.registry` for the
entry catalog. Import :mod:`~nof1_causal_lab.evaluation.seeds` to populate the
default rows (the Stage 2 identification gates).
"""

from __future__ import annotations

from .contracts import (
    Cadence,
    Cost,
    Mode,
    Scenario,
    Score,
    Stage,
    StageRunner,
    StageScorer,
)
from .registry import REGISTRY, RegistryEntry, evaluate, register, select

__all__ = [
    "REGISTRY",
    "Cadence",
    "Cost",
    "Mode",
    "RegistryEntry",
    "Scenario",
    "Score",
    "Stage",
    "StageRunner",
    "StageScorer",
    "evaluate",
    "register",
    "select",
]
