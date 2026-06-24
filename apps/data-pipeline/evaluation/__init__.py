"""Capability-evaluation registry: ``(scenario x stage)`` cells across the pipeline.

See :mod:`~evaluation.contracts` for the Scenario / StageRunner /
StageScorer protocols and :mod:`~evaluation.registry` for the
entry catalog. Import :mod:`~evaluation.seeds` to populate the
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
